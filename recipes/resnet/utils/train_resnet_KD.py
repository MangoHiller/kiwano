#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajout de 6 modes de distillation:
1) MSE sur logits ('mse')
2) pseudo-label ('pseudo_label')
3) MSE sur embeddings ('mse_emb')
4) CosineEmbeddingLoss sur embeddings ('cos_emb')
5) Distillation "Hinton" paramétrée (combinaison CrossEntropy + KL-Div) => 'hinton_kd'
6) Distillation par MSE sur feature maps intermédiaires => 'feature_mse'

Usage exemple:
    python train_resnet_KD.py --musan data/musan --rirs_noises data/rirs_noises \
           --teacher_checkpoint /chemin/teacher_resnet101.ckpt \
           --kd_mode feature_mse \
           /chemin/vers/data \
           /chemin/vers/exp_dir
"""
import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import idr_torch
import hostlist

# Kiwano imports
from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import (
    Augmentation, Noise, Normal, Sometimes, Linear, CMVN, Crop, SpecAugment, Reverb
)
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNetV2, IDRDScheduler, JeffreysLoss, SELayer, BasicBlock, SEBasicBlock


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Récupère le learning rate actuel depuis l'optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    """Ensemble de segments audio pour la vérification du locuteur."""

    def __init__(
        self,
        audio_transforms: List[Augmentation] = None,
        feature_extractor=None,
        feature_transforms: List[Augmentation] = None
    ):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]):
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(
                val for idx, val in enumerate(self.segments.values())
                if idx == segment_id_or_index
            )
        try:
            audio, sample_rate = segment.load_audio()
            if audio.shape[0] == 0:
                logger.warning(f"⚠️ Segment audio vide : {segment_id_or_index}")
                return torch.zeros((1, 81, 350)), -1

            if self.audio_transforms is not None:
                audio, sample_rate = self.audio_transforms(audio, sample_rate)

            feature = None
            if self.feature_extractor is not None:
                feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

            if self.feature_transforms is not None:
                feature = self.feature_transforms(feature)

            return feature, self.labels[segment.spkid]

        except Exception as e:
            logger.error(f"❌ Erreur sur le segment {segment.spkid}: {e}")
            return torch.zeros((1, 81, 350)), -1
    
"""def grad_stats_hook(name):
    def hook(grad):
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            print(f"[Gradient Hook] ❌ NaN/Inf gradient in param {name}")
        else:
            # On peut aussi imprimer les stats si on veut plus de détail
            gmin = grad.min().item()
            gmax = grad.max().item()
            gmean = grad.mean().item()
            print(f"[Gradient Hook] {name} => min:{gmin:.4f}, max:{gmax:.4f}, mean:{gmean:.4f}")
    return hook

def activation_hook(module, input_, output):
    # input_ est un tuple, output est un Tensor (ou tuple)
    if isinstance(input_, tuple):
        input_ = input_[0]
    # Stats input
    if torch.isnan(input_).any() or torch.isinf(input_).any():
        print(f"[Forward Hook] ❌ NaN/Inf in input of {module.__class__.__name__}")
    # Stats output
    if torch.isnan(output).any() or torch.isinf(output).any():
        print(f"[Forward Hook] ❌ NaN/Inf in output of {module.__class__.__name__}")
    # On peut imprimer des stats succinctes
    print(f"[Forward Hook] {module.__class__.__name__} => out.min:{output.min():.4f}, out.max:{output.max():.4f}")

def register_forward_hooks(model):
    # On parcourt quelques modules-clés
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, SELayer, BasicBlock, SEBasicBlock)):
            module.register_forward_hook(activation_hook)"""




if __name__ == "__main__":
    # Impression des variables internes IDR
    print(str(idr_torch.master_addr))
    print(str(idr_torch.master_port))
    print(str(idr_torch.local_rank))

    NODE_ID = os.environ["SLURM_NODEID"]
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    if idr_torch.rank == 0:
        print(
            ">>> Training on ",
            len(idr_torch.hostname),
            " nodes and ",
            idr_torch.size,
            " processes, master node is ",
            MASTER_ADDR
        )
    print(
        f"- Process {idr_torch.rank} corresponds to GPU {idr_torch.local_rank} of node {NODE_ID}"
    )

    # Arguments supplémentaires pour la distillation
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Mode debug ultra-rapide avec dataset réduit.") #A VIRER APRES DEBUG de hinton_KD

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--musan", type=str, default="data/musan/")
    parser.add_argument("--rirs_noises", type=str, default="data/rirs_noises/")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Chemin du checkpoint du Teacher (pour knowledge distillation)")
    parser.add_argument(
        "--kd_mode",
        type=str,
        default=None,
        choices=["mse", "pseudo_label", "mse_emb", "cos_emb", "hinton_kd", "feature_mse", "feature_cos"],
        help=(
            "Mode de Knowledge Distillation à utiliser. "
            "'mse' pour MSE sur logits, "
            "'pseudo_label' pour argmax du teacher, "
            "'mse_emb' pour MSE directe sur embeddings finaux."
            "'cos_emb' pour CosineEmbeddingLoss sur embeddings."
            "'hinton_kd' pour combiner CrossEntropy + KL-Div (Hinton et al. 2015)."
            "'feature_mse' pour les feature maps intermédiaires du Teacher et du Student via une MSE"
            "'feature_cos' pour Cosine Similarity sur feature maps intermédiaires."

        )
    )

    # Spécifiques à Hinton (température + alpha)
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Température pour la distillation Hinton (softmax)."
    )

    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Balance entre la CrossEntropy sur vrais labels et la KL-distillation (Hinton)."
    )

    parser.add_argument("training_corpus", type=str, metavar="training_corpus")
    parser.add_argument("exp_dir", type=str, metavar="exp_dir")

    args = parser.parse_args()

    print("# " + " ".join(sys.argv[0:]))
    print("# Started at " + time.ctime())
    print("#")

    # Chargement d'un checkpoint existant au besoin
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location={"cuda": "cpu"})

    epochs_start = 0
    if checkpoint:
        epochs_start = checkpoint["epochs"]

    # Initialisation du backend distribué
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=idr_torch.rank,
        world_size=idr_torch.size
    )
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")

    # Chargement musan & rirs_noises
    musan = SegmentSet()
    musan.from_dict(Path(args.musan))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    # Préparation des données
    from kiwano.augmentation import Codec, Filtering  # Gardés en commentaires en exemple
    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=Sometimes([
            Noise(musan_music, snr_range=[5, 15]),
            Noise(musan_speech, snr_range=[13, 20]),
            Noise(musan_noise, snr_range=[0, 15]),
            Normal(),
            Reverb(reverb)
        ]),
        feature_extractor=Fbank(),
        feature_transforms=Linear([
            CMVN(),
            Crop(350),
            SpecAugment()
        ]),
    )
    training_data.from_dict(Path(args.training_corpus))
    training_data.describe()

    if args.debug:      #A VIRER APRES DEBUG de hinton_KD
        logger.warning("⚠️ MODE DEBUG ACTIVÉ : réduction à 64 segments.")            #A VIRER APRES DEBUG de hinton_KD
        training_data.segments = dict(list(training_data.segments.items())[:64])    #A VIRER APRES DEBUG de hinton_KD


    total_segments = len(training_data)
    logger.info(f"📊 Vérification des segments : {total_segments} segments chargés.")

    # Sampler pour le multi-GPU distribué
    train_sampler = DistributedSampler(
        training_data, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )
    train_dataloader = DataLoader(
        training_data,
        batch_size=8,
        drop_last=True,
        shuffle=False,
        num_workers=10,
        sampler=train_sampler,
        pin_memory=True
    )

    # Nombre de classes
    num_classes = len(set(training_data.labels.values()))
    logger.info(f"Nombre de classes (locuteurs) détecté : {num_classes}")

    # Instanciation du modèle Student
    resnet_model = ResNetV2(num_classes=num_classes, num_blocks=[3, 4, 6, 3], block=BasicBlock, block_se=SEBasicBlock)
    print(resnet_model, flush=True)
    if checkpoint:
        resnet_model.load_state_dict(checkpoint["model"])
    resnet_model = nn.SyncBatchNorm.convert_sync_batchnorm(resnet_model)
    resnet_model.to(gpu)
    resnet_model = DDP(resnet_model, device_ids=[idr_torch.local_rank])
    #resnet_model = DDP(resnet_model, device_ids=[idr_torch.local_rank], find_unused_parameters=True) #a decommenter pr la distilation MSE embed pr eviter le souci du calcul des logit inutle dans ce mode

    """    # ------ AJOUT DES HOOKS ICI ------
    for n, p in resnet_model.module.named_parameters():
        if p.requires_grad:
            p.register_hook(grad_stats_hook(n))

    register_forward_hooks(resnet_model.module)
    # ----------------------------------"""


    # Optimiseur / Scheduler
    optimizer = torch.optim.SGD([
        {"params": resnet_model.module.preresnet.parameters(), "weight_decay": 0.0001, "lr": 1e-5},
        {"params": resnet_model.module.temporal_pooling.parameters(), "weight_decay": 0.0001, "lr": 1e-5},
        {"params": resnet_model.module.embedding.parameters(), "weight_decay": 0.0001, "lr": 1e-5},
        {"params": resnet_model.module.output.parameters(), "lr": 1e-5}
    ], momentum=0.9)

    if checkpoint and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Perte par défaut (CrossEntropy sur les logits)
    criterion = nn.CrossEntropyLoss()  # JeffreysLoss(coeff1=0.1, coeff2=0.025) si besoin

    scheduler = IDRDScheduler(
        optimizer,
        num_epochs=150,
        initial_lr=0.2,
        warm_up_epoch=5,
        plateau_epoch=15,
        patience=10,
        factor=5,
        amsmloss=0.3
    )
    if checkpoint:
        scheduler.set_epoch(checkpoint["epochs"])

    running_loss = [np.nan] * 500
    # --- GradScaler : Désactivé pour le debug initial ---
    scaler = GradScaler(enabled=True)
    #scaler = GradScaler(enabled=False) # <<< DEBUG >>> Désactiver AMP pour commencer
    #logger.warning("<<< DEBUG >>> AMP désactivé via GradScaler(enabled=False)")

    # Chargement du Teacher si --teacher_checkpoint
    teacher_model = None
    if args.teacher_checkpoint is not None and args.kd_mode is not None:
        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location={"cuda": "cpu"})
        teacher_model = ResNetV2(num_classes=5994, num_blocks=[3, 4, 23, 3], block=BasicBlock, block_se=SEBasicBlock)
        teacher_model.load_state_dict(teacher_ckpt["model"])
        teacher_model = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
        teacher_model.to(gpu)
        #teacher_model = DDP(teacher_model, device_ids=[idr_torch.local_rank]) #on retire Teacher du DDP car il na pas besoin d'etre train il est gelé donc inutile ds le calcul du gradient
        teacher_model.eval()

        # Geler les poids du Teacher
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Teacher chargé depuis {args.teacher_checkpoint} en mode {args.kd_mode}")
    
    # Critères supplémentaires
    mse_criterion = nn.MSELoss()
    cos_criterion = nn.CosineEmbeddingLoss()
    cos_criterion_features = nn.CosineEmbeddingLoss(reduction='mean')

    # Pour la distillation Hinton: on utilisera F.kl_div() pour la KL-Div
    #
    # Note: log( softmax(student/T) ) vs. softmax( teacher/T )
    # On met le Student en log-prob --> cf. doc pytorch pr F.kl_div()
    # On met le Teacher en prob (pas log)
    # n applique la même température T aux deux
    # C’est la forme requise pour utiliser F.kl_div(log_probs_student, probs_teacher) dans PyTorch 
    # et correspond à la formulation mathématique de la KL-Divergence pour la distillation de Hinton.

    def hinton_kl_div_loss(student_logits, teacher_logits, T=1.0, eps=1e-8):
        """Calcule la KL-Divergence pour la distillation Hinton:
        F.kl_div( log_softmax(student/T), softmax(teacher/T) ) * T^2, en batchmean."""
        # Distribution Teacher
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        
        # Clamp pour éviter les probabilités nulles strictes
        teacher_probs = teacher_probs.clamp(min=eps) #A vireer si useless
        # Log-distribution Student
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        student_log_probs = student_log_probs.clamp(min=eps) #A vireer si useless

        # KL-Div
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        # Vérification post-calcul (optionnel mais utile pour debug)
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            print(f"  NaN/Inf détecté DANS hinton_kl_div_loss")
            print(f"  Teacher probs min/max: {teacher_probs.min().item()} / {teacher_probs.max().item()}")
            print(f"  Student log_probs min/max: {student_log_probs.min().item()} / {student_log_probs.max().item()}")
        return kl_div * (T * T)
    
    # --- Activer la détection d'anomalies Autograd ---
    #torch.autograd.set_detect_anomaly(True) 
    #logger.warning("<<< DEBUG >>> torch.autograd.set_detect_anomaly(True) activé")
    # -------------------------------------------------

    print("START TRAINING ...")
    for epoch in range(epochs_start, 150):
        iterations = 0
        train_sampler.set_epoch(epoch)
        # Mise à jour de margin (dans AMSMLoss) avc IDRDScheduler
        resnet_model.module.set_m(scheduler.get_amsmloss())

        torch.distributed.barrier()  # Synchro

        for feats, iden in train_dataloader:

            feats = feats.unsqueeze(1).float().to(gpu)
            iden = iden.to(gpu)

            optimizer.zero_grad()

            # New
            manual_step = False 

            with autocast(enabled=True): #mis a false pr le trainign en mode kd/debug


                # --- Calcul de ce_loss (placé avant la structure if/elif) ---
                student_final_logits = resnet_model(feats, iden)
                ce_loss = criterion(student_final_logits, iden)

                loss = 0.0 # Initialiser loss
                feature_distil_loss = 0.0 # Initialiser
                #backward_handled_internally = False # Flag pour savoir si backward a été fait dans le bloc
                # ---------------------------------------------------------------


                # -----------------------------
                # DISTILLATION PAR PSEUDO-LABEL
                # -----------------------------
                if args.kd_mode == "pseudo_label":
                    # teacher pour générer un pseudo_label
                    with torch.no_grad():
                        t_emb = teacher_model(feats, None)  # Embedding teacher
                        teacher_logits = teacher_model.output(t_emb, None)  # Logits teacher
                        pseudo_label = torch.argmax(teacher_logits, dim=1)

                    preds = resnet_model(feats, pseudo_label)  # Student logit cross-ent sur pseudo-label
                    loss = criterion(preds, pseudo_label)

                # ------------------------------------
                # DISTILLATION PAR MSE SUR LES LOGITS
                # ------------------------------------
                elif args.kd_mode == "mse":
                    # On compare directement les logits du Student et du Teacher
                    with torch.no_grad():
                        t_emb = teacher_model(feats, None)  # Embedding teacher
                        teacher_logits = teacher_model.output(t_emb, None)  # shape B x num_classes

                    student_logits = resnet_model(feats, iden)  # ATTENTION: il faut donner Iden = vrai label 
                                                                                   #pr calculer les logit du student
                    loss = mse_criterion(student_logits, teacher_logits)
                # -----------------------------------------------------
                # DISTILLATION PAR MSE ON EMBEDDINGS (mse_emb), sans classification
                # -----------------------------------------------------
                elif args.kd_mode == "mse_emb":
                    # On compare directement les embeddings finaux Student et Teacher
                    with torch.no_grad():
                        teacher_emb = teacher_model(feats, None)  # Embedding final du Teacher

                    student_emb = resnet_model(feats, None)  # Embedding final du Student
                    loss = mse_criterion(student_emb, teacher_emb)

                # ------------------------------------------------
                # 4) NEW DISTILLATION MODE: COSINE ON EMBEDDINGS (cos_emb)
                # ------------------------------------------------
                elif args.kd_mode == "cos_emb":
                    # On compare les embeddings finaux (Teacher vs Student) via CosineEmbeddingLoss
                    with torch.no_grad():
                        teacher_emb = teacher_model(feats, None)

                    student_emb = resnet_model(feats, None)
                    # CosineEmbeddingLoss prend un "target" => +1 si embeddings "similaires"
                    target = torch.ones(student_emb.size(0), device=gpu)
                    loss = cos_criterion(student_emb, teacher_emb, target)
                
                # ----------------------------------------------------------------
                # 5) NEW DISTILLATION MODE: HINTON KD (combinaison CE + KL-Div)
                # ----------------------------------------------------------------
                elif args.kd_mode == "hinton_kd":
                    manual_step = True
                    # alpha: pondération, T: température
                    alpha = args.alpha
                    T = args.temperature
                    
                    # ---------- 1ère passe  : logits AVEC marge + CE ---------
                    # 1. Obtenir les logits modifiés par la marge du Student (pour CE Loss)
                    student_margin_logits = resnet_model(feats, iden) # Logits AVEC marge

                    # 2. Calculer CE Loss sur les logits AVEC marge
                    ce_loss = criterion(student_margin_logits, iden)

                    # backward partiel (grad sur tout le réseau)
                    scaler.scale(alpha * ce_loss).backward(retain_graph=True)
            
                    # ---------- 2ᵉ passe : logits SANS marge + KLDiv ---------
                    # NB: pas de no_grad -> on VEUT les gradients sur output & embed
                    student_emb = resnet_model(feats, None)          # même réseau, nouvelle passe
                    logits_raw = resnet_model.module.output(student_emb, None)

                    # Perte de distillation (KL) sur les distributions soft du teacher
                    with torch.no_grad():
                        teacher_emb = teacher_model(feats, None)
                        teacher_logits = teacher_model.output(teacher_emb, None) #on a retiré teacher_model.module.output (car l’accès à .module n’est plus nécessaire, car ce n’est plus un wrapper DDP.)

                    kd_loss = hinton_kl_div_loss(logits_raw, teacher_logits, T=T)

                    # backward final
                    scaler.scale((1-alpha) * kd_loss).backward()
                    
                    # step / update
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    loss = alpha * ce_loss + (1-alpha) * kd_loss          # pour le log seulement


                # 6) NEW DISTILLATION MODE: feature_mse (MSE sur feature maps + Cross-Entropy) 
                # ----------------------------------------------------------
                # Note: Cette méthode de distillation est séparée en deux passes avec backward séparés
                elif args.kd_mode == "feature_mse":
                    manual_step = True
                    # ---- hyper-paramètres locaux ------------------------------------
                    block_ids = [3, 4]
                    lambda_list = [0.5, 0.5]  # ex: plus d'importance aux couches profondes

                    # --- 1) Passe 1 : Calcul et Backward pour CE Loss ---
                    # Note: Exécute une passe avant complète juste pour cette partie
                    student_final_logits_for_ce = resnet_model(feats, iden)

                    ce_loss = criterion(student_final_logits_for_ce, iden)

                    # --- Backward pour CE Loss (pondéré par alpha) ---
                    # Pas besoin de scaler.scale() car scaler est désactivé pour debug
                    #scaled_ce_loss = args.alpha * ce_loss

                    # Important: retain_graph=True car on va faire un autre backward pour la loss MSE
                    #scaled_ce_loss.backward(retain_graph=True)
                    # Les gradients pour cette partie sont maintenant accumulés dans param.grad

                    scaler.scale(args.alpha * ce_loss).backward(retain_graph=True) #pour le debug

                    # --- 2) Passe 2 : Calcul et Backward pour Feature MSE Loss ---
                    with torch.no_grad():
                        teacher_feats = teacher_model.extract_intermediate_features(feats, detach_features=True)
                    # Extraction features Student (attachées, nouvelle passe avant partielle)
                    student_feats = resnet_model.module.extract_intermediate_features(feats, detach_features=False)

                    total_mse_loss = 0.0
                    for b, weight in zip(block_ids, lambda_list): # Utilisation de 'weight' ici est OK
                        s_f = student_feats[b]
                        t_f = teacher_feats[b]

                        # Utilisation de mse_criterion défini plus haut
                        block_loss = mse_criterion(s_f, t_f)

                        total_mse_loss += weight * block_loss # Accumulation pondérée

                    # feature_distil_loss est le terme MSE total pondéré
                    feature_distil_loss = total_mse_loss

                    # --- Backward pour Feature MSE Loss (pondéré par 1-alpha) ---
                    # Pas besoin de scaler.scale()
                    #scaled_feature_loss = (1 - args.alpha) * feature_distil_loss

                    scaler.scale((1 - args.alpha) * feature_distil_loss).backward()

                    # Le graphe de la CE Loss n'est plus pertinent, mais le graphe de la feature loss est utilisé ici.
                    # Pas besoin de retain_graph=True car c'est le dernier backward pour cette itération.
                    #scaled_feature_loss.backward()
                    # Les gradients de cette partie sont maintenant ajoutés à ceux de la CE loss dans param.grad

                    # --- Perte combinée (pour logging/affichage uniquement) ---
                    # Calculer la valeur non-scalée pour le log basé sur les composantes
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    loss = args.alpha * ce_loss + (1 - args.alpha) * feature_distil_loss

                    #backward_handled_internally = True # Indiquer que backward est fait

                # --- Fin du bloc feature_mse ---
                
                # 7) --- NEW DISTILLATION MODE: feature_cos (COS feature maps + CE)---
                elif args.kd_mode == "feature_cos":
                    manual_step = True
                    block_ids = [3, 4]
                    lambda_list = [0.5, 0.5]  # ex: plus d'importance aux couches profondes
                    
                    # ---------- 1) Passe 1: passe : CE (logits complets) ----------------------
                    logits_ce  = resnet_model(feats, iden)
                    ce_loss    = criterion(logits_ce, iden)
                    scaler.scale(args.alpha * ce_loss).backward(retain_graph=True)   # gardez le graphe ▼

                    # ---------- 2ᵉ passe : CosineEmbeddingLoss sur features ----------
                    with torch.no_grad():
                        t_feats = teacher_model.extract_intermediate_features(feats, detach_features=True)

                    s_feats = resnet_model.module.extract_intermediate_features(feats, detach_features=False)

                    cos_loss = 0.0
                    for b, w in zip(block_ids, lambda_list):
                        s  = s_feats[b].view(s_feats[b].size(0), -1)   # (N, C·H·W) <-- le view(N, -1) sert pr convertir la map (C, H, W) en (N, C.H.W) pr calculer la dist cos
                        t  = t_feats[b].view(t_feats[b].size(0), -1)

                        target = torch.ones(s.size(0), device=gpu)        # même classe = +1
                        cos_loss = cos_loss + w * cos_criterion_features(s, t, target)

                    scaler.scale((1-args.alpha) * cos_loss).backward()          # backward final

                    # ---------- step / update (identique à feature_mse) --------------
                    scaler.unscale_(optimizer)
                    #torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), 1.0) #test de desactiver la clipping de la norme
                    scaler.step(optimizer)
                    scaler.update()

                    loss = args.alpha * ce_loss + (1-args.alpha) * cos_loss              # log only
                # ----------------------------------------------------------

                else:
                    # Entraînement classique
                    preds = resnet_model(feats, iden)
                    loss = criterion(preds, iden)

            # ---------- FIN DES MODES ---------------------------------------

            # === chemin générique : un seul backward/step ======================
            if not manual_step:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            running_loss.pop(0)
            running_loss.append(loss.item())
            rmean_loss = float(np.nanmean(np.array(running_loss)))

            if iterations % 100 == 0:
                msg = (
                    f"{time.ctime()}: Epoch: [{epoch}/150] "
                    f"({iterations}/{len(train_dataloader)})\tAvgLoss:{rmean_loss:.4f}\t"
                    f"C-Loss:{loss.item():.4f}\tLR:{get_lr(optimizer):.8f}\t"
                    f"Margin:{resnet_model.module.get_m():.4f}"
                )
                print(msg)

            iterations += 1

        # Step du scheduler
        scheduler.step()

        # Sauvegarde du modèle côté rank=0
        #if dist.get_rank() == 0 and not args.debug: #a remplacer pr le debug
        if dist.get_rank() == 0 : #A VIRER APRES DEBUG de hinton_KD virer 'and not args.debug:'
            ckpt = {
                "epochs": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "model": resnet_model.module.state_dict(),
                "name": type(resnet_model.module).__name__,
                "config": resnet_model.module.extra_repr(),
            }
            torch.save(ckpt, os.path.join(args.exp_dir, f"model{epoch}.ckpt"))

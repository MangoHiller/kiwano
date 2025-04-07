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
        "--temperature", type=float, default=4.0,
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
        batch_size=32,
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
    resnet_model = ResNetV2(num_classes=num_classes, num_blocks=[3, 4, 6, 3])
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
    #scaler = GradScaler(enabled=True)
    scaler = GradScaler(enabled=False) # <<< DEBUG >>> Désactiver AMP pour commencer
    logger.warning("<<< DEBUG >>> AMP désactivé via GradScaler(enabled=False)")

    # Chargement du Teacher si --teacher_checkpoint
    teacher_model = None
    if args.teacher_checkpoint is not None and args.kd_mode is not None:
        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location={"cuda": "cpu"})
        teacher_model = ResNetV2(num_classes=5994, num_blocks=[3, 4, 23, 3])
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

    def hinton_kl_div_loss(student_logits, teacher_logits, T=4.0, eps=1e-8):
        """Calcule la KL-Divergence pour la distillation Hinton:
        F.kl_div( log_softmax(student/T), softmax(teacher/T) ) * T^2, en batchmean."""
        # Distribution Teacher
        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        
        # Clamp pour éviter les probabilités nulles strictes
        #teacher_probs = teacher_probs.clamp(min=eps) #A vireer si useless
        # Log-distribution Student
        student_log_probs = F.log_softmax(student_logits / T, dim=1)

        # KL-Div
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

        # Vérification post-calcul (optionnel mais utile pour debug)
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            print(f"  NaN/Inf détecté DANS hinton_kl_div_loss")
            print(f"  Teacher probs min/max: {teacher_probs.min().item()} / {teacher_probs.max().item()}")
            print(f"  Student log_probs min/max: {student_log_probs.min().item()} / {student_log_probs.max().item()}")
        return kl_div * (T * T)
    
    # --- Activer la détection d'anomalies Autograd ---
    torch.autograd.set_detect_anomaly(True) 
    logger.warning("<<< DEBUG >>> torch.autograd.set_detect_anomaly(True) activé")
    # -------------------------------------------------

    print("START TRAINING ...")
    for epoch in range(epochs_start, 150):
        iterations = 0
        train_sampler.set_epoch(epoch)
        # Mise à jour de margin (dans AMSMLoss) avc IDRDScheduler
        resnet_model.module.set_m(scheduler.get_amsmloss())

        torch.distributed.barrier()  # Synchro

        for feats, iden in train_dataloader:
            print(f"\n--- Iteration {iterations} (Epoch {epoch}) ---") # <<< DEBUG >>>

            feats = feats.unsqueeze(1).float().to(gpu)
            iden = iden.to(gpu)

            optimizer.zero_grad()

            # Forward Student
            with autocast(enabled=False): #mis a false pr le trainign en mode kd/debug


                # --- Calcul de ce_loss (placé avant la structure if/elif) ---
                print("<<< DEBUG >>> Calcul de student_final_logits...") # <<< DEBUG >>>
                student_final_logits = resnet_model(feats, iden)
                print(f"<<< DEBUG >>> student_final_logits shape: {student_final_logits.shape}") # <<< DEBUG >>>
                if torch.isnan(student_final_logits).any(): print("<<< DEBUG >>> ❌ NaN DANS student_final_logits!") # <<< DEBUG >>>
                print(f"<<< DEBUG >>> Calcul de ce_loss (valeur avant combinaison)...") # <<< DEBUG >>>
                ce_loss = criterion(student_final_logits, iden)
                print(f"<<< DEBUG >>> ce_loss: {ce_loss.item()}") # <<< DEBUG >>>
                if torch.isnan(ce_loss).any(): print("<<< DEBUG >>> ❌ NaN DANS ce_loss!") # <<< DEBUG >>>

                loss = 0.0 # Initialiser loss
                feature_distil_loss = 0.0 # Initialiser
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
                    # alpha: pondération, T: température
                    alpha = args.alpha
                    T = args.temperature

                    # 1. Obtenir les logits modifiés par la marge du Student (pour CE Loss)
                    student_margin_logits = resnet_model(feats, iden) # Logits AVEC marge

                    # 2. Calculer CE Loss sur les logits AVEC marge
                    ce_loss = criterion(student_margin_logits, iden)
            
                    # 3. Obtenir les logits BRUTS du Student (pour KD Loss)
                    #    Pour éviter une passe complète redondante, on refait juste la dernière étape
                    with torch.no_grad(): # Pas besoin de gradient pour cette partie si on recalcule depuis l'embedding
                        student_emb = resnet_model(feats, None) # Récupère l'embedding 
                    student_raw_logits = resnet_model.module.output(student_emb, None) # Logits BRUTS (sans marge)

                    # Perte de classification standard (CE) sur les vrais labels
                    #ce_loss = criterion(student_logits, iden) # 3. CE Loss sur logits BRUTS

                    # Perte de distillation (KL) sur les distributions soft du teacher
                    with torch.no_grad():
                        teacher_emb = teacher_model(feats, None)
                        teacher_logits = teacher_model.output(teacher_emb, None) #on a retiré teacher_model.module.output (car l’accès à .module n’est plus nécessaire, car ce n’est plus un wrapper DDP.)

                    kd_loss = hinton_kl_div_loss(student_raw_logits, teacher_logits, T=T)


                    # Combinaison
                    loss = alpha * ce_loss + (1 - alpha) * kd_loss


                
                # 6) NEW DISTILLATION MODE: feature_mse (MSE sur feature maps + Cross-Entropy)
                elif args.kd_mode == "feature_mse":
                    # On définit par exemple 4 blocs => [layer1..layer4]
                    # On leur associe des poids lambda pour la pondération
                    print("<<< DEBUG >>> Mode: feature_mse") # <<< DEBUG >>>
                    block_ids = [3, 4]
                    lambda_list = [0.5, 0.5]  # ex: plus d'importance aux couches profondes



                    # Extraire les features Teacher (no_grad)
                    with torch.no_grad():
                        print("<<< DEBUG >>> Extraction Teacher features...") # <<< DEBUG >>>
                        teacher_feats = teacher_model.extract_intermediate_features(feats, detach_features=True)
                        #teacher_feats = {k: v.detach().clone() for k, v in teacher_model.extract_intermediate_features(feats).items()}


                    # Extraire les features Student
                    print("<<< DEBUG >>> Extraction Student features...") # <<< DEBUG >>>
                    student_feats = resnet_model.module.extract_intermediate_features(feats, detach_features=False)
                    #student_feats = {k: v.clone() for k, v in resnet_model.module.extract_intermediate_features(feats).items()}


                    # Calcul MSE + pondération
                    total_mse_loss = 0.0
                    for b, weight in zip(block_ids, lambda_list):
                        print(f"<<< DEBUG >>> Traitement Bloc {b} (poids {weight})...") # <<< DEBUG >>>

                        t_f = teacher_feats[b]  # feature map du bloc b (Teacher)
                        s_f = student_feats[b]  # feature map du bloc b (Student)
                        print(f"<<< DEBUG >>>   s_f shape: {s_f.shape}, t_f shape: {t_f.shape}") # <<< DEBUG >>>
                        if torch.isnan(s_f).any(): print(f"<<< DEBUG >>>   ❌ NaN DANS s_f (bloc {b})!") # <<< DEBUG >>>
                        
                        #Calcul MSE
                        print(f"<<< DEBUG >>>   Calcul block_mse...") # <<< DEBUG >>>
                        block_mse = F.mse_loss(s_f, t_f, reduction="mean")
                        print(f"<<< DEBUG >>>   block_loss (bloc {b}): {block_mse.item()}") # <<< DEBUG >>>
                        if torch.isnan(block_mse).any(): print(f"<<< DEBUG >>>   ❌ NaN DANS block_loss (bloc {b})!") # <<< DEBUG >>>


                        # On multiplie par le poids lambda
                        total_mse_loss = total_mse_loss + weight * block_mse  # Évite la modification in-place
                        print(f"<<< DEBUG >>>   total_feature_loss intermédiaire: {total_mse_loss.item()}") # <<< DEBUG >>>
                        if torch.isnan(total_mse_loss).any(): print(f"<<< DEBUG >>>   ❌ NaN DANS total_feature_loss après bloc {b}!") # <<< DEBUG >>>

                    print(f"<<< DEBUG >>> feature_mse_loss finale: {total_mse_loss.item()}") # <<< DEBUG >>>

                    # Cross-Entropy Loss sur les vraies classes (tâche principale)
                    #ce_loss = F.cross_entropy(resnet_model(feats, iden), iden) #de base ?
                    ce_loss = criterion(resnet_model(feats, iden), iden)

                    # Combinaison des deux pertes
                    loss = args.alpha * ce_loss + (1 - args.alpha) * total_mse_loss
                    print(f"<<< DEBUG >>> Loss combinée: {loss.item()} (alpha={args.alpha}, ce={ce_loss.item()}, feat={total_mse_loss.item()})") # <<< DEBUG >>>
                    if torch.isnan(loss).any(): print("<<< DEBUG >>> ❌ NaN DANS loss finale!") # <<< DEBUG >>>
                
                # 7) --- NEW DISTILLATION MODE: feature_cos (COS feature maps + CE)---
                elif args.kd_mode == "feature_cos":

                    block_ids = [3, 4]
                    lambda_list = [0.5, 0.5]  # ex: plus d'importance aux couches profondes

                    # Extraction features Teacher (détachées)
                    with torch.no_grad():
                        teacher_inter_feats = teacher_model.extract_intermediate_features(feats, detach_features=True)
                    # Extraction features Student (attachées)
                    student_inter_feats = resnet_model.module.extract_intermediate_features(feats, detach_features=False)

                    total_feature_loss = 0.0
                    for b, weight in zip(block_ids, lambda_list):
                        s_f = student_inter_feats[b]
                        t_f = teacher_inter_feats[b]

                        # Vérif dimensions spatiales
                        if s_f.shape[2:] != t_f.shape[2:]:
                            logger.warning(f"Différence spatiale bloc {b}. S:{s_f.shape} T:{t_f.shape}. Aplatissement nécessaire.")
                            # Potentiellement ajouter un AdaptiveAvgPool2d ici si nécessaire avant flatten
                            # s_f = F.adaptive_avg_pool2d(s_f, (1, 1))
                            # t_f = F.adaptive_avg_pool2d(t_f, (1, 1))

                        # Aplatir pour CosineEmbeddingLoss (N, C, H, W) -> (N, C*H*W) ou (N*H*W, C)
                        # (N, C*H*W) est plus simple pour CosineEmbeddingLoss qui attend (N, D)
                        s_f_flat = s_f.view(s_f.size(0), -1)
                        t_f_flat = t_f.view(t_f.size(0), -1)

                        # Target = 1 pour maximiser similarité
                        target = torch.ones(s_f_flat.size(0), device=gpu)
                        block_loss = cos_criterion_features(s_f_flat, t_f_flat, target)

                        # Pondération et accumulation
                        #weight = args.feature_weights[b]
                        total_feature_loss = total_feature_loss + weight * block_loss
                    
                    # Cross-Entropy Loss sur les vraies classes (tâche principale)
                    student_final_logits_for_ce = resnet_model(feats, iden)
                    ce_loss = criterion(student_final_logits_for_ce, iden)

                    feature_distil_loss = total_feature_loss
                    loss = args.alpha * ce_loss + (1 - args.alpha) * feature_distil_loss
                # ----------------------------------------------------------

                else:
                    # Entraînement classique
                    preds = resnet_model(feats, iden)
                    loss = criterion(preds, iden)

            print("<<< DEBUG >>> Appel de loss.backward()...")
            scaler.scale(loss).backward()
            print("<<< DEBUG >>> loss.backward() terminé.") 
            
            print("<<< DEBUG >>> Vérification des gradients...")
            found_nan_grad = False
            max_grad_norm = 0.0
            for name, param in resnet_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"<<< DEBUG >>> ❌ NaN DÉTECTÉ DANS GRADIENT de {name}") # <<< DEBUG >>>
                        found_nan_grad = True
                    else:
                         # Calculer la norme L2 du gradient pour ce paramètre
                         param_norm = param.grad.data.norm(2).item()
                         max_grad_norm = max(max_grad_norm, param_norm)
                         # Afficher si la norme est très grande (ex > 1000)
                         if param_norm > 1000:
                              print(f"<<< DEBUG >>>   Gradient élevé pour {name}: Norme L2 = {param_norm:.4f}") # <<< DEBUG >>>
            print(f"<<< DEBUG >>> Vérification gradients terminée. NaN trouvé: {found_nan_grad}. Norme Max observée: {max_grad_norm:.4f}") # <<< DEBUG >>>
                  
            #for name, param in resnet_model.named_parameters():
            #    if param.grad is not None and torch.isnan(param.grad).any():
            #        print(f"❌ NaN detected in gradients of {name}", flush=True)
            scaler.unscale_(optimizer)
            #torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), 1.0) # A VIRER obj cliper les gradient
            print("<<< DEBUG >>> Appel de clip_grad_norm_...") # <<< DEBUG >>>
            torch.nn.utils.clip_grad_norm_(resnet_model.parameters(), max_norm=1.0)  # pzreil a retirer si pas de HInton KD
            print("<<< DEBUG >>> Appel de optimizer.step()...") # <<< DEBUG >>>
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

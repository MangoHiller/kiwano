#!/usr/bin/env python3
# kd_train.py
# -*- coding: utf-8 -*-

"""
Script principal pour la Knowledge Distillation d'un Student (EfficientNet-B0)
à partir d'un Teacher (EfficientNet-B2) pour la vérification du locuteur.

 - Utilise la même approche de chargement des données que train_efficientnet.py
   (mêmes audio_transforms, feature_extractor, feature_transforms).
 - Combine CrossEntropy + KL-Div + (optionnel) Cosine Loss.
 - Gèle le Teacher, fine-tune le Student.
 - Sauvegarde les checkpoints du Student.

Usage (exemple) :
  python kd_train.py \
    --teacher_ckpt teacher_effB2.ckpt \
    --student_ckpt student_effB0_pretrained.ckpt \
    --musan data/musan \
    --rirs_noises data/rirs_noises \
    --local_rank 0 \
    data/voxceleb2_segments.json \
    exp_kd

"""

import os
import time
import argparse
import logging
from pathlib import Path

import hostlist
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.cuda.amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from kiwano.dataset import SegmentSet
from kiwano.augmentation import Noise, Normal, Reverb, Sometimes, Linear, CMVN, Crop, SpecAugment
from kiwano.features import Fbank
from kiwano.utils import Pathlike

# Import 
from dataset import SpeakerTrainingSegmentSet, To3Channels
from kd_teacher_student import TeacherStudentWrapper
from kd_loss import KDLoss, CosineEmbedLoss
from utils import save_checkpoint, AverageMeter

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KD Training (EfficientNet-B0) depuis un Teacher EfficientNet-B2")
    
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Rang local pour l'entraînement distribué (NCCL).")
    parser.add_argument("--musan", type=str, default="data/musan/",
                        help="Chemin vers le dossier musan (bruits).")
    parser.add_argument("--rirs_noises", type=str, default="data/rirs_noises/",
                        help="Chemin vers le dossier rirs_noises.")
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Checkpoint du Teacher (EfficientNet-B2).")
    parser.add_argument("--student_ckpt", type=str, default=None,
                        help="Checkpoint du Student (EfficientNet-B0), si pré-entraîné.")
    
    # Hyperparamètres KD
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Poids de la KL-Div vs CrossEntropy. alpha \u2208 [0,1]")
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="Température pour la distillation (logits).")
    parser.add_argument("--use_cosine_loss", action="store_true",
                        help="Activer la Cosine Loss sur les embeddings Teacher/Student.")
    parser.add_argument("--cosine_weight", type=float, default=0.5,
                        help="Poids de la Cosine Loss si use_cosine_loss.")
    
    # Entraînement
    parser.add_argument("--epochs", type=int, default=30,
                        help="Nombre d'époques KD.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Taille du batch.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    
    parser.add_argument("training_corpus", type=str,
                        help="Chemin vers le corpus d'entraînement (fichier .json) pour SpeakerTrainingSegmentSet.")
    parser.add_argument("exp_dir", type=str,
                        help="Dossier de sortie (sauvegarde checkpoints Student).")
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)

    # Initialisation logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Lancement du script KD (Teacher -> Student)")

    # SLURM / DDP
    rank = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    hostnames = hostlist.expand_hostlist(os.environ.get("SLURM_JOB_NODELIST", "localhost"))
    os.environ["MASTER_ADDR"] = hostnames[0]
    master_port = os.environ.get("MASTER_PORT", "12345")
    os.environ["MASTER_PORT"] = master_port

    print(f"[Process {rank}] MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"[Process {rank}] MASTER_PORT: {master_port}")
    print(f"[Process {rank}] RANK: {rank}")
    print(f"[Process {rank}] LOCAL_RANK: {local_rank}")
    print(f"[Process {rank}] WORLD_SIZE: {world_size}")

    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[Process {rank}] Device: {device}")

    # Charger musan
    musan = SegmentSet()
    musan.from_dict(Path(args.musan))
    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    # Charger reverb
    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    # ------------------------------
    #  dataset & dataloader
    # ------------------------------
    audio_transforms = Sometimes([
        Noise(musan_music, snr_range=[5, 15]),
        Noise(musan_speech, snr_range=[13, 20]),
        Noise(musan_noise, snr_range=[0, 15]),
        Normal(),
        Reverb(reverb),
    ])
    feature_extractor = Fbank()
    feature_transforms = Linear([
        CMVN(),
        Crop(350),
        SpecAugment(),
        # To3Channels(),
    ])

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=audio_transforms,
        feature_extractor=feature_extractor,
        feature_transforms=feature_transforms
    )
    training_data.from_dict(Path(args.training_corpus))

    # Sampler distribué
    train_sampler = DistributedSampler(
        training_data,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    # DataLoader
    train_loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=4,
        sampler=train_sampler,
        pin_memory=True
    )

    num_classes = len(set(training_data.labels.values()))
    logger.info(f"Nombre de classes (locuteurs) détecté : {num_classes}")

    # ------------------------------
    # Teacher & Student
    # ------------------------------
    # On encapsule tout dans TeacherStudentWrapper
    from kd_teacher_student import TeacherStudentWrapper
    model_wrapper = TeacherStudentWrapper(
        num_classes=num_classes,
        teacher_ckpt=args.teacher_ckpt,
        student_ckpt=args.student_ckpt,
        teacher_model_name="efficientnet-b2",
        student_model_name="efficientnet-b0"
    )
    teacher = model_wrapper.teacher
    student = model_wrapper.student

    # Geler le teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Envoyer sur le device
    teacher.to(device)
    student.to(device)

    # DDP
    teacher = DDP(teacher, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    student = DDP(student, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    # ------------------------------
    # Définition des pertes
    # ------------------------------
    ce_loss = nn.CrossEntropyLoss()
    from kd_loss import KDLoss, CosineEmbedLoss
    kd_loss = KDLoss(temperature=args.temperature)
    cos_loss_module = None
    if args.use_cosine_loss:
        cos_loss_module = CosineEmbedLoss()

    # ------------------------------
    # Optimiseur
    # ------------------------------
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)

    # Amp GradScaler
    scaler = torch.cuda.amp.GradScaler()

    # Boucle d'entraînement
    # ---------------------
    best_loss = float('inf')
    for epoch in range(args.epochs):
        student.train()
        train_sampler.set_epoch(epoch)

        losses_ce = AverageMeter()
        losses_kd = AverageMeter()
        losses_cos = AverageMeter()
        losses_total = AverageMeter()

        for batch_idx, (feats, iden) in enumerate(train_loader):
            # feats : [B, time=350, freq=81] 
            feats = feats.unsqueeze(1).to(device, dtype=torch.float32)  # => [B,1,350,81]
            iden = iden.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # 1) Teacher forward
                with torch.no_grad():
                    # logits teacher => (B, num_classes)
                    teacher_logits = teacher(feats, iden=iden)
                    # embeddings teacher => (B, embed_dim)
                    teacher_emb = teacher(feats, iden=None)

                # 2) Student forward
                student_logits = student(feats, iden=iden)
                student_emb = student(feats, iden=None)

                # 3) CrossEntropy
                loss_ce = ce_loss(student_logits, iden)

                # 4) Distillation (logits)
                loss_kd = kd_loss(student_logits, teacher_logits)

                # 5) Cosine embedding
                loss_cos = 0.0
                if cos_loss_module is not None:
                    # cos_label=1 => on veut proches
                    cos_label = torch.ones(student_emb.size(0)).to(device)
                    loss_cos = cos_loss_module(student_emb, teacher_emb, cos_label)

                # 6) Combinaison
                total_loss = args.alpha * loss_kd + (1 - args.alpha) * loss_ce
                if cos_loss_module is not None:
                    total_loss += args.cosine_weight * loss_cos

            # backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Mises à jour logs
            bs = feats.size(0)
            losses_ce.update(loss_ce.item(), bs)
            losses_kd.update(loss_kd.item(), bs)
            if cos_loss_module is not None:
                losses_cos.update(loss_cos.item(), bs)
            losses_total.update(total_loss.item(), bs)

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] - Step {batch_idx+1}/{len(train_loader)} | "
                      f"CE: {losses_ce.avg:.4f} | KD: {losses_kd.avg:.4f} "
                      + (f"| Cos: {losses_cos.avg:.4f} " if cos_loss_module else "")
                      + f"| Total: {losses_total.avg:.4f}")

        epoch_loss = losses_total.avg
        print(f"====> Fin epoch {epoch+1}/{args.epochs}, Loss = {epoch_loss:.4f}")

        # Sauvegarde checkpoint si meilleur
        if epoch_loss < best_loss and dist.get_rank() == 0:
            best_loss = epoch_loss
            ckpt_path = os.path.join(args.exp_dir, "best_student.ckpt")
            save_checkpoint(student.module, optimizer, epoch+1, ckpt_path)
            print("  * Meilleure loss atteinte, checkpoint sauvegardé.")

    print("Fin de l'entraînement KD.")


if __name__ == "__main__":
    main()

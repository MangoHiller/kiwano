#!/usr/bin/env python3
"""
train_efficientnet.py

Script d'entraînement d'un modèle EfficientNetV2 pour la vérification du locuteur,
inspiré de la structure de train_resnet.py.

Exemple d'utilisation (en job SLURM HPC) :
  python train_efficientnet.py \
    --local_rank 0 \
    --musan data/musan \
    --rirs_noises data/rirs_noises \
    --checkpoint /chemin/vers/mon_checkpoint.ckpt \
    data/voxceleb1 \
    exp_effnet

Note : ce script attend une configuration SLURM (variables d'environnement)
       pour le DistributedDataParallel. Ajustez selon vos besoins.
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path
from typing import Union, List

import hostlist
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from kiwano.dataset import Segment, SegmentSet
from kiwano.augmentation import (
    Augmentation, Noise, Codec, Filtering, Normal, Sometimes,
    Linear, CMVN, Crop, SpecAugment, Reverb
)
from kiwano.features import Fbank
from kiwano.model import EfficientNetV2, IDRDScheduler, JeffreysLoss
from kiwano.model import AMSMLoss  # si besoin
from kiwano.utils import Pathlike

import socket

logger = logging.getLogger(__name__)

def find_free_port():
    """Trouve un port TCP libre sur la machine"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Laisse le système choisir un port libre
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # Récupère le numéro du port


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Récupère le learning rate du premier param_group de l'optimiseur.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]

# Définition d'une classe de transformation pour convertir les tenseurs en 3 canaux
class To3Channels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (time, freq) = (350, 81) après Crop, SpecAugment, etc.
        Retourne un tenseur (3, freq, time) pour coller au format EfficientNet (C,H,W).
        """
        # => on insère la dimension canal et on répète
        x = x.unsqueeze(0).repeat(3, 1, 1)  # => (3, freq=81, time=350)
        return x

class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    """
    Jeu de données pour l'entraînement de la vérification du locuteur.
    Hérite à la fois de Dataset et de SegmentSet (kiwano).
    Gère le chargement audio, les transformations, l'extraction FBANK, etc.
    """
    def __init__(
        self,
        audio_transforms: List[Augmentation] = None,
        feature_extractor=None,
        feature_transforms: List[Augmentation] = None
    ):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_extractor = feature_extractor
        self.feature_transforms = feature_transforms

    def __getitem__(self, idx_or_key: Union[int, str]):
        """
        Récupère un item (feature, label) :
          - idx_or_key : soit un index, soit la clé str de 'segments'.
        """
        if isinstance(idx_or_key, str):
            segment = self.segments[idx_or_key]
        else:
            # Récupérer la i-ème entrée dans self.segments
            segment = next(
                val for i, val in enumerate(self.segments.values())
                if i == idx_or_key
            )

        # Chargement audio
        audio, sample_rate = segment.load_audio()
        # Transformations audio
        if self.audio_transforms is not None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        # Extraction de features FBANK
        if self.feature_extractor is not None:
            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)
        else:
            raise ValueError("feature_extractor ne doit pas être None.")

        # Transformations sur les features (CMVN, Crop, SpecAug, etc.)
        if self.feature_transforms is not None:
            feature = self.feature_transforms(feature)

        # On mappe le spkid -> label ID
        # self.labels est un dict : {spkid: label_id, ...}
        label = self.labels[segment.spkid]

        return feature, label

    def __len__(self):
        return len(self.segments)


def main():
    """
    Point d'entrée principal.
    1) Parse arguments
    2) init_process_group (DDP)
    3) Charger dataset + data loader
    4) Instancier EfficientNetV2 + Optim + IDRDScheduler
    5) Boucle d'entraînement
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Rang local pour l'entraînement distribué (NCCL).")
    parser.add_argument("--musan", type=str, default="data/musan/",
                        help="Chemin vers le dossier musan (bruits).")
    parser.add_argument("--rirs_noises", type=str, default="data/rirs_noises/",
                        help="Chemin vers le dossier rirs_noises.")
    parser.add_argument("--checkpoint", type=str,
                        help="Chemin vers un checkpoint .ckpt à reprendre.")
    parser.add_argument("training_corpus", type=str,
                        help="Chemin vers le corpus d'entraînement (fichier .json ou dir).")
    parser.add_argument("exp_dir", type=str,
                        help="Chemin vers le dossier d'export (ckpt).")

    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", 0))  # Récupérer le rang dans SLURM
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))  # Rang local
    world_size = int(os.environ.get("SLURM_NTASKS", 1))  # Nombre total de tâches
    #node_list = os.environ.get("SLURM_JOB_NODELIST", "localhost")
    #world = int(os.environ["SLURM_JOB_NUM_NODES"])
    #world_size = int(os.environ["SLURM_NTASKS"])

    # Initialisation du logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Démarrage du script train_efficientnet.py")

    # Configuration SLURM / DDP
    hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
    
    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")



    os.environ["MASTER_ADDR"] = hostnames[0]
    master_port = os.environ["MASTER_PORT"]  # Port dynamique

    print(f"[Process {rank}] SLURM_JOB_NODELIST: {os.environ['SLURM_JOB_NODELIST']}")
    print(f"[Process {rank}] Expanded hostnames: {hostnames}")
    print(f"[Process {rank}] MASTER_ADDR: {os.environ['MASTER_ADDR']}")
    print(f"[Process {rank}] MASTER_PORT: {master_port}")
    print(f"[Process {rank}] RANK: {rank}")
    print(f"[Process {rank}] LOCAL_RANK: {local_rank}")
    print(f"[Process {rank}] WORLD_SIZE: {world_size}")

    # Initialisation du process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    
    print(f"Process {rank}/{world_size} initialized on {os.uname().nodename}.")
    print(f"Process {dist.get_rank()} running on {os.uname().nodename}.")
    

    # Définir device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.current_device()}, Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9} GB")
    print(f"Available memory: {torch.cuda.memory_reserved(0) / 1e9} GB")

    # Charger musan
    musan = SegmentSet()
    musan.from_dict(Path(args.musan))

    musan_music = musan.get_speaker("music")
    musan_speech = musan.get_speaker("speech")
    musan_noise = musan.get_speaker("noise")

    # Charger reverb
    reverb = SegmentSet()
    reverb.from_dict(Path(args.rirs_noises))

    # Construire le dataset
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
        #To3Channels(),
    ])

    training_data = SpeakerTrainingSegmentSet(
        audio_transforms=audio_transforms,
        feature_extractor=feature_extractor,
        feature_transforms=feature_transforms
    )
    # Charger la description (segments + labels) depuis training_corpus
    training_data.from_dict(Path(args.training_corpus))

    # Sampler distribué
    train_sampler = DistributedSampler(
        training_data,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    # DataLoader
    train_dataloader = DataLoader(
        training_data,
        batch_size=128,
        drop_last=True,
        shuffle=False,  # shuffle déjà géré par train_sampler
        num_workers=10,
        sampler=train_sampler,
        pin_memory=True
    )

    # Instancier le modèle
    #  => On détermine le nombre de classes
    num_classes = len(set(training_data.labels.values()))
    logger.info(f"Nombre de classes (locuteurs) détecté : {num_classes}")

    eff_model = EfficientNetV2(
        num_classes=num_classes,
        input_features=81,
        embed_features=256,
        model_name="efficientnet-b2"
    )
    
    print(eff_model, flush=True)
    # Convertir BatchNorm en SyncBatchNorm
    eff_model = nn.SyncBatchNorm.convert_sync_batchnorm(eff_model)

    # Envoyer sur le device
    eff_model.to(device)

    # Créer l'optimiseur
    optimizer = torch.optim.SGD(
        [{'params': eff_model.parameters(), 'weight_decay': 1e-4, 'lr': 1e-2}],
        momentum=0.9
    )
     #Optimiseur Adam pr tester comme hyperion

    """optimizer = torch.optim.Adam(
        eff_model.parameters(),
        lr=0.01,
        betas=(0.9, 0.95),
        weight_decay=1e-5,
        amsgrad=True
    )"""

    # Charger checkpoint si demandé
    epochs_start = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        eff_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_start = checkpoint.get("epochs", 0)
        logger.info(f"Checkpoint chargé, reprise à l'epoch {epochs_start}.")

    # Instancier la loss
    criterion = JeffreysLoss(coeff1=0.1, coeff2=0.025)

    # Scheduler IDRD
    scheduler = IDRDScheduler(
        optimizer=optimizer,
        num_epochs=150,
        initial_lr=0.01,
        warm_up_epoch=5,
        plateau_epoch=15,
        patience=10,
        factor=5,
        amsmloss=0.3
    )
    if args.checkpoint:
        # Ajuster le scheduler si besoin
        scheduler.set_epoch(epochs_start)

    # Préparer le modèle en DDP
    eff_model = DDP(eff_model, device_ids=[local_rank], output_device=local_rank)

    # Instancier GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Dossier de sortie
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    # Boucle d'entraînement
    # ======================

    def train_one_epoch(epoch):
        eff_model.train()
        train_sampler.set_epoch(epoch)  # important pour DistributedSampler
        total_loss = 0.0
        total_examples = 0
        iterations = 0

        # Mettre la marge AMSMLoss dynamique
        # eff_model.module => car DDP
        eff_model.module.output.set_m(scheduler.get_amsmloss())

        for feats, iden in train_dataloader:
            feats = feats.unsqueeze(1)  # [B, 1, 350, 81]
            feats = feats.to(device, dtype=torch.float32)
            iden = iden.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                # forward
                preds = eff_model(feats, iden)
                loss = criterion(preds, iden)

            # backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * feats.size(0)
            total_examples += feats.size(0)

            if iterations % 100 == 0:
                current_lr = get_lr(optimizer)
                margin = eff_model.module.output.get_m()
                msg = (f"{time.ctime()} | Epoch [{epoch}/150] (iter {iterations}/{len(train_dataloader)}) "
                       f"C-Loss: {loss.item():.4f} | LR: {current_lr:.8f} | Margin: {margin:.4f}")
                print(msg)
            iterations += 1

        epoch_loss = total_loss / total_examples
        return epoch_loss

    # Pas de vrai set de validation (pour HPC distribué) => si besoin, code additionnel
    # on peut juste sauvegarder un checkpoint

    for epoch in range(epochs_start, 150):
        epoch_loss = train_one_epoch(epoch)
        logger.info(f"Fin epoch {epoch}/{150}, C-Loss = {epoch_loss:.4f}")

        # Step du scheduler
        scheduler.step()

        # Sauvegarde checkpoint sur le rang 0 seulement
        if dist.get_rank() == 0:
            ckpt = {
                "epochs": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "model_state_dict": eff_model.module.state_dict(),
            }
            torch.save(ckpt, os.path.join(exp_dir, f"model_effnet_{epoch}.ckpt"))

    logger.info("Fin de l'entraînement.")


if __name__ == "__main__":
    main()

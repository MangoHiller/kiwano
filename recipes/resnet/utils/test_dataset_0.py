#!/usr/bin/env python3
"""
test_load_efficientnet.py

Script minimal pour tester la lecture du dataset (VoxCeleb1) 
et le chargement d'un checkpoint EfficientNetV2, 
mais sans forward pass (pas d'extraction d'embeddings).

On se limite à:
- Charger 5 segments (pour le dataset).
- Charger le checkpoint EfficientNetV2 et imprimer quelques infos.
- Ne pas faire de forward pass.
"""

import sys
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from kiwano.dataset import SegmentSet
from kiwano.features import Fbank
from kiwano.augmentation import Linear, CMVN, Crop  # ou PadOrTrunc
from kiwano.embedding import EmbeddingSet
from kiwano.model import EfficientNetV2
from kiwano.utils import Pathlike

# =========================================
# Classe minimaliste Dataset
# =========================================
from typing import Union, List
from torch.utils.data import Dataset

class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
    def __init__(self, feature_extractor=None, feature_transforms=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_transforms = feature_transforms

    def __getitem__(self, idx_or_key: Union[int,str]):
        if isinstance(idx_or_key, str):
            segment = self.segments[idx_or_key]
        else:
            # Conversion index -> key
            segment = next(
                val for i, val in enumerate(self.segments.values()) if i == idx_or_key
            )

        audio, sr = segment.load_audio()

        # Extraction de features (Fbank)
        feats = self.feature_extractor.extract(audio, sampling_rate=sr)

        # Transformations (CMVN, Crop, etc.)
        if self.feature_transforms:
            feats = self.feature_transforms(feats)

        return feats, segment.segmentid

    def __len__(self):
        return len(self.segments)

def get_parser():
    parser = argparse.ArgumentParser(description="Test dataset + load EfficientNet checkpoint (no forward).")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Nombre total de répliques (DDP).")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rang DDP (0..world_size-1).")
    parser.add_argument("data_dir", type=str,
                        help="Chemin vers le dataset (ex. data/voxceleb1/).")
    parser.add_argument("model_ckpt", type=str,
                        help="Chemin vers le checkpoint .ckpt EfficientNetV2.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"# Started test_load_efficientnet.py at {time.ctime()}", flush=True)
    print(f"# data_dir    = {args.data_dir}", flush=True)
    print(f"# model_ckpt  = {args.model_ckpt}", flush=True)
    print(f"# world_size  = {args.world_size}, rank = {args.rank}", flush=True)

    device = torch.device("cuda")
    print(f"# device = {device}", flush=True)

    # ====================== DATASET ======================
    dataset = SpeakerExtractingSegmentSet(
        feature_extractor=Fbank(),
        feature_transforms=Linear([
            CMVN(),
            Crop(350, random=False), 
            # ou PadOrTrunc(350) si vous préférez
        ])
    )

    print(">>> Avant from_dict(...)", flush=True)
    dataset.from_dict(Path(args.data_dir))
    print(">>> Après from_dict(...)", flush=True)

    nb_segments = len(dataset)
    print(f"# Nombre de segments charges : {nb_segments}", flush=True)
    if nb_segments == 0:
        print("# ERREUR: Aucun segment n'a été chargé. On s'arrête.", flush=True)
        return

    # On va juste lire 5 segments pour test
    sampler = DistributedSampler(dataset, 
                                 num_replicas=args.world_size, 
                                 rank=args.rank, 
                                 shuffle=False)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,   # pour éviter blocage multiprocess
        sampler=sampler,
        pin_memory=False
    )

    print(">>> DataLoader créé. On va juste itérer sur 5 segments pour valider la pipeline.", flush=True)
    max_iter = 5
    for idx, (feats, segid) in enumerate(dataloader):
        if idx >= max_iter:
            break
        print(f"Exemple idx={idx}, segid={segid[0]}, feats shape={feats.shape}")

    # ====================== MODELE EFFICIENTNET ======================
    print(f">>> Chargement du checkpoint : {args.model_ckpt}", flush=True)
    ckpt = torch.load(args.model_ckpt, map_location="cpu")
    # On suppose que la clé s'appelle 'model_state_dict'
    state_dict = ckpt["model_state_dict"]

    eff_model = EfficientNetV2(
        num_classes=5994,  # la valeur n'est pas critique ici
        input_features=81,
        embed_features=256,
        model_name="efficientnet-b0"
    )

    eff_model.load_state_dict(state_dict, strict=True)

    # Juste imprimer un résumé / architecture
    print(">>> Modèle EfficientNetV2 chargé. Voici un print(eff_model) :", flush=True)
    print(eff_model)
    
    print(f"Avant to(device): {torch.cuda.memory_allocated() / 1e6} MB", flush=True)
    torch.cuda.empty_cache()
    eff_model.to(device)
    print(f"Après to(device): {torch.cuda.memory_allocated() / 1e6} MB", flush=True)
    eff_model.eval()

    # On itère sur 5 segments pour un test rapide
    max_iter = 5
    with torch.no_grad():
        for idx, (feats, segid) in enumerate(dataloader):
            if idx >= max_iter:
                break

            # feats: [1, frames, 81], on insère dimension channel => [1, 1, frames, 81]
            feats = feats.unsqueeze(1).float().to(device)
            
            embed = eff_model(feats)  # [1, 256] en principe

            print(f"Exemple idx={idx}, segid={segid[0]}, feats shape={feats.shape}, embed shape={embed.shape}", flush=True)


    # Pas de forward pass
    print("# Fin test_load_efficientnet.py", flush=True)

if __name__ == "__main__":
    main()

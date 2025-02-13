#!/usr/bin/env python3
"""
extract_efficientnet.py

Extrait les embeddings (xvectors) d'un modèle EfficientNetV2 pour chaque segment
dans un dataset (par ex. VoxCeleb1). Sauvegarde les embeddings dans un .pkl.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

import numpy as np

from kiwano.utils import Pathlike
from kiwano.features import Fbank
from kiwano.augmentation import Linear, CMVN, Crop, PadOrTrunc
from kiwano.dataset import SegmentSet, Segment
from kiwano.model import EfficientNetV2
from kiwano.embedding import EmbeddingSet, write_pkl

# ======================
# Dataset "SpeakerExtractingSegmentSet"
# ======================
class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
    
    #Dataset pour l'extraction d'embeddings xvectors (sans labels).
    #Charge l'audio, extrait les features, renvoie (feature, segment_id).
    
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
                val for i,val in enumerate(self.segments.values()) if i==idx_or_key
            )

        audio, sr = segment.load_audio()

        # Extraction de features (Fbank)
        features = self.feature_extractor.extract(audio, sampling_rate=sr)
        
        # Optionnel: transformations style CMVN, Crop, etc.
        if self.feature_transforms:
            features = self.feature_transforms(features)

        return features, segment.segmentid

    def __len__(self):
        return len(self.segments)

#############################################
# Créer la classe To3Channels (même que pour l'entraînement)
#############################################
class To3Channels:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (time, freq) = (T, F)
        Retourne un tenseur (3, T, F) ou (3, F, T) selon la convention choisie.
        Ici, on reprend exactement la logique de train_efficientnetV2.py :
           x = x.unsqueeze(0).repeat(3, 1, 1)
        """
        # Ajoute la dimension "canal" et la duplique sur 3 canaux
        x = x.unsqueeze(0).repeat(3, 1, 1)  # => (3, T, F) si x.shape=(T,F)
        return x

"""class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
    def __init__(self, feature_extractor=None, feature_transforms=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_transforms = feature_transforms

    def skip_too_short_segments(self, min_frames=350):
        
        #Parcourt tous les segments, calcule rapidement les features FBANK,
        #et retire ceux qui font moins de `min_frames`.
        
        original_count = len(self.segments)
        too_short_keys = []

        # Petit extracteur local pour mesurer la taille
        tmp_fbank = self.feature_extractor if self.feature_extractor else Fbank()

        for key, seg in list(self.segments.items()):
            audio, sr = seg.load_audio()
            feats = tmp_fbank.extract(audio, sampling_rate=sr)

            if feats.shape[0] < min_frames:
                too_short_keys.append(key)

        # Retirer du dictionnaire
        for k in too_short_keys:
            del self.segments[k]

        print(f"[skip_too_short_segments] Total segments : {original_count}")
        print(f"[skip_too_short_segments] Trop courts (<{min_frames}) : {len(too_short_keys)}")
        print(f"[skip_too_short_segments] Restant : {len(self.segments)}")

    def __getitem__(self, idx_or_key):
        if isinstance(idx_or_key, str):
            segment = self.segments[idx_or_key]
        else:
            segment = next(
                val for i, val in enumerate(self.segments.values()) if i == idx_or_key
            )

        audio, sr = segment.load_audio()

        feats = self.feature_extractor.extract(audio, sampling_rate=sr) \
                if self.feature_extractor else None

        if self.feature_transforms:
            feats = self.feature_transforms(feats)

        return feats, segment.segmentid

    def __len__(self):
        return len(self.segments)"""


def get_parser():
    parser = argparse.ArgumentParser(description="Extraction embeddings EfficientNetV2")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Nombre total de répliques (DDP).")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rang DDP (0..world_size-1).")
    parser.add_argument("data_dir", type=str,
                        help="Chemin vers le dataset (ex. data/voxceleb1/).")
    parser.add_argument("model_ckpt", type=str,
                        help="Chemin vers le checkpoint du modèle EfficientNetV2.")
    parser.add_argument("output_path", type=str,
                        help="pkl:out.pkl => chemin de sortie du .pkl contenant les embeddings.")

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"# Command line : {' '.join(sys.argv)}", flush=True)
    print(f"# Started at {time.ctime()}", flush=True)
    print(f"# data_dir = {args.data_dir}", flush=True)
    print(f"# model_ckpt = {args.model_ckpt}", flush=True)
    print(f"# output_path = {args.output_path}", flush=True)

    device = torch.device("cuda")

    # -- Construire le dataset --
    dataset = SpeakerExtractingSegmentSet(
        feature_extractor=Fbank(),
        feature_transforms=Linear([
            CMVN(),
            PadOrTrunc(350),
            #To3Channels(),
            #Crop(350, random=False),
            
        ]),
    )
    dataset.from_dict(Path(args.data_dir))

    nb_segments = len(dataset)
    print(f"# Nombre de segments charges : {nb_segments}", flush=True)
    if nb_segments == 0:
        print("# ERREUR : Aucun segment n'a ete charge. Verifiez data_dir ou from_dict.")
        print("# Le script s'arrete.")
        return

    # 3) Avant d'itérer, filtrer les segments trop courts
    #dataset.skip_too_short_segments(min_frames=350)


    # -- Sampler distribué si >1 job array --
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)

    # -- DataLoader --
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4,
                            sampler=sampler, pin_memory=True)

    # -- Charger le modèle EfficientNetV2 --
    model_ckpt = torch.load(args.model_ckpt, map_location="cpu")
    eff_model = EfficientNetV2(num_classes=5994,  # la valeur n'est pas critique pour extraction
                               input_features=81,
                               embed_features=256,
                               model_name="efficientnet-b2")

    # On a sauvegardé "model_state_dict" ou similaire
    #  => si c'est "model_state_dict" dans le .ckpt, adapter ici
    eff_model.load_state_dict(model_ckpt["model_state_dict"], strict=True)
    print(">>> Modèle EfficientNetV2 chargé. Voici un print(eff_model) :", flush=True)
    print(eff_model)

    print(f"Avant to(device): {torch.cuda.memory_allocated() / 1e6} MB", flush=True)
    torch.cuda.empty_cache()

    eff_model.to(device)
    print(f"Après to(device): {torch.cuda.memory_allocated() / 1e6} MB", flush=True)

    eff_model.eval()

    # -- On stocke les embeddings dans un EmbeddingSet --
    emb = EmbeddingSet()

    print("# Debut de la boucle d'extraction ...", flush=True)
    with torch.no_grad():
        for feats, segid in dataloader:
            # feats: [B, 3, T, F] => déjà 3 canaux, aucune insertion à faire.
            feats = feats.unsqueeze(1).float().to(device)
            feats = feats.float().to(device)

            # L’output typique de EfficientNetV2 => vecteur d’embedding [batch, 256]
            embed = eff_model(feats)

            # On stocke le 1er (et unique) batch
            emb[segid[0]] = embed.cpu().squeeze(0)
            
            print(f"Processed x-vector for key: {segid[0]}")

    # -- Sauvegarder dans le .pkl --
    write_pkl(args.output_path, emb)

    print(f"# Embeddings saved to {args.output_path}")
    print(f"# Ended at {time.ctime()}")

if __name__ == "__main__":
    main()

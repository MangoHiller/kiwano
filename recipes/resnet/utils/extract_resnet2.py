#!/usr/bin/env python3

import sys, os, time
from pathlib import Path
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn

from kiwano.utils import Pathlike
from kiwano.features import Fbank, FbankV2
from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop
from kiwano.dataset import Segment, SegmentSet
from kiwano.model import ResNet, ResNetV2, ResNetV3, ResNetV4, ResNetV5, BasicBlock, SEBasicBlock, MiniBasicBlock, MiniSEBasicBlock
from kiwano.embedding import EmbeddingSet, write_pkl

from torch.utils.data.distributed import DistributedSampler

import soundfile as sf

from torch.utils.data import Dataset, DataLoader, Sampler

import argparse

class SpeakerExtractingSegmentSet(Dataset, SegmentSet):
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor = None, feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

        audio, sample_rate = segment.load_audio()
        if self.audio_transforms != None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        if self.feature_extractor != None:
            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

        if self.feature_transforms != None:
            feature = self.feature_transforms(feature)

        return feature, segment.segmentid


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--rank",
        default=0,
        type=int,
    )

    parser.add_argument("--model_size",
                        type=str,
                        choices=["resnet18","resnet36","resnet50","resnet101","resnet101eq","resnet200","resnet400","resnet800"],
                        default="resnet18",
                        help="Taille du modèle à entraîner")

    parser.add_argument("--width_mult", type=float, default=1.0, help="Multiplicateur sur la largeur (nombre de channels) des couches du ResNet.") #permet de faire varier la largeur du modèle, facteur sur les features map.

    parser.add_argument(
        "data_dir",
        type=str,
        help="data_dir",
    )

    parser.add_argument(
        "model",
        type=str,
        help="Model",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="pkl:output.pkl",
    )

    return parser

def get_resnet_model(model_size: str, num_classes=5994, width_mult: float = 1.0):
    """Retourne un ResNetV2 configuré pour la taille demandée."""
    base_channels = {
        "resnet18":  [128, 128, 256, 256],
        "resnet36":  [128, 128, 256, 256],
        "resnet50":  [128, 128, 256, 256],
        "resnet101eq": [128, 128, 256, 256],
        "resnet101": [128, 128, 256, 256],
        "resnet200": [128, 128, 256, 256],
        "resnet400": [128, 128, 256, 256],
        "resnet800": [128, 128, 256, 256],
    }

    resnet_config = {
        "resnet18": ([2, 2, 2, 2], MiniBasicBlock, MiniSEBasicBlock),
        "resnet36": ([3, 4, 6, 3], MiniBasicBlock, MiniSEBasicBlock),
        "resnet50": ([3, 4, 6, 3], BasicBlock, SEBasicBlock),
        "resnet101": ([3, 4, 23, 3], BasicBlock, SEBasicBlock),
        "resnet101eq": ([3, 10, 17, 3], BasicBlock, SEBasicBlock), 
        "resnet200": ([3, 24, 36, 3], BasicBlock, SEBasicBlock),
        "resnet400": ([4, 44, 87, 4], BasicBlock, SEBasicBlock),
        "resnet800": ([8, 88, 174, 8], BasicBlock, SEBasicBlock),
    }
    if model_size not in resnet_config:
        raise ValueError(f"model_size invalide ({model_size}), choix possibles : {list(resnet_config)}")
    num_blocks, block, block_se = resnet_config[model_size]
    channels = [int(c * width_mult) for c in base_channels[model_size]]
    return ResNetV2(
        num_classes=num_classes,
        channels=channels,
        num_blocks=num_blocks,
        block=block,
        block_se=block_se
    )


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    print("#"+" ".join( sys.argv[0:]  ))
    print("# Started at "+time.ctime())
    print("#")

    device = torch.device("cuda")


    extracting_data = SpeakerExtractingSegmentSet(
                                    feature_extractor=Fbank(),
                                    feature_transforms=Linear( [
                                        CMVN(),
                                        Crop(1400, random=False),
                                    ] ),
                                )

    extracting_data.from_dict(Path(args.data_dir))

    nb_segments = len(extracting_data.segments)
    print(f"[INFO] Nombre total de segments chargés : {nb_segments}")

    extracting_sampler = DistributedSampler(extracting_data, num_replicas=args.world_size, rank=args.rank)

    extracting_dataloader = DataLoader(extracting_data, batch_size=1, num_workers=10, sampler=extracting_sampler, pin_memory=True)
    iterator = iter(extracting_dataloader)

    #resnet_model = ResNet(num_classes=18000)
    #resnet_model = ResNetV2(num_classes=18000)
    #resnet_model = ResNetV2(num_classes=5994, num_blocks=[3,24,36,3], block=BasicBlock, block_se=SEBasicBlock) #pr extraire le resnet200
    #resnet_model = ResNetV2(num_classes=5994, num_blocks=[3,4,23,3], block=BasicBlock, block_se=SEBasicBlock) #pr extraire resnet101_5994
    #resnet_model = ResNetV2(num_classes=6000, num_blocks=[3,4,6,3], block=BasicBlock, block_se=SEBasicBlock) #pr extraire resnet50_6000
    #resnet_model = ResNetV2(num_classes=5994, num_blocks=[3,4,6,3], block=BasicBlock, block_se=SEBasicBlock) #pr extraire resnet50_5994 cos_embed
    #resnet_model = ResNetV5()
    #resnet_model = ResNetV3(k=3)
    #resnet_model = ResNetV4()

    resnet_model = get_resnet_model(args.model_size, width_mult=args.width_mult) # remplace la selection manuelle du modele

    resnet_model.load_state_dict(torch.load(args.model)["model"])
    resnet_model.eval().to(device)
    #resnet_model.to(device).half() #passage en precision FP16

    #resnet_model.eval()


    emb = EmbeddingSet()

    #count_extracted = 0


    count = 0

    with torch.no_grad():
        for feat, key in extracting_dataloader:
            feat = feat.unsqueeze(1).to(device)

            pred = resnet_model(feat).squeeze(0)          # (256,)
            # ────────────────────────────────────────────────
            # FILTRE : si NaN / Inf ⇒ on ignore complètement ce segment
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                continue
            # ────────────────────────────────────────────────
            emb[key[0]] = pred.cpu()

            count += 1
            if count % 1000 == 0:
                print(f"  → {count} segments valides extraits")

    write_pkl(args.output_dir, emb)

    print(f"[INFO] Extraction terminée. {count} segments extraits.")

    print("# Ended at "+time.ctime())




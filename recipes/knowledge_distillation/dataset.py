# dataset.py

import torch
from torch.utils.data import Dataset
from kiwano.dataset import SegmentSet
from typing import Union, List
from pathlib import Path

class To3Channels:
    """
    Transformation pour dupliquer les features sur 3 canaux
    (si on veut coller au format (3, freq, time) d'EfficientNet).
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x : (time, freq)
        return x.unsqueeze(0).repeat(3, 1, 1)  # => (3, time, freq)


class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    """
    Jeu de données pour l'entraînement de la vérification du locuteur,
    identique à la logique de train_efficientnet.py.
    Gère :
      - chargement audio
      - audio augmentations
      - extraction FBANK
      - transforms sur features (CMVN, crop, SpecAugment, etc.)
      - mapping spkid -> label
    """
    def __init__(self, audio_transforms=None, feature_extractor=None, feature_transforms=None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_extractor = feature_extractor
        self.feature_transforms = feature_transforms

    def __getitem__(self, idx_or_key: Union[int, str]):
        if isinstance(idx_or_key, str):
            segment = self.segments[idx_or_key]
        else:
            # Récupérer la i-ème entrée dans self.segments (liste)
            segment = next(
                val for i, val in enumerate(self.segments.values())
                if i == idx_or_key
            )

        # Chargement audio
        audio, sample_rate = segment.load_audio()

        # Transforms audio
        if self.audio_transforms:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)

        # Extraction features (Fbank)
        if self.feature_extractor is None:
            raise ValueError("feature_extractor ne doit pas être None.")
        feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

        # Transforms sur features
        if self.feature_transforms:
            feature = self.feature_transforms(feature)

        # Récupérer label
        label = self.labels[segment.spkid]

        return feature, label

    def __len__(self):
        return len(self.segments)

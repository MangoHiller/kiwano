from typing import Dict, List, Tuple

import numpy as np


def make_calibration_splits_no_speaker_overlap(
    segment_ids: List[str],
    speaker_ids: List[str],
    cluster_labels: List[int],
    ratio: float = 0.8,
    min_val_speakers: int = 5,
    random_seed: int = 0,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Build train_cal/val_cal per cluster with no speaker overlap.

    Returns:
        dict[k] -> {"train": [segment_id,...], "val": [segment_id,...],
                    "report": {"n_spk_train": int, "n_spk_val": int, "n_seg_train": int, "n_seg_val": int}}
    """
    rng = np.random.default_rng(random_seed)
    segs = np.asarray(segment_ids)
    spks = np.asarray(speaker_ids)
    labs = np.asarray(cluster_labels)

    out: Dict[int, Dict[str, List[str]]] = {}

    for k in range(int(labs.max()) + 1):
        mask = labs == k
        seg_k = segs[mask]
        spk_k = spks[mask]

        unique_spk = np.unique(spk_k)
        n_spk = unique_spk.size
        if n_spk == 0:
            out[k] = {"train": [], "val": [], "report": {"n_spk_train": 0, "n_spk_val": 0, "n_seg_train": 0, "n_seg_val": 0}}
            continue

        # split speakers
        n_val = max(int(round((1.0 - ratio) * n_spk)), min_val_speakers)
        n_val = min(n_val, n_spk - 1) if n_spk > 1 else 1
        perm = rng.permutation(n_spk)
        val_spk = set(unique_spk[perm[:n_val]].tolist())
        train_spk = set(unique_spk[perm[n_val:]].tolist())
        if not train_spk:  # degenerate small cluster
            train_spk = set(val_spk)
            val_spk = set()

        train_mask = np.array([s in train_spk for s in spk_k])
        val_mask = ~train_mask if val_spk else np.zeros_like(train_mask, dtype=bool)

        train_ids = seg_k[train_mask].tolist()
        val_ids = seg_k[val_mask].tolist()

        out[k] = {
            "train": train_ids,
            "val": val_ids,
            "report": {
                "n_spk_train": int(len(train_spk)),
                "n_spk_val": int(len(val_spk)),
                "n_seg_train": int(len(train_ids)),
                "n_seg_val": int(len(val_ids)),
            },
        }

    return out

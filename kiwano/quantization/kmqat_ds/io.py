import glob
import json
import os
import pickle
from typing import Dict, Iterable, List, Tuple

import numpy as np


def safe_makedirs(path: str) -> None:
    """Create directory if it doesn't exist (like mkdir -p)."""
    os.makedirs(path, exist_ok=True)


def read_embedding_pkl_dict(pkl_patterns: List[str]) -> Dict[str, np.ndarray]:
    """
    Load EmbeddingSet pickles (as produced by extract_resnet2.py) into a dict {segment_id: vector}.
    Accepts one or several glob patterns.

    Notes:
        - Values in PKL may be torch tensors or numpy arrays; we convert everything to np.ndarray (float32).
    """
    merged: Dict[str, np.ndarray] = {}
    nfiles = 0
    for pattern in pkl_patterns:
        for p in sorted(glob.glob(pattern)):
            with open(p, "rb") as f:
                # EmbeddingSet.h was pickled. It's a dict-like {segid: tensor/array}
                payload = pickle.load(f)
            # payload can be EmbeddingSet or plain dict; try both
            if hasattr(payload, "h"):
                items = payload.h.items()
            else:
                items = payload.items()
            for k, v in items:
                arr = _to_numpy_f32(v)
                merged[str(k)] = arr
            nfiles += 1
    if nfiles == 0:
        raise FileNotFoundError(f"No PKL found for patterns: {pkl_patterns}")
    return merged


def _to_numpy_f32(x) -> np.ndarray:
    try:
        import torch  # lazy import

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def load_list_file(list_path: str) -> List[Tuple[str, str, float, str]]:
    """
    Load 'liste' format rows: 'segmentid spkid duration path'.

    Returns:
        List of tuples: (segment_id, speaker_id, duration, path)
    """
    rows: List[Tuple[str, str, float, str]] = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seg_id, spk_id, dur, path = line.split(maxsplit=3)
            rows.append((seg_id, spk_id, float(dur), path))
    if not rows:
        raise ValueError(f"Empty list file: {list_path}")
    return rows


def write_npy(path: str, array: np.ndarray) -> None:
    """Save numpy array to .npy."""
    np.save(path, array)


def write_csv_meta(path: str, meta_rows: List[Tuple[str, str, float, str]], split: str = "train") -> None:
    """
    Write meta CSV with columns: segment_id, speaker_id, duration, path, split.
    """
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "speaker_id", "duration", "path", "split"])
        for seg_id, spk_id, dur, p in meta_rows:
            w.writerow([seg_id, spk_id, f"{dur:.6f}", p, split])


def write_json(path: str, payload: dict) -> None:
    """Write a JSON file with UTF-8 and pretty indentation."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def compute_l2_norms_histogram(x: np.ndarray, bins: int = 50) -> dict:
    """
    Compute a histogram of L2 norms for sanity checks.
    Returns a dict with mean, std, min, max, and (hist, edges).
    """
    norms = np.linalg.norm(x, axis=1)
    hist, edges = np.histogram(norms, bins=bins)
    return {
        "mean": float(norms.mean()),
        "std": float(norms.std()),
        "min": float(norms.min()),
        "max": float(norms.max()),
        "hist": hist.tolist(),
        "edges": edges.tolist(),
    }

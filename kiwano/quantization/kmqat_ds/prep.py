from typing import Dict, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA


def apply_l2_norm(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row vector. Zero vectors (if any) are left unchanged.
    """
    eps = 1e-12
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    scale = 1.0 / np.maximum(norms, eps)
    return x * scale


def fit_cmvn_global(x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fit global CMVN (mean/std across samples) on the input matrix (N x D).
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-12] = 1.0  # avoid div-by-zero
    return {"mean": mean, "std": std}


def apply_cmvn_global(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply previously fitted CMVN stats = {'mean': ..., 'std': ...}."""
    return (x - stats["mean"]) / stats["std"]


def fit_pca(x: np.ndarray, out_dim: int) -> PCA:
    """
    Fit a PCA object on x (N x D) with out_dim components.
    Returns sklearn PCA (which can be pickled as-is).
    """
    pca = PCA(n_components=out_dim, svd_solver="auto", whiten=False, random_state=0)
    pca.fit(x)
    return pca


def apply_pca(x: np.ndarray, pca: PCA) -> np.ndarray:
    """Apply fitted PCA to x."""
    return pca.transform(x)

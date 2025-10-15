from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _kmeans_fit_predict(x: np.ndarray, k: int, seed: int, max_iter: int = 300) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run KMeans euclidean (assumes x rows are L2-normalized to emulate cosine).
    Returns (labels, centers, inertia).
    """
    # If dataset is huge and memory-constrained, one can switch to MiniBatchKMeans here.
    km = KMeans(n_clusters=k, n_init=1, random_state=seed, max_iter=max_iter, verbose=0)
    labels = km.fit_predict(x)
    centers = km.cluster_centers_
    inertia = km.inertia_
    return labels, centers, inertia


def cosine_kmeans_sweep(
    x: np.ndarray,
    k_grid: Iterable[int],
    seeds_per_k: int = 10,
    max_iter: int = 300,
    min_cluster_frac: float = 0.05,
) -> Dict:
    """
    Sweep K over k_grid with multiple random seeds.
    Assumes x rows are L2-normalized => euclidean ~ cosine.

    Returns:
        dict with per-K and per-seed metrics, plus best per-K summary.
    """
    n = x.shape[0]
    results = {"per_k": {}, "summary": {}}

    for k in k_grid:
        per_seed = []
        labels_runs = []
        inertias = []
        for s in range(seeds_per_k):
            labels, centers, inertia = _kmeans_fit_predict(x, k, seed=s, max_iter=max_iter)
            # Basic size check
            sizes = np.bincount(labels, minlength=k)
            frac = sizes / float(n)
            too_small = bool((frac < min_cluster_frac).any())

            # scores
            try:
                sil = float(silhouette_score(x, labels, metric="cosine"))
            except Exception:
                sil = float("nan")
            try:
                db = float(davies_bouldin_score(x, labels))
            except Exception:
                db = float("nan")
            try:
                ch = float(calinski_harabasz_score(x, labels))
            except Exception:
                ch = float("nan")

            per_seed.append(
                {
                    "seed": s,
                    "silhouette_cosine": sil,
                    "davies_bouldin": db,
                    "calinski_harabasz": ch,
                    "inertia": float(inertia),
                    "sizes": sizes.tolist(),
                    "too_small": too_small,
                }
            )
            labels_runs.append(labels)
            inertias.append(float(inertia))

        # stability (ARI/NMI) across seeds
        S = len(labels_runs)
        ari_vals, nmi_vals = [], []
        for i in range(S):
            for j in range(i + 1, S):
                ari_vals.append(float(adjusted_rand_score(labels_runs[i], labels_runs[j])))
                nmi_vals.append(float(normalized_mutual_info_score(labels_runs[i], labels_runs[j])))

        summary = {
            "mean_silhouette": _nanmean([r["silhouette_cosine"] for r in per_seed]),
            "mean_db": _nanmean([r["davies_bouldin"] for r in per_seed]),
            "mean_ch": _nanmean([r["calinski_harabasz"] for r in per_seed]),
            "mean_inertia": float(np.mean(inertias)),
            "mean_ari": float(np.mean(ari_vals)) if ari_vals else float("nan"),
            "mean_nmi": float(np.mean(nmi_vals)) if nmi_vals else float("nan"),
            "any_too_small": any(r["too_small"] for r in per_seed),
        }
        results["per_k"][k] = {"per_seed": per_seed, "summary": summary}

    return results


def choose_best_k_from_scores(results: Dict) -> int:
    """
    Heuristic: prefer K with
      - no 'too small' clusters,
      - highest mean silhouette (cosine),
      - low mean DB (as tiebreaker),
      - decent stability (mean ARI).
    """
    candidates = []
    for k, rec in results["per_k"].items():
        s = rec["summary"]
        if s["any_too_small"]:
            continue
        candidates.append(
            (k, s["mean_silhouette"], -s["mean_db"], s["mean_ari"])
        )
    if not candidates:
        # fall back on max silhouette regardless
        for k, rec in results["per_k"].items():
            s = rec["summary"]
            candidates.append((k, s["mean_silhouette"], -s["mean_db"], s["mean_ari"]))
    # sort by silhouette desc, DB asc (via minus), ARI desc
    candidates.sort(key=lambda t: (t[1], t[2], t[3]), reverse=True)
    best_k = candidates[0][0]
    return int(best_k)


def summarize_cluster_stats(x: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Compute sizes, and entropy of assignment as a simple dispersion indicator.
    """
    from math import log

    k = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=k)
    p = sizes / float(labels.shape[0])
    eps = 1e-12
    entropy = -float(np.sum(p * np.log(p + eps)))
    return {"sizes": sizes.tolist(), "entropy": entropy}


def _nanmean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())

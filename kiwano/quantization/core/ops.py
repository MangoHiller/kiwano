# core/ops.py
from __future__ import annotations
import torch as t
import torch.nn.functional as F
from .helpers import _central_clip, _build_codebook_from_quantiles, _assign_and_alpha

@t.no_grad()
def init_kmqat_params_for_weight(
    W: t.Tensor,
    n_bits: int,
    retention_ratio: float,
    symmetric_rescale: bool,
    kind: str,              # "conv2d" | "conv1d" | "linear"
    per_channel: bool = True,
):
    """
    Calcule (codebook, indices, alpha) pour KMQAT selon la forme de W.

    Args:
        W: poids FP32 (Conv2d: [out,in,kh,kw]; Conv1d: [out,in,k]; Linear: [out,in])
        n_bits: nb de bits (K = 2**n_bits)
        retention_ratio: r (0<r<=1) pour _central_clip
        symmetric_rescale: True => rescale [-1,1] symétrique ; False => affine
        kind: "conv2d" | "conv1d" | "linear"
        per_channel: si True, 1 codebook/alpha par canal de sortie

    Returns:
        codebook, indices, alpha
        - codebook:  [out, K]  si per_channel else [K]
        - indices:   mêmes dims que W (entiers [0..K-1])
        - alpha:     [out]     si per_channel else scalaire tensor([])
    """
    assert kind in {"conv2d", "conv1d", "linear"}
    maxK = 1 << n_bits
    out = W.shape[0]

    if per_channel:
        # Mise à plat par canal de sortie
        if kind in {"conv2d", "conv1d"}:
            flat = W.reshape(out, -1)        # [out, L]
            L = flat.shape[1]
        else:  # linear
            flat = W                         # [out, in]
            L = flat.shape[1]

        K_eff = min(maxK, L)
        codebook_eff = t.empty((out, K_eff), device=W.device, dtype=W.dtype)
        # IMPORTANT: indices au FORMAT WRAPPER => [out, L]
        indices = t.empty((out, L), device=W.device, dtype=t.long)
        alpha = t.empty((out,), device=W.device, dtype=W.dtype)

        for o in range(out):
            w = flat[o]                                    # [L]
            kept = _central_clip(w, retention_ratio)
            c_eff = _build_codebook_from_quantiles(kept, K_eff, symmetric_rescale)  # [K_eff]
            q, a, idx = _assign_and_alpha(w, kept, c_eff, symmetric_rescale)        # idx:[L] (vectorisé)

            codebook_eff[o] = c_eff
            alpha[o] = a
            indices[o] = idx                               # [L]  <-- vectorisé par canal

        # Padding à droite vers [out, maxK] si besoin
        if K_eff < maxK:
            pad = (0, maxK - K_eff)
            codebook = F.pad(codebook_eff, pad)            # [out, maxK]
        else:
            codebook = codebook_eff                        # [out, maxK == K_eff]

        return codebook, indices, alpha

    else:
        # Per-tensor (rare)
        w = W.reshape(-1)                                  # [N]
        N = w.numel()
        K_eff = min(maxK, N)

        kept = _central_clip(w, retention_ratio)
        c_eff = _build_codebook_from_quantiles(kept, K_eff, symmetric_rescale)       # [K_eff]
        q, a, idx = _assign_and_alpha(w, kept, c_eff, symmetric_rescale)             # idx:[N]

        # Padding du codebook global → [maxK]
        if K_eff < maxK:
            pad = (0, maxK - K_eff)
            codebook = F.pad(c_eff, pad)                                             # [maxK]
        else:
            codebook = c_eff

        # FORMAT WRAPPER: indices vectorisés 1D [N] (pas view_as)
        a_out = a.expand(W.shape[0])                                                  # alpha [out]
        return codebook, idx.to(t.long), a_out

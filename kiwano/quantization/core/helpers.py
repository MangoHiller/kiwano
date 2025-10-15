from __future__ import annotations
import torch as t
import torch.nn.functional as F

@t.no_grad()
def _central_clip(w: t.Tensor, r: float) -> t.Tensor:
    """
    But : garder les poids dans le quantile [ (1-r)/2 , (1+r)/2 ]
    Ex: r=0.9 > garder 90% des poids centraux, couper 5% aux extrémités
    0 < r <= 1.0
    """
    if r >= 1.0:
        return w
    qlow  = (1.0 - r) / 2.0
    qhigh = (1.0 + r) / 2.0
    lo = w.quantile(qlow)
    hi = w.quantile(qhigh)
    return w.clamp(min=lo.item(), max=hi.item())

@t.no_grad()
def _build_codebook_from_quantiles(w_kept: t.Tensor, K: int, symmetric_rescale: bool = True,
                    partition_mode: str = "equal_mass") -> t.Tensor:
    """
    Construit K centroids (k=1/bin) selon partition_mode:
      - "equal_mass": bords = quantiles (bins d'égale masse)
      - "equal_width": bords = linspace(min,max) (bins d'égale largeur en valeur)
    Puis rescale dans [-1,1] si symmetric_rescale=True.
    """
    if K <= 0:
        raise ValueError("K must be >= 1")

    if partition_mode not in ("equal_mass", "equal_width"):
        raise ValueError(f"partition_mode invalide: {partition_mode}")

    # Bords des intervalles
    if partition_mode == "equal_mass":
        qs = t.linspace(0, 1, steps=K + 1, device=w_kept.device)
        edges = t.quantile(w_kept, qs)
    else:  # "equal_width"
        wmin = w_kept.min()
        wmax = w_kept.max()
        # évite les bords identiques (poids tous égaux)
        if (wmax - wmin).abs() < 1e-12:
            edges = t.linspace(wmin - 1e-6, wmax + 1e-6, steps=K + 1, device=w_kept.device)
        else:
            edges = t.linspace(wmin, wmax, steps=K + 1, device=w_kept.device)

    # Centroid = moyenne des points du bin (dernier bin inclut le bord droit)
    cents = []
    for i in range(K):
        left, right = edges[i], edges[i + 1]
        mask = (w_kept >= left) & (w_kept <= right) if i == K - 1 else ((w_kept >= left) & (w_kept < right))
        if mask.any():
            cents.append(w_kept[mask].mean())
        else:
            # bin vide → centre du segment (évite NaN)
            cents.append(0.5 * (left + right))
    c = t.stack(cents)

    # Rescale
    if symmetric_rescale:
        m = c.abs().max().clamp_min(1e-8)
        c = c / m
    else:
        cmin, cmax = c.min(), c.max()
        m = (cmax - cmin).clamp_min(1e-8)
        c = (c - cmin) / m * 2 - 1
    return c

def _assign_and_alpha(w: t.Tensor, kept: t.Tensor, c: t.Tensor, symmetric_rescale: bool):
    """
    Assigne chaque poids w -> centroid c (en espace rescalé) et calcule alpha optimal:
      alpha = argmin ||w - alpha*q||^2 = (w·q) / (q·q)
    """
    if symmetric_rescale:
        m = kept.abs().max().clamp_min(1e-8)
        w_scaled = (w / m).clamp(-1, 1)
    else:
        wmin, wmax = kept.min(), kept.max()
        m = (wmax - wmin).clamp_min(1e-8)
        w_scaled = ((w - wmin) / m * 2 - 1).clamp(-1, 1)

    d = (w_scaled[:, None] - c[None, :]).abs()
    idx = d.argmin(dim=1)
    q = c[idx]

    num = (w * q).sum()
    den = (q.pow(2)).sum().clamp_min(1e-8)
    alpha = (num / den).clamp(min=1e-5)
    return q, alpha, idx

@t.no_grad()
def _ema_refresh_codebook(
    weight: t.Tensor,
    alpha_raw: t.Tensor,
    codebook: t.Tensor,
    indices: t.Tensor,
    per_channel: bool,
    alpha_softplus: bool = True,
    alpha_eps: float = 1e-5,
    beta: float = 0.1,
    reassign: bool = False,
):
    """
    Met à jour IN-PLACE les centroids par EMA vers la moyenne des points assignés.
    Optionnellement, réassigne (nearest-centroid L1) avant EMA.
    Retourne des stats dict: {"inertia", "usage", "entropy", "Keff_mean"}.
    Shapes attendues:
      - per_channel=True:  weight [O, ...], codebook [O, Kmax], indices [O, L]
      - per_channel=False: weight [...],    codebook [Kmax],   indices [L]
    """
    if codebook is None:
        return None

    # alpha effectif (compat reparam softplus)
    alpha_eff = (F.softplus(alpha_raw) + alpha_eps) if alpha_softplus else alpha_raw.clamp_min(1e-5)

    device = weight.device
    stats = {"inertia": 0.0, "usage": 0.0, "entropy": 0.0, "Keff_mean": 0.0}

    # --------- flatten par canal: [O, L] ---------
    O = weight.shape[0] if per_channel else 1
    Wv = weight.view(O, -1) if per_channel else weight.view(1, -1)  # [O, L]
    idx_all = indices if per_channel else indices.view(1, -1)        # [O, L]

    inertias, usages, ents, keffs = [], [], [], []
    for oc in range(O):
        w = Wv[oc]                  # [L]
        a = alpha_eff[oc] if per_channel else alpha_eff.mean()
        q_proxy = (w / a).clamp(-1, 1)    # cible dans l’espace des centroids
        idx = idx_all[oc]                 # [L]
        Keff = int(idx.max().item()) + 1
        keffs.append(Keff)

        # (Option) réassignation L1
        if reassign:
            cb = codebook[oc, :Keff] if per_channel else codebook[:Keff]
            d = (q_proxy[:, None] - cb[None, :]).abs()
            idx = d.argmin(dim=1)
            if per_channel:
                idx_all[oc].copy_(idx)
            else:
                indices.copy_(idx)

        cb_old = (codebook[oc, :Keff] if per_channel else codebook[:Keff]).clone()
        cb_new = cb_old.clone()
        counts = t.zeros(Keff, device=device, dtype=t.long)

        for k in range(Keff):
            mk = (idx == k)
            counts[k] = mk.sum()
            if mk.any():
                mean_k = q_proxy[mk].mean()
                cb_new[k] = (1.0 - beta) * cb_old[k] + beta * mean_k

        cb_new.clamp_(-1, 1)
        if per_channel:
            codebook[oc, :Keff].copy_(cb_new)
            q_hat = codebook[oc, :Keff][idx]
        else:
            codebook[:Keff].copy_(cb_new)
            q_hat = codebook[:Keff][idx]

        # Stats
        inertia = ((q_proxy - q_hat) ** 2).mean()
        inertias.append(inertia)

        used = (counts > 0).float().mean()
        usages.append(used)

        p = (counts.float() / counts.sum().clamp_min(1))
        entropy = -(p[p > 0] * p[p > 0].log()).sum() / (p.numel() + 1e-9)
        ents.append(entropy)

    stats["inertia"]   = t.stack([x if t.is_tensor(x) else t.tensor(x, device=device) for x in inertias]).mean().item()
    stats["usage"]     = t.stack([x if t.is_tensor(x) else t.tensor(x, device=device) for x in usages]).mean().item()
    stats["entropy"]   = t.stack([x if t.is_tensor(x) else t.tensor(x, device=device) for x in ents]).mean().item()
    stats["Keff_mean"] = t.tensor(keffs, device=device, dtype=t.float32).mean().item()
    return stats

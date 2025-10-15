# search.py (Kiwano – MPQ voir Liu et al. 2024)
import itertools
import numpy as np
import torch as t
import torch.nn as nn
from tqdm import tqdm

from .core.helpers import _central_clip, _build_codebook_from_quantiles, _assign_and_alpha

@t.no_grad()
def _kmqat_error_weight_only(W: t.Tensor, n_bits: int, r: float = 0.9,
                             symmetric_rescale: bool = True, per_channel: bool = True) -> float:
    """
    Sert à l'estimation d'erreur de quantification KMQAT (Liu et al. 2024).
    Renvoie ||W - Q||_2^2 avec Q = alpha * q_idx
    (quantiles + mean + rescale, assignation en espace [-1,1], alpha LSQ-like).
    Gère Conv1d, Conv2d (per‑channel par sortie) et Linear.
    """
    if W.ndim == 4:  # Conv2d [O, I, KH, KW]
        O, I, KH, KW = W.shape
        NperOut = I * KH * KW
        K = min(2 ** n_bits, NperOut)
        if per_channel: # 1 codebook/alpha par canal de sortie
            err = 0.0
            Wv = W.view(O, -1)
            for oc in range(O): # par canal de sortie
                w = Wv[oc]
                kept = _central_clip(w, r)
                if kept.numel() < 2:
                    continue
                c = _build_codebook_from_quantiles(kept, K, symmetric_rescale) # [K]
                q, alpha, _ = _assign_and_alpha(w, kept, c, symmetric_rescale) # q:[NperOut]
                err += (w - alpha * q).pow(2).sum().item() # erreur partielle par canal
            return float(err)
        else:
            w = W.view(-1)
            kept = _central_clip(w, r)
            if kept.numel() < 2:
                return 0.0
            K = min(2 ** n_bits, kept.numel())
            c = _build_codebook_from_quantiles(kept, K, symmetric_rescale)
            q, alpha, _ = _assign_and_alpha(w, kept, c, symmetric_rescale)
            return float((w - alpha * q).pow(2).sum().item())

    elif W.ndim == 3:  # Conv1d [O, I, K]
        O, I, Klen = W.shape
        NperOut = I * Klen
        Kc = min(2 ** n_bits, NperOut)
        err = 0.0
        Wv = W.view(O, -1)  # [O, I*K]
        for oc in range(O):
            w = Wv[oc]
            kept = _central_clip(w, r)
            if kept.numel() < 2:
                continue
            c = _build_codebook_from_quantiles(kept, Kc, symmetric_rescale)
            q, alpha, _ = _assign_and_alpha(w, kept, c, symmetric_rescale)
            err += (w - alpha * q).pow(2).sum().item()
        return float(err)

    elif W.ndim == 2:  # Linear [O, I]
        O, I = W.shape
        Kc = min(2 ** n_bits, I)
        err = 0.0
        for oc in range(O):
            w = W[oc]
            kept = _central_clip(w, r)
            if kept.numel() < 2:
                continue
            c = _build_codebook_from_quantiles(kept, Kc, symmetric_rescale)
            q, alpha, _ = _assign_and_alpha(w, kept, c, symmetric_rescale)
            err += (w - alpha * q).pow(2).sum().item()
        return float(err)

    else:
        return 0.0

@t.no_grad()
def precompute_kmqat_errors(model: nn.Module, candidate_bits, r=0.9,
                            symmetric_rescale=True, per_channel=True):
    """
    But : Précalculer les erreurs de quantification KMQAT par couche et par nombre de bits.
          permet d'éviter de recalculer à chaque évaluation dans la recherche.
    Retourne un dict {(layer_name, b): ||W - Q||^2} pour Conv1d/Conv2d/Linear.
    """
    q_errors = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)) and getattr(m, "weight", None) is not None:
            W = m.weight.detach()
            for b in candidate_bits:
                q_errors[(name, b)] = _kmqat_error_weight_only(
                    W, n_bits=b, r=r,
                    symmetric_rescale=symmetric_rescale,
                    per_channel=(isinstance(m, (nn.Conv1d, nn.Conv2d)) and per_channel)
                ) 
    return q_errors

def _layer_num_params(model: nn.Module):
    layer_params = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)) and getattr(m, "weight", None) is not None:
            layer_params[name] = m.weight.numel()
    return layer_params

# -------------------------
# recherche MPQ (4 sections + 1 bit par section, comme l’article)
# -------------------------

def find_optimal_bit_assignment(
    model: nn.Module,
    candidate_bits,                 # ex: [2,3,4]
    target_ratio: float,            # ex: 0.25  -> taille_quant ≤ 0.25 * taille_FP32
    hessian_traces: dict,           # {layer_name: trace}
    quantization_errors: dict,      # {(layer_name, bit): ||W-Q||^2}
    num_sections: int = 4,          # segmentation (article)
):
    """
    Minimise Omega_s = Σ_i Tr(H_i) * ||W_i - Q_i||^2
    sous contrainte de taille ≤ target_ratio * taille_FP32.
    """
    # 1) couches candidates (avec trace connue)
    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)) and n in hessian_traces]
    if not layer_names:
        print("Aucune couche éligible trouvée.")
        return {}

    # 2) tri par sensibilité décroissante
    sorted_layers = sorted(layer_names, key=lambda k: hessian_traces[k], reverse=True)

    # 3) segmentation en sections
    G = min(max(1, num_sections), len(sorted_layers))
    layer_groups = np.array_split(np.array(sorted_layers, dtype=object), G)

    # 4) espace de recherche: un bit par section
    bits_sorted = sorted(candidate_bits, reverse=True)   # + sensibles → + de bits
    search_space = list(itertools.product(bits_sorted, repeat=G))

    # 5) tailles
    layer_params = _layer_num_params(model)
    fp32_total_bits = sum(layer_params[n] for n in layer_names) * 32
    target_size_bits = fp32_total_bits * target_ratio

    # 6) évaluation
    results = []
    for combo in tqdm(search_space, desc="Évaluation des combinaisons"):
        total_cost = 0.0
        total_bits = 0
        feasible = True
        for g_idx, b in enumerate(combo):
            for lname in layer_groups[g_idx]:
                lname = str(lname)
                if (lname, b) not in quantization_errors:
                    feasible = False
                    break
                total_cost += hessian_traces[lname] * quantization_errors[(lname, b)]
                total_bits += layer_params[lname] * b
            if not feasible:
                break
        if feasible:
            results.append((combo, total_cost, total_bits))

    # 7) sélection sous contrainte
    valid = [(c, cost, bits) for (c, cost, bits) in results if bits <= target_size_bits]
    best = (min(valid, key=lambda x: x[1]) if valid
            else (min(results, key=lambda x: x[2]) if results else None))

    if best is None:
        print("Recherche impossible (pas de combinaisons évaluables).")
        return {}

    combo, cost, bits = best
    print("\n--- Politique MPQ trouvée ---")
    print(f"combinaison (par section): {combo}")
    print(f"coût Omega_s: {cost:.4e}")
    print(f"taille estimée: {bits/(8*1024):.2f} KiB | compression ≈ {fp32_total_bits/max(bits,1):.2f}x")

    # 8) dictionnaire final (par couche)
    bit_assignment = {}
    for g_idx, b in enumerate(combo):
        for lname in layer_groups[g_idx]:
            bit_assignment[str(lname)] = int(b)
    return bit_assignment

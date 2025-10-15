import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrappers KMQAT
from .wrappersBETA import (
    KMeansQuantConv1d,
    KMeansQuantConv2d,
    KMeansQuantLinear,
)

# ============================================================
# MAPPING couches -> wrappers KMQAT
# ============================================================
QWRAP = {
    nn.Conv1d: KMeansQuantConv1d,
    nn.Conv2d: KMeansQuantConv2d,
    nn.Linear: KMeansQuantLinear,
}

# ============================================================
# QAT UNIFORME: wrapping récursif + (option) activation
# ============================================================
def prepare_model_for_uniform_qat(
    model: nn.Module,
    n_bits: int = 8,
    retention_ratio: float = 0.9,
    symmetric_rescale: bool = True,
    per_channel: bool = True,
    debug_stats: bool = False,
    exclude_names=None,
    start_enabled: bool = True,
) -> nn.Module:
    """
        Rôle. Remplacer récursivement les Conv2d/Linear FP32 
        par les wrappers KMeansQuant* et appeler enable_quantization() 
        pour construire les paramètres KMQAT (codebook/indices/Alpha).
    """
    if exclude_names is None:
        exclude_names = []

    def _print_stats(layer, layer_name, kind="BEFORE"):
        w = layer.weight.data.detach().cpu().numpy()
        print(f"[{kind}] {layer_name}: min={w.min():.4f} max={w.max():.4f} "
              f"mean={w.mean():.4f} std={w.std():.4f} shape={w.shape}")

    def _recursive_prepare(module, prefix=""):
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name

            # descente récursive
            if len(list(child.children())):
                _recursive_prepare(child, prefix=full)

            # wrapping si type supporté
            if type(child) in QWRAP:
                if any(ex in full for ex in exclude_names):
                    print(f"[SKIP QUANT] {full} ({type(child).__name__})")
                    continue

                if debug_stats:
                    _print_stats(child, full, "BEFORE")

                Wrapped = QWRAP[type(child)]
                wrapped = Wrapped(
                    child,
                    n_bits=n_bits,
                    retention_ratio=retention_ratio,
                    symmetric_rescale=symmetric_rescale,
                    per_channel=per_channel,
                    debug=debug_stats,
                )
                if start_enabled:
                    wrapped.enable_quantization()

                setattr(module, name, wrapped)
                if debug_stats and start_enabled:
                    qw = wrapped._quantized_weight().detach().cpu().numpy()
                    print(f"[AFTER ] {full}: min={qw.min():.4f} max={qw.max():.4f} "
                          f"mean={qw.mean():.4f} std={qw.std():.4f} shape={qw.shape}")
                print(f"→ {full} ({type(child).__name__}) → {n_bits}-bit")

    _recursive_prepare(model)
    return model


# ============================================================
# QAT MIXTE (MPQ): wrapping selon bit_assignment
# ============================================================
def prepare_model_for_mixed_qat(
    model: nn.Module,
    bit_assignment: Dict[str, int],
    retention_ratio: float = 0.9,
    symmetric_rescale: bool = True,
    per_channel: bool = True,
    start_enabled: bool = False,
    debug_stats: bool = False,
) -> nn.Module:
    """
    Remplace uniquement les modules présents dans bit_assignment par KMQAT wrappers
    configurés avec leur n_bits respectif. start_enabled=False pour MSFT (on active par paliers).
    """
    def replace_one(full_name: str, mod: nn.Module, n_bits: int):
        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        child_name  = full_name.rsplit(".", 1)[1] if "." in full_name else full_name
        parent      = model.get_submodule(parent_name) if parent_name else model

        Wrapped = QWRAP[type(mod)]
        wrapped = Wrapped(
            mod,
            n_bits=n_bits,
            retention_ratio=retention_ratio,
            symmetric_rescale=symmetric_rescale,
            per_channel=per_channel,
            debug=debug_stats,
        )
        if start_enabled:
            wrapped.enable_quantization()
        setattr(parent, child_name, wrapped)
        print(f"[wrap] {full_name} → {n_bits}-bit")

    for name, mod in model.named_modules():
        if name in bit_assignment and type(mod) in QWRAP:
            replace_one(name, mod, bit_assignment[name])

    return model


# ============================================================
# BN Recalibration (utile après wrapping/activation)
# ============================================================
@torch.no_grad()
def recalibrate_bn(model, loader, device, num_passes: int = 2):
    """
    Remet les BN en mode train, fait quelques passes *sans grad* avec
    les mêmes shapes que pendant le training pour mettre à jour
    running_mean/var. Puis rend au mode initial.
    """
    was_training = model.training
    model.train()

    for _ in range(num_passes):
        for x, _ in loader:
            # même pré-processing que dans la boucle train
            if x.dim() == 3:              # [B, 81, 350]
                x = x.unsqueeze(1)        # -> [B, 1, 81, 350]
            x = x.float().to(device)

            # ResNetV2.forward(x, iden=None) retourne les embeddings,
            # c’est suffisant pour faire traverser toutes les BN.
            try:
                _ = model(x, None) if hasattr(model, 'forward') else model(x)
            except TypeError:
                # Si 'model' est un DDP-wrapped module, l'appel simple marche aussi,
                # sinon, on tente sans l'argument 'None'.
                _ = model(x)

    if not was_training:
        model.eval()


# ============================================================
# Kick-start Alpha: matcher l'écart-type par canal
# ============================================================
@torch.no_grad()
def match_weight_std_per_channel(model: nn.Module):
    """
    Pour chaque couche quantifiée, ajuste alpha pour approx. égaliser
    std(W) et std(q) canal par canal: alpha <- alpha * (sW / sQ).

    Q = ALPHA·q par canal égale ~ celui des poids FP32.
    Géré pour Conv1d/Conv2d/Linear, per-channel et per-tensor.

    idée proche de https://arxiv.org/pdf/1902.08153
    """
    for m in model.modules():
        # Conv1d
        if isinstance(m, KMeansQuantConv1d) and getattr(m, "quantization_enabled", False) and m.codebook is not None:
            W = m.weight           # [O, I, K]
            O = W.size(0)
            rows = []
            if m.codebook.dim() == 2:  # Pour chaque canal de sortie [O,K]
                for oc in range(O):
                    cb = m.codebook[oc]; idx = m.indices[oc]; q = cb[idx]; rows.append(q)
                Qv = torch.stack(rows).view_as(W)
                sW = W.view(O, -1).std(dim=1).clamp_min(1e-8)
                sQ = Qv.view(O, -1).std(dim=1).clamp_min(1e-8)
                m.alpha.mul_(sW / sQ) # Écart-type des poids FP32 / Écart-type des poids quantifiés 
            else:  # per-tensor
                q = m.codebook[m.indices].view_as(W)
                sW = W.std().clamp_min(1e-8)
                sQ = q.std().clamp_min(1e-8)
                # alpha est [O] → on met l'échelle moyenne
                m.alpha.mul_((sW / sQ)) 
        # Conv2d
        if isinstance(m, KMeansQuantConv2d) and getattr(m, "quantization_enabled", False) and m.codebook is not None:
            W = m.weight           # [O, I, KH, KW]
            O = W.size(0)
            rows = []
            if m.codebook.dim() == 2:  # Pour chaque canal de sortie  [O,K]
                for oc in range(O):
                    cb = m.codebook[oc]; idx = m.indices[oc]; q = cb[idx]; rows.append(q)
                Qv = torch.stack(rows).view_as(W)
                sW = W.view(O, -1).std(dim=1).clamp_min(1e-8)
                sQ = Qv.view(O, -1).std(dim=1).clamp_min(1e-8)
                m.alpha.mul_(sW / sQ) # Écart-type des poids FP32 / Écart-type des poids quantifiés 
            else:  # per-tensor
                q = m.codebook[m.indices].view_as(W)
                sW = W.std().clamp_min(1e-8)
                sQ = q.std().clamp_min(1e-8)
                m.alpha.mul_((sW / sQ))

        # Linear
        if isinstance(m, KMeansQuantLinear) and getattr(m, "quantization_enabled", False) and m.codebook is not None:
            W = m.weight           # [O, I]
            O = W.size(0)
            if m.codebook.dim() == 2:  # Pour chaque canal de sortie  [O,K]
                rows = []
                for oc in range(O):
                    cb = m.codebook[oc]; idx = m.indices[oc]; q = cb[idx]; rows.append(q)
                Q = torch.stack(rows).view_as(W)
                sW = W.view(O, -1).std(dim=1).clamp_min(1e-8)
                sQ = Q.view(O, -1).std(dim=1).clamp_min(1e-8)
                m.alpha.mul_(sW / sQ) # Écart-type des poids FP32 / Écart-type des poids quantifiés 
            else:                      # per-tensor
                q = m.codebook[m.indices].view_as(W)
                sW = W.std().clamp_min(1e-8)
                sQ = q.std().clamp_min(1e-8)
                m.alpha.mul_(sW / sQ) 


# ============================================================
# Optimiseur: LR(alpha) ↑, WD(alpha)=0  (+ option codebooks)
# ============================================================
def make_optimizer_with_param_groups(model: nn.Module,
                                     base_lr_w: float = 5e-4,
                                     lr_alpha: float = 5e-3,
                                     weight_decay_w: float = 1e-4,
                                     momentum: float = 0.9):
    """
    But : Créer un optimiseur avec des groupes de paramètres distincts pour les poids, les alpha et le codebook.
    - Les poids (weight_params) ont un LR de base et un weight decay.
    - Les alpha (alpha_params) ont un LR plus élevé et pas de weight decay.
    - Le codebook (codebook_params) peut être ajouté si on veut l'entraîner en Phase B.

    """
    alpha_params, weight_params, codebook_params = [], [], []
    for m in model.modules():
        if hasattr(m, "alpha") and isinstance(m.alpha, torch.Tensor):
            alpha_params.append(m.alpha)
        if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor):
            weight_params.append(m.weight)
        if hasattr(m, "codebook") and isinstance(m.codebook, torch.Tensor):
            codebook_params.append(m.codebook)

    optim = torch.optim.SGD([
        {"params": weight_params,  "lr": base_lr_w, "weight_decay": weight_decay_w, "momentum": momentum},
        {"params": alpha_params,   "lr": lr_alpha,  "weight_decay": 0.0,            "momentum": momentum},
        # Pour phase B (si on veux aussi finetune le codebook) : décommente ci‑dessous
        # {"params": codebook_params, "lr": 1e-3, "weight_decay": 0.0, "momentum": momentum},
    ])
    return optim


# ============================================================
# Helpers MSFT: staging par seuil de bits
# ============================================================
def enable_quant_for_stage(model: nn.Module, bit_assignment: Dict[str, int], threshold_bit: int, verbose: bool = True):
    """
    Active la quantif pour les couches dont bit <= threshold_bit.
    Utilisé par MSFT: on commence par les plus bas bits, puis on élargit.
    """
    for name, m in model.named_modules():
        if name in bit_assignment and bit_assignment[name] <= threshold_bit:
            if hasattr(m, "enable_quantization") and not getattr(m, "quantization_enabled", False):
                m.enable_quantization()
                if verbose:
                    print(f"[MSFT] enabled: {name} ({bit_assignment[name]}-bit)")


def freeze_bn_stats(model: nn.Module):
    """
    Met en eval les BN pour figer running stats (bien après quelques epochs).
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def train_epochs_with_optimizer(model, train_loader, val_loader, device, optimizer, num_epochs=5):
    """
    Boucle simple d'entraînement (classification CE), pratique pour CIFAR10
    ou un petit finetune de validation. Pour VoxCeleb, voir DDP.
    """
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, y) if model.__class__.__name__ == "ResNetV2" else model(x)  # ResNetV2 attend (x, iden) pour la loss
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_corrects += (logits.argmax(1) == y).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc  = running_corrects / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss, val_acc = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                logits = model(x, y) if model.__class__.__name__ == "ResNetV2" else model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_acc  += (logits.argmax(1) == y).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc  /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")


# ============================================================
# Checkpoints (sauve eff_codebook = alpha × codebook)
# ============================================================
def save_quantized_checkpoint(model, optimizer, epoch, filepath, **kwargs):
    """
    Sauvegarde QAT complète:
      - "model": state_dict() (reprend le FT; inclut alpha, codebook, etc.)
      - "quantization_data": centroids effectifs (alpha×codebook), indices, bias, shape
    """
    quantization_data = {}
    for name, module in model.named_modules():
        has_q = hasattr(module, "codebook") and module.codebook is not None \
                and hasattr(module, "indices") and module.indices is not None
        if not has_q:
            continue

        # centroids effectifs
        alpha_eff = F.softplus(module.alpha.detach()) + getattr(module, "_alpha_eps", 1e-5) # alpha > 0 garanti

        if module.codebook.dim() == 2:   # per-channel [O, K]
            eff_cb = module.codebook.detach() * alpha_eff.view(-1, 1)
        else:                             # per-tensor [K]
            eff_cb = module.codebook.detach() * alpha_eff.mean()

        quantization_data[name] = {
            "eff_codebook": eff_cb.cpu(),                                 # (O,K) ou (K,)
            "indices": module.indices.detach().cpu().to(torch.int64),     # [..] int64
            "bias": module.bias.detach().cpu() if module.bias is not None else None,
            "shape": tuple(module.weight.shape),
        }

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "quantization_data": quantization_data,
        **kwargs
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, filepath)
    print(f"[save] QAT checkpoint époch {epoch} → {filepath}")


# ============================================================
# Checkpoints (Charge chpt et reconstruit Wq)
# ============================================================

def _reconstruct_Wq_from_quant_data(mod, d):
    """
    Reconstruit le TENSEUR DE POIDS DÉQUANTIFIÉ Wq (au format FP32) à partir des
    artefacts de quantification stockés dans le checkpoint d'entraînement KMQAT :

      - eff_codebook : centroids *déjà scalés* (Alpha x codebook)
                       shape (O,K) en per-channel ou (K,) en per-tensor.
      - indices      : index entiers vers eff_codebook
                       shape (O,NperOut) ou (N,)
      - shape        : shape originale du poids (ex. [O,I,KH,KW]).

    ! Important :
    Wq est retourné en FP32 pour ré-initialiser les poids du module PyTorch standard.
    Les VALEURS sont quantifiées (prises sur la grille), mais le TENSEUR est FP32.
    On évalue donc l'impact accuracy d'une W-quantization, sans chemin d'exécution INT8.
    """
    eff_cb = d.get("eff_codebook", None) #lit d["eff_codebook"]
    idx    = d.get("indices", None) #lit d["indices"]
    shape  = tuple(d["shape"])
    if eff_cb is None or idx is None:
        return None

    eff_cb = eff_cb.to(torch.float32) # cast en FP32 pour l'inference !
    idx    = idx.to(torch.int64)

    # --- per-channel: eff_cb [O,K], idx [O,N] ---
    if eff_cb.dim() == 2 and idx.dim() == 2:
        O, K = eff_cb.shape
        if idx.size(0) != O:
            return None
        rows = []
        for oc in range(O):
            # idx[oc]: [NperOut] # Pour chaque out-channel oc : on remplace les indices par leurs centroids FP32
            rows.append(eff_cb[oc].index_select(0, idx[oc]))
        Qv = torch.stack(rows, dim=0)         # [O, NperOut]
        Wq = Qv.view(*shape)
        return Wq

    # --- per-tensor: eff_cb [K], idx [N] ---
    if eff_cb.dim() == 1 and idx.dim() == 1:
        q  = eff_cb.index_select(0, idx)      # [N]
        Wq = q.view(*shape)
        return Wq

    return None  # format inattendu → fallback

def load_inference_model_from_checkpoint(filepath, base_model_architecture):
    """
    Recharge un checkpoint KMQAT (eff_codebook, indices, bias) et reconstruit un
    modèle d'inférence en écrasant uniquement les poids des couches quantifiées.
    """
    ckpt  = torch.load(filepath, map_location='cpu')
    qdata = ckpt.get("quantization_data", {})

    model = base_model_architecture()
    model.load_state_dict(ckpt["model"], strict=False)

    for name, mod in model.named_modules():
        if name not in qdata:
            continue
        d = qdata[name]

        # Compat anciens ckpts (où seule "codebook" existait)
        if "eff_codebook" not in d and "codebook" in d:
            d = {
                "eff_codebook": d["codebook"],
                "indices": d["indices"],
                "bias": d.get("bias", None),
                "shape": d["shape"],
            }

        Wq = _reconstruct_Wq_from_quant_data(mod, d)
        if Wq is None:
            print(f"[WARN] Reconstruction quantifiée échouée pour '{name}'. Fallback FP32.")
            continue

        with torch.no_grad():
            mod.weight.data.copy_(Wq) # mod.weight est un tenseur FP32 mais les valeurs sont en 8bits donc 'quantifié en contenu et FP32 en format'
            if d.get("bias", None) is not None and mod.bias is not None:
                mod.bias.data.copy_(d["bias"].to(mod.bias.dtype))

    model.eval()
    return model

# ============================================================
# Rafraîchissement global des codebooks (EMA + option réassignation)
# ============================================================
@torch.no_grad()
def ema_refresh_codebooks(model: nn.Module, beta: float = 0.1, reassign: bool = False, verbose: bool = True):
    """
    Parcourt le modèle et appelle refresh_codebook() sur chaque wrapper KMQAT activé.
    Retourne des stats agrégées (inertie/usage/entropie/Keff).
    """
    
    stats_all = {"inertia": [], "usage": [], "entropy": [], "Keff_mean": []}
    for m in model.modules():
        if isinstance(m, (KMeansQuantConv1d, KMeansQuantConv2d, KMeansQuantLinear)):
            if getattr(m, "quantization_enabled", False) and (m.codebook is not None):
                st = m.refresh_codebook(beta=beta, reassign=reassign)
                if st is not None:
                    for k in stats_all.keys():
                        stats_all[k].append(torch.tensor(st[k], dtype=torch.float32))

    out = {}
    for k, arr in stats_all.items():
        if len(arr):
            out[k] = torch.stack(arr).mean().item()

    if verbose and len(out):
        print(f"[CB-REFRESH] beta={beta:.3f} reassign={reassign} | "
              f"inertia={out.get('inertia', float('nan')):.4e}  "
              f"usage={out.get('usage', float('nan')):.3f}  "
              f"entropy={out.get('entropy', float('nan')):.3f}  "
              f"Keff≈{out.get('Keff_mean', float('nan')):.1f}")
    return out

# ============================================================
# Réassignation + recalage fermé de alpha (par couche)
# ============================================================

@torch.no_grad()
def _module_alpha_eff(m: nn.Module, eps: float = 1e-5):
    """alpha 'effectif' compatible softplus si présent."""
    if hasattr(m, "_alpha_softplus") and getattr(m, "_alpha_softplus", False):
        return F.softplus(m.alpha) + getattr(m, "_alpha_eps", eps)
    return m.alpha.clamp_min(eps)

@torch.no_grad()
def recompute_alpha_closed_form_(m: nn.Module, eps: float = 1e-5):
    """
    Recalcule alpha canal-par-canal par la formule fermée:
        alpha = <W, Q> / <Q, Q>   avec Q = codebook[indices]
    en utilisant les indices courants.
    """
    if not hasattr(m, "codebook") or m.codebook is None:
        return None
    if not hasattr(m, "indices") or m.indices is None:
        return None
    if not hasattr(m, "alpha"):
        return None

    w = m.weight
    alpha_old = m.alpha.clone()
    if m.per_channel and m.codebook.dim() == 2:
        out_c = w.shape[0]
        new_alpha = torch.zeros_like(m.alpha)
        for oc in range(out_c):
            # Q canal courant
            idx = m.indices[oc]                 # [L]
            Keff = int(idx.max().item()) + 1
            q = m.codebook[oc, :Keff][idx]      # [L] (dans [-1,1])
            # remettre à l'échelle pour comparer à W
            q = q * _module_alpha_eff(m, eps=eps)[oc]
            wvec = w[oc].reshape(-1)
            num = (wvec * q).sum()
            den = (q * q).sum().clamp_min(eps)
            new_alpha[oc] = (num / den).clamp_min(eps)
        m.alpha.copy_(new_alpha)
    else:
        idx = m.indices
        Keff = int(idx.max().item()) + 1
        q = m.codebook[:Keff][idx] * _module_alpha_eff(m, eps=eps).mean()
        wvec = w.reshape(-1)
        num = (wvec * q).sum()
        den = (q * q).sum().clamp_min(eps)
        m.alpha.copy_((num / den).clamp_min(eps))

    delta = (m.alpha - alpha_old).abs().mean().item()
    return {"alpha_l1mean": float(delta)}

@torch.no_grad()
def reassign_indices_(m: nn.Module):
    """
    Réassigne les indices par nearest-centroid dans l'espace des centroids:
        q_proxy = clamp(w / alpha_eff, -1, 1)
    puis idx = argmin |q_proxy - c_k|.
    """
    if not hasattr(m, "codebook") or m.codebook is None:
        return None
    if not hasattr(m, "indices") or m.indices is None:
        return None

    w = m.weight
    aeff = _module_alpha_eff(m)
    moved = 0.0

    if m.per_channel and m.codebook.dim() == 2:
        out_c = w.shape[0]
        for oc in range(out_c):
            wvec = w[oc].reshape(-1)               # [L]
            q_proxy = (wvec / aeff[oc]).clamp(-1, 1)
            idx_old = m.indices[oc]
            Keff = int(idx_old.max().item()) + 1
            cb = m.codebook[oc, :Keff]             # [Keff]
            d = (q_proxy[:, None] - cb[None, :]).abs()  # L1 robuste
            idx_new = d.argmin(dim=1)
            moved += (idx_new != idx_old).float().mean().item()
            m.indices[oc] = idx_new
        moved /= float(out_c)
    else:
        wvec = w.reshape(-1)
        q_proxy = (wvec / aeff.mean()).clamp(-1, 1)
        idx_old = m.indices
        Keff = int(idx_old.max().item()) + 1
        cb = m.codebook[:Keff]
        d = (q_proxy[:, None] - cb[None, :]).abs()
        idx_new = d.argmin(dim=1)
        moved = (idx_new != idx_old).float().mean().item()
        m.indices.copy_(idx_new)

    return {"moved_frac": float(moved)}

@torch.no_grad()
def reassign_and_recalc_alpha_all(model: nn.Module, verbose: bool = True):
    """
    Passe sur toutes les couches KMQAT et :
      1) réassigne les indices,
      2) recalcule alpha (formule fermée).

    But : réduire l'inertie + ajuster alpha. Permet de meilleure convergence.
    """

    moved_fracs, dalphas = [], []
    for m in model.modules():
        if isinstance(m, (KMeansQuantConv1d, KMeansQuantConv2d, KMeansQuantLinear)):
            if not getattr(m, "quantization_enabled", False):
                continue
            if m.codebook is None or m.indices is None:
                continue
            st1 = reassign_indices_(m)
            st2 = recompute_alpha_closed_form_(m)
            if st1:
                moved_fracs.append(torch.tensor(st1["moved_frac"]))
            if st2:
                dalphas.append(torch.tensor(st2["alpha_l1mean"]))
    if verbose and (len(moved_fracs) or len(dalphas)):
        mf = torch.stack(moved_fracs).mean().item() if moved_fracs else 0.0
        da = torch.stack(dalphas).mean().item() if dalphas else 0.0
        print(f"[REASSIGN+ALPHA] moved_frac≈{mf:.3f}  Δ|alpha|_mean≈{da:.3e}")
    return

# ============================================================
# --- EMA adaptative + metriques Keff/usage ---

@torch.no_grad()
def estimate_usage_keff_entropy(model: nn.Module):
    """
    Heuristique globale (moyennée) : usage moyen, Keff total, entropie moyenne
    en se basant sur m.indices/m.codebook des wrappers KMQAT.
    """
    from .wrappersBETA import (KMeansQuantConv1d, KMeansQuantConv2d, KMeansQuantLinear)
    total_used, total_K = 0, 0
    usages, ents = [], []
    for m in model.modules():
        if isinstance(m, (KMeansQuantConv1d, KMeansQuantConv2d, KMeansQuantLinear)):
            idx = getattr(m, "indices", None)
            cb  = getattr(m, "codebook", None)
            if isinstance(idx, torch.Tensor) and isinstance(cb, torch.Tensor) and cb.numel() > 0:
                if m.codebook.dim() == 2:
                    # moyenne par canal
                    K = m.codebook.size(1)
                    hist_used = []
                    ent_local = []
                    for oc in range(idx.size(0)):
                        h = torch.bincount(idx[oc].view(-1), minlength=K).float()
                        used = (h > 0).float().mean().item()
                        p = h / (h.sum() + 1e-12)
                        ent = -(p[p>0]*p[p>0].log()).sum().item() / (torch.log(torch.tensor(float(K))).item()+1e-12)
                        hist_used.append(used); ent_local.append(ent)
                    usages.append(float(torch.tensor(hist_used).mean()))
                    ents.append(float(torch.tensor(ent_local).mean()))
                    total_used += K  # approx
                    total_K    += K
                else:
                    K = m.codebook.size(0)
                    h = torch.bincount(idx.view(-1), minlength=K).float()
                    used = (h > 0).float().mean().item()
                    p = h / (h.sum() + 1e-12)
                    ent = -(p[p>0]*p[p>0].log()).sum().item() / (torch.log(torch.tensor(float(K))).item()+1e-12)
                    usages.append(used); ents.append(ent)
                    total_used += (h > 0).sum().item()
                    total_K    += K
    if total_K == 0:
        return None
    return dict(
        usage=float(torch.tensor(usages).mean()) if usages else 0.0,
        Keff=int(total_used),  # approx globale
        entropy=float(torch.tensor(ents).mean()) if ents else 0.0
    )

def cosine_beta(ep, T, bmin, bmax):
    import math
    return bmin + 0.5*(bmax-bmin)*(1 - math.cos(math.pi * (ep/(max(T-1,1)))))

def decide_beta(policy, epoch, T, bmin, bmax, inertia=None, usage=None, Keff=None):
    """
    - fixed: retourne bmin
    - cosine: planning cos
    - adaptive: pousse beta si inertia haute / usage<.90 / Keff<240 ; sinon garde bas
    """
    if policy == "fixed":
        return bmin
    if policy == "cosine":
        return cosine_beta(epoch, T, bmin, bmax)
    # adaptive
    beta = bmin
    if (inertia is not None and inertia > 1e-2) or (usage is not None and usage < 0.90) or (Keff is not None and Keff < 240):
        beta = min(bmax, bmin + 0.10)  # boost d’un cran
    elif (inertia is not None and inertia < 5e-3) and (usage is not None and usage > 0.95) and (Keff is not None and Keff >= 245):
        beta = max(bmin, bmin)         # reste bas
    return float(beta)
# ============================================================
# kiwano/quantization/monitor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import idr_torch

@torch.no_grad()
def log_alpha_stats_rank0(model_ddp, epoch, step=None, total_steps=None):
    if idr_torch.rank != 0:
        return
    vals = []
    for m in model_ddp.module.modules():
        a = getattr(m, "alpha", None)
        if isinstance(a, torch.Tensor):
            a_eff = F.softplus(a.detach()) + getattr(m, "_alpha_eps", 0.0)
            vals.append(a_eff.view(-1))
    if not vals:
        return
    a = torch.cat(vals, dim=0).float()
    txt = (f"[ALPHA] epoch {epoch+1}" if step is None else f"[ALPHA] epoch {epoch+1} step {step}/{total_steps}")
    print(f"{txt}: min={a.min():.4e} mean={a.mean():.4e} max={a.max():.4e}")

@torch.no_grad()
def codebook_fingerprint(model: nn.Module):
    vals = []
    for m in model.modules():
        cb = getattr(m, "codebook", None)
        if isinstance(cb, torch.Tensor):
            c = cb.detach().float().view(-1)
            if c.numel() > 0:
                vals.append(torch.stack([c.mean(), c.std(), c.abs().mean()]))
    if not vals:
        return None
    return torch.stack(vals, dim=0).mean(dim=0)

@torch.no_grad()
def log_quant_error_layers(model_ddp, tag=""):
    if idr_torch.rank != 0:
        return
    picks = ["preresnet.pre_conv1", "preresnet.layer2.0.downsample.0", "embedding.fc_embed"]
    lines = []
    for name, m in model_ddp.module.named_modules():
        if any(p in name for p in picks) and hasattr(m, "_quantized_weight"):
            W = m.weight.detach().float()
            Wq = m._quantized_weight().detach().float()
            err = (W - Wq).norm() / (W.norm() + 1e-12)
            cs  = torch.nn.functional.cosine_similarity(W.view(1, -1), Wq.view(1, -1)).item()
            lines.append(f"{name}: q_err={err:.3e} cos={cs:.4f}")
    if lines:
        print(f"[QERR]{(' '+tag) if tag else ''} " + " | ".join(lines))

def set_ste_weight_only(model: nn.Module, flag: bool):
    for m in model.modules():
        if hasattr(m, "ste_weight_only"):
            m.ste_weight_only = flag

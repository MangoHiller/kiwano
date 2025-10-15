# sensitivity.py
import torch
import torch.nn as nn

@torch.no_grad()
def _next_batch(it, loader):
    """Récupère le prochain batch, sinon réinitialise l'itérateur."""
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it

def estimate_hessian_traces(model: nn.Module,
                            data_loader,
                            criterion,
                            device,
                            num_iterations: int = 10, forward_fn=None,):
    """
    Estime Tr(H) par couche (Conv1d/Conv2d/Linear) via Hutchinson:
        E_v[v^T H v] ~ (1/T) * Σ (v^T H v).
    Le ResNetV2 (speaker‑verif) peut attendre (x, labels) au forward.

    Notes:
    - On veut des LOGITS pour calculer la perte: `loss = criterion(logits, labels)`.
    - `forward_fn` (optionnel) doit être un callable: forward_fn(x) -> logits.
    - Si `forward_fn` n'est pas fourni, on fait:
        1) model.forward_logits(x) si présent,
        2) sinon model(x) (qui doit alors retourner des logits).
    """
    # Cibles: modules avec 'weight' quantifiables
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear))
              and (getattr(m, "weight", None) is not None)]
    traces = {n: 0.0 for n, _ in layers}
    counts = {n: 0   for n, _ in layers}

    model.eval().to(device)   # on évite de bouger les BN stats
    data_iter = iter(data_loader)

    for _ in range(num_iterations):
        (x, y), data_iter = _next_batch(data_iter, data_loader)

        # Mise en forme entrée: (N,1,H,W) si besoin
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.to(device)
        y = y.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)

        # 1) FORWARD: produire des LOGITS (pas de label passé au modèle)
        if forward_fn is not None:
            logits = forward_fn(x)
        elif hasattr(model, "forward_logits"):
            logits = model.forward_logits(x)
        else:
            # fallback: on suppose que model(x) renvoie déjà des logits
            logits = model(x)

        # 2) Perte CE sur logits
        loss = criterion(logits, y)

        # 2) pour chaque couche, g = dL/dW (graph), puis hv = d(g·v)/dW
        for name, m in layers:
            g = torch.autograd.grad(loss, m.weight,
                                    create_graph=True,
                                    retain_graph=True,
                                    allow_unused=True)[0]
            if g is None:
                continue

            # v ~ Rademacher estimation
            v = torch.empty_like(m.weight, device=device).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            gv = (g * v).sum()

            hv = torch.autograd.grad(gv, m.weight,
                                     retain_graph=True,
                                     allow_unused=True)[0]
            if hv is None:
                continue

            traces[name] += (hv * v).sum().item()
            counts[name] += 1

            del g, v, gv, hv

        # libère le graphe de cette itération
        del loss, logits
        model.zero_grad(set_to_none=True)

    # Moyenne par couche
    for n in traces:
        if counts[n] > 0:
            traces[n] /= counts[n]

    return traces

#!/usr/bin/env python3

"""
Uniform KMQAT for ResNetV2 (speaker verification, VoxCeleb2) – DDP

- Wrappe toutes les Conv1d/Conv2d/Linear avec KMeansQuant* au même n_bits
- Active la quantization dès le départ (uniform QAT)
- Reprise depuis un checkpoint QAT uniforme
- Sauvegarde des checkpoints avec "eff_codebook" (alpha×codebook) + indices (utils.save_quantized_checkpoint)
"""

import os
import sys
import argparse
import copy
import time
import logging
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

# KIwano: modèles / dataset / features / aug
from kiwano.model import ResNetV2, MiniBasicBlock, MiniSEBasicBlock, BasicBlock, SEBasicBlock
from kiwano.dataset import Segment, SegmentSet
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Linear, CMVN, Crop

# Quantization toolbox (MAJ)
from kiwano.quantization.utils import (
    prepare_model_for_uniform_qat,
    save_quantized_checkpoint,
    recalibrate_bn,            # BN recal utile mais optionnelle
    ema_refresh_codebooks,
    estimate_usage_keff_entropy,
    decide_beta,

)

# Quantization Monitors / logs alpha, quant error, ste_weight_only
from kiwano.quantization.monitor import (  
    log_alpha_stats_rank0,
    codebook_fingerprint,
    log_quant_error_layers,
    set_ste_weight_only,
)

# DDP / SLURM
import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ===== DEBUG/INSPECT =====
PRINT_MODELS_BEFORE_AFTER = False  # mettre True ponctuellement pour imprimer avant/après wrapping (rank-0 seule)
ALPHA_LOG_EVERY_STEPS = 500        # log alpha toutes les N itérations (rank-0 seule)
# =========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_resnet_model(model_size: str, num_classes: int, width_mult: float = 1.0):
    """Construit un ResNetV2 KIwano."""
    base_channels = {
        "resnet18":  [128, 128, 256, 256],
        "resnet36":  [128, 128, 256, 256],
        "resnet50":  [128, 128, 256, 256],
        "resnet101eq": [128, 128, 256, 256],
        "resnet101": [128, 128, 256, 256],
        "resnet200": [128, 128, 256, 256],
        "resnet400": [128, 128, 256, 256],
        "resnet800": [128, 128, 256, 256],
    }
    resnet_config = {
        "resnet18":   ([2, 2, 2, 2],  MiniBasicBlock, MiniSEBasicBlock),
        "resnet36":   ([3, 4, 6, 3],  MiniBasicBlock, MiniSEBasicBlock),
        "resnet50":   ([3, 4, 6, 3],  BasicBlock,    SEBasicBlock),
        "resnet101":  ([3, 4, 23, 3], BasicBlock,    SEBasicBlock),
        "resnet101eq":([3, 10, 17, 3],BasicBlock,    SEBasicBlock),
        "resnet200":  ([3, 24, 36, 3],BasicBlock,    SEBasicBlock),
        "resnet400":  ([4, 44, 87, 4],BasicBlock,    SEBasicBlock),
        "resnet800":  ([8, 88, 174, 8],BasicBlock,   SEBasicBlock),
    }
    if model_size not in resnet_config:
        raise ValueError(f"model_size invalide ({model_size}), choix possibles : {list(resnet_config)}")
    num_blocks, block, block_se = resnet_config[model_size]
    channels = [int(c * width_mult) for c in base_channels[model_size]]
    return ResNetV2(
        num_classes=num_classes,
        channels=channels,
        num_blocks=num_blocks,
        block=block,
        block_se=block_se
    )

class SpeakerTrainingSegmentSet(Dataset, SegmentSet):
    """Dataset KIwano (audio -> fbank -> CMVN + Crop).
       NB: si un item est invalide, on renvoie (zeros, -1)"""
    def __init__(self,
                 audio_transforms: List[Augmentation] = None,
                 feature_extractor=None,
                 feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]):
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)
        try:
            audio, sample_rate = segment.load_audio()
            if audio.shape[0] == 0:
                logger.warning(f" Segment audio vide : {segment_id_or_index}")
                return torch.zeros((1, 81, 350)), -1

            if self.audio_transforms is not None:
                audio, sample_rate = self.audio_transforms(audio, sample_rate)

            feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate) if self.feature_extractor else None
            if self.feature_transforms is not None:
                feature = self.feature_transforms(feature)

            return feature, self.labels[segment.spkid]

        except Exception as e:
            spk = getattr(segment, "spkid", "UNK")
            logger.error(f" Erreur sur le segment {spk}: {e}")
            return torch.zeros((1, 81, 350)), -1

def _flat_types(model):
    return {name: type(mod).__name__ for name, mod in model.named_modules()}

def print_wrap_diff(before_model, after_model):
    before = _flat_types(before_model)
    after  = _flat_types(after_model)
    names = sorted(set(before.keys()) | set(after.keys()))
    wrapped, skipped = [], []
    for n in names:
        tb = before.get(n); ta = after.get(n)
        if tb is None or ta is None:
            continue
        if tb != ta:
            wrapped.append((n, tb, ta))
        elif ("Conv" in tb or "Linear" in tb) and ("KMeansQuant" not in ta):
            skipped.append((n, tb))
    print("\n=== WRAP DIFF (avant -> après) ===")
    for n, tb, ta in wrapped:
        print(f"[WRAPPED] {n}: {tb}  ->  {ta}")
    for n, tb in skipped:
        print(f"[SKIPPED] {n}: {tb} (resté FP)")

# ------------------------------------------------------------------
# Script principal
# ------------------------------------------------------------------

def main():
    # Logs SLURM/DDP
    print(str(idr_torch.master_addr))
    print(str(idr_torch.master_port))
    print(str(idr_torch.local_rank))

    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']

    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostname), " nodes and ", idr_torch.size, " processes, master node is ", MASTER_ADDR)
    print(f"- Process {idr_torch.rank} corresponds to GPU {idr_torch.local_rank} of node {NODE_ID}")

    parser = argparse.ArgumentParser(description="Uniform KMQAT pour ResNetV2 (speaker verification)")

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--model_path", type=Path, required=True, help="Checkpoint FP32 pré‑entraîné (source).")
    parser.add_argument("--output_path", type=Path, required=True, help="Dossier de sortie (checkpoints).")
    parser.add_argument("--data_path", type=Path, required=True, help="Dossier données (calibration/fine‑tuning).")
    parser.add_argument("--resume_from", type=Path, default=None, help="Checkpoint QAT uniforme pour reprise.")

    parser.add_argument("--model_size",
                        type=str,
                        choices=["resnet18","resnet36","resnet50","resnet101","resnet101eq","resnet200","resnet400","resnet800"],
                        default="resnet36",
                        help="Taille du modèle.")
    parser.add_argument("--width_mult", type=float, default=1.0, help="Multiplicateur de largeur.")

    # Uniform QAT
    parser.add_argument("--n_bits", type=int, default=8, help="Nombre de bits (uniform KMQAT).")

    # Fine‑tuning
    parser.add_argument("--lr", type=float, default=1e-4, help="LR base.")
    parser.add_argument("--ft_epochs", type=int, default=40, help="Nb d’époques de FT.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--nb_worker", type=int, default=10, help="Workers dataloader.")

    # Divers
    parser.add_argument("--bn_recal_passes", type=int, default=2, help="Mini passes de BN recal au tout début (0 pour désactiver).")
    
    parser.add_argument("--warmup_epochs", type=int, default=2,
                    help="Nb d’époques en STE weight-only avant d’autoriser l’apprentissage de α.")
    parser.add_argument("--alpha_clip", type=float, default=1.5,
                    help="Clip de norme L2 des gradients d’alpha (0 pour désactiver).")
    parser.add_argument("--alpha_lr_mult", type=float, default=0.75,
                    help="Multiplicateur de LR pour le groupe α (ex: 2.0 => LR(α)=2×LR_base).")
        
    # Gestion des refresh des centroids
    parser.add_argument("--cb_refresh_every", type=int, default=1,
                    help="0=off ; sinon, fréquence (en époques) de l’EMA-refresh des codebooks.")
    
    #parser.add_argument("--cb_refresh_beta", type=float, default=0.1, help="Pas EMA pour le rafraîchissement des centroids.")
    #parser.add_argument("--cb_reassign_epoch", type=int, default=-1,  help="Époque (1-based) à laquelle on force une réassignation unique. -1=off.")

    #parser.add_argument("--bn_recal_after_reassign", type=int, default=2, help="Nombre de passes BN-recal juste après une réassignation (0=off).")
    #parser.add_argument("--mw_restart_epochs", type=int, default=1, help="Warm-restart temporaire des LR pendant N époques après réassignation (0=off).")
    #parser.add_argument("--mw_restart_mult", type=float, default=2.0, help="Facteur multiplicatif sur tous les LR pendant le warm-restart post-réassignation.")
    #parser.add_argument("--post_reassign_beta", type=float, default=0.05, help="β EMA temporaire pendant post_reassign_beta_epochs époques après une réassignation.")
    #parser.add_argument("--post_reassign_beta_epochs", type=int, default=1, help="Durée (en époques) durant laquelle on utilise post_reassign_beta après réassignation.")
    
    # EMA scheduler (NOUVEAU)
    parser.add_argument("--cb_beta_min", type=float, default=0.10)
    parser.add_argument("--cb_beta_max", type=float, default=0.30)
    parser.add_argument("--cb_beta_policy", type=str, choices=["fixed","cosine","adaptive"], default="adaptive")

    args = parser.parse_args()

    print("#"+" ".join(sys.argv[0:]))
    print("# Started at "+time.ctime())
    print("#")

    # DDP init
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=idr_torch.rank, world_size=idr_torch.size)
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")

    # --------------- Data ----------------
    dataset = SpeakerTrainingSegmentSet(
                                feature_extractor=Fbank(),
                                feature_transforms=Linear([
                                    CMVN(),
                                    Crop(350)
                                ] ),
                                )
    dataset.from_dict(Path(args.data_path))

    train_sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=args.nb_worker,
        sampler=train_sampler,
        pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()

    # --------------- Modèle / reprise ----------------
    start_epoch = 0
    model_qat = None
    optimizer = None
    scheduler = None
    num_classes = 5994  # VoxCeleb2 par défaut

    if args.resume_from:
        if idr_torch.rank == 0:
            print(f"\n[RESUME] Reprise fine-tuning depuis checkpoint : {args.resume_from}")

        checkpoint = torch.load(args.resume_from, map_location='cpu')
        num_classes = checkpoint.get("num_classes", num_classes)

        # Reconstruire l’archi
        model_qat = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)

        #exclude = ["preresnet.pre_conv1", "embedding.fc_embed"] #exclude temporaire de certaines couches sensibles

        # IMPORTANT: wrapping UNIFORME (tout à n_bits) et activation quant
        model_qat = prepare_model_for_uniform_qat(model_qat, n_bits=args.n_bits)
        #model_qat = prepare_model_for_uniform_qat(model_qat, n_bits=args.n_bits, exclude_names=exclude)


        # Charger le state_dict (wrappers ont alpha/codebook → strict=False)
        missing, unexpected = model_qat.load_state_dict(checkpoint["model"], strict=False)
        if idr_torch.rank == 0:
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

        start_epoch = checkpoint.get("epoch", 0)

    else:
        if idr_torch.rank == 0:
            print(f"\n[DÉMARRAGE] depuis modèle FP32 : {args.model_path}")

        ckpt_fp32 = torch.load(args.model_path, map_location='cpu')
        model_fp32 = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)
        model_fp32.load_state_dict(ckpt_fp32["model"], strict=False)  # robustesse
        model_fp32.to(gpu).eval()

        # Copie et wrapping UNIFORME (+ activation quant)
        model_qat = copy.deepcopy(model_fp32)

        #exclude = ["preresnet.pre_conv1", "embedding.fc_embed"] # on exclut deux couches sensibles

        model_qat = prepare_model_for_uniform_qat(model_qat, n_bits=args.n_bits)
        #model_qat = prepare_model_for_uniform_qat(model_qat, n_bits=args.n_bits, exclude_names=exclude)
        
        if PRINT_MODELS_BEFORE_AFTER and idr_torch.rank == 0:
            print("\n=== MODÈLE FP32 (abrégé) ===")
            print(model_fp32)
            print("\n=== MODÈLE APRÈS WRAP (abrégé) ===")
            print(model_qat)
            print_wrap_diff(model_fp32, model_qat)
            print("\n[DEBUG] Fin après impression. Quitte sans entraîner.")
            return

    # --------------- SyncBN / DDP ----------------
    model_qat = nn.SyncBatchNorm.convert_sync_batchnorm(model_qat)
    model_qat.to(gpu)

    # NOTE: garde find_unused_parameters=True s’il y a des branches conditionnelles.
    model_qat_ddp = DDP(model_qat, device_ids=[idr_torch.local_rank], find_unused_parameters=True) # besoin de True ?

    # Dossier de reprise
    resume_ckpt_dir = args.output_path / "resume_checkpoints"
    if idr_torch.rank == 0:
        os.makedirs(resume_ckpt_dir, exist_ok=True)

    # --------------- Optim / Sched ----------------
    base_lr  = args.lr
    alpha_lr = base_lr * float(args.alpha_lr_mult)

    # Prendre le "vrai" modèle (hors DDP wrapper)
    model_core = model_qat_ddp.module

    from torch.nn.modules.batchnorm import _BatchNorm  # couvre BatchNorm1d/2d + SyncBatchNorm

    weight_params, bn_bias_params, alpha_params = [], [], []
    seen = set()

    for mod_name, mod in model_core.named_modules():
        for p_name, p in mod.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if id(p) in seen:
                # Normalement impossible avec recurse=False, mais on reste prudent
                continue
            seen.add(id(p))

            if p_name == "alpha" or f"{mod_name}.{p_name}".endswith(".alpha"):
                alpha_params.append(p)
            elif p_name == "bias" or isinstance(mod, _BatchNorm):
                bn_bias_params.append(p)
            else:
                weight_params.append(p)

    if idr_torch.rank == 0:
        print(f"[PGROUPS] weights={len(weight_params)}  bn_bias={len(bn_bias_params)}  alpha={len(alpha_params)}  unique={len(seen)}")

    # Sécurité: chaque param est dans un seul groupe
    assert len(seen) == (len(weight_params) + len(bn_bias_params) + len(alpha_params)), \
        "[PGROUPS] Un paramètre a été classé dans plusieurs groupes !"

    optimizer = torch.optim.SGD(
        [
            {'params': weight_params, 'lr': base_lr,  'weight_decay': 1e-4, 'momentum': 0.9},
            {'params': bn_bias_params, 'lr': base_lr, 'weight_decay': 0.0,  'momentum': 0.9},
            {'params': alpha_params,   'lr': alpha_lr, 'weight_decay': 0.0,  'momentum': 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs, eta_min=1e-7)

    # Pour tes logs de gradients
    alpha_param_ids = {id(p) for p in alpha_params}


    # Reprise optim/sched si présents
    if args.resume_from and 'optimizer' in checkpoint:
        try:
            # protège contre les changements de nb de param_groups
            ck = checkpoint['optimizer']
            if 'param_groups' in ck and len(ck['param_groups']) != len(optimizer.param_groups):
                if idr_torch.rank == 0:
                    print(f"[WARN] Incompatibilité param_groups (ckpt={len(ck['param_groups'])} vs now={len(optimizer.param_groups)}). "
                        "Je charge seulement les 'state' tensors.")
                # Charge uniquement l'état des tensors, laisse les groupes actuels
                optimizer.state = ck.get('state', {})
            else:
                optimizer.load_state_dict(ck)
        except Exception as e:
            if idr_torch.rank == 0:
                print(f"[WARN] Impossible de recharger optimizer: {e}")

    if args.resume_from and 'scheduler' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            if idr_torch.rank == 0:
                print(f"[WARN] Impossible de recharger scheduler: {e}")

    # --------------- (Option) BN Recalibration ----------------
    if args.bn_recal_passes > 0:
        try:
            if idr_torch.rank == 0:
                print(f"[BN RECAL] passes={args.bn_recal_passes}")
            recalibrate_bn(model_qat_ddp, data_loader, gpu, num_passes=args.bn_recal_passes)
        except Exception as e:
            if idr_torch.rank == 0:
                print(f"[BN RECAL] ignorée ({e})")

    # --------------- Entraînement ----------------
    global_epoch_counter = start_epoch
    last_saved_epoch = None

    for epoch in range(start_epoch, args.ft_epochs):
        train_sampler.set_epoch(epoch)
        model_qat_ddp.train()

        # WARMUP STE weight-only + alpha gelé
        is_warmup = (epoch < args.warmup_epochs)
        set_ste_weight_only(model_qat_ddp, is_warmup)

        # Permet de requires_grad=True, mais annule les gradients d’α pendant le warm-up
        if is_warmup:
            for p in alpha_params:
                # hook one-shot : g → 0
                if getattr(p, "_mask_hook", None) is None:
                    p._mask_hook = p.register_hook(lambda g: g.zero_())
        else:
            # enlève le hook quand le warm-up est fini
            for p in alpha_params:
                h = getattr(p, "_mask_hook", None)
                if h is not None:
                    h.remove()
                    p._mask_hook = None
        if idr_torch.rank == 0 and (epoch == 0 or epoch == args.warmup_epochs):
            print(f"[STE] epoch {epoch+1}: ste_weight_only={is_warmup} | alpha_trainable={not is_warmup}")     

        running_loss = [np.nan] * 100
        for i, (feats, labels) in enumerate(data_loader):
            feats = feats.unsqueeze(1).float().to(gpu)
            labels = labels.to(gpu)

            optimizer.zero_grad()
            outputs = model_qat_ddp(feats, labels)  # ResNetV2.forward(x, iden) → logits
            loss = criterion(outputs, labels)
            loss.backward()

            # Clip alpha
            if args.alpha_clip > 0 :
                torch.nn.utils.clip_grad_norm_(alpha_params, args.alpha_clip)
            
            if idr_torch.rank == 0 and (i + 1) % 100 == 0:
                def _grad_norm(tensors):
                    s = 0.0
                    for t in tensors:
                        if t.grad is not None:
                            s += t.grad.detach().float().norm().item() ** 2
                    return s ** 0.5
                g_alpha = _grad_norm(alpha_params)
                # 'other' = tous les params de l'optimizer non dans alpha_params
                other_tensors = [p for gid, g in enumerate(optimizer.param_groups) for p in g['params'] if gid in (0,1)]  # groupes weights & bn_bias
                g_other = _grad_norm(other_tensors)
                ratio = g_alpha / (g_other + 1e-12)
                print(f"[GRAD] ||∇α||={g_alpha:.3e}  ||∇others||={g_other:.3e}  ratio={ratio:.3f}")

            optimizer.step()

            if idr_torch.rank == 0:
                running_loss.pop(0)
                running_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    rmean_loss = np.nanmean(np.array(running_loss))
                    lr_w  = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else float('nan')
                    lr_bb = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else float('nan')
                    lr_a  = optimizer.param_groups[2]['lr'] if len(optimizer.param_groups) > 2 else float('nan')
                    #current_lr = get_lr(optimizer)

                    print("{}: Epoch [{}/{}] ({}/{})  "
                          "AvgLoss:{:.4f}  Loss:{:.4f}  LRw:{:.6f}  LRbb:{:.6f}  LRα:{:.6f}".format(
                              time.ctime(),
                              epoch + 1, args.ft_epochs,
                              i + 1, len(data_loader),
                              rmean_loss, loss.item(), lr_w, lr_bb, lr_a
                          ),
                          flush=True)
                
                # Log ALPHA rarement (rank-0 seulement)
                if (i + 1) % ALPHA_LOG_EVERY_STEPS == 0:
                    log_alpha_stats_rank0(model_qat_ddp, epoch, step=i+1, total_steps=len(data_loader))
                    log_quant_error_layers(model_qat_ddp, tag=f"ep{epoch+1}/it{i+1}")

    

        scheduler.step()
        global_epoch_counter = epoch + 1

        # Log alpha en fin d’époque (rank-0 seulement)
        log_alpha_stats_rank0(model_qat_ddp, epoch)
        log_quant_error_layers(model_qat_ddp, tag=f"epoch{epoch+1}")

        # --- Refresh codebooks (EMA) périodique + gestion du beta post reasignation ---
        if args.cb_refresh_every > 0 and ((epoch + 1) % args.cb_refresh_every == 0):
            fp_before = codebook_fingerprint(model_qat_ddp.module) if idr_torch.rank == 0 else None

            # lecture état global pour piloter β
            m = estimate_usage_keff_entropy(model_qat_ddp.module)
            usage = m['usage'] if m else None
            keff  = m['Keff']  if m else None

            # on peut aussi passer une "inertie" mesurée sur la frame précédente ;
            # si tu veux, remplace inertia=None par delta_l1 (voir juste après).
            beta_now = decide_beta(
                policy=args.cb_beta_policy,
                epoch=epoch, T=args.ft_epochs,
                bmin=args.cb_beta_min, bmax=args.cb_beta_max,
                inertia=None, usage=usage, Keff=keff
            )

            if idr_torch.rank == 0:
                u = float('nan') if usage is None else float(usage)
                k = keff if keff is not None else -1
                print(f"[CB-REFRESH] epoch {epoch+1}: policy={args.cb_beta_policy} beta={beta_now:.3f} usage={u:.3f} Keff={k}")

            stats = ema_refresh_codebooks(model_qat_ddp.module, beta=beta_now, reassign=False, verbose=(idr_torch.rank==0))

            if idr_torch.rank == 0 and fp_before is not None:
                fp_after = codebook_fingerprint(model_qat_ddp.module)
                delta = (fp_after - fp_before).abs() / (fp_before.abs() + 1e-12)
                # si tu veux réinjecter "inertia", garde delta[2].item()
                print(f"[CB-REFRESH-Δ] meanΔ={delta[0]:.3e}  stdΔ={delta[1]:.3e}  l1Δ={delta[2]:.3e}")

        # --- Réassignation unique (optionnelle) à mi-parcours (~60–70%) ---
        """if args.cb_reassign_epoch > 0 and (epoch + 1) == args.cb_reassign_epoch:
            if idr_torch.rank == 0:
                print(f"[CB-REFRESH] epoch {epoch+1}: REASSIGN (nearest-centroid)")
            # 1) réassign indices
            from kiwano.quantization.utils import reassign_and_recalc_alpha_all, ema_refresh_codebooks
            # réassign + alpha fermé immédiatement (remet l'échelle en ligne)
            reassign_and_recalc_alpha_all(model_qat_ddp.module, verbose=(idr_torch.rank==0))

            # 2) BN recal court si demandé
            if args.bn_recal_after_reassign > 0:
                try:
                    if idr_torch.rank == 0:
                        print(f"[BN RECAL | post-reassign] passes={args.bn_recal_after_reassign}")
                    recalibrate_bn(model_qat_ddp, data_loader, gpu, num_passes=args.bn_recal_after_reassign)
                except Exception as e:
                    if idr_torch.rank == 0:
                        print(f"[BN RECAL | post-reassign] ignorée ({e})")

            # 3) premier EMA doux (même époque) avec beta réduit (évite sur-correction)
            beta_now = args.post_reassign_beta if args.post_reassign_beta > 0 else args.cb_refresh_beta
            ema_refresh_codebooks(model_qat_ddp.module, beta=beta_now, reassign=False, verbose=(idr_torch.rank==0))

            # 4) programme la fenêtre post-reassign pour beta temporaire
            if args.post_reassign_beta_epochs > 0:
                post_reassign_beta_until_epoch = (epoch + 1) + args.post_reassign_beta_epochs

            # 5) micro warm-restart des LR pendant N époques
            if args.mw_restart_epochs > 0 and args.mw_restart_mult > 1.0:
                if idr_torch.rank == 0:
                    print(f"[LR WARM-RESTART] ×{args.mw_restart_mult} pendant {args.mw_restart_epochs} époque(s)")
                _scale_all_lrs(args.mw_restart_mult)
                warm_restart_until_epoch = (epoch + 1) + args.mw_restart_epochs"""

        # Save fin d’époque
        if idr_torch.rank == 0:
            out_path = resume_ckpt_dir / f"model{epoch}.ckpt"
            print("-" * 30)
            print(f"Sauvegarde : {out_path}")
            print("-" * 30)
            save_quantized_checkpoint(
                model=model_qat_ddp.module,
                optimizer=optimizer,
                epoch=global_epoch_counter,
                filepath=out_path,
                scheduler=scheduler.state_dict(),
                # méta-infos utiles pour reprise/traçabilité
                mode="qat",
                uniform_n_bits=args.n_bits,
                num_classes=num_classes
            )
            last_saved_epoch = epoch

    # ---- BN recal finale (2–3 passes) AVANT la sauvegarde/éval ----
    if args.bn_recal_passes >= 0:
        try:
            if idr_torch.rank == 0:
                print(f"[BN RECAL - FINAL] passes=2")
            recalibrate_bn(model_qat_ddp, data_loader, gpu, num_passes=2)
        except Exception as e:
            if idr_torch.rank == 0:
                print(f"[BN RECAL - FINAL] ignorée ({e})")

    # --------------- Final ----------------
    if idr_torch.rank == 0:
        print("=" * 70)
        print("\nUniform KMQAT terminé.")

        if last_saved_epoch is not None:
            final_src = resume_ckpt_dir / f"model{last_saved_epoch}.ckpt"
            final_dst = args.output_path / f"model{last_saved_epoch}.ckpt"
            if os.path.exists(final_src):
                print(f"Copie checkpoint final vers : {final_dst}")
                import shutil
                shutil.copyfile(final_src, final_dst)
                print("Copie terminée.")
            else:
                print(f"AVERTISSEMENT: checkpoint {final_src} introuvable.")
        else:
            print("AVERTISSEMENT: aucun checkpoint sauvegardé durant l’entraînement ?")

        print("\nOpération de quantization uniforme terminée avec succès !")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

"""
Mixed-Precision KMQAT with Multi-Stage Fine-Tuning (MSFT) for ResNetV2 (speaker verification, VoxCeleb2) – DDP ready.

Pipeline:
- Si --resume_from absent:
    1) charge le modèle FP32
    2) estime les traces Hessiennes (proxy logits)
    3) pré-calcul des erreurs de quantization KMQAT (pour les wrappers)
    4) recherche d'une politique de bits sous contrainte de taille (MPQ)
    5) wrappe le modèle (prepare_model_for_mixed_qat), quantization désactivée au départ
- MSFT:
    - pour chaque stage (bits triés croissant), active les couches dont bit ≤ target_bit
    - (option) BN recalibration courte
    - fine-tune quelques époques, sauvegarde checkpoint (eff_codebook+indices)
- Reprise:
    - recharge la politique ("bit_assignment") depuis le checkpoint
    - re-wrappe et recharge state_dict (strict=False)
    - reprend au bon global_epoch (et restaure optimizer/scheduler si dispo)
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

from kiwano.model import ResNetV2, MiniBasicBlock, MiniSEBasicBlock, BasicBlock, SEBasicBlock
from kiwano.dataset import Segment, SegmentSet
from kiwano.features import Fbank
from kiwano.augmentation import Augmentation, Linear, CMVN, Crop

from kiwano.quantization.utils import (
    prepare_model_for_mixed_qat,
    save_quantized_checkpoint,
    recalibrate_bn,        # BN recalibration (utile avant/entre stages)
)
from kiwano.quantization.sensitivity import estimate_hessian_traces
from kiwano.quantization.search import (
    precompute_kmqat_errors,
    find_optimal_bit_assignment,
)

# DDP / SLURM
import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

# --------------------
# pour Hessian
# --------------------

# 1) Hessian traces sur le modèle FP32 en passant par forward_logits
#def _forward_logits_fn(x):
#    return model_fp32.forward_logits(x)

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

    parser = argparse.ArgumentParser(description="MPQ + MSFT KMQAT pour ResNetV2 (speaker verification)")

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--model_path", type=Path, required=True, help="Checkpoint FP32 pré‑entraîné (source).")
    parser.add_argument("--output_path", type=Path, required=True, help="Dossier de sortie (checkpoints).")
    parser.add_argument("--data_path", type=Path, required=True, help="Dossier données (calibration/fine‑tuning).")
    parser.add_argument("--resume_from", type=Path, default=None, help="Checkpoint QAT MPQ pour reprise.")

    parser.add_argument("--model_size",
                        type=str,
                        choices=["resnet18","resnet36","resnet50","resnet101","resnet101eq","resnet200","resnet400","resnet800"],
                        default="resnet36",
                        help="Taille du modèle.")
    parser.add_argument("--width_mult", type=float, default=1.0, help="Multiplicateur de largeur.")

    # MPQ search
    parser.add_argument("--candidate_bits", type=int, nargs='+', default=[2, 3, 4], help="Bits candidats pour MPQ.")
    parser.add_argument("--target_ratio", type=float, default=0.10, help="Taille cible / FP32 (ex. 0.10 = 10%).")

    # MSFT / FT
    parser.add_argument("--lr", type=float, default=1e-4, help="LR base.")
    parser.add_argument("--ft_epochs", type=int, default=40, help="Nb total d’époques (réparties par stage).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--nb_worker", type=int, default=10, help="Workers dataloader.")
    parser.add_argument("--bn_recal_per_stage", type=int, default=1, help="Passes BN recal à chaque début de stage (0 pour désactiver).")

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

    # --------------- Modèle / reprise / recherche policy ----------------
    start_epoch = 0
    model_qat = None
    optimizer = None
    scheduler = None
    bit_assignment = None
    num_classes = 5994  # VoxCeleb2 par défaut

    if args.resume_from:
        if idr_torch.rank == 0:
            print(f"\n[RESUME] Reprise fine-tuning depuis checkpoint : {args.resume_from}")

        checkpoint = torch.load(args.resume_from, map_location='cpu')
        num_classes = checkpoint.get("num_classes", num_classes)

        # Reconstruire archi
        model_qat = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)

        # Politique de bits récupérée du checkpoint (OBLIGATOIRE pour MPQ)
        bit_assignment = checkpoint.get("bit_assignment", None)
        if bit_assignment is None:
            sys.exit("Erreur : checkpoint sans politique de bits (bit_assignment).")

        # Wrapping (quantization désactivée par défaut)
        prepare_model_for_mixed_qat(model_qat, bit_assignment)

        # Charger l'état du modèle wrappé
        missing, unexpected = model_qat.load_state_dict(checkpoint["model"], strict=False)
        if idr_torch.rank == 0:
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

        start_epoch = checkpoint.get("epoch", 0)

    else:
        if idr_torch.rank == 0:
            print(f"\n[DÉMARRAGE] depuis modèle FP32 : {args.model_path}")

        ckpt_fp32 = torch.load(args.model_path, map_location='cpu')
        model_fp32 = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)
        model_fp32.load_state_dict(ckpt_fp32["model"], strict=False)
        model_fp32.to(gpu).eval()

        # Recherche MPQ sur rank 0 puis broadcast
        if idr_torch.rank == 0:
            print("\n--- MPQ (recherche politique) ---")

            # Proxy logits pour Hessian CE
            #proxy = ResNetV2LogitProxy(model_fp32).to(gpu).eval()

            # 1) Hessian traces
            hessian_traces = estimate_hessian_traces(
                model_fp32, data_loader, criterion, gpu, num_iterations=10, forward_fn=model_fp32.forward_logits
            )

            # 2) KMQAT weight errors (cohérents wrappers)
            quant_errors = precompute_kmqat_errors(
                model_fp32, candidate_bits=args.candidate_bits,
                r=0.9, symmetric_rescale=True, per_channel=True
            )

            # 3) Politique optimale sous contrainte
            bit_assignment = find_optimal_bit_assignment(
                model=model_fp32,
                candidate_bits=args.candidate_bits,
                target_ratio=args.target_ratio,
                hessian_traces=hessian_traces,
                quantization_errors=quant_errors,
                num_sections=4
            )
            objects_to_broadcast = [bit_assignment]
        else:
            objects_to_broadcast = [None]

        # Broadcast aux autres ranks
        dist.broadcast_object_list(objects_to_broadcast, src=0)
        bit_assignment = objects_to_broadcast[0]

        if idr_torch.rank == 0:
            print("\nPréparation modèle pour QAT mixte...")
        model_qat = copy.deepcopy(model_fp32)
        prepare_model_for_mixed_qat(model_qat, bit_assignment)  # quant disabled; enable per-stage ci-dessous

    # --------------- SyncBN / DDP ----------------
    model_qat = nn.SyncBatchNorm.convert_sync_batchnorm(model_qat)
    model_qat.to(gpu)

    # NOTE: garde find_unused_parameters=True s’il y a des branches conditionnelles.
    model_qat_ddp = DDP(model_qat, device_ids=[idr_torch.local_rank], find_unused_parameters=True)

    # Dossier de reprise
    resume_ckpt_dir = args.output_path / "resume_checkpoints"
    if idr_torch.rank == 0:
        os.makedirs(resume_ckpt_dir, exist_ok=True)

    # --------------- Staging ----------------
    # Bits présents dans la politique (ordre croissant → 2,3,4,…)
    candidate_bits_sorted = sorted(list(set(bit_assignment.values())))
    # Répartition des époques par stage (au moins 1)
    epochs_per_stage = max(1, args.ft_epochs // len(candidate_bits_sorted))

    # Si reprise, on détermine le stage courant
    current_stage_from_resume = start_epoch // epochs_per_stage if epochs_per_stage > 0 else 0

    # --------------- Boucle MSFT ----------------
    global_epoch_counter = start_epoch
    last_saved_epoch = None

    for stage_idx, target_bit in enumerate(candidate_bits_sorted):
        # Skip si stage déjà couvert par reprise
        if stage_idx < current_stage_from_resume:
            if idr_torch.rank == 0:
                print(f"\n--- Stage {stage_idx + 1}/{len(candidate_bits_sorted)} (<= {target_bit}-bit) déjà complété (resume).")
            # Assure que tout ce qui doit être activé l'est (utile si reprise milieu / changements)
            for name, module in model_qat_ddp.module.named_modules():
                if name in bit_assignment and bit_assignment[name] <= target_bit:
                    if hasattr(module, 'enable_quantization') and not getattr(module, 'quantization_enabled', False):
                        module.enable_quantization()
            continue

        if idr_torch.rank == 0:
            print("-" * 70)
            print(f"--- DÉBUT STAGE {stage_idx + 1}/{len(candidate_bits_sorted)} (<= {target_bit}-bit) ---")
            print("-" * 70)

        # Activer la quantization pour les couches ≤ target_bit
        for name, module in model_qat_ddp.module.named_modules():
            if name in bit_assignment and bit_assignment[name] <= target_bit:
                if hasattr(module, 'enable_quantization') and not getattr(module, 'quantization_enabled', False):
                    module.enable_quantization()
                    if idr_torch.rank == 0:
                        print(f"  - enable: '{name}' -> {bit_assignment[name]}-bit")

        # (Option) BN recalibration courte
        if args.bn_recal_per_stage > 0:
            try:
                if idr_torch.rank == 0:
                    print(f"[BN RECAL] stage={stage_idx+1}, passes={args.bn_recal_per_stage}")
                recalibrate_bn(model_qat_ddp, data_loader, gpu, num_passes=args.bn_recal_per_stage)
            except Exception as e:
                if idr_torch.rank == 0:
                    print(f"[BN RECAL] ignorée ({e})")

        # Optimiseur par stage : LR pour poids, LR*10 pour alpha
        base_lr = args.lr
        alpha_lr = base_lr * 10
        alpha_params, other_params = [], []
        for n, p in model_qat_ddp.named_parameters():
            if not p.requires_grad:
                continue
            (alpha_params if n.endswith('.alpha') else other_params).append(p)

        optimizer = torch.optim.SGD(
            [
                {'params': other_params, 'lr': base_lr, 'weight_decay': 1e-4, 'momentum': 0.9},
                {'params': alpha_params, 'lr': alpha_lr, 'weight_decay': 0.0, 'momentum': 0.0},
            ],
            momentum=0.0, weight_decay=0.0
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_stage, eta_min=1e-7)

        # Reprise optimizer/scheduler si on est pile au stage correspondant
        if args.resume_from and stage_idx == current_stage_from_resume and 'checkpoint' in locals():
            try:
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as e:
                if idr_torch.rank == 0:
                    print(f"[WARN] Rechargement optimizer/scheduler échoué: {e}")

        # Départ interna du stage si reprise au milieu
        start_epoch_in_stage = 0
        if stage_idx == current_stage_from_resume:
            start_epoch_in_stage = start_epoch - (stage_idx * epochs_per_stage)
            start_epoch_in_stage = max(0, start_epoch_in_stage)

        # ---- Entraînement du stage ----
        for epoch_in_stage in range(start_epoch_in_stage, epochs_per_stage):
            current_global_epoch = stage_idx * epochs_per_stage + epoch_in_stage
            train_sampler.set_epoch(current_global_epoch)
            model_qat_ddp.train()

            running_loss = [np.nan] * 100
            for i, (feats, labels) in enumerate(data_loader):
                feats = feats.unsqueeze(1).float().to(gpu)
                labels = labels.to(gpu)

                optimizer.zero_grad()
                outputs = model_qat_ddp(feats, labels)  # ResNetV2.forward(x, iden) → logits (AMSMLoss)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if idr_torch.rank == 0:
                    running_loss.pop(0); running_loss.append(loss.item())
                    if (i + 1) % 100 == 0:
                        rmean_loss = np.nanmean(np.array(running_loss))
                        current_lr = get_lr(optimizer)
                        print("{}: Stage [{}/{}] Epoch [{}/{}] ({}/{})  "
                              "AvgLoss:{:.4f}  Loss:{:.4f}  LR:{:.6f}".format(
                                  time.ctime(),
                                  stage_idx + 1, len(candidate_bits_sorted),
                                  epoch_in_stage + 1, epochs_per_stage,
                                  i + 1, len(data_loader),
                                  rmean_loss, loss.item(), current_lr
                              ),
                              flush=True)

            scheduler.step()
            global_epoch_counter = current_global_epoch + 1

            # Save fin d’époque (rank 0)
            if idr_torch.rank == 0:
                out_path = resume_ckpt_dir / f"model{current_global_epoch}.ckpt"
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
                    mode="mpq_msft",
                    bit_assignment=bit_assignment,
                    num_classes=num_classes
                )
                last_saved_epoch = current_global_epoch

    # --------------- Final ----------------
    if idr_torch.rank == 0:
        print("=" * 70)
        print("\nMSFT KMQAT terminé.")

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

        print("\nOpération de quantization MPQ + MSFT terminée avec succès !")

if __name__ == '__main__':
    main()

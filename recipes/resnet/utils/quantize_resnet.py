#!/usr/bin/env python3

import os
import sys
import torch
import argparse
from pathlib import Path
from collections import OrderedDict
import copy
from typing import Optional, Union, List

import numpy as np
import torch
import time
from torch import nn
from tqdm import tqdm  
import logging

from torch.utils.data import Dataset, DataLoader, Sampler

# Assurer que les modules personnalisés peuvent être importés
# Ajustez le chemin si nécessaire pour correspondre à votre structure
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Imports depuis notre code existant
from kiwano.model import ResNetV2, MiniBasicBlock, MiniSEBasicBlock, BasicBlock, SEBasicBlock
from kiwano.dataset import Segment, SegmentSet
#from kiwano.utils.train_resnet import get_resnet_model, SpeakerTrainingSegmentSet # On réutilise les fonctions de train_resnet
from kiwano.features import Fbank
from kiwano.utils import Pathlike

from kiwano.augmentation import Augmentation, Noise, Codec, Filtering, Normal, Sometimes, Linear, CMVN, Crop, SpecAugment, Reverb

from kiwano.quantization.utils import prepare_model_for_mixed_qat, save_quantized_checkpoint, load_inference_model_from_checkpoint
from kiwano.quantization.sensitivity import estimate_hessian_traces
from kiwano.quantization.search import find_optimal_bit_assignment

# DDP: Imports pour l'environnement distribué
import idr_torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_resnet_model(model_size: str, num_classes: int, width_mult: float = 1.0):
    """Retourne un ResNetV2 configuré pour la taille demandée."""
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
        "resnet18": ([2, 2, 2, 2], MiniBasicBlock, MiniSEBasicBlock),
        "resnet36": ([3, 4, 6, 3], MiniBasicBlock, MiniSEBasicBlock),
        "resnet50": ([3, 4, 6, 3], BasicBlock, SEBasicBlock),
        "resnet101": ([3, 4, 23, 3], BasicBlock, SEBasicBlock),
        "resnet101eq": ([3, 10, 17, 3], BasicBlock, SEBasicBlock), 
        "resnet200": ([3, 24, 36, 3], BasicBlock, SEBasicBlock),
        "resnet400": ([4, 44, 87, 4], BasicBlock, SEBasicBlock),
        "resnet800": ([8, 88, 174, 8], BasicBlock, SEBasicBlock),
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
    def __init__(self, audio_transforms: List[Augmentation] = None, feature_extractor = None, feature_transforms: List[Augmentation] = None):
        super().__init__()
        self.audio_transforms = audio_transforms
        self.feature_transforms = feature_transforms
        self.feature_extractor = feature_extractor

    def __getitem__(self, segment_id_or_index: Union[int, str]) -> Segment:
        segment = None
        if isinstance(segment_id_or_index, str):
            segment = self.segments[segment_id_or_index]
        else:
            segment = next(val for idx, val in enumerate(self.segments.values()) if idx == segment_id_or_index)

        try:
            audio, sample_rate = segment.load_audio()
            
            if audio.shape[0] == 0:
                logger.warning(f"⚠️ Segment audio vide : {segment_id_or_index}")
                return torch.zeros((1, 81, 350)), -1            

            if self.audio_transforms != None:
                audio, sample_rate = self.audio_transforms(audio, sample_rate)

            if self.feature_extractor != None:
                feature = self.feature_extractor.extract(audio, sampling_rate=sample_rate)

            if self.feature_transforms != None:
                feature = self.feature_transforms(feature)

            return feature, self.labels[ segment.spkid ]

        except Exception as e:
            logger.error(f"❌ Erreur sur le segment {segment.spkid}: {e}")
            return torch.zeros((1, 81, 350)), -1


# Imports de la logique KMQAT que nous devons externaliser
# Normalement, cette fonction serait dans un fichier comme 'kiwano/quantization/core.py'
def kmqat_quantize_layer(weights, n_bits, retention_ratio=0.9):
    """
    Fonction "boîte noire" de KMQAT.
    (Copiez ici l'implémentation de la fonction de la réponse précédente)
    """
    with torch.no_grad():
        weights_flat = weights.flatten()
        sorted_weights, _ = torch.sort(weights_flat)
        num_weights = sorted_weights.numel()
        low_idx = int(num_weights * (1 - retention_ratio) / 2)
        high_idx = int(num_weights * (1 + retention_ratio) / 2)
        central_weights = sorted_weights[low_idx:high_idx]
        num_weights_to_group = central_weights.numel()
        if num_weights_to_group < 2: return weights
        num_centroids = min(2**n_bits, num_weights_to_group)
        intervals = torch.chunk(central_weights, num_centroids)
        codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)
        distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
        indices = torch.argmin(distances, dim=1)
        return codebook[indices].reshape(weights.shape)


def main():

    print(str(idr_torch.master_addr))
    print(str(idr_torch.master_port))
    print(str(idr_torch.local_rank))

    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']

    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostname), " nodes and ", idr_torch.size, " processes, master node is ", MASTER_ADDR)
    print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))

    parser = argparse.ArgumentParser(description="Script de quantification pour les modèles ResNet.")

    parser.add_argument("--local_rank", type=int)

    parser.add_argument("--mode", type=str, required=True, choices=['qat', 'mpq'], help="Mode de quantification: 'qat' (uniforme) ou 'mpq' (mixte).")
    parser.add_argument("--model_path", type=Path, required=True, help="Chemin vers le checkpoint du modèle float32 pré-entraîné.")
    parser.add_argument("--output_path", type=Path, required=True, help="Chemin pour sauvegarder le checkpoint quantifié final.")
    parser.add_argument("--data_path", type=Path, required=True, help="Chemin vers les données de calibration/fine-tuning.")

    parser.add_argument("--resume_from", type=Path, default=None, help="Chemin vers un checkpoint QAT pour reprendre le fine-tuning.")
    
    parser.add_argument("--model_size",
                        type=str,
                        choices=["resnet18","resnet36","resnet50","resnet101","resnet101eq","resnet200","resnet400","resnet800"],
                        default="resnet18",
                        help="Taille du modèle à finetuner")
    
    parser.add_argument("--width_mult", type=float, default=1.0, help="Multiplicateur sur la largeur (nombre de channels) des couches du ResNet.") #permet de faire varier la largeur du modèle, facteur sur les features map.  
    
    # Arguments pour le QAT uniforme
    parser.add_argument("--n_bits", type=int, default=8, help="Nombre de bits pour le QAT uniforme.")
    
    # Arguments pour le MPQ
    parser.add_argument("--candidate_bits", type=int, nargs='+', default=[2, 4, 8], help="Liste des bits candidats pour MPQ.")
    parser.add_argument("--target_ratio", type=float, default=0.25, help="Ratio de compression cible pour MPQ.")
    
    # Arguments pour le fine-tuning
    parser.add_argument("--lr", type=float, default=1e-5, help="Taux d'apprentissage pour le fine-tuning.")
    parser.add_argument("--ft_epochs", type=int, default=5, help="Nombre d'époques pour le fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size pour le fine-tuning.")
    
    parser.add_argument(
        "--nb_worker",
        type=int,
        default=10,
        help="Nbr de worker"
    )

    args = parser.parse_args()

    print("#"+" ".join( sys.argv[0:]  ))
    print("# Started at "+time.ctime())
    print("#")

    # DDP: Initialisation du processus distribué
    torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=idr_torch.rank, world_size=idr_torch.size)
    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du device : {device}")
    
    # --- ÉTAPE 3.1 : Initialiser les variables pour le fine-tuning ---
    start_epoch = 0
    model_qat = None
    bit_assignment = None
    optimizer = None
    scheduler = None

    # --- SETUP DU DATALOADER ET CRITERION ---
    dataset = SpeakerTrainingSegmentSet(
                                feature_extractor=Fbank(),
                                feature_transforms=Linear([
                                    CMVN(),
                                    Crop(350),
                                    SpecAugment(),
                                ] ),
                                )
    dataset.from_dict(Path(args.data_path))
    
    # DDP: Utiliser DistributedSampler
    train_sampler = DistributedSampler(dataset, num_replicas=idr_torch.size, rank=idr_torch.rank, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=True, shuffle=False, num_workers=args.nb_worker, sampler=train_sampler, pin_memory=True)

    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume_from:
        # Cas 2 : Reprendre le fine-tuning à partir d'un checkpoint QAT
        if idr_torch.rank == 0:
            print(f"\nReprise du fine-tuning à partir du checkpoint : {args.resume_from}")

        checkpoint = torch.load(args.resume_from, map_location='cpu')

        # Recréer l'architecture du modèle à partir du checkpoint
        num_classes = checkpoint.get("num_classes", 5994)  # Exemple, à adapter
        model_qat = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)

        bit_assignment = checkpoint.get("bit_assignment", None)
        if bit_assignment is None:
            sys.exit("Erreur : Le checkpoint ne contient pas de politique de bits.")

        # Préparer le modele avec les wrappers avec la politique de bits
        prepare_model_for_mixed_qat(model_qat, bit_assignment)

        # Charger les poids du modèle (poids FP32 des wrappers)
        model_qat.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get("epoch", 0)  # Récupérer l'époque de reprise
    
    else:
        # Cas 1 : On démarre un nouveau fine-tuning à partir d'un modèle FP32
        if idr_torch.rank == 0:
            print(f"\nDémarrage d'un nouveau fine-tuning à partir du modèle float32 : {args.model_path}")

        checkpoint = torch.load(args.model_path, map_location='cpu')
        num_classes = 5994 # a adapater ou exttraire depuis le modèle en FP32
        model_fp32 = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)
        model_fp32.load_state_dict(checkpoint["model"])
        model_fp32.to(gpu).eval()


    # --- ÉTAPE 3.2 - 1 : Charger le modèle ResNetV2 float32 ---

    #print(f"\n1. Chargement du modèle float32 depuis : {args.model_path}")
    #checkpoint = torch.load(args.model_path, map_location='cpu')

    # Recréer l'architecture du modèle à partir du checkpoint
    #model_name = checkpoint.get("name", "ResNetV2") # Supposons ResNetV2 si non trouvé

    # Ici, nous aurions besoin de plus d'infos pour recréer le modèle,
    # comme la taille, num_classes, etc. Simplifions pour l'instant.
    # Pour l'exemple, nous allons utiliser le 'get_resnet_model'
    #num_classes = 5994 # Exemple, à adapter
    #model_fp32 = get_resnet_model(args.model_size, num_classes=num_classes, width_mult=args.width_mult)
    #model_fp32.load_state_dict(checkpoint["model"])
    #model_fp32.to(gpu).eval()
    


        if args.mode == 'mpq':
            if idr_torch.rank == 0:
                print("\n--- Mode de Quantification à Précision Mixte (MPQ) activé ---")
                
                # --- ÉTAPE 3.2 - 2a : Estimer la sensibilité Hessienne ---
                hessian_traces = estimate_hessian_traces(model_fp32, data_loader, criterion, device, num_iterations=10)
                
                # Pré-calculer les erreurs de quantification
                print("\nPré-calcul des erreurs de quantification...")
                quant_errors = {}
                for name, module in model_fp32.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)): #rajout de conv1d 
                        for b in args.candidate_bits:
                            q_weights = kmqat_quantize_layer(module.weight.detach(), n_bits=b)
                            error = torch.sum((module.weight.detach() - q_weights)**2).item()
                            quant_errors[(name, b)] = error

                # --- ÉTAPE 3.2 - 2b : Trouver la politique de bits optimale ---

                bit_assignment = find_optimal_bit_assignment(
                    model_fp32, args.candidate_bits, args.target_ratio, hessian_traces, quant_errors
                    )

                objects_to_broadcast = [bit_assignment] # Liste des objets à diffuser
            
            else:
                objects_to_broadcast = [None] # Placeholder pour les autres processus

            # DDP: Diffuser l'objet du rang 0 à tous les autres processus
            dist.broadcast_object_list(objects_to_broadcast, src=0)

            # Tous les processus ont maintenant la même politique de bits
            bit_assignment = objects_to_broadcast[0]

        elif args.mode == 'qat':
            print(f"\n--- Mode de Quantification Uniforme (QAT) activé à {args.n_bits} bits ---")
            # --- ÉTAPE 3.2 - 3 : Créer une politique de bits uniforme ---
            for name, module in model_fp32.named_modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear)):
                    bit_assignment[name] = args.n_bits
        
        # --- ÉTAPE 3.2 - 4 : Préparer le modèle pour le QAT avec la politique de bits ---
        print("\nPréparation du modèle pour le fine-tuning QAT...")
        # On travaille sur une copie pour ne pas altérer le modèle FP32
        model_qat = copy.deepcopy(model_fp32)
        prepare_model_for_mixed_qat(model_qat, bit_assignment)

    # DDP: Conversion SyncBatchNorm et envoi au GPU
    model_qat = nn.SyncBatchNorm.convert_sync_batchnorm(model_qat)
    model_qat.to(gpu)
    
    
    # DDP: Encapsuler le modèle avec DDP
    model_qat_ddp = DDP(model_qat, device_ids=[idr_torch.local_rank], find_unused_parameters=True)

    optimizer = torch.optim.SGD(model_qat_ddp.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs - start_epoch, eta_min=1e-7)

    # Si on reprend, charger l'état de l'optimiseur/scheduler
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

    """optimizer = torch.optim.SGD([
    {'params': model_qat_ddp.module.preresnet.parameters()},
    {'params': model_qat_ddp.module.temporal_pooling.parameters()},
    {'params': model_qat_ddp.module.embedding.parameters()},
    {'params': model_qat_ddp.module.output.parameters()}
], lr=args.lr, momentum=0.9)""" # On peut ajouter le weight_decay si besoin plus tard

    # --- ÉTAPE 3.2 - 5 & 6 : Lancer le fine-tuning ---
    print("\nDébut du fine-tuning QAT...")


    running_loss = [np.nan] * 100

    for epoch in range(start_epoch, args.ft_epochs):
        train_sampler.set_epoch(epoch)

        model_qat_ddp.train()

        for i, (feats, labels) in enumerate(data_loader):
            feats, labels = feats.unsqueeze(1).float().to(gpu), labels.to(gpu)
            optimizer.zero_grad()
            outputs = model_qat_ddp(feats, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if idr_torch.rank == 0:
                running_loss.pop(0)
                running_loss.append(loss.item())
        
                # Le processus maître affiche le progrès tous les 100 pas
                if (i + 1) % 100 == 0:
                    rmean_loss = np.nanmean(np.array(running_loss))
                    current_lr = get_lr(optimizer)
                    
                    msg = "{}: Epoch: [{}/{}] ({}/{}) \t AvgLoss:{:.4f} \t C-Loss:{:.4f} \t LR:{:.8f}".format(
                                time.ctime(), 
                                epoch + 1, args.ft_epochs, 
                                i + 1, len(data_loader),
                                rmean_loss, 
                                loss.item(),
                                current_lr
                            )
                    print(msg, flush=True)
                    #print(f"  Epoch [{epoch+1}/{args.ft_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        
        if idr_torch.rank == 0:
            # Construire un nom de fichier qui inclut l'époque
            # ex: /path/to/exp/quantized_epoch_0.ckpt
            output_path_epoch = args.output_path / f"model{epoch}.ckpt"
            
            # On passe le numéro de l'époque actuelle + 1
            save_quantized_checkpoint(
                model=model_qat_ddp.module, 
                optimizer=optimizer, 
                epoch=epoch + 1, 
                filepath=output_path_epoch,
                scheduler=scheduler.state_dict(),
                bit_assignment=bit_assignment,
                num_classes=num_classes
            )
            
    # --- ÉTAPE 3.2 - 7 : Sauvegarder le modèle final ---
    #print(f"\nFine-tuning terminé. Sauvegarde du modèle compact final sous : {args.output_path}")
    # Nous utilisons une fonction de sauvegarde qui stocke à la fois les poids FP32 et les données compactes
    #save_quantized_checkpoint(model_qat_ddp.module, optimizer, args.ft_epochs, args.output_path)
    
    print("\nOpération de quantification terminée", flush=True)


if __name__ == '__main__':
    main()
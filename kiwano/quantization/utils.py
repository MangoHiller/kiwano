import torch
import torch.nn as nn
from collections import OrderedDict
import copy

# Importer nos wrappers personnalisés du même package
from .wrappers import KMeansQuantConv1d, KMeansQuantConv2d, KMeansQuantLinear

# ==============================================================================
# FONCTIONS DE PRÉPARATION DU MODÈLE POUR LE QAT
# ==============================================================================
def prepare_model_for_uniform_qat(model, n_bits=8):
    """
    Prépare un modèle pour le QAT uniforme en remplaçant récursivement les couches
    par leurs wrappers quantifiés, avec un n_bits unique.
    """
    # Dictionnaire de mapping: type de couche -> classe de wrapper
    MAPPING = {
        nn.Conv1d: KMeansQuantConv1d,
        nn.Conv2d: KMeansQuantConv2d,
        nn.Linear: KMeansQuantLinear
    }
    
    def _recursive_prepare(module):
        for name, child in module.named_children():
            # Si le fils a lui-même des enfants, on continue la récursion
            if len(list(child.children())) > 0:
                _recursive_prepare(child)

            # Si le fils est un type de couche que nous voulons quantifier
            if type(child) in MAPPING:
                print(f"Remplacement de '{name}' ({type(child).__name__}) -> wrapper {n_bits}-bit.")
                # Créer le wrapper avec la couche originale et le n_bits
                wrapped_layer = MAPPING[type(child)](child, n_bits=n_bits)
                # Remplacer le module dans le parent
                setattr(module, name, wrapped_layer)
                
    _recursive_prepare(model)

def prepare_model_for_mixed_qat(model, bit_assignment):
    """
    Prépare un modèle pour le QAT à précision mixte, en assignant
    à chaque couche le bit-width spécifié dans le dictionnaire bit_assignment.
    """
    MAPPING = {
        nn.Conv1d: KMeansQuantConv1d,
        nn.Conv2d: KMeansQuantConv2d,
        nn.Linear: KMeansQuantLinear
    }
    
    # On parcourt toutes les couches nommées pour retrouver celles de bit_assignment
    for name, module in model.named_modules():
        if name in bit_assignment:
            if type(module) in MAPPING:
                # Trouver le module parent pour pouvoir remplacer le fils
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                parent_module = model.get_submodule(parent_name)
                
                assigned_bit = bit_assignment[name]
                print(f"Remplacement de '{name}' ({type(module).__name__}) -> wrapper {assigned_bit}-bit.")
                wrapped_layer = MAPPING[type(module)](module, n_bits=assigned_bit)
                setattr(parent_module, child_name, wrapped_layer)

# ==============================================================================
# FONCTIONS DE GESTION DES CHECKPOINTS QAT
# ==============================================================================

def save_quantized_checkpoint(model, optimizer, epoch, filepath, **kwargs):
    """
    Sauvegarde un checkpoint complet pour un modèle en cours de QAT.
    Ce checkpoint contient à la fois les poids float32 pour reprendre l'entraînement
    ET les données compactes pour créer un modèle d'inférence.
    """
    
    # 1. Extraire les données de quantification compactes
    quantization_data = {}
    for name, module in model.named_modules():
        if hasattr(module, 'get_quantization_components'):
            codebook, indices, bias = module.get_quantization_components()
            if codebook is not None:
                quantization_data[name] = {
                    'codebook': codebook.cpu(),
                    'indices': indices.cpu().to(torch.uint8),
                    'bias': bias.cpu() if bias is not None else None,
                    'shape': module.weight.shape,
                }
    
    # 2. Créer le dictionnaire de checkpoint en incluant les données standards
    #    ET nos données de quantification.
    checkpoint = {
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(), # Contient les poids float32
        "name": type(model).__name__, # Nom de la classe du modèle
        "config": model.extra_repr(),
        "quantization_data": quantization_data, # Contient les données compactes
        **kwargs # Pour tout autre info que vous voulez sauvegarder (loss, etc.)
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint QAT (époque {epoch}) sauvegardé sous {filepath}")


def load_inference_model_from_checkpoint(filepath, base_model_architecture):
    """
    Charge un checkpoint quantifié et reconstruit un modèle d'INFÉRENCE
    léger et rapide, directement à partir des données compactes.
    Ne peut pas être utilisé pour reprendre l'entraînement.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    quantization_data = checkpoint["quantization_data"]
    
    # Créer une nouvelle instance vierge du modèle
    inference_model = base_model_architecture()
    
    # Itérer sur les couches du nouveau modèle et les peupler avec les données compactes
    for name, module in inference_model.named_modules():
        if name in quantization_data:
            data = quantization_data[name]
            codebook = data['codebook']
            indices = data['indices']
            shape = data['shape']
            
            # Reconstruire les poids
            reconstructed_weights = codebook[indices.to(torch.long)].reshape(shape)
            
            # Assigner les poids et le biais
            module.weight.data.copy_(reconstructed_weights)
            if data['bias'] is not None and module.bias is not None:
                module.bias.data.copy_(data['bias'])
                
    return inference_model

def load_qat_model_for_finetuning(filepath, model_qat, optimizer=None):
    """
    Charge un checkpoint QAT pour reprende un fine-tuning.
    Cette fonction charge les poids dans un modèle déjà créé.
    """
    print(f"Reprise du fine-tuning depuis le checkpoint : {filepath}")
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Charger le state_dict du modèle
    model_qat.load_state_dict(checkpoint["model"])
    
    # Charger l'état de l'optimiseur si fourni
    if optimizer and "optimizer":
        optimizer.load_state_dict(checkpoint["optimizer"])
        
    start_epoch = checkpoint.get("epochs", 0)
    
    print(f"Checkpoint chargé. Reprise à partir de l'époque {start_epoch}.")
    return model_qat, optimizer, start_epoch
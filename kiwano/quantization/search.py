import torch
import numpy as np
import itertools
from tqdm import tqdm # Pour une barre de progression sympathique

def find_optimal_bit_assignment(
    model, 
    candidate_bits, 
    target_ratio, 
    hessian_traces, 
    quantization_errors
):
    """
    Trouve la combinaison optimale de bits par couche pour minimiser la sensibilité totale
    sous une contrainte de taille.

    Args:
        model (nn.Module): Le modèle pré-entraîné en float32.
        candidate_bits (list): La liste des bits candidats (ex: [2, 4, 8]).
        target_ratio (float): Le ratio de compression cible (ex: 0.25).
        hessian_traces (dict): Dictionnaire {layer_name: trace_hessian}.
        quantization_errors (dict): Dictionnaire {(layer_name, bit): error}.

    Returns:
        dict: Un dictionnaire mappant chaque nom de couche à son bit-width optimal.
    """
    print("\n--- Début de la Recherche de la Politique de Précision Mixte Optimale ---")
    
    # ÉTAPE 5 : Classement et groupement des couches
    print("1. Classement et groupement des couches par sensibilité...")
    layer_names = list(hessian_traces.keys())
    
    # Trier les couches par leur trace Hessienne décroissante
    sorted_layers = sorted(layer_names, key=lambda k: hessian_traces[k], reverse=True)
    
    # Diviser en G groupes (G = nombre de précisions candidates)
    num_groups = len(candidate_bits)
    if not sorted_layers:
        print("AVERTISSEMENT: Aucune couche à quantifier trouvée.")
        return {}
        
    layer_groups = np.array_split(sorted_layers, num_groups)
    
    # --- VÉRIFICATION VISUELLE ---
    print(f"Couches classées en {num_groups} groupes de sensibilité.")
    for i, group in enumerate(layer_groups):
        print(f"  - Groupe {i+1} (le plus sensible): {list(group)}")

    # ÉTAPE 6 : Définition de l’espace de recherche réduit
    print("\n2. Génération de l'espace de recherche...")
    # On assigne les bits les plus élevés aux groupes les plus sensibles
    sorted_bits = sorted(candidate_bits, reverse=True)
    search_space = list(itertools.product(sorted_bits, repeat=num_groups))
    print(f"Espace de recherche réduit à {len(search_space)} combinaisons.")

    # ÉTAPE 7 : Calcul du coût pour chaque combinaison
    print("\n3. Évaluation du coût et de la taille pour chaque combinaison...")
    search_results = []
    
    # Pré-calculer les nombres de paramètres pour éviter les accès répétés
    layer_params = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            layer_params[name] = module.weight.numel()

    for combination in tqdm(search_space, desc="Évaluation des combinaisons"):
        total_cost = 0
        total_size_bits = 0
        
        for group_idx, assigned_bit in enumerate(combination):
            for layer_name in layer_groups[group_idx]:
                error = quantization_errors[(layer_name, assigned_bit)]
                sensitivity = hessian_traces[layer_name]
                
                total_cost += sensitivity * error
                total_size_bits += layer_params[layer_name] * assigned_bit
        
        search_results.append({
            'combination': combination,
            'cost': total_cost,
            'size_bits': total_size_bits
        })

    # ÉTAPE 8 : Optimisation et recherche de la meilleure combinaison
    print("\n4. Recherche de la combinaison optimale sous contrainte de taille...")
    fp32_total_size = sum(layer_params.values()) * 32
    target_size_bits = fp32_total_size * target_ratio

    # Filtrer les combinaisons qui respectent la contrainte
    valid_combinations = [res for res in search_results if res['size_bits'] <= target_size_bits]
    
    if not valid_combinations:
        print("ERREUR: Aucune combinaison ne respecte la contrainte de taille. Essayez d'augmenter le target_ratio.")
        # On retourne la combinaison la plus petite en taille comme "meilleur effort"
        best_effort = min(search_results, key=lambda x: x['size_bits'])
        best_result = best_effort
    else:
        # Trouver la combinaison avec le coût minimal parmi les valides
        best_result = min(valid_combinations, key=lambda x: x['cost'])

    optimal_combination = best_result['combination']
    
    # Créer le dictionnaire final d'assignation
    optimal_bit_assignment = {}
    for group_idx, assigned_bit in enumerate(optimal_combination):
        for layer_name in layer_groups[group_idx]:
            optimal_bit_assignment[layer_name] = assigned_bit

    # --- VÉRIFICATION VISUELLE FINALE ---
    print("\n--- Recherche Terminée ---")
    print(f"Combinaison de bits optimale trouvée : {optimal_combination}")
    print(f"  - Coût (sensibilité totale) : {best_result['cost']:.4e}")
    print(f"  - Taille du modèle estimée : {best_result['size_bits'] / (8*1024):.2f} Ko")
    print(f"  - Ratio de compression estimé : {fp32_total_size / best_result['size_bits']:.2f}x")
    
    return optimal_bit_assignment
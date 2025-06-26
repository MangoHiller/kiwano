import torch
import torch.nn as nn
import time

def estimate_hessian_traces(model, data_loader, criterion, device, num_iterations=10):
    """
    Estime la trace de la matrice Hessienne pour chaque couche pondérée d'un modèle
    en utilisant l'estimateur de Hutchinson.

    Args:
        model (nn.Module): Le modèle pré-entraîné en float32.
        data_loader (DataLoader): Un DataLoader pour fournir des batches de données.
        criterion (nn.Module): La fonction de perte (ex: nn.CrossEntropyLoss).
        device (torch.device): Le device sur lequel effectuer les calculs ('cuda' ou 'cpu').
        num_iterations (int): Le nombre de vecteurs aléatoires à utiliser pour l'estimation.

    Returns:
        dict: Un dictionnaire mappant le nom de chaque couche à sa trace Hessienne estimée.
    """
    
    hessian_traces = {}
    
    # Identifier toutes les couches à analyser (celles avec des poids)
    layers_to_analyze = {
        name: module for name, module in model.named_modules()
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear))
    }
    
    # Créer un itérateur pour piocher des batches
    data_iterator = iter(data_loader)
    
    print(f"Début de l'estimation de la trace Hessienne pour {len(layers_to_analyze)} couches...")
    start_time = time.time()
    
    for name, module in layers_to_analyze.items():
        trace_estimations = []
        for i in range(num_iterations):
            # Obtenir un nouveau batch de données
            try:
                feats, labels = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                feats, labels = next(data_iterator)
            
            if feats.dim() == 3: # Si on a (N, H, W)
                feats = feats.unsqueeze(1) # Le transformer en (N, 1, H, W)
            
            feats, labels = feats.to(device), labels.to(device)

            # Le modèle doit être en mode .train() pour permettre la création du graphe
            model.train()
            model.zero_grad()

            # Forward pass
            outputs = model(feats, labels)
            loss = criterion(outputs, labels)
            
            # Calculer les gradients par rapport aux poids de la couche actuelle
            grads = torch.autograd.grad(loss, module.weight, create_graph=True)[0]
            
            # Vecteur de Rademacher
            v = torch.randint_like(module.weight, low=0, high=2, device=device) * 2.0 - 1.0
            
            # Produit gradient-vecteur
            grad_v_prod = (grads * v).sum()
            
            # Produit Hessienne-vecteur
            h_v_prod = torch.autograd.grad(grad_v_prod, module.weight, retain_graph=False)[0]
            
            # Estimer la trace
            trace_estimations.append((h_v_prod * v).sum().item())

        # Moyenne des estimations pour la couche
        avg_trace = sum(trace_estimations) / len(trace_estimations)
        hessian_traces[name] = avg_trace
        print(f"  - Couche '{name}': Trace Hessienne estimée = {avg_trace:.4f}")

    model.eval() # Remettre le modèle en mode évaluation
    
    end_time = time.time()
    print(f"\nEstimation Hessienne terminée en {end_time - start_time:.2f} secondes.")
    
    return hessian_traces

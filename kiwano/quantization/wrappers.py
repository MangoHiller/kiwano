# kiwano/quantization/wrappers.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# WRAPPER POUR LES COUCHES DE CONVOLUTION 2D
# ==============================================================================
class KMeansQuantConv2d(nn.Module):
    """
    Wrapper pour appliquer la quantification KMQAT aux couches nn.Conv2d.
    """
    def __init__(self, original_conv_layer, n_bits=8, retention_ratio=0.9, debug=False):
        super(KMeansQuantConv2d, self).__init__()

        # On garde une ref à la couche originale pour ses poids et biais
        #self.original_layer = original_conv_layer
        
        self.n_bits = n_bits
        self.retention_ratio = retention_ratio

        self.debug = debug

        # 1. Copier les attributs de configuration (stride, padding, etc.)
        self.in_channels = original_conv_layer.in_channels
        self.out_channels = original_conv_layer.out_channels
        self.kernel_size = original_conv_layer.kernel_size
        self.stride = original_conv_layer.stride
        self.padding = original_conv_layer.padding
        self.dilation = original_conv_layer.dilation
        self.groups = original_conv_layer.groups

        # 2. Réassigner les nn.Parameter. C'est l'étape la plus importante.
        # Le wrapper devient maintenant le propriétaire direct des poids et biais.
        self.weight = original_conv_layer.weight
        if original_conv_layer.bias is not None:
            self.bias = original_conv_layer.bias
        else:
            self.register_parameter('bias', None) # Important pour la cohérence  

    def _quantize_weights(self, weights):
        """Contient le pipeline complet de KMQAT."""
        
        # Étape 3a : Récupération des poids (déjà passés en argument)
        current_weights = weights

        # Étape 3b : Exclusion des outliers (rétention ratio)
        # Aplatir les poids en un vecteur 1D
        weights_flat = current_weights.flatten()

        # Trier les poids pour identifier les outliers
        sorted_weights, _ = torch.sort(weights_flat)


        # Calculer les indices pour conserver r% des poids centraux
        num_weights = sorted_weights.numel()
        low_idx = int(num_weights * (1 - self.retention_ratio) / 2)
        high_idx = int(num_weights * (1 + self.retention_ratio) / 2)
        
        # Sélectionner uniquement les poids centraux pour le calcul des centroids
        central_weights = sorted_weights[low_idx:high_idx]
        
        # Étape 3c : Partitionnement en intervalles égaux 
        num_weights_to_group = central_weights.numel()
        
        # On crée un nombre de centroids égal à 2^n_bits ou au nombre de poids centraux,
        # selon le plus petit des deux.
        # Cela permet de s'adapter à des couches avec peu de poids centraux.
        if num_weights_to_group < 2:
            return weights, torch.tensor(1.0, device=weights.device) # Ne peut pas quantifier

        num_centroids = min(2**self.n_bits, num_weights_to_group)
        intervals = torch.chunk(central_weights, num_centroids)

        # Étape 3d : Calcul des centroids
        # Pour chaque intervalle, on calcule sa moyenne.
        codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)
        
        # Étape 3e : RESCALING SYMÉTRIQUE DES CENTROIDS
                # On divise tous les centroids par cette valeur pour les ramener dans [-1, 1]
                # On ajoute un petit epsilon pour éviter la division par zéro si max_abs_val est 0
        max_abs_val = torch.max(torch.abs(codebook))
        rescaled_codebook = codebook / (max_abs_val + 1e-8)

        # Étape 3f : ASSIGNATION DE CHAQUE POIDS AU CENTROID LE PLUS PROCHE
        distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
        indices = torch.argmin(distances, dim=1)

        quantized_weights_rescaled = rescaled_codebook[indices].reshape(weights.shape)


        if self.debug:
        # (temporaire) Vérification visuelle des résultats intermédiaires
            print(f"[Quantization] Poids originaux : {num_weights}")
            print(f"[Quantization] Poids centraux retenus ({self.retention_ratio*100}%): {central_weights.numel()}")
            print(f"[Quantization] Nombre d'intervalles créés : {len(intervals)}")
            print(f"[Quantization] Taille du premier intervalle : {intervals[0].numel()}")

            # Vérifications visuelles du calcul des centroids
            print(f"  [Quantization] Poids centraux: {central_weights.numel()}, Intervalles créés: {len(intervals)}")
            print(f"  [Quantization] Premiers 3 centroids calculés: {codebook[:3].tolist()}")

            # Verification du rescaling des centroids 
            print(f"  [Quantization] Max abs centroid value: {max_abs_val.item():.4f}")
            print(f"  [Quantization] Premiers 3 centroids rescalés: {rescaled_codebook[:3].tolist()}")

            # Verification de l'assignation des poids

            print(f"  [Quantization] Premiers 3 poids quantifiés : {quantized_weights_rescaled[:3].tolist()}")


        return quantized_weights_rescaled, max_abs_val

    def forward(self, x):
        float_weight = self.weight
        
        quantized_weight_rescaled, alpha = self._quantize_weights(float_weight)
        
        # Étape 3g: Remise à l'échelle
        final_quantized_weight = quantized_weight_rescaled * alpha
        
        # STE (Straight-Through Estimator)
        ste_quantized_weight = final_quantized_weight + (float_weight - final_quantized_weight).detach()
        
        return F.conv2d(x, ste_quantized_weight, self.bias, 
                        self.stride, self.padding, 
                        self.dilation, self.groups)

    def get_quantization_components(self):
        """
        Exécute le pipeline de quantification une dernière fois pour l'export
        et retourne les composants nécessaires pour une sauvegarde compacte : 
        le dictionnaire (codebook) et les indices.
        """
        with torch.no_grad():
            # Utilise le poids float32 actuel du wrapper
            weights = self.weight
            weights_flat = weights.flatten()

            # Recalculer les centroids une dernière fois sur la base des poids finaux
            sorted_weights, _ = torch.sort(weights_flat)
            num_weights = sorted_weights.numel()
            low_idx = int(num_weights * (1 - self.retention_ratio) / 2)
            high_idx = int(num_weights * (1 + self.retention_ratio) / 2)
            central_weights = sorted_weights[low_idx:high_idx]

            num_weights_to_group = central_weights.numel()
            if num_weights_to_group < 2:
                # Si la quantification n'est pas possible, on retourne des composants vides
                return None, None, self.bias

            num_centroids = min(2**self.n_bits, num_weights_to_group)
            intervals = torch.chunk(central_weights, num_centroids)
            
            # Le dictionnaire (codebook) à l'échelle d'origine
            codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)

            # Calculer les indices finaux
            distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
            indices = torch.argmin(distances, dim=1)

            # Retourner le codebook, les indices, et le biais
            return codebook, indices, self.bias

# ==============================================================================
# WRAPPER POUR LES COUCHES DE CONVOLUTION 1D
# ==============================================================================
class KMeansQuantConv1d(nn.Module):
    """
    Wrapper pour appliquer la quantification KMQAT aux couches nn.Conv1d.
    Identique à Conv2d, sauf pour l'appel à F.conv1d.
    """
    def __init__(self, original_conv_layer, n_bits=8, retention_ratio=0.9, debug=False):
        super(KMeansQuantConv1d, self).__init__()

        # On garde une ref à la couche originale pour ses poids et biais
        #self.original_layer = original_conv_layer
        self.n_bits = n_bits
        self.retention_ratio = retention_ratio

        self.debug = debug

        # 1. Copier les attributs de configuration (stride, padding, etc.)
        self.in_channels = original_conv_layer.in_channels
        self.out_channels = original_conv_layer.out_channels
        self.kernel_size = original_conv_layer.kernel_size
        self.stride = original_conv_layer.stride
        self.padding = original_conv_layer.padding
        self.dilation = original_conv_layer.dilation
        self.groups = original_conv_layer.groups

        # 2. Réassigner les nn.Parameter. C'est l'étape la plus importante.
        # Le wrapper devient maintenant le propriétaire direct des poids et biais.
        self.weight = original_conv_layer.weight
        if original_conv_layer.bias is not None:
            self.bias = original_conv_layer.bias
        else:
            self.register_parameter('bias', None) # Important pour la cohérence  

    def _quantize_weights(self, weights):
        """Contient le pipeline complet de KMQAT."""
        
        # Étape 3a : Récupération des poids (déjà passés en argument)
        current_weights = weights

        # Étape 3b : Exclusion des outliers (rétention ratio)
        # Aplatir les poids en un vecteur 1D
        weights_flat = current_weights.flatten()

        # Trier les poids pour identifier les outliers
        sorted_weights, _ = torch.sort(weights_flat)


        # Calculer les indices pour conserver r% des poids centraux
        num_weights = sorted_weights.numel()
        low_idx = int(num_weights * (1 - self.retention_ratio) / 2)
        high_idx = int(num_weights * (1 + self.retention_ratio) / 2)
        
        # Sélectionner uniquement les poids centraux pour le calcul des centroids
        central_weights = sorted_weights[low_idx:high_idx]
        
        # Étape 3c : Partitionnement en intervalles égaux 
        num_weights_to_group = central_weights.numel()
        
        # On crée un nombre de centroids égal à 2^n_bits ou au nombre de poids centraux,
        # selon le plus petit des deux.
        # Cela permet de s'adapter à des couches avec peu de poids centraux.
        if num_weights_to_group < 2:
            return weights, torch.tensor(1.0, device=weights.device) # Ne peut pas quantifier

        num_centroids = min(2**self.n_bits, num_weights_to_group)
        intervals = torch.chunk(central_weights, num_centroids)

        # Étape 3d : Calcul des centroids
        # Pour chaque intervalle, on calcule sa moyenne.
        codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)
        
        # Étape 3e : RESCALING SYMÉTRIQUE DES CENTROIDS
                # On divise tous les centroids par cette valeur pour les ramener dans [-1, 1]
                # On ajoute un petit epsilon pour éviter la division par zéro si max_abs_val est 0
        max_abs_val = torch.max(torch.abs(codebook))
        rescaled_codebook = codebook / (max_abs_val + 1e-8)

        # Étape 3f : ASSIGNATION DE CHAQUE POIDS AU CENTROID LE PLUS PROCHE
        distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
        indices = torch.argmin(distances, dim=1)

        quantized_weights_rescaled = rescaled_codebook[indices].reshape(weights.shape)


        if self.debug:
        # (temporaire) Vérification visuelle des résultats intermédiaires
            print(f"[Quantization] Poids originaux : {num_weights}")
            print(f"[Quantization] Poids centraux retenus ({self.retention_ratio*100}%): {central_weights.numel()}")
            print(f"[Quantization] Nombre d'intervalles créés : {len(intervals)}")
            print(f"[Quantization] Taille du premier intervalle : {intervals[0].numel()}")

            # Vérifications visuelles du calcul des centroids
            print(f"  [Quantization] Poids centraux: {central_weights.numel()}, Intervalles créés: {len(intervals)}")
            print(f"  [Quantization] Premiers 3 centroids calculés: {codebook[:3].tolist()}")

            # Verification du rescaling des centroids 
            print(f"  [Quantization] Max abs centroid value: {max_abs_val.item():.4f}")
            print(f"  [Quantization] Premiers 3 centroids rescalés: {rescaled_codebook[:3].tolist()}")

            # Verification de l'assignation des poids

            print(f"  [Quantization] Premiers 3 poids quantifiés : {quantized_weights_rescaled[:3].tolist()}")


        return quantized_weights_rescaled, max_abs_val

    def forward(self, x):
        float_weight = self.weight
        
        quantized_weight_rescaled, alpha = self._quantize_weights(float_weight)
        
        # Étape 3g: Remise à l'échelle
        final_quantized_weight = quantized_weight_rescaled * alpha
        
        # STE (Straight-Through Estimator)
        ste_quantized_weight = final_quantized_weight + (float_weight - final_quantized_weight).detach()
        
        return F.conv1d(x, ste_quantized_weight, self.bias, 
                        self.stride, self.padding, 
                        self.dilation, self.groups)

    def get_quantization_components(self):
        """
        Exécute le pipeline de quantification une dernière fois pour l'export
        et retourne les composants nécessaires pour une sauvegarde compacte : 
        le dictionnaire (codebook) et les indices.
        """
        with torch.no_grad():
            # Utilise le poids float32 actuel du wrapper
            weights = self.weight
            weights_flat = weights.flatten()

            # Recalculer les centroids une dernière fois sur la base des poids finaux
            sorted_weights, _ = torch.sort(weights_flat)
            num_weights = sorted_weights.numel()
            low_idx = int(num_weights * (1 - self.retention_ratio) / 2)
            high_idx = int(num_weights * (1 + self.retention_ratio) / 2)
            central_weights = sorted_weights[low_idx:high_idx]

            num_weights_to_group = central_weights.numel()
            if num_weights_to_group < 2:
                # Si la quantification n'est pas possible, on retourne des composants vides
                return None, None, self.bias

            num_centroids = min(2**self.n_bits, num_weights_to_group)
            intervals = torch.chunk(central_weights, num_centroids)
            
            # Le dictionnaire (codebook) à l'échelle d'origine
            codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)

            # Calculer les indices finaux
            distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
            indices = torch.argmin(distances, dim=1)

            # Retourner le codebook, les indices, et le biais
            return codebook, indices, self.bias

# ==============================================================================
# WRAPPER POUR LES COUCHES LINÉAIRES (DENSES)
# ==============================================================================
class KMeansQuantLinear(nn.Module):
    """
    Wrapper pour appliquer la quantification KMQAT aux couches nn.Linear.
    """
    def __init__(self, original_linear_layer, n_bits=8, retention_ratio=0.9, debug=False):
        super().__init__()
        # On garde une référence à la couche originale pour ses poids et biais
        #self.original_layer = original_linear_layer
        self.n_bits = n_bits
        self.retention_ratio = retention_ratio

        self.debug = debug

        # 1. Copier les attributs pertinents d'une couche nn.Linear
        self.in_features = original_linear_layer.in_features
        self.out_features = original_linear_layer.out_features

        # 2. Réassigner les nn.Parameter.
        self.weight = original_linear_layer.weight
        if original_linear_layer.bias is not None:
            self.bias = original_linear_layer.bias
        else:
            self.register_parameter('bias', None)
        
    def _quantize_weights(self, weights):
        """Contient le pipeline complet de KMQAT."""
        
        # Étape 3a : Récupération des poids (déjà passés en argument)
        current_weights = weights

        # Étape 3b : Exclusion des outliers (rétention ratio)
        # Aplatir les poids en un vecteur 1D
        weights_flat = current_weights.flatten()

        # Trier les poids pour identifier les outliers
        sorted_weights, _ = torch.sort(weights_flat)


        # Calculer les indices pour conserver r% des poids centraux
        num_weights = sorted_weights.numel()
        low_idx = int(num_weights * (1 - self.retention_ratio) / 2)
        high_idx = int(num_weights * (1 + self.retention_ratio) / 2)
        
        # Sélectionner uniquement les poids centraux pour le calcul des centroids
        central_weights = sorted_weights[low_idx:high_idx]
        
        # Étape 3c : Partitionnement en intervalles égaux 
        num_weights_to_group = central_weights.numel()
        
        # On crée un nombre de centroids égal à 2^n_bits ou au nombre de poids centraux,
        # selon le plus petit des deux.
        # Cela permet de s'adapter à des couches avec peu de poids centraux.
        if num_weights_to_group < 2:
            return weights, torch.tensor(1.0, device=weights.device) # Ne peut pas quantifier

        num_centroids = min(2**self.n_bits, num_weights_to_group)
        intervals = torch.chunk(central_weights, num_centroids)

        # Étape 3d : Calcul des centroids
        # Pour chaque intervalle, on calcule sa moyenne.
        codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)
        
        # Étape 3e : RESCALING SYMÉTRIQUE DES CENTROIDS
                # On divise tous les centroids par cette valeur pour les ramener dans [-1, 1]
                # On ajoute un petit epsilon pour éviter la division par zéro si max_abs_val est 0
        max_abs_val = torch.max(torch.abs(codebook))
        rescaled_codebook = codebook / (max_abs_val + 1e-8)

        # Étape 3f : ASSIGNATION DE CHAQUE POIDS AU CENTROID LE PLUS PROCHE
        distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
        indices = torch.argmin(distances, dim=1)

        quantized_weights_rescaled = rescaled_codebook[indices].reshape(weights.shape)


        if self.debug:
        # (temporaire) Vérification visuelle des résultats intermédiaires
            print(f"[Quantization] Poids originaux : {num_weights}")
            print(f"[Quantization] Poids centraux retenus ({self.retention_ratio*100}%): {central_weights.numel()}")
            print(f"[Quantization] Nombre d'intervalles créés : {len(intervals)}")
            print(f"[Quantization] Taille du premier intervalle : {intervals[0].numel()}")

            # Vérifications visuelles du calcul des centroids
            print(f"  [Quantization] Poids centraux: {central_weights.numel()}, Intervalles créés: {len(intervals)}")
            print(f"  [Quantization] Premiers 3 centroids calculés: {codebook[:3].tolist()}")

            # Verification du rescaling des centroids 
            print(f"  [Quantization] Max abs centroid value: {max_abs_val.item():.4f}")
            print(f"  [Quantization] Premiers 3 centroids rescalés: {rescaled_codebook[:3].tolist()}")

            # Verification de l'assignation des poids

            print(f"  [Quantization] Premiers 3 poids quantifiés : {quantized_weights_rescaled[:3].tolist()}")


        return quantized_weights_rescaled, max_abs_val
    
    def forward(self, x):
        # 1. Récupérer les poids float32 de la couche originale
        float_weight = self.weight
        
        # 2. Simuler la quantification
        quantized_weight_rescaled, alpha = self._quantize_weights(float_weight)

        final_quantized_weight = quantized_weight_rescaled * alpha

        # 3. Appliquer le STE
        ste_quantized_weight = final_quantized_weight + (float_weight - final_quantized_weight).detach()
        
        # 4. Effectuer l'opération linéaire avec les poids simulés
        return F.linear(x, ste_quantized_weight, self.bias)

    def get_quantization_components(self):
        """
        Exécute le pipeline de quantification une dernière fois pour l'export
        et retourne les composants nécessaires pour une sauvegarde compacte : 
        le dictionnaire (codebook) et les indices.
        """
        with torch.no_grad():
            # Utilise le poids float32 actuel du wrapper
            weights = self.weight
            weights_flat = weights.flatten()

            # Recalculer les centroids une dernière fois sur la base des poids finaux
            sorted_weights, _ = torch.sort(weights_flat)
            num_weights = sorted_weights.numel()
            low_idx = int(num_weights * (1 - self.retention_ratio) / 2)
            high_idx = int(num_weights * (1 + self.retention_ratio) / 2)
            central_weights = sorted_weights[low_idx:high_idx]

            num_weights_to_group = central_weights.numel()
            if num_weights_to_group < 2:
                # Si la quantification n'est pas possible, on retourne des composants vides
                return None, None, self.bias

            num_centroids = min(2**self.n_bits, num_weights_to_group)
            intervals = torch.chunk(central_weights, num_centroids)
            
            # Le dictionnaire (codebook) à l'échelle d'origine
            codebook = torch.tensor([chunk.mean() for chunk in intervals], device=weights.device)

            # Calculer les indices finaux
            distances = torch.abs(weights_flat.unsqueeze(1) - codebook.unsqueeze(0))
            indices = torch.argmin(distances, dim=1)

            # Retourner le codebook, les indices, et le biais
            return codebook, indices, self.bias


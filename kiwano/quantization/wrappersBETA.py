# kiwano/quantization/wrappers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.helpers import _central_clip, _build_codebook_from_quantiles, _ema_refresh_codebook
from .core.ops import init_kmqat_params_for_weight


# ==============================================================================
# WRAPPER POUR LES COUCHES DE CONVOLUTION 2D
# ==============================================================================
class KMeansQuantConv2d(nn.Module):
    """
    Quantification KMQAT pour nn.Conv2d.
    - Coupe r% d'outliers (step 1)
    - Code-book = mean par intervalle (k-means(k=1), step 2)
    - Réscaling symétrique des centroids (step 3)
    - alpha scalaire entraînable (LSQ-like init)
    """

    # ------------------------------------------------------------------ #
    #                        INITIALISATION                              #
    # ------------------------------------------------------------------ #
    def __init__(self, original_conv_layer: nn.Conv2d, n_bits=8, retention_ratio=0.9, symmetric_rescale=True, per_channel=True, debug=False):
        super(KMeansQuantConv2d, self).__init__()

        # Hyper KMQAT
        self.n_bits           = n_bits
        self.retention_ratio  = retention_ratio
        self.symmetric_rescale= symmetric_rescale
        self.per_channel      = per_channel
        self.debug            = debug
        
        # ----------------- paramètres de la couche d'origine -----------------
        self.in_channels  = original_conv_layer.in_channels
        self.out_channels = original_conv_layer.out_channels
        self.kernel_size  = original_conv_layer.kernel_size
        self.stride       = original_conv_layer.stride
        self.padding      = original_conv_layer.padding
        self.dilation     = original_conv_layer.dilation
        self.groups       = original_conv_layer.groups

        # ------------------------- poids / biais -----------------------------
        self.weight = nn.Parameter(original_conv_layer.weight.detach().clone())
        if original_conv_layer.bias is not None:
            self.bias = nn.Parameter(original_conv_layer.bias.detach().clone())
        else:
            self.register_parameter('bias', None)

        # --------------------- buffers & paramètres QAT ----------------------
        self.register_buffer('codebook', None)            # (K,) les centroids
        self.register_buffer('indices',  None)            # (N,) assignation de chaque poids au centroid
        #self.alpha = nn.Parameter(torch.tensor(1.0))      # scalaire entraînable
        self.alpha = nn.Parameter(torch.ones(self.out_channels)) # scale alpha pour chaque canal de sortie


        # --- warmup & reparam alpha ---
        self.ste_weight_only = False    # piloté par le script d'entraînement (warm-up)
        self._alpha_eps = 1e-5          # alpha_effectif = softplus(alpha_raw) + eps
        self._alpha_softplus = True     # active l'usage softplus
        self.quantization_enabled = False

    # ------------------------------------------------------------------ #
    #                    CONSTRUCTION DES PARAMÈTRES PTQ                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _init_quant_params(self):
        """
        But: Construire les paramètres PTQ initiaux (codebook, indices, alpha) une fois, au moment où on active la quantif.
            Dans LSQ et QAT classiques : on sait que pour Conv2d, la distribution des poids varie par canal de sortie.

            Exemple : certains filtres détectent des bords → poids centrés et petits.

                    D’autres capturent des patterns plus “forts” → poids plus grands.
        
        Entrée: Poids FP32 W = [out, in, kh, kw] (nn.Conv2d)
        
        Sorties stockées:
            self.codebook: [out, K] (Conv per‑channel)

            self.indices: [out, NperOut]

            self.alpha: [out]
        """
        codebook, indices, alpha = init_kmqat_params_for_weight(
            self.weight.data,
            n_bits=self.n_bits,
            retention_ratio=self.retention_ratio,
            symmetric_rescale=self.symmetric_rescale,
            kind="conv2d",
            per_channel=True,
        )

        self.codebook = codebook
        self.indices = indices

        alpha = alpha.to(self.alpha.device, self.alpha.dtype)
        # inverse softplus approx: a0 = log(exp(alpha) - 1)
        a0 = torch.log(torch.exp(alpha) - 1.0 + 1e-6)
        self.alpha.data.copy_(a0)  # 'alpha' devient le paramètre "raw"

        if self.debug:
            print(f"[KMQAT/Conv2d] K={2**self.n_bits}, alpha_mean={self.alpha.mean().item():.4f}")

    # ------------------------------------------------------------------ #


    def enable_quantization(self) -> None:
        """
        Active la simulation de quantification pour ce wrapper.
        """
        if not self.quantization_enabled:
            self._init_quant_params()
            self.quantization_enabled = True
            print("KMQAT Conv2d enabled")

    def disable_quantization(self) -> None:
        """
        Désactive la simulation, le wrapper se comporte en FP32.
        """
        self.quantization_enabled = False

    # ------------------------------------------------------------------ #
    #                RÉCUPÈRE LES POIDS QUANTIFIÉS COURANTS              #
    # ------------------------------------------------------------------ #
    def _quantized_weight(self) -> torch.Tensor:
        """
        But: Reconstruire W_q = alpha x q (sans STE).

        Pour chaque canal de sortie, on récupère le centroid le plus proche de chaque poids. q = codebook[oc][indices].

        Entrée: self.codebook (K,) et self.indices (N) ou (out, NperOut)
        Sortie: W_q = [out, in, kh, kw] (ou [out, NperOut, kh, kw] si per-channel)
        """
        if not self.quantization_enabled or self.codebook is None:
            return self.weight
        O, I, KH, KW = self.weight.shape
        if self.per_channel and self.codebook.dim() == 2:
            rows = []
            for oc in range(O):
                cb = self.codebook[oc]      # [K]
                idx = self.indices[oc]      # [NperOut]
                q = cb[idx]                 # [NperOut]
                rows.append(q)
            Qv = torch.stack(rows)          # [out, NperOut]
            Q  = Qv.view_as(self.weight)    # [out,in,kh,kw]
            alpha_eff = F.softplus(self.alpha) + self._alpha_eps # alpha > 0 garanti au forward alpha=softplus(alpha)+eps
            return Q * alpha_eff.view(-1,1,1,1)
        else:
            cb = self.codebook              # [K]
            idx= self.indices               # [N]
            q  = cb[idx].view_as(self.weight)
            return q * self.alpha.view(-1,1,1,1)

    # ------------------------------------------------------------------ #
    #                RECALCUL DES CODEBOOK / REFRESH                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_codebook(self, beta: float = 0.1, reassign: bool = False):
        """
        EMA des centroids (et option réassignation). 

        But: Mettre à jour les centroids par EMA vers la moyenne des points assignés.

        """
        if not getattr(self, "quantization_enabled", False) or (self.codebook is None):
            return None
        return _ema_refresh_codebook(
            weight=self.weight,
            alpha_raw=self.alpha,
            codebook=self.codebook,
            indices=self.indices,
            per_channel=self.per_channel,
            alpha_softplus=getattr(self, "_alpha_softplus", True),
            alpha_eps=getattr(self, "_alpha_eps", 1e-5),
            beta=beta,
            reassign=reassign,
        )

    # ------------------------------------------------------------------ #
    #                       CHEMIN DE FORWARD                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        But: Simuler la quantification des poids lors de l'entraînement (QAT)  avec STE durant le backward ou utiliser les poids quantifiés lors de l'inférence.
        """
        if not self.quantization_enabled:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        Wq = self._quantized_weight() # poid quantifié (alpha·q) reconstruit

        if self.training:
            if self.ste_weight_only:
                # Warm-up: alpha gelé, seul W reçoit du gradient
                Wused = (Wq - self.weight).detach() + self.weight 
            else:
                # KMQAT complet: alpha et W reçoivent du gradient W et alpha apprennent
                Wused = Wq + (self.weight - self.weight.detach())
        else:
            Wused = Wq.detach() # en inférence, on utilise Wq sans gradient
        # STE
        #Wused = (Wq - self.weight).detach() + self.weight if self.training else Wq.detach() # Wused = (Wq - W).detach() + W → Weight-only QAT:W learns, alpha ne reçoit pas de gradient.
        #Wused = Wq + (self.weight - self.weight.detach()) if self.training else Wq.detach() #Wused = Wq + (W - W.detach()) → KMQAT complet: W learns, alpha aussi

        return F.conv2d(x, Wused, self.bias, self.stride, self.padding, self.dilation, self.groups)


    # ------------------------------------------------------------------ #
    #                 EXPORT DES COMPOSANTS DE QUANTIZATION              #
    # ------------------------------------------------------------------ #
    def get_quantization_components(self):
        """
        But: Récupérer les paramètres de quantification pour la sauvegarde.
        Renvoie (codebook_rescalé, indices, bias) pour la sauvegarde.
        """
        if self.codebook is None or self.indices is None:
            return None, None, self.bias
        # on stocke les centroids dans l’échelle rescalée ; alpha est sauvegardé à part via state_dict
        return self.codebook, self.indices, self.bias

# ==============================================================================
# WRAPPER POUR LES COUCHES DE CONVOLUTION 1D
# ==============================================================================
class KMeansQuantConv1d(nn.Module):
    """
    Quantification KMQAT pour nn.Conv1d.
    - Coupe r% d'outliers (step 1)
    - Code-book = mean par intervalle (k-means(k=1), step 2)
    - Réscaling symétrique des centroids (step 3)
    - alpha par canal de sortie (LSQ-like init)
    """

    def __init__(self, original_conv_layer: nn.Conv1d, n_bits=8, retention_ratio=0.9,
                 symmetric_rescale=True, per_channel=True, debug=False):
        super(KMeansQuantConv1d, self).__init__()

        # Hyper KMQAT
        self.n_bits            = n_bits
        self.retention_ratio   = retention_ratio
        self.symmetric_rescale = symmetric_rescale
        self.per_channel       = per_channel
        self.debug             = debug

        # ----------------- paramètres de la couche d'origine -----------------
        self.in_channels  = original_conv_layer.in_channels
        self.out_channels = original_conv_layer.out_channels
        self.kernel_size  = original_conv_layer.kernel_size
        self.stride       = original_conv_layer.stride
        self.padding      = original_conv_layer.padding
        self.dilation     = original_conv_layer.dilation
        self.groups       = original_conv_layer.groups

        # ------------------------- poids / biais -----------------------------
        self.weight = nn.Parameter(original_conv_layer.weight.detach().clone())  # [O, I, K]
        if original_conv_layer.bias is not None:
            self.bias = nn.Parameter(original_conv_layer.bias.detach().clone())
        else:
            self.register_parameter('bias', None)

        # --------------------- buffers & paramètres QAT ----------------------
        self.register_buffer('codebook', None)   # [O, K] en per-channel, sinon [K]
        self.register_buffer('indices',  None)   # [O, NperOut] en per-channel, sinon [N]
        self.alpha = nn.Parameter(torch.ones(self.out_channels))  # [O]

        # --- warmup & reparam alpha ---
        self.ste_weight_only = False    # piloté par le script d'entraînement (warm-up)
        self._alpha_eps = 1e-5          # alpha_effectif = softplus(alpha_raw) + eps
        self._alpha_softplus = True     # active l'usage softplus
        self.quantization_enabled = False

    # ------------------------------------------------------------------ #
    #                    CONSTRUCTION DES PARAMÈTRES PTQ                 #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _init_quant_params(self):
        """
        Construit codebook/indices/alpha à l’activation de la quantif.
        W: [O, I, K]  (Conv1d)
        per-channel: on traite chaque sortie (filtre) séparément.
        """
        codebook, indices, alpha = init_kmqat_params_for_weight(
            self.weight.data,
            n_bits=self.n_bits,
            retention_ratio=self.retention_ratio,
            symmetric_rescale=self.symmetric_rescale,
            kind="conv1d",
            per_channel=True,
        )

        self.codebook = codebook
        self.indices = indices

        alpha = alpha.to(self.alpha.device, self.alpha.dtype)
        # inverse softplus approx: a0 = log(exp(alpha) - 1)
        a0 = torch.log(torch.exp(alpha) - 1.0 + 1e-6)
        self.alpha.data.copy_(a0)  # 'alpha' devient le paramètre "raw"

        if self.debug:
            print(f"[KMQAT/Conv1d] K={2**self.n_bits}, alpha_mean={self.alpha.mean().item():.4f}")

    # ------------------------------------------------------------------ #
    def enable_quantization(self) -> None:
        if not self.quantization_enabled:
            self._init_quant_params()
            self.quantization_enabled = True
            print("KMQAT Conv1d enabled")

    def disable_quantization(self) -> None:
        self.quantization_enabled = False

    # ------------------------------------------------------------------ #
    #                RÉCUPÈRE LES POIDS QUANTIFIÉS COURANTS              #
    # ------------------------------------------------------------------ #
    def _quantized_weight(self) -> torch.Tensor:
        """
        Reconstruit Wq = α × q.
        W shape: [O, I, K1]
        α shape: [O] → reshape [O,1,1]
        """
        if not self.quantization_enabled or self.codebook is None:
            return self.weight

        O, I, K1 = self.weight.shape
        if self.per_channel and self.codebook.dim() == 2:
            rows = []
            for oc in range(O):
                cb = self.codebook[oc]   # [K]
                idx = self.indices[oc]   # [NperOut]
                q = cb[idx]              # [NperOut]
                rows.append(q)
            Qv = torch.stack(rows)       # [O, NperOut]
            Q  = Qv.view_as(self.weight) # [O, I, K1]
            alpha_eff = F.softplus(self.alpha) + self._alpha_eps # alpha > 0 garanti au forward alpha=softplus(alpha)+eps
            return Q * alpha_eff.view(-1, 1, 1)

        else:
            cb = self.codebook
            idx= self.indices
            q  = cb[idx].view_as(self.weight)  # [O, I, K1]
            return q * self.alpha.view(-1, 1, 1)

    # ------------------------------------------------------------------ #
    #                RECALCUL DES CODEBOOK / REFRESH                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_codebook(self, beta: float = 0.1, reassign: bool = False):
        """
        EMA des centroids (et option réassignation).
        But: Mettre à jour les centroids par EMA vers la moyenne des points assignés.
        """
        if not getattr(self, "quantization_enabled", False) or (self.codebook is None):
            return None
        return _ema_refresh_codebook(
            weight=self.weight,
            alpha_raw=self.alpha,
            codebook=self.codebook,
            indices=self.indices,
            per_channel=self.per_channel,
            alpha_softplus=getattr(self, "_alpha_softplus", True),
            alpha_eps=getattr(self, "_alpha_eps", 1e-5),
            beta=beta,
            reassign=reassign,
        )

    # ------------------------------------------------------------------ #
    #                       CHEMIN DE FORWARD                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        Wq = self._quantized_weight()
        if self.training:
            if self.ste_weight_only:
                # Warm-up: alpha gelé, seul W reçoit du gradient
                Wused = (Wq - self.weight).detach() + self.weight 
            else:
                # KMQAT complet: alpha et W reçoivent du gradient W et alpha apprennent
                Wused = Wq + (self.weight - self.weight.detach())
        else:
            Wused = Wq.detach() # en inférence, on utilise Wq sans gradient
        
        # STE: Wused = W + (Wq - W).detach() = (Wq - W).detach() + W
        #Wused = (Wq - self.weight).detach() + self.weight if self.training else Wq.detach() # Weight only
        #Wused = Wq + (self.weight - self.weight.detach()) if self.training else Wq.detach() # Weight and alpha learn

        return F.conv1d(x, Wused, self.bias, self.stride, self.padding, self.dilation, self.groups)

    # ------------------------------------------------------------------ #
    #                 EXPORT DES COMPOSANTS DE QUANTIZATION              #
    # ------------------------------------------------------------------ #
    def get_quantization_components(self):
        """
        Renvoie (codebook_rescalé, indices, bias) pour sauvegarde.
        α est sauvegardé à part via state_dict, ou tu peux exporter eff_codebook ailleurs.
        """
        if self.codebook is None or self.indices is None:
            return None, None, self.bias
        return self.codebook, self.indices, self.bias



# ==============================================================================
# KMQAT Linear — per-out-channel codebook/indices + alpha[out]
# ==============================================================================
class KMeansQuantLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, n_bits=8, retention_ratio=0.9,
                 symmetric_rescale=True, per_channel=True, debug=False):
        super().__init__()
        self.n_bits           = n_bits
        self.retention_ratio  = retention_ratio
        self.symmetric_rescale= symmetric_rescale
        self.per_channel      = per_channel
        self.debug            = debug

        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features

        self.weight = nn.Parameter(original_linear.weight.detach().clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.detach().clone())
        else:
            self.register_parameter('bias', None)

        self.register_buffer('codebook', None)   # [out,K] ou [K]
        self.register_buffer('indices',  None)   # [out,NperOut] ou [N]
        self.alpha = nn.Parameter(torch.ones(self.out_features))

        # --- warmup & reparam alpha ---
        self.ste_weight_only = False    # piloté par le script d'entraînement (warm-up)
        self._alpha_eps = 1e-5          # alpha_effectif = softplus(alpha_raw) + eps
        self._alpha_softplus = True     # active l'usage softplus
        self.quantization_enabled = False

    @torch.no_grad()
    def _init_quant_params(self):
        codebook, indices, alpha = init_kmqat_params_for_weight(
            self.weight.data,
            n_bits=self.n_bits,
            retention_ratio=self.retention_ratio,
            symmetric_rescale=self.symmetric_rescale,
            kind="linear",
            per_channel=True,
        )
        
        self.codebook = codebook
        self.indices = indices

        alpha = alpha.to(self.alpha.device, self.alpha.dtype)
        # inverse softplus approx: a0 = log(exp(alpha) - 1)
        a0 = torch.log(torch.exp(alpha) - 1.0 + 1e-6)
        self.alpha.data.copy_(a0)  # 'alpha' devient le paramètre "raw"
        
        if self.debug:
            print(f"[KMQAT/Linear] K={2**self.n_bits}, alpha_mean={self.alpha.mean().item():.4f}")

    def enable_quantization(self):
        if not self.quantization_enabled:
            self._init_quant_params()
            self.quantization_enabled = True
            if self.debug:
                print("KMQAT Linear enabled")

    def disable_quantization(self):
        self.quantization_enabled = False

    def _quantized_weight(self) -> torch.Tensor:
        if not self.quantization_enabled or self.codebook is None:
            return self.weight
        O, I = self.weight.shape
        if self.per_channel and self.codebook.dim() == 2:
            rows = []
            for oc in range(O):
                cb = self.codebook[oc]   # [K]
                idx= self.indices[oc]    # [in]
                q  = cb[idx]             # [in]
                rows.append(q)
            Q = torch.stack(rows)        # [out,in]
            alpha_eff = F.softplus(self.alpha) + self._alpha_eps # alpha > 0 garanti au forward alpha=softplus(alpha)+eps
            return Q * alpha_eff.view(-1,1)
        else:
            cb = self.codebook
            idx= self.indices
            q  = cb[idx].view_as(self.weight)
            return q * self.alpha.view(-1,1)

    # ------------------------------------------------------------------ #
    #                RECALCUL DES CODEBOOK / REFRESH                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def refresh_codebook(self, beta: float = 0.1, reassign: bool = False):
        """
        EMA des centroids (et option réassignation).
        But: Mettre à jour les centroids par EMA vers la moyenne des points assignés.
        """
        if not getattr(self, "quantization_enabled", False) or (self.codebook is None):
            return None
        return _ema_refresh_codebook(
            weight=self.weight,
            alpha_raw=self.alpha,
            codebook=self.codebook,
            indices=self.indices,
            per_channel=self.per_channel,
            alpha_softplus=getattr(self, "_alpha_softplus", True),
            alpha_eps=getattr(self, "_alpha_eps", 1e-5),
            beta=beta,
            reassign=reassign,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantization_enabled:
            return F.linear(x, self.weight, self.bias)
        Wq = self._quantized_weight()
        if self.training:
            if self.ste_weight_only:
                # Warm-up: alpha gelé, seul W reçoit du gradient
                Wused = (Wq - self.weight).detach() + self.weight 
            else:
                # KMQAT complet: alpha et W reçoivent du gradient W et alpha apprennent
                Wused = Wq + (self.weight - self.weight.detach())
        else:
            Wused = Wq.detach() # en inférence, on utilise Wq sans gradient

        #Wused = (Wq - self.weight).detach() + self.weight if self.training else Wq.detach() #weight only
        #Wused = Wq + (self.weight - self.weight.detach()) if self.training else Wq.detach() # weight W et alpha learn
        return F.linear(x, Wused, self.bias)
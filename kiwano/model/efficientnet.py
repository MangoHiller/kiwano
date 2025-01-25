import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet as PretrainedEfficientNet
from kiwano.model import ASTP, SpeakerEmbedding, AMSMLoss

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes, input_features=81, embed_features=256, model_name="efficientnet-b0"):
        """
        Classe EfficientNetV2 adaptée pour la vérification du locuteur.

        Args:
            num_classes (int): Nombre de classes (nombre de locuteurs).
            input_features (int): Dimensions des caractéristiques d'entrée (par défaut 81 pour FBANK).
            embed_features (int): Dimensions des embeddings.
            model_name (str): Nom du modèle EfficientNet pré-entraîné.
        """
        super(EfficientNetV2, self).__init__()

        # Charger le modèle pré-entraîné EfficientNet
        self.base_model = PretrainedEfficientNet.from_pretrained(model_name)

        # Adapter la première couche pour accepter 1 canal (FBANKs)
        self.base_model._conv_stem = nn.Conv2d(
            in_channels=1,
            out_channels=self.base_model._conv_stem.out_channels,
            kernel_size=self.base_model._conv_stem.kernel_size,
            stride=self.base_model._conv_stem.stride,
            padding=self.base_model._conv_stem.padding,
            bias=False
        )

        #######################################################################
        # 3) Supprimer la FC et le pooling adaptatif par défaut
        #    car on veut récupérer un tenseur [B, C, H, W], puis manipuler
        #    directement la dimension temporelle/frequency.
        #######################################################################
        self.base_model._fc = nn.Identity()               # plus de FC
        self.base_model._avg_pooling = nn.Identity()      # plus de GlobalAvgPool
        self.base_model._dropout = nn.Identity()          # plus de dropout

        #######################################################################
        # 4) Définir les modules pr la SV : ASTP, Embedding, AMSMLoss
        #######################################################################
        # - In dim : dépendra de la forme finale de self.base_model.extract_features
        #   Généralement, pour efficientnet-b0, on a [B, 1280, H, W].
        #
        # - On veut mimer ResNetV2 : on obtient un tenseur [B, C, T],
        #   puis ASTP => [B, 2*C], etc.
        #
        # ICI : "1280" est le nombre de canaux final d'EfficientNet-B0
        #######################################################################
       
        self.temporal_pooling = ASTP(in_dim=2560, bottleneck_dim=1280)  
        self.embedding = SpeakerEmbedding(in_dim=2 * 2560, embed_dim=embed_features) 
        #self.dropout = nn.Dropout(p=0.1)
        self.output = AMSMLoss(num_features=embed_features, num_classes=num_classes, s=10, m=0.1)  
        
        #self.output = nn.Linear(embed_features, num_classes) #sans AMSMLoss


    def forward(self, x, iden=None):
        """
        Forward pass.

        Args:
        x : [batch_size, 1, time, freq] = [B, 1, 350, 81] par ex.
        labels (iden): Lebel, ou None

        Returns:
            Tensor: Embeddings ou logits. : [B, embed_features] ou [B, num_classes]
        """

        #######################################################################
        # Étape A : Extraction des features
        #######################################################################
        # Après extract_features, on obtient [B, 1280, H', W']
        feats = self.base_model.extract_features(x)  # shape ~ [B, 1280, 10, 2], par ex.


        #######################################################################
        # Étape B : Placer la fréquence dans les canaux
        # feats => [B, 1280, 10, 2]
        # On transpose(2,3) => [B, 1280, 2, 10]
        # Puis on fusionne canaux = 1280×2 = 2560, time = 10
        # On obtient [B, 2560, 10]
        #######################################################################

        feats = feats.transpose(2, 3)  # => [B, 1280, 2, 10]
        feats = feats.flatten(1, 2)    # => [B, (1280×2), 10] = [B, 2560, 10]

        #######################################################################
        # Étape C : ASTP => [B, 2*C'] => [B, 5120]
        #######################################################################
        feats = self.temporal_pooling(feats) 

        #######################################################################
        # Étape D : Embedding => [B, embed_features], ex. [B, 256]
        #######################################################################
        feats = self.embedding(feats) 

        #######################################################################
        # Étape E : Si labels= None, on renvoie l'embedding (pour scoring).
        #           Sinon, on applique la couche AMSMLoss => logits
        #######################################################################

        if iden is not None:
        # Mode training => on renvoie des logits pour classification
            return self.output(feats, iden)  
        else:
        # Mode inference => on renvoie l'embedding
            return feats

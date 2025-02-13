# kd_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    """
    KL-Divergence entre logits Student / Teacher, avec ajustement de température.
    """
    def __init__(self, temperature=4.0, reduction='batchmean'):
        """
        Args:
            temperature (float) : température T pour le lissage des logits.
            reduction (str) : réduction pour KLDivLoss (ex: 'batchmean').
        """
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, student_logits, teacher_logits):
        # Diviser logits par T, prendre log-softmax / softmax
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)

        # Multiplier par T^2
        loss_kl = self.kl_div(student_log_probs, teacher_probs) * (T * T)
        return loss_kl


class CosineEmbedLoss(nn.Module):
    """
    Encourage le Student à apprendre des embeddings proches de ceux du Teacher.
    Utilise nn.CosineEmbeddingLoss.
    """
    def __init__(self, margin=0.0, reduction='mean'):
        super().__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)

    def forward(self, student_emb, teacher_emb, target):
        return self.cosine_loss(student_emb, teacher_emb, target)

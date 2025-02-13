# utils.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    """
    Sauvegarde un checkpoint contenant :
      - epoch
      - state_dict du model
      - state_dict de l'optimizer
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, path)
    print(f"[utils] Checkpoint sauvegardé : {path}")


class AverageMeter:
    """
    Classe pour suivre la valeur moyenne/mise à jour d'une métrique
    (loss, accuracy, etc.) sur un epoch.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val  = val
        self.sum += val * n
        self.count += n
        self.avg  = self.sum / self.count if self.count != 0 else 0

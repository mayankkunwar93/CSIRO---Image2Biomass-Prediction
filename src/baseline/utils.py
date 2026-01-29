import torch
import numpy as np
from sklearn.metrics import r2_score

# Target weights for the baseline (Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM)
TARGET_WEIGHTS = torch.tensor([0.1, 0.1, 0.1, 0.5, 0.2])

def weighted_mse_loss(p, y, device):
    weights = TARGET_WEIGHTS.to(device)
    base_loss = ((p - y)**2 * weights).sum() / weights.sum()
    return base_loss

def weighted_r2_score(y_true, y_pred):
    w = np.array([0.1, 0.1, 0.1, 0.5, 0.2])
    num = ((y_true - y_pred)**2 * w).sum()
    den = ((y_true - y_true.mean(axis=0))**2 * w).sum()
    return 1 - num / (den + 1e-8)

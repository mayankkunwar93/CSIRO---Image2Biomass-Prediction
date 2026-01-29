import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

# Taxonomy logic for multi-hot encoding
species_to_subspecies = {

    # Grassy types
    "BarleyGrass": ["grass", "barley"],
    "Barleygrass": ["grass", "barley"],
    "Bromegrass": ["grass", "brome"],
    "Fescue": ["grass", "fescue"],
    "Phalaris": ["grass", "phalaris"],
    "Ryegrass": ["grass", "rye"],
    "SilverGrass": ["grass", "silver"],
    "SpearGrass": ["grass", "spear"],

    # Clovers
    "Clover": ["clover"],
    "SubcloverDalkeith": ["clover", "subclover", "dalkeith"],
    "SubcloverLosa": ["clover", "subclover", "losa"],
    "WhiteClover": ["clover", "white"],

    # Legumes
    "Lucerne": ["legume", "lucerne"],

    # Broadleaf weeds
    "Capeweed": ["broadleaf", "capeweed"],
    "CrumbWeed": ["broadleaf", "crumbweed"],

    # Mixed
    "Mixed": ["mixed"]
}

def weighted_huber_loss(p, y, target_std, target_mean, weights=None, teacher_pred=None, supervise_weight=0.5, delta=1.0):
    t_std = torch.tensor(target_std, device=p.device).float().clamp(min=1e-3)
    t_mean = torch.tensor(target_mean, device=p.device).float()

    # Total Biomass Loss
    loss_total = F.huber_loss(p[:, 0], y[:, 0], delta=delta)

    # Ratio Protection Logic
    y_log = y * t_std + t_mean
    y_grams = torch.expm1(torch.clamp(y_log, max=10)).clamp(min=0.0)
    dead_grams = torch.clamp(y_grams[:, 0] - y_grams[:, 1], min=0.0)

    true_ratio_gdm_dead = torch.stack([y_grams[:, 1], dead_grams], dim=1)
    denom1 = true_ratio_gdm_dead.sum(dim=1, keepdim=True)
    true_ratio_gdm_dead = torch.where(denom1 > 1e-8, true_ratio_gdm_dead / (denom1 + 1e-8), torch.full_like(true_ratio_gdm_dead, 0.5))

    p_clamped = torch.clamp(p, min=-15, max=15)
    pred_ratio_gdm_dead_log = F.log_softmax(p_clamped[:, 1:3], dim=1)
    loss_ratio = F.kl_div(pred_ratio_gdm_dead_log, true_ratio_gdm_dead, reduction='batchmean')

    student_loss = (3.0 * loss_total) + (1.2 * loss_ratio)

    # Teacher Clamping
    loss_supervise = torch.tensor(0.0, device=p.device)
    if teacher_pred is not None and supervise_weight > 0:
        teacher_norm = (torch.clamp(teacher_pred, 0, 10) - t_mean) / t_std
        loss_supervise = F.huber_loss(p_clamped[:, :4], teacher_norm, delta=delta)

    total_loss = student_loss + (supervise_weight * loss_supervise)

    if weights is not None:
        return (total_loss * torch.clamp(weights, 0.1, 5.0)).mean()
    return total_loss

def weighted_r2_score(y_true, y_pred):
    w = np.array([0.5, 0.2, 0.1, 0.2])
    num = ((y_true - y_pred)**2 * w).sum()
    den = ((y_true - y_true.mean(axis=0))**2 * w).sum()
    return 1 - num / (den + 1e-8)

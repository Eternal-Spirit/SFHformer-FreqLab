"""
utils/losses.py - Segmentation Losses and Comprehensive Clinical Evaluation Metrics.
Includes:
- DiceLoss, BCEDiceLoss (Combined Loss)
- compute_segmentation_metrics (Dice, IoU, Sensitivity, Specificity, Precision, Accuracy)
- compute_stratified_thin_vessel_metrics (Evaluated strictly on <= 3.5 px ground truth skeleton)
- compute_vascular_continuity (Connected components fragmentation ratio)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import morphology
import scipy.ndimage as ndi

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice

class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss for robust boundary and area optimization."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class SoftSkeletonize(nn.Module):
    """Differentiable morphological soft-skeletonization (FP32) for topology preservation."""
    def __init__(self, num_iter=12):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

class SoftclDiceLoss(nn.Module):
    """
    Differentiable Centerline Dice (clDice) loss computed strictly in FP32.
    Masks the outer 2px patch border so vessels cut by crop boundaries do not
    introduce artificial edge skeletons into topology gradients.
    """
    def __init__(self, iter_=12, smooth=1.0, border_ignore=2):
        super().__init__()
        self.smooth = smooth
        self.border_ignore = border_ignore
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, logits, targets):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            y_pred = torch.sigmoid(logits.float())
            y_true = targets.float()
            skel_pred = self.soft_skeletonize(y_pred)
            skel_true = self.soft_skeletonize(y_true)

            b = self.border_ignore
            if b > 0 and skel_pred.shape[-1] > 2 * b and skel_pred.shape[-2] > 2 * b:
                skel_pred = skel_pred[:, :, b:-b, b:-b]
                skel_true = skel_true[:, :, b:-b, b:-b]
                y_pred = y_pred[:, :, b:-b, b:-b]
                y_true = y_true[:, :, b:-b, b:-b]

            tprec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
            tsens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
            cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + 1e-7)
            return 1.0 - cl_dice

class HybridTopologyLoss(nn.Module):
    """Compound BCE + Soft-Dice + FP32 Soft-clDice loss for Dual-Stream Hybrid U-Net."""
    def __init__(self, bce_weight=0.4, dice_weight=0.4, cldice_weight=0.2, skel_iter=12, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.cldice = SoftclDiceLoss(iter_=skel_iter)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        cl_loss = self.cldice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss + self.cldice_weight * cl_loss

def compute_segmentation_metrics(pred_binary, target_binary, fov_mask=None):
    """
    Computes clinical segmentation metrics strictly within the valid FOV.
    Inputs: boolean or uint8 binary numpy arrays.
    """
    if fov_mask is not None:
        p = pred_binary[fov_mask]
        t = target_binary[fov_mask]
    else:
        p = pred_binary.flatten()
        t = target_binary.flatten()

    tp = np.sum(p & t)
    fp = np.sum(p & (~t))
    fn = np.sum((~p) & t)
    tn = np.sum((~p) & (~t))

    eps = 1e-7
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps) # Recall
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "accuracy": float(accuracy)
    }

def compute_thin_vessel_sensitivity(pred_binary, target_binary, fov_mask=None):
    """
    Measures detection sensitivity strictly on the smallest annotated visible vessels (<= 3.5 px skeleton).
    """
    skel = morphology.skeletonize(target_binary > 0)
    dist = ndi.distance_transform_edt(target_binary > 0)
    thin_skel = skel & (dist * 2.0 <= 3.5)

    if fov_mask is not None:
        thin_skel = thin_skel & fov_mask

    total_thin = np.sum(thin_skel)
    if total_thin == 0:
        return 1.0

    detected_thin = np.sum(pred_binary & thin_skel)
    return float(detected_thin / total_thin)

def compute_vascular_continuity_ratio(pred_binary, target_binary, fov_mask=None):
    """
    Evaluates fragmentation by comparing the number of connected components in predictions vs ground truth.
    A high ratio (e.g. > 2.0) indicates severe fragmentation / broken vessel segments.
    """
    if fov_mask is not None:
        p = pred_binary & fov_mask
        t = target_binary & fov_mask
    else:
        p = pred_binary
        t = target_binary

    _, num_pred_cc = ndi.label(p)
    _, num_gt_cc = ndi.label(t)

    ratio = num_pred_cc / max(num_gt_cc, 1)
    return float(ratio), int(num_pred_cc), int(num_gt_cc)

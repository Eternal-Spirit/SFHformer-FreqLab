"""
Image quality metrics: PSNR and SSIM.

All functions operate on numpy arrays (HxWx3, float32, [0,1]) or
torch tensors (BxCxHxW, float32, [0,1]).
"""

import numpy as np
import torch


def psnr_np(img1, img2, max_val=1.0):
    """Compute PSNR between two numpy images.
    
    Args:
        img1, img2: HxWx3 float32 in [0, max_val].
        max_val: Maximum pixel value.
    
    Returns:
        PSNR in dB (float).
    """
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(max_val ** 2 / mse))


def ssim_np(img1, img2, max_val=1.0):
    """Compute SSIM between two numpy images (simplified, per-channel mean).
    
    Uses the standard SSIM formula with default constants.
    
    Args:
        img1, img2: HxWx3 float32 in [0, max_val].
        max_val: Maximum pixel value.
    
    Returns:
        Mean SSIM across channels (float).
    """
    from skimage.metrics import structural_similarity
    return float(structural_similarity(img1, img2, data_range=max_val,
                                        channel_axis=2))


def psnr_torch(pred, target, max_val=1.0):
    """Compute PSNR for a batch of torch tensors.
    
    Args:
        pred, target: (B, C, H, W) float tensors.
        max_val: Maximum pixel value.
    
    Returns:
        Mean PSNR across the batch (float).
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-10))
    return float(psnr.mean().item())

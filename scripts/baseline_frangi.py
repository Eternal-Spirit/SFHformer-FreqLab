"""
scripts/baseline_frangi.py - Classical Frangi Vessel Enhancement Baseline (Experiments C & D).
Implements:
1. Green-channel extraction & CLAHE preprocessing.
2. Multiscale Frangi Hessian filter (skimage.filters.frangi).
3. Clean validation threshold tuning (N=59 clean images) and metric evaluation.
"""

import os
import json
import glob
import numpy as np
from PIL import Image
import skimage.exposure as exposure
from skimage.filters import frangi
import scipy.ndimage as ndi
from utils.losses import compute_segmentation_metrics, compute_thin_vessel_sensitivity, compute_vascular_continuity_ratio

REPO_DIR = r"D:\SFHformer-FreqLab"
DATA_DIR = os.path.join(REPO_DIR, "data")
FIVES_DIR = os.path.join(DATA_DIR, "fives")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "benchmarks", "frangi_baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

def apply_frangi(rgb_image_arr, sigmas=(1.0, 2.0, 3.0, 4.0, 5.0), alpha=0.5, beta=0.5):
    """
    Applies Frangi vessel enhancement filter to RGB fundus image.
    Uses inverted green channel with CLAHE contrast enhancement.
    """
    # Retinal vessels appear darkest in the green channel
    green = rgb_image_arr[:, :, 1].astype(np.float32) / 255.0
    # Invert so vessels are bright ridges on dark background
    inverted = 1.0 - green
    # CLAHE contrast enhancement
    clahe_img = exposure.equalize_adapthist(inverted, clip_limit=0.01)
    # Multiscale Frangi filter
    vesselness = frangi(clahe_img, sigmas=sigmas, alpha=alpha, beta=beta, black_ridges=False)
    # Normalize to [0, 1]
    v_min, v_max = vesselness.min(), vesselness.max()
    if v_max > v_min:
        vesselness = (vesselness - v_min) / (v_max - v_min)
    return vesselness

def evaluate_frangi_on_image(img_path, mask_path, fov_path, threshold=0.15):
    """Evaluates Frangi filter on a single full fundus image."""
    im = np.array(Image.open(img_path).convert('RGB'))
    mk = np.array(Image.open(mask_path).convert('L')) > 0
    fov = np.array(Image.open(fov_path).convert('L')) > 0 if fov_path and os.path.exists(fov_path) else (im.mean(axis=2) > 12)

    vesselness = apply_frangi(im)
    pred_binary = (vesselness >= threshold) & fov

    metrics = compute_segmentation_metrics(pred_binary, mk, fov_mask=fov)
    thin_sens = compute_thin_vessel_sensitivity(pred_binary, mk, fov_mask=fov)
    cont_ratio, p_cc, gt_cc = compute_vascular_continuity_ratio(pred_binary, mk, fov_mask=fov)

    metrics["thin_vessel_sensitivity"] = thin_sens
    metrics["vascular_continuity_ratio"] = cont_ratio
    metrics["pred_connected_components"] = p_cc
    metrics["gt_connected_components"] = gt_cc
    return metrics, vesselness, pred_binary

if __name__ == '__main__':
    print("Frangi vessel enhancement baseline module loaded successfully.")

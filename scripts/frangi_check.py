import os
import sys
sys.path.insert(0, r"D:\SFHformer-FreqLab")
import numpy as np
from PIL import Image
from scripts.baseline_frangi import apply_frangi
from utils.losses import compute_segmentation_metrics

raw_p = r'data\fives\raw\val\images\train_100_A.png'
rest_p = r'data\fives\restored_val\images\train_100_A.png'
mk_p = r'data\fives\raw\val\masks\train_100_A.png'
fov_p = r'data\fives\raw\val\fov\train_100_A.png'

im_raw = np.array(Image.open(raw_p).convert('RGB'))
im_rest = np.array(Image.open(rest_p).convert('RGB'))
mk = np.array(Image.open(mk_p).convert('L')) > 0
fov = np.array(Image.open(fov_p).convert('L')) > 0

v_raw = apply_frangi(im_raw)
v_rest = apply_frangi(im_rest)

print("--- Threshold Sweep with Calibrated Frangi Percentiles ---")
for t in [0.001, 0.003, 0.005, 0.008, 0.012, 0.020]:
    mr = compute_segmentation_metrics((v_raw >= t) & fov, mk, fov_mask=fov)
    ms = compute_segmentation_metrics((v_rest >= t) & fov, mk, fov_mask=fov)
    print(f"Thresh {t:.4f} -> Raw Dice: {mr['dice']:.4f} (Sens: {mr['sensitivity']:.4f}, Prec: {mr['precision']:.4f}) | Restored Dice: {ms['dice']:.4f} (Sens: {ms['sensitivity']:.4f}, Prec: {ms['precision']:.4f})")

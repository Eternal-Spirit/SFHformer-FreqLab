"""
scripts/milestone2_5_verification.py - Milestone 2.5 Verification Suite:
1. Paired per-image thin-vessel sensitivity (all 59 clean validation images).
   - Evaluated on skeleton pixels (caliber <= 3.5 px).
   - Evaluated on all area pixels (caliber <= 3.5 px).
   - Paired Wilcoxon signed-rank test & Paired bootstrap 95% CI.
2. Vascular Continuity & Topological Analysis:
   - clDice (Centerline Dice).
   - Skeleton connected components & endpoints before and after morphological closing.
   - Ground-truth vessel break rate.
3. Frangi Response Map & Dynamic Range Investigation:
   - Max, mean, p90, p99 Hessian eigenvalue responses on raw vs restored images.
4. Optimization & Convergence Audit:
   - Loss trajectories, LR decay schedule, plateau verification.
"""

import os
import sys
import time
import json
import glob
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
from scipy import stats
from skimage import morphology
import torch

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.unet import get_unet
from scripts.baseline_frangi import apply_frangi

DATA_DIR = os.path.join(REPO_DIR, "data", "fives")
RAW_VAL_DIR = os.path.join(DATA_DIR, "raw", "val", "images")
REST_VAL_DIR = os.path.join(DATA_DIR, "restored_val", "images")
MASKS_VAL_DIR = os.path.join(DATA_DIR, "raw", "val", "masks")
FOV_VAL_DIR = os.path.join(DATA_DIR, "raw", "val", "fov")

VAL_MANIFEST = os.path.join(DATA_DIR, "patches_512", "val_full_image_tile_manifest.json")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "dataset_audit", "milestone2_5_verification")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    print("[1/5] Loading Best Checkpoints for Exp A (Raw) and Exp B (Restored)...")
    m_raw = get_unet(in_channels=3, out_channels=1, base_channels=32).to(DEVICE)
    ck_raw = torch.load("checkpoints/unet_raw/best.pth", map_location=DEVICE, weights_only=False)
    m_raw.load_state_dict(ck_raw["model"])
    m_raw.eval()
    
    m_rest = get_unet(in_channels=3, out_channels=1, base_channels=32).to(DEVICE)
    ck_rest = torch.load("checkpoints/unet_restored/best.pth", map_location=DEVICE, weights_only=False)
    m_rest.load_state_dict(ck_rest["model"])
    m_rest.eval()
    return m_raw, m_rest

def predict_full_image(model, img_arr, coords, tile_size=512):
    H, W = img_arr.shape[:2]
    accum_prob = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
    accum_weight = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
    
    wy = np.hamming(tile_size)
    wx = np.hamming(tile_size)
    window_2d = torch.from_numpy(np.outer(wy, wx).astype(np.float32)).to(DEVICE)
    
    tile_tensors = []
    for c in coords:
        t, l = c["top"], c["left"]
        tile = img_arr[t:t+tile_size, l:l+tile_size]
        tile_tensors.append(torch.from_numpy(tile).permute(2, 0, 1))
    x_tiles = torch.stack(tile_tensors, dim=0).to(DEVICE)
    
    chunk_sz = 5
    prob_chunks = []
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for ch in range(0, len(x_tiles), chunk_sz):
                logits = model(x_tiles[ch:ch+chunk_sz])
                probs = torch.sigmoid(logits).squeeze(1).float()
                prob_chunks.append(probs)
    pred_tiles = torch.cat(prob_chunks, dim=0)
    
    for k, c in enumerate(coords):
        t, l = c["top"], c["left"]
        accum_prob[t:t+tile_size, l:l+tile_size] += pred_tiles[k] * window_2d
        accum_weight[t:t+tile_size, l:l+tile_size] += window_2d
        
    return (accum_prob / accum_weight).cpu().numpy()

def compute_cldice(v_p, v_l):
    """
    Computes clDice (Centerline Dice) between predicted binary mask v_p and ground truth v_l.
    """
    tprec = 0.0
    tsens = 0.0
    skel_p = morphology.skeletonize(v_p > 0)
    skel_l = morphology.skeletonize(v_l > 0)
    
    skel_p_sum = np.sum(skel_p)
    skel_l_sum = np.sum(skel_l)
    
    if skel_p_sum > 0:
        tprec = np.sum(v_l[skel_p]) / skel_p_sum
    else:
        tprec = 1.0 if skel_l_sum == 0 else 0.0
        
    if skel_l_sum > 0:
        tsens = np.sum(v_p[skel_l]) / skel_l_sum
    else:
        tsens = 1.0 if skel_p_sum == 0 else 0.0
        
    if (tprec + tsens) > 0:
        return 2.0 * tprec * tsens / (tprec + tsens)
    return 0.0

def count_skeleton_endpoints(skel_binary):
    """Counts endpoints in a 1-pixel skeleton using 3x3 neighborhood convolution."""
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    conv = ndi.convolve(skel_binary.astype(np.uint8), kernel, mode='constant', cval=0)
    # An endpoint has central pixel active (value >= 10) and exactly 1 neighbor (total == 11)
    endpoints = (conv == 11)
    return int(np.sum(endpoints))

def analyze_vessel_breaks(pred_binary, gt_skel_continuous):
    """
    Vectorized vessel break analysis:
    Finds how many continuous ground-truth vessel segments are severed/broken
    into multiple disconnected fragments by the prediction.
    """
    labeled_gt, num_gt_comp = ndi.label(gt_skel_continuous)
    if num_gt_comp == 0:
        return 0, 0.0
        
    overlap = gt_skel_continuous & pred_binary
    if np.sum(overlap) == 0:
        return num_gt_comp, 1.0
        
    labeled_overlap, _ = ndi.label(overlap)
    coords = np.where(overlap)
    gt_labels = labeled_gt[coords]
    pred_labels = labeled_overlap[coords]
    
    pairs = np.unique(np.column_stack([gt_labels, pred_labels]), axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    broken_segments = int(np.sum(counts > 1))
    break_rate = float(broken_segments / num_gt_comp)
    return broken_segments, break_rate

def run_paired_verification():
    print("=" * 80)
    print(" MILESTONE 2.5: RIGOROUS PAIRED STATISTICAL & TOPOLOGICAL VERIFICATION ")
    print("=" * 80)
    
    m_raw, m_rest = load_models()
    
    with open(VAL_MANIFEST, 'r') as f:
        records = json.load(f)
        
    clean_records = [r for r in records if r.get("is_clean_val", True)]
    print(f" -> Evaluating strictly on {len(clean_records)} clean duplicate-free validation images.")
    
    tile_size = clean_records[0]["tile_size"]
    coords = clean_records[0]["tile_coordinates"]
    
    # Storage for per-image statistics
    per_image_results = []
    
    raw_thin_skel_sens = []
    rest_thin_skel_sens = []
    
    raw_thin_area_sens = []
    rest_thin_area_sens = []
    
    raw_dice_list = []
    rest_dice_list = []
    
    raw_cldice_raw = []
    rest_cldice_raw = []
    
    raw_cldice_closed = []
    rest_cldice_closed = []
    
    raw_cc_raw = []
    rest_cc_raw = []
    
    raw_cc_closed = []
    rest_cc_closed = []
    
    raw_endpoints = []
    rest_endpoints = []
    
    raw_breaks = []
    rest_breaks = []
    
    closing_elem = morphology.disk(2)
    
    t0 = time.time()
    for idx, rec in enumerate(clean_records):
        img_id = rec.get("image_id") or rec.get("source_image_id")
        
        raw_p = os.path.join(RAW_VAL_DIR, f"{img_id}.png")
        rest_p = os.path.join(REST_VAL_DIR, f"{img_id}.png")
        mk_p = os.path.join(MASKS_VAL_DIR, f"{img_id}.png")
        fov_p = os.path.join(FOV_VAL_DIR, f"{img_id}.png")
        
        raw_arr = np.array(Image.open(raw_p).convert('RGB'), dtype=np.float32) / 255.0
        rest_arr = np.array(Image.open(rest_p).convert('RGB'), dtype=np.float32) / 255.0
        mk = np.array(Image.open(mk_p).convert('L')) > 0
        fov = np.array(Image.open(fov_p).convert('L')) > 0 if os.path.exists(fov_p) else (mk.shape[0] > 0)
        
        # Ground truth skeleton and distance transform
        gt_skel = morphology.skeletonize(mk)
        dist = ndi.distance_transform_edt(mk)
        caliper_diam = dist * 2.0
        
        # Target 1: Skeleton pixels of thin vessels (<= 3.5 px)
        thin_skel_mask = gt_skel & (caliper_diam <= 3.5) & fov
        # Target 2: ALL area pixels of thin vessels (<= 3.5 px)
        thin_area_mask = mk & (caliper_diam <= 3.5) & fov
        
        tot_thin_skel = int(np.sum(thin_skel_mask))
        tot_thin_area = int(np.sum(thin_area_mask))
        
        # Predictions
        prob_raw = predict_full_image(m_raw, raw_arr, coords, tile_size)
        pred_raw_bin = (prob_raw >= 0.5) & fov
        pred_raw_closed = morphology.closing(pred_raw_bin, closing_elem) & fov
        
        prob_rest = predict_full_image(m_rest, rest_arr, coords, tile_size)
        pred_rest_bin = (prob_rest >= 0.5) & fov
        pred_rest_closed = morphology.closing(pred_rest_bin, closing_elem) & fov
        
        # 1. Thin vessel sensitivity
        raw_ts = float(np.sum(pred_raw_bin & thin_skel_mask) / max(tot_thin_skel, 1))
        rest_ts = float(np.sum(pred_rest_bin & thin_skel_mask) / max(tot_thin_skel, 1))
        raw_thin_skel_sens.append(raw_ts)
        rest_thin_skel_sens.append(rest_ts)
        
        raw_ta = float(np.sum(pred_raw_bin & thin_area_mask) / max(tot_thin_area, 1))
        rest_ta = float(np.sum(pred_rest_bin & thin_area_mask) / max(tot_thin_area, 1))
        raw_thin_area_sens.append(raw_ta)
        rest_thin_area_sens.append(rest_ta)
        
        # 2. Overall Dice
        dice_raw = float(2 * np.sum(pred_raw_bin[fov] & mk[fov]) / (np.sum(pred_raw_bin[fov]) + np.sum(mk[fov]) + 1e-7))
        dice_rest = float(2 * np.sum(pred_rest_bin[fov] & mk[fov]) / (np.sum(pred_rest_bin[fov]) + np.sum(mk[fov]) + 1e-7))
        raw_dice_list.append(dice_raw)
        rest_dice_list.append(dice_rest)
        
        # 3. clDice (Before and After closing)
        cld_raw_b = float(compute_cldice(pred_raw_bin, mk))
        cld_rest_b = float(compute_cldice(pred_rest_bin, mk))
        raw_cldice_raw.append(cld_raw_b)
        rest_cldice_raw.append(cld_rest_b)
        
        cld_raw_a = float(compute_cldice(pred_raw_closed, mk))
        cld_rest_a = float(compute_cldice(pred_rest_closed, mk))
        raw_cldice_closed.append(cld_raw_a)
        rest_cldice_closed.append(cld_rest_a)
        
        # 4. Connected components
        _, cc_raw_b = ndi.label(pred_raw_bin & fov)
        _, cc_rest_b = ndi.label(pred_rest_bin & fov)
        raw_cc_raw.append(cc_raw_b)
        rest_cc_raw.append(cc_rest_b)
        
        _, cc_raw_a = ndi.label(pred_raw_closed & fov)
        _, cc_rest_a = ndi.label(pred_rest_closed & fov)
        raw_cc_closed.append(cc_raw_a)
        rest_cc_closed.append(cc_rest_a)
        
        # 5. Endpoints & Breaks
        skel_raw = morphology.skeletonize(pred_raw_bin & fov)
        skel_rest = morphology.skeletonize(pred_rest_bin & fov)
        ep_raw = count_skeleton_endpoints(skel_raw)
        ep_rest = count_skeleton_endpoints(skel_rest)
        raw_endpoints.append(ep_raw)
        rest_endpoints.append(ep_rest)
        
        brk_raw, brate_raw = analyze_vessel_breaks(pred_raw_bin, gt_skel & fov)
        brk_rest, brate_rest = analyze_vessel_breaks(pred_rest_bin, gt_skel & fov)
        raw_breaks.append(brate_raw)
        rest_breaks.append(brate_rest)
        
        per_image_results.append({
            "image_id": img_id,
            "thin_skel_sens_raw": round(raw_ts, 4),
            "thin_skel_sens_restored": round(rest_ts, 4),
            "delta_thin_skel_sens": round(rest_ts - raw_ts, 4),
            "thin_area_sens_raw": round(raw_ta, 4),
            "thin_area_sens_restored": round(rest_ta, 4),
            "delta_thin_area_sens": round(rest_ta - raw_ta, 4),
            "dice_raw": round(dice_raw, 4),
            "dice_restored": round(dice_rest, 4),
            "delta_dice": round(dice_rest - dice_raw, 4),
            "cldice_raw": round(cld_raw_b, 4),
            "cldice_restored": round(cld_rest_b, 4),
            "cldice_closed_raw": round(cld_raw_a, 4),
            "cldice_closed_restored": round(cld_rest_a, 4),
            "cc_raw": cc_raw_b,
            "cc_restored": cc_rest_b,
            "cc_closed_raw": cc_raw_a,
            "cc_closed_restored": cc_rest_a,
            "endpoints_raw": ep_raw,
            "endpoints_restored": ep_rest,
            "break_rate_raw": round(brate_raw, 4),
            "break_rate_restored": round(brate_rest, 4)
        })
        
        if (idx + 1) % 15 == 0 or (idx + 1) == len(clean_records):
            print(f"  Processed {idx + 1} / {len(clean_records)} images ({time.time() - t0:.1f}s)...")
            
    # Paired Statistics Computation
    delta_skel = np.array(rest_thin_skel_sens) - np.array(raw_thin_skel_sens)
    delta_area = np.array(rest_thin_area_sens) - np.array(raw_thin_area_sens)
    delta_dice = np.array(rest_dice_list) - np.array(raw_dice_list)
    delta_cldice = np.array(rest_cldice_raw) - np.array(raw_cldice_raw)
    
    # Wilcoxon signed-rank test
    w_skel_stat, w_skel_pval = stats.wilcoxon(rest_thin_skel_sens, raw_thin_skel_sens)
    w_area_stat, w_area_pval = stats.wilcoxon(rest_thin_area_sens, raw_thin_area_sens)
    w_dice_stat, w_dice_pval = stats.wilcoxon(rest_dice_list, raw_dice_list)
    
    # Paired t-test
    t_skel_stat, t_skel_pval = stats.ttest_rel(rest_thin_skel_sens, raw_thin_skel_sens)
    
    # 95% Bootstrap Confidence Interval for Delta
    rng = np.random.default_rng(2026)
    B = 10000
    boot_skel_means = [np.mean(rng.choice(delta_skel, size=len(delta_skel), replace=True)) for _ in range(B)]
    ci_skel_lower = float(np.percentile(boot_skel_means, 2.5))
    ci_skel_upper = float(np.percentile(boot_skel_means, 97.5))
    
    boot_area_means = [np.mean(rng.choice(delta_area, size=len(delta_area), replace=True)) for _ in range(B)]
    ci_area_lower = float(np.percentile(boot_area_means, 2.5))
    ci_area_upper = float(np.percentile(boot_area_means, 97.5))
    
    stats_summary = {
        "n_images": len(clean_records),
        "thin_vessel_skeleton_sensitivity": {
            "raw_mean": round(float(np.mean(raw_thin_skel_sens)), 4),
            "raw_std": round(float(np.std(raw_thin_skel_sens)), 4),
            "restored_mean": round(float(np.mean(rest_thin_skel_sens)), 4),
            "restored_std": round(float(np.std(rest_thin_skel_sens)), 4),
            "mean_delta": round(float(np.mean(delta_skel)), 4),
            "bootstrap_95_ci": [round(ci_skel_lower, 4), round(ci_skel_upper, 4)],
            "wilcoxon_stat": float(w_skel_stat),
            "wilcoxon_pval": float(w_skel_pval),
            "ttest_stat": round(float(t_skel_stat), 3),
            "ttest_pval": float(t_skel_pval),
            "is_statistically_significant": bool(w_skel_pval < 0.05)
        },
        "thin_vessel_area_sensitivity": {
            "raw_mean": round(float(np.mean(raw_thin_area_sens)), 4),
            "raw_std": round(float(np.std(raw_thin_area_sens)), 4),
            "restored_mean": round(float(np.mean(rest_thin_area_sens)), 4),
            "restored_std": round(float(np.std(rest_thin_area_sens)), 4),
            "mean_delta": round(float(np.mean(delta_area)), 4),
            "bootstrap_95_ci": [round(ci_area_lower, 4), round(ci_area_upper, 4)],
            "wilcoxon_pval": float(w_area_pval)
        },
        "overall_dice": {
            "raw_mean": round(float(np.mean(raw_dice_list)), 4),
            "raw_std": round(float(np.std(raw_dice_list)), 4),
            "restored_mean": round(float(np.mean(rest_dice_list)), 4),
            "restored_std": round(float(np.std(rest_dice_list)), 4),
            "mean_delta": round(float(np.mean(delta_dice)), 4),
            "wilcoxon_pval": float(w_dice_pval)
        },
        "cldice_and_topology": {
            "raw_cldice_before_closing": round(float(np.mean(raw_cldice_raw)), 4),
            "restored_cldice_before_closing": round(float(np.mean(rest_cldice_raw)), 4),
            "raw_cldice_after_closing": round(float(np.mean(raw_cldice_closed)), 4),
            "restored_cldice_after_closing": round(float(np.mean(rest_cldice_closed)), 4),
            "raw_mean_components_before_closing": round(float(np.mean(raw_cc_raw)), 1),
            "restored_mean_components_before_closing": round(float(np.mean(rest_cc_raw)), 1),
            "raw_mean_components_after_closing": round(float(np.mean(raw_cc_closed)), 1),
            "restored_mean_components_after_closing": round(float(np.mean(rest_cc_closed)), 1),
            "raw_mean_endpoints": round(float(np.mean(raw_endpoints)), 1),
            "restored_mean_endpoints": round(float(np.mean(rest_endpoints)), 1),
            "raw_mean_break_rate": round(float(np.mean(raw_breaks)), 4),
            "restored_mean_break_rate": round(float(np.mean(rest_breaks)), 4)
        }
    }
    
    print("\n" + "=" * 80)
    print(" PAIRED STATISTICAL VERIFICATION SUMMARY (N=59 Clean Images) ")
    print("=" * 80)
    print(f" Thin-Vessel SKELETON Sensitivity (<= 3.5 px):")
    print(f"   Raw U-Net:      {stats_summary['thin_vessel_skeleton_sensitivity']['raw_mean']:.4f} +/- {stats_summary['thin_vessel_skeleton_sensitivity']['raw_std']:.4f}")
    print(f"   Restored U-Net: {stats_summary['thin_vessel_skeleton_sensitivity']['restored_mean']:.4f} +/- {stats_summary['thin_vessel_skeleton_sensitivity']['restored_std']:.4f}")
    print(f"   Mean Paired Delta: {stats_summary['thin_vessel_skeleton_sensitivity']['mean_delta']:+.4f}")
    print(f"   95% Bootstrap CI: [{ci_skel_lower:+.4f}, {ci_skel_upper:+.4f}]")
    print(f"   Wilcoxon p-value: {w_skel_pval:.4e} (Statistically Significant: {w_skel_pval < 0.05})")
    print(f"   Paired t-test:    t = {t_skel_stat:.3f}, p = {t_skel_pval:.4e}")
    print("-" * 80)
    print(f" clDice (Centerline Dice Overlap):")
    print(f"   Raw U-Net:      {stats_summary['cldice_and_topology']['raw_cldice_before_closing']:.4f} (After Closing: {stats_summary['cldice_and_topology']['raw_cldice_after_closing']:.4f})")
    print(f"   Restored U-Net: {stats_summary['cldice_and_topology']['restored_cldice_before_closing']:.4f} (After Closing: {stats_summary['cldice_and_topology']['restored_cldice_after_closing']:.4f})")
    print("-" * 80)
    print(f" Topological Components & Endpoints:")
    print(f"   Components Before Closing: Raw = {stats_summary['cldice_and_topology']['raw_mean_components_before_closing']} vs Restored = {stats_summary['cldice_and_topology']['restored_mean_components_before_closing']}")
    print(f"   Components After Closing:  Raw = {stats_summary['cldice_and_topology']['raw_mean_components_after_closing']} vs Restored = {stats_summary['cldice_and_topology']['restored_mean_components_after_closing']}")
    print(f"   Skeleton Endpoints:        Raw = {stats_summary['cldice_and_topology']['raw_mean_endpoints']} vs Restored = {stats_summary['cldice_and_topology']['restored_mean_endpoints']}")
    print(f"   Vessel Break Rate:         Raw = {stats_summary['cldice_and_topology']['raw_mean_break_rate']*100:.1f}% vs Restored = {stats_summary['cldice_and_topology']['restored_mean_break_rate']*100:.1f}%")
    print("=" * 80)
    
    # Save full reports
    out_master = os.path.join(RESULTS_DIR, "paired_verification_summary.json")
    with open(out_master, 'w') as f:
        json.dump(stats_summary, f, indent=2)
        
    out_per_image = os.path.join(RESULTS_DIR, "per_image_paired_metrics.json")
    with open(out_per_image, 'w') as f:
        json.dump(per_image_results, f, indent=2)
        
    print(f"Saved master summary to: {out_master}")
    print(f"Saved per-image table to: {out_per_image}")
    return stats_summary

if __name__ == '__main__':
    run_paired_verification()

"""
scripts/run_frangi_experiments_cd.py - Evaluates Experiments C & D:
Experiment C: Raw fundus -> Frangi
Experiment D: SFHFormer-restored fundus -> Frangi
Evaluated strictly using Duplicate-Free Image-Level Validation (N=59 clean, N=60 sensitivity).
"""

import os
import sys
import json
import glob
import numpy as np
from PIL import Image

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from scripts.baseline_frangi import apply_frangi
from utils.losses import compute_segmentation_metrics, compute_thin_vessel_sensitivity, compute_vascular_continuity_ratio

DATA_DIR = os.path.join(REPO_DIR, "data", "fives")
RAW_VAL_DIR = os.path.join(DATA_DIR, "raw", "val", "images")
REST_VAL_DIR = os.path.join(DATA_DIR, "restored_val", "images")
MASKS_VAL_DIR = os.path.join(DATA_DIR, "raw", "val", "masks")
FOV_VAL_DIR = os.path.join(DATA_DIR, "raw", "val", "fov")

MANIFEST_PATH = os.path.join(DATA_DIR, "patches_512", "val_full_image_tile_manifest.json")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "benchmarks", "frangi_baseline")
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_subset(img_dir, mode_name, records, thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
    print(f"\n[Evaluating {mode_name}] Running Frangi across {len(records)} images...")
    
    # Precompute vesselness maps
    vesselness_cache = {}
    for r in records:
        img_id = r.get("image_id") or r.get("source_image_id")
        img_p = os.path.join(img_dir, f"{img_id}.png")
        im_arr = np.array(Image.open(img_p).convert('RGB'))
        vesselness_cache[img_id] = apply_frangi(im_arr)
        
    best_t = None
    best_clean_dice = -1.0
    sweep_results = {}
    
    for t in thresholds:
        clean_dices, full_dices = [], []
        clean_ious, full_ious = [], []
        clean_sens, full_sens = [], []
        clean_spec, full_spec = [], []
        clean_prec, full_prec = [], []
        clean_thin_sens, full_thin_sens = [], []
        clean_cont, full_cont = [], []
        
        for r in records:
            img_id = r.get("image_id") or r.get("source_image_id")
            is_clean = r.get("is_clean_val", True)
            
            mk_p = os.path.join(MASKS_VAL_DIR, f"{img_id}.png")
            fov_p = os.path.join(FOV_VAL_DIR, f"{img_id}.png")
            
            mk = np.array(Image.open(mk_p).convert('L')) > 0
            fov = np.array(Image.open(fov_p).convert('L')) > 0 if os.path.exists(fov_p) else (mk.shape[0] > 0)
            
            v = vesselness_cache[img_id]
            pred = (v >= t) & fov
            
            m = compute_segmentation_metrics(pred, mk, fov_mask=fov)
            ts = compute_thin_vessel_sensitivity(pred, mk, fov_mask=fov)
            cont, _, _ = compute_vascular_continuity_ratio(pred, mk, fov_mask=fov)
            
            full_dices.append(m["dice"])
            full_ious.append(m["iou"])
            full_sens.append(m["sensitivity"])
            full_spec.append(m["specificity"])
            full_prec.append(m["precision"])
            full_thin_sens.append(ts)
            full_cont.append(cont)
            
            if is_clean:
                clean_dices.append(m["dice"])
                clean_ious.append(m["iou"])
                clean_sens.append(m["sensitivity"])
                clean_spec.append(m["specificity"])
                clean_prec.append(m["precision"])
                clean_thin_sens.append(ts)
                clean_cont.append(cont)
                
        mean_clean_dice = float(np.mean(clean_dices))
        sweep_results[f"thresh_{t:.2f}"] = {
            "threshold": t,
            "clean_n59": {
                "dice": round(float(np.mean(clean_dices)), 4),
                "iou": round(float(np.mean(clean_ious)), 4),
                "sensitivity": round(float(np.mean(clean_sens)), 4),
                "specificity": round(float(np.mean(clean_spec)), 4),
                "precision": round(float(np.mean(clean_prec)), 4),
                "thin_vessel_sensitivity": round(float(np.mean(clean_thin_sens)), 4),
                "vascular_continuity_ratio": round(float(np.mean(clean_cont)), 2)
            },
            "full_n60": {
                "dice": round(float(np.mean(full_dices)), 4),
                "iou": round(float(np.mean(full_ious)), 4),
                "sensitivity": round(float(np.mean(full_sens)), 4),
                "specificity": round(float(np.mean(full_spec)), 4),
                "precision": round(float(np.mean(full_prec)), 4),
                "thin_vessel_sensitivity": round(float(np.mean(full_thin_sens)), 4),
                "vascular_continuity_ratio": round(float(np.mean(full_cont)), 2)
            }
        }
        print(f"  Threshold {t:.2f} -> Clean Dice: {mean_clean_dice:.4f} | Thin Sens: {np.mean(clean_thin_sens):.4f} | Cont Ratio: {np.mean(clean_cont):.1f}")
        
        if mean_clean_dice > best_clean_dice:
            best_clean_dice = mean_clean_dice
            best_t = t
            
    print(f" -> Best Threshold for {mode_name}: {best_t:.2f} (Clean Dice: {best_clean_dice:.4f})")
    return {
        "best_threshold": best_t,
        "best_metrics": sweep_results[f"thresh_{best_t:.2f}"],
        "all_thresholds": sweep_results
    }

def main():
    print("=" * 80)
    print(" EXPERIMENTS C & D: CLASSICAL FRANGI VESSEL ENHANCEMENT BENCHMARK ")
    print("=" * 80)
    
    with open(MANIFEST_PATH, 'r') as f:
        records = json.load(f)
        
    clean_n = sum(1 for r in records if r.get("is_clean_val", True))
    print(f"Validation Manifest Loaded: {len(records)} images ({clean_n} clean, {len(records) - clean_n} contaminated duplicate)")
    
    # Experiment C: Raw -> Frangi
    exp_c = evaluate_subset(RAW_VAL_DIR, "Experiment C (Raw -> Frangi)", records)
    
    # Experiment D: Restored -> Frangi
    exp_d = evaluate_subset(REST_VAL_DIR, "Experiment D (SFHFormer-Restored -> Frangi)", records)
    
    comparison = {
        "experiment_c_raw": exp_c["best_metrics"]["clean_n59"],
        "experiment_d_restored": exp_d["best_metrics"]["clean_n59"],
        "delta_restored_minus_raw": {
            "dice_delta": round(exp_d["best_metrics"]["clean_n59"]["dice"] - exp_c["best_metrics"]["clean_n59"]["dice"], 4),
            "iou_delta": round(exp_d["best_metrics"]["clean_n59"]["iou"] - exp_c["best_metrics"]["clean_n59"]["iou"], 4),
            "thin_vessel_sensitivity_delta": round(exp_d["best_metrics"]["clean_n59"]["thin_vessel_sensitivity"] - exp_c["best_metrics"]["clean_n59"]["thin_vessel_sensitivity"], 4),
            "continuity_ratio_delta": round(exp_d["best_metrics"]["clean_n59"]["vascular_continuity_ratio"] - exp_c["best_metrics"]["clean_n59"]["vascular_continuity_ratio"], 2)
        }
    }
    
    print("\n" + "=" * 80)
    print(" COMPARISON: RAW VS SFHFORMER-RESTORED FRANGI ENHANCEMENT (N=59 Clean) ")
    print("=" * 80)
    print(f" Metric                     Raw (Exp C)    Restored (Exp D)    Delta")
    print("-" * 70)
    print(f" Dice (F1 Score):           {comparison['experiment_c_raw']['dice']:.4f}         {comparison['experiment_d_restored']['dice']:.4f}              {comparison['delta_restored_minus_raw']['dice_delta']:+.4f}")
    print(f" IoU (Jaccard):             {comparison['experiment_c_raw']['iou']:.4f}         {comparison['experiment_d_restored']['iou']:.4f}              {comparison['delta_restored_minus_raw']['iou_delta']:+.4f}")
    print(f" Thin-Vessel Sensitivity:   {comparison['experiment_c_raw']['thin_vessel_sensitivity']:.4f}         {comparison['experiment_d_restored']['thin_vessel_sensitivity']:.4f}              {comparison['delta_restored_minus_raw']['thin_vessel_sensitivity_delta']:+.4f}")
    print(f" Continuity Ratio (CC/GT):  {comparison['experiment_c_raw']['vascular_continuity_ratio']:.2f}          {comparison['experiment_d_restored']['vascular_continuity_ratio']:.2f}               {comparison['delta_restored_minus_raw']['continuity_ratio_delta']:+.2f}")
    print("=" * 80)
    
    out_p = os.path.join(RESULTS_DIR, "frangi_raw_vs_restored_validation.json")
    with open(out_p, 'w') as f:
        json.dump({
            "experiment_c": exp_c,
            "experiment_d": exp_d,
            "clean_benchmark_comparison": comparison
        }, f, indent=2)
    print(f"Results saved to: {out_p}")

if __name__ == '__main__':
    main()

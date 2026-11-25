"""
scripts/evaluate_milestone3.py
Milestone 3: Final Generalization Evaluation
Evaluates:
- FIVES 200 untouched test images (Raw U-Net, Restored U-Net, Raw Frangi, Restored Frangi)
- HRF 45 external images (Separated into 40 exposed vs 5 unexposed)

Strict safeguards:
- Frozen thresholds: U-Net = 0.50, Frangi = 0.0010
- Checkpoint SHA-256 verification before and after
- No test-driven tuning or retraining
- Duplicate pair tracking (test_32_A vs test_48_A)
- Caliber-stratified skeleton sensitivity, area sensitivity, and branch-level recall
- Topological continuity (clDice, CC count, endpoints, break rate)
"""

import os
import sys
import time
import json
import glob
import hashlib
import cv2
import numpy as np
from PIL import Image
import torch
from scipy import stats
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.unet import UNet
from scripts.baseline_frangi import apply_frangi

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.70)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPTS = {
    "raw": {
        "path": os.path.join(REPO_DIR, "checkpoints", "unet_raw", "best.pth"),
        "sha": "97639a65b6cc3d09d0638036e04ea018c236f3e8df0bf556282de6beba6f9d50"
    },
    "restored": {
        "path": os.path.join(REPO_DIR, "checkpoints", "unet_restored", "best.pth"),
        "sha": "2cfb37371cc5c0038bba6f7c3dcde808a4571c246a3c5df7bddba460ff9011cb"
    },
    "sfhformer": {
        "path": os.path.join(REPO_DIR, "checkpoints", "fives_retina_finetune", "best.pth"),
        "sha": "d74e267c6ac664686528d6285564fcefcfddc97db1c1f0f676cc00ab5a49563c"
    }
}

UNET_THRESHOLD = 0.50
FRANGI_THRESHOLD = 0.0010

HRF_UNEXPOSED = {"14_g.png", "14_h.png", "15_dr.png", "15_g.png", "15_h.png"}

def sha256_file(fname):
    h = hashlib.sha256()
    with open(fname, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def verify_checkpoints():
    print("[Audit] Verifying model checkpoint integrity...")
    for k, v in CKPTS.items():
        curr = sha256_file(v["path"])
        assert curr == v["sha"], f"Integrity failure on {k}: expected {v['sha']}, got {curr}"
        print(f" -> {k.upper()} checkpoint SHA-256 confirmed: {curr[:16]}...")
    print(" -> All checkpoints intact.")

def get_hamming_window(tile_size=512):
    w1d = np.hamming(tile_size).astype(np.float32)
    w2d = np.outer(w1d, w1d)
    w2d = np.maximum(w2d, 1e-4)
    return torch.from_numpy(w2d).to(DEVICE)

def load_unet(mode="raw"):
    model = UNet(in_channels=3, out_channels=1, features=(32, 64, 128, 256)).to(DEVICE)
    ck = torch.load(CKPTS[mode]["path"], map_location=DEVICE, weights_only=False)
    st = ck['model'] if 'model' in ck else ck
    model.load_state_dict(st)
    model.eval()
    return model

def predict_sliding_window(model, img_arr, coords, tile_size=512, chunk_sz=5, window_2d=None):
    H, W = img_arr.shape[:2]
    accum_prob = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
    accum_weight = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
    
    tile_tensors = []
    for c in coords:
        t, l = c["top"], c["left"]
        tile = img_arr[t:t+tile_size, l:l+tile_size]
        tile_tensors.append(torch.from_numpy(tile).permute(2, 0, 1))
        
    x_tiles = torch.stack(tile_tensors, dim=0).to(DEVICE)
    
    out_probs = []
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for ch in range(0, len(x_tiles), chunk_sz):
                logits = model(x_tiles[ch:ch+chunk_sz])
                probs = torch.sigmoid(logits).squeeze(1).float()
                out_probs.append(probs)
                
    full_probs = torch.cat(out_probs, dim=0)
    
    for k, c in enumerate(coords):
        t, l = c["top"], c["left"]
        accum_prob[t:t+tile_size, l:l+tile_size] += full_probs[k] * window_2d
        accum_weight[t:t+tile_size, l:l+tile_size] += window_2d
        
    prob_map = (accum_prob / accum_weight).cpu().numpy()
    return prob_map

def compute_clDice(v_p, v_l):
    s_p = skeletonize(v_p).astype(bool)
    s_l = skeletonize(v_l).astype(bool)
    tprec = (s_p & v_l).sum() / max(s_p.sum(), 1)
    tsens = (s_l & v_p).sum() / max(s_l.sum(), 1)
    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))

def analyze_topology_and_branches(pred_binary, gt_mask, fov):
    pred = (pred_binary & fov).astype(bool)
    gt = (gt_mask & fov).astype(bool)
    
    # 1. Connected components
    num_pred_cc, _ = cv2.connectedComponents(pred.astype(np.uint8))
    num_gt_cc, gt_labels = cv2.connectedComponents(gt.astype(np.uint8))
    
    # 2. Skeleton endpoints
    skel_pred = skeletonize(pred)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    n_pred = convolve(skel_pred.astype(np.uint8), kernel, mode='constant') * skel_pred
    endpoints_pred = int(((n_pred == 1) & skel_pred).sum())
    
    # 3. Ground truth vessel break rate (vectorized)
    gt_fg = gt_labels > 0
    gt_v = gt_labels[gt_fg]
    pr_v = pred[gt_fg].astype(np.int32)
    pairs = np.column_stack((gt_v, pr_v))
    unique_pairs = np.unique(pairs, axis=0)
    
    gt_recovered = unique_pairs[unique_pairs[:, 1] == 1, 0]
    gt_unrecovered = unique_pairs[unique_pairs[:, 1] == 0, 0]
    severed_gt = np.intersect1d(gt_recovered, gt_unrecovered)
    total_gt_vessels = max(num_gt_cc - 1, 1)
    break_rate = float(len(severed_gt) / total_gt_vessels * 100.0)
    
    # 4. Caliber separation
    dist = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 3)
    skel_gt = skeletonize(gt)
    
    skel_thin = skel_gt & (dist <= 3.5)
    skel_med = skel_gt & (dist > 3.5) & (dist <= 7.0)
    skel_thick = skel_gt & (dist > 7.0)
    area_thin = gt & (dist <= 3.5)
    
    thin_skel_sens = float((pred & skel_thin).sum() / max(skel_thin.sum(), 1))
    med_skel_sens = float((pred & skel_med).sum() / max(skel_med.sum(), 1))
    thick_skel_sens = float((pred & skel_thick).sum() / max(skel_thick.sum(), 1))
    thin_area_sens = float((pred & area_thin).sum() / max(area_thin.sum(), 1))
    
    # 5. Branch-level recall (split at junctions)
    n_gt = convolve(skel_gt.astype(np.uint8), kernel, mode='constant') * skel_gt
    branch_points = (n_gt > 2) & skel_gt
    branches_mask = skel_gt & (~branch_points)
    
    num_branches, branch_labels = cv2.connectedComponents(branches_mask.astype(np.uint8))
    hit_pixels = pred & branches_mask
    
    branch_lengths = np.bincount(branch_labels.ravel())
    branch_hits = np.bincount(branch_labels[hit_pixels].ravel(), minlength=len(branch_lengths))
    
    valid_branch_idx = np.where(branch_lengths >= 5)[0]
    valid_branch_idx = valid_branch_idx[valid_branch_idx > 0]
    
    if len(valid_branch_idx) > 0:
        cov = branch_hits[valid_branch_idx] / branch_lengths[valid_branch_idx]
        recalled = cov >= 0.50
        branch_recall = float(np.mean(recalled))
        
        branch_calibers = np.bincount(branch_labels.ravel(), weights=dist.ravel()) / np.maximum(branch_lengths, 1)
        thin_idx = valid_branch_idx[branch_calibers[valid_branch_idx] <= 3.5]
        if len(thin_idx) > 0:
            thin_branch_recall = float(np.mean(np.isin(thin_idx, valid_branch_idx[recalled])))
        else:
            thin_branch_recall = 0.0
    else:
        branch_recall = 0.0
        thin_branch_recall = 0.0
        
    cldice = compute_clDice(pred, gt)
    
    return {
        "num_pred_cc": int(num_pred_cc),
        "num_gt_cc": int(num_gt_cc),
        "endpoints_pred": endpoints_pred,
        "break_rate": break_rate,
        "thin_skel_sens": thin_skel_sens,
        "med_skel_sens": med_skel_sens,
        "thick_skel_sens": thick_skel_sens,
        "thin_area_sens": thin_area_sens,
        "branch_recall": branch_recall,
        "thin_branch_recall": thin_branch_recall,
        "cldice": cldice
    }

def compute_pixel_metrics(pred_binary, gt_mask, fov):
    p = pred_binary[fov]
    g = gt_mask[fov]
    
    tp = int((p & g).sum())
    fp = int((p & (~g)).sum())
    fn = int(((~p) & g).sum())
    tn = int(((~p) & (~g)).sum())
    
    dice = float(2 * tp / max(2 * tp + fp + fn, 1))
    iou = float(tp / max(tp + fp + fn, 1))
    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    prec = float(tp / max(tp + fp, 1))
    
    return {"dice": dice, "iou": iou, "sens": sens, "spec": spec, "prec": prec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

def bootstrap_ci(deltas, n_boot=10000, ci=95):
    if len(deltas) == 0:
        return (0.0, 0.0)
    arr = np.array(deltas)
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(sample))
    lower = float(np.percentile(boot_means, (100 - ci) / 2))
    upper = float(np.percentile(boot_means, 100 - (100 - ci) / 2))
    return lower, upper

def run_fives_test_evaluation(raw_model, restored_model):
    print("\n" + "=" * 80)
    print(" [STAGE 1/2] FIVES Test Set Evaluation (200 Images, Untouched) ")
    print("=" * 80)
    
    split_p = os.path.join(REPO_DIR, "data", "fives", "dataset_split.json")
    with open(split_p, 'r') as f:
        split = json.load(f)
    test_meta = {x['filename']: x['disease'] for x in split['test_images']}
    
    raw_img_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "images")
    rest_img_dir = os.path.join(REPO_DIR, "data", "fives", "restored_test", "images")
    mask_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "masks")
    fov_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "fov")
    
    files = sorted(glob.glob(os.path.join(raw_img_dir, "*.png")))
    assert len(files) == 200, f"Expected 200 test images, got {len(files)}"
    
    # 25-tile coordinates
    coords = []
    offsets = [0, 384, 768, 1152, 1536]
    for top in offsets:
        for left in offsets:
            coords.append({"top": top, "left": left})
    window_2d = get_hamming_window(512)
    
    results = []
    t0 = time.time()
    
    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        base = os.path.splitext(fname)[0]
        disease = test_meta.get(fname, "Unknown")
        
        raw_np = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
        rest_p = os.path.join(rest_img_dir, fname)
        rest_np = np.array(Image.open(rest_p).convert('RGB'), dtype=np.float32) / 255.0
        
        gt_p = os.path.join(mask_dir, fname)
        gt_mask = np.array(Image.open(gt_p).convert('L')) > 127
        
        fov_p = os.path.join(fov_dir, fname)
        if os.path.exists(fov_p):
            fov = np.array(Image.open(fov_p).convert('L')) > 127
        else:
            fov = raw_np.mean(axis=2) > (10.0 / 255.0)
            
        # 1. Raw U-Net
        prob_raw = predict_sliding_window(raw_model, raw_np, coords, 512, 5, window_2d)
        pred_raw = (prob_raw >= UNET_THRESHOLD) & fov
        pix_raw = compute_pixel_metrics(pred_raw, gt_mask, fov)
        topo_raw = analyze_topology_and_branches(pred_raw, gt_mask, fov)
        
        # 2. Restored U-Net
        prob_rest = predict_sliding_window(restored_model, rest_np, coords, 512, 5, window_2d)
        pred_rest = (prob_rest >= UNET_THRESHOLD) & fov
        pix_rest = compute_pixel_metrics(pred_rest, gt_mask, fov)
        topo_rest = analyze_topology_and_branches(pred_rest, gt_mask, fov)
        
        # 3. Raw Frangi
        frangi_raw_resp = apply_frangi(raw_np)
        pred_fr_raw = (frangi_raw_resp >= FRANGI_THRESHOLD) & fov
        pix_fr_raw = compute_pixel_metrics(pred_fr_raw, gt_mask, fov)
        topo_fr_raw = analyze_topology_and_branches(pred_fr_raw, gt_mask, fov)
        
        # 4. Restored Frangi
        frangi_rest_resp = apply_frangi(rest_np)
        pred_fr_rest = (frangi_rest_resp >= FRANGI_THRESHOLD) & fov
        pix_fr_rest = compute_pixel_metrics(pred_fr_rest, gt_mask, fov)
        topo_fr_rest = analyze_topology_and_branches(pred_fr_rest, gt_mask, fov)
        
        rec = {
            "image_id": base,
            "filename": fname,
            "disease": disease,
            "raw_unet": {**pix_raw, **topo_raw},
            "restored_unet": {**pix_rest, **topo_rest},
            "raw_frangi": {**pix_fr_raw, **topo_fr_raw},
            "restored_frangi": {**pix_fr_rest, **topo_fr_rest}
        }
        results.append(rec)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == 200:
            print(f" -> Evaluated [{idx+1}/200] FIVES test images ({(time.time()-t0)/(idx+1):.2f}s/img)")
            
    print(f" -> FIVES Test evaluation completed in {time.time()-t0:.1f}s.")
    return results

def run_hrf_evaluation(raw_model, restored_model):
    print("\n" + "=" * 80)
    print(" [STAGE 2/2] HRF External Evaluation (45 Images, 3504x2336) ")
    print("=" * 80)
    
    manifest_p = os.path.join(REPO_DIR, "data", "hrf", "hrf_tile_manifest_512.json")
    with open(manifest_p, 'r') as f:
        records = json.load(f)
    coords_by_id = {r['image_id']: r['tile_coordinates'] for r in records}
    
    raw_img_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "images")
    rest_img_dir = os.path.join(REPO_DIR, "data", "hrf", "restored", "images")
    mask_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "masks")
    fov_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "fov")
    
    files = sorted(glob.glob(os.path.join(raw_img_dir, "*.png")))
    assert len(files) == 45, f"Expected 45 HRF images, got {len(files)}"
    
    window_2d = get_hamming_window(512)
    results = []
    t0 = time.time()
    
    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        base = os.path.splitext(fname)[0]
        coords = coords_by_id[base]
        is_unexposed = (fname in HRF_UNEXPOSED)
        
        raw_np = np.array(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0
        rest_p = os.path.join(rest_img_dir, fname)
        rest_np = np.array(Image.open(rest_p).convert('RGB'), dtype=np.float32) / 255.0
        
        gt_p = os.path.join(mask_dir, fname)
        gt_mask = np.array(Image.open(gt_p).convert('L')) > 127
        
        fov_p = os.path.join(fov_dir, fname)
        if os.path.exists(fov_p):
            fov = np.array(Image.open(fov_p).convert('L')) > 127
        else:
            fov = raw_np.mean(axis=2) > (10.0 / 255.0)
            
        # 1. Raw U-Net
        prob_raw = predict_sliding_window(raw_model, raw_np, coords, 512, 5, window_2d)
        pred_raw = (prob_raw >= UNET_THRESHOLD) & fov
        pix_raw = compute_pixel_metrics(pred_raw, gt_mask, fov)
        topo_raw = analyze_topology_and_branches(pred_raw, gt_mask, fov)
        
        # 2. Restored U-Net
        prob_rest = predict_sliding_window(restored_model, rest_np, coords, 512, 5, window_2d)
        pred_rest = (prob_rest >= UNET_THRESHOLD) & fov
        pix_rest = compute_pixel_metrics(pred_rest, gt_mask, fov)
        topo_rest = analyze_topology_and_branches(pred_rest, gt_mask, fov)
        
        # 3. Raw Frangi
        frangi_raw_resp = apply_frangi(raw_np)
        pred_fr_raw = (frangi_raw_resp >= FRANGI_THRESHOLD) & fov
        pix_fr_raw = compute_pixel_metrics(pred_fr_raw, gt_mask, fov)
        topo_fr_raw = analyze_topology_and_branches(pred_fr_raw, gt_mask, fov)
        
        # 4. Restored Frangi
        frangi_rest_resp = apply_frangi(rest_np)
        pred_fr_rest = (frangi_rest_resp >= FRANGI_THRESHOLD) & fov
        pix_fr_rest = compute_pixel_metrics(pred_fr_rest, gt_mask, fov)
        topo_fr_rest = analyze_topology_and_branches(pred_fr_rest, gt_mask, fov)
        
        rec = {
            "image_id": base,
            "filename": fname,
            "is_unexposed": is_unexposed,
            "raw_unet": {**pix_raw, **topo_raw},
            "restored_unet": {**pix_rest, **topo_rest},
            "raw_frangi": {**pix_fr_raw, **topo_fr_raw},
            "restored_frangi": {**pix_fr_rest, **topo_fr_rest}
        }
        results.append(rec)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == 45:
            print(f" -> Evaluated [{idx+1}/45] HRF images ({(time.time()-t0)/(idx+1):.2f}s/img)")
            
    print(f" -> HRF evaluation completed in {time.time()-t0:.1f}s.")
    return results

def compute_summary_statistics(results, dataset_name="FIVES"):
    systems = ["raw_unet", "restored_unet", "raw_frangi", "restored_frangi"]
    metrics = ["dice", "iou", "sens", "spec", "prec", "thin_skel_sens", "med_skel_sens", "thick_skel_sens", 
               "thin_area_sens", "branch_recall", "thin_branch_recall", "cldice", "num_pred_cc", "endpoints_pred", "break_rate"]
    
    summary = {}
    for sys_name in systems:
        summary[sys_name] = {}
        for m in metrics:
            vals = [r[sys_name][m] for r in results]
            summary[sys_name][m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals))
            }
            
    # Paired comparisons: restored_unet vs raw_unet
    paired = {}
    for m in metrics:
        v_raw = np.array([r["raw_unet"][m] for r in results])
        v_rest = np.array([r["restored_unet"][m] for r in results])
        diff = v_rest - v_raw
        
        mean_diff = float(np.mean(diff))
        ci_low, ci_high = bootstrap_ci(diff)
        
        try:
            w_stat, w_p = stats.wilcoxon(v_rest, v_raw)
            w_p = float(w_p)
        except Exception:
            w_p = 1.0
            
        try:
            t_stat, t_p = stats.ttest_rel(v_rest, v_raw)
            t_p = float(t_p)
        except Exception:
            t_p = 1.0
            
        paired[m] = {
            "mean_delta": mean_diff,
            "ci_95": [ci_low, ci_high],
            "wilcoxon_p": w_p,
            "ttest_p": t_p
        }
        
    return {"per_system": summary, "paired_delta_restored_minus_raw": paired}

def main():
    t_start = time.time()
    verify_checkpoints()
    
    out_dir = os.path.join(REPO_DIR, "results", "milestone3")
    os.makedirs(out_dir, exist_ok=True)
    
    raw_model = load_unet("raw")
    restored_model = load_unet("restored")
    
    # 1. FIVES Test Set Evaluation
    fives_results = run_fives_test_evaluation(raw_model, restored_model)
    fives_summary = compute_summary_statistics(fives_results, "FIVES_Test")
    
    # Disease stratified summaries
    disease_summaries = {}
    for d in ["N", "A", "D", "G"]:
        sub = [r for r in fives_results if r["disease"] == d]
        disease_summaries[d] = compute_summary_statistics(sub, f"FIVES_{d}")
        
    # Duplicate sensitivity analysis (test_32_A vs test_48_A)
    fives_clean = [r for r in fives_results if r["filename"] != "test_48_A.png"]
    fives_clean_summary = compute_summary_statistics(fives_clean, "FIVES_Test_Clean199")
    
    with open(os.path.join(out_dir, "fives_test_results.json"), 'w') as f:
        json.dump({
            "results": fives_results,
            "summary_overall_200": fives_summary,
            "summary_clean_199": fives_clean_summary,
            "disease_breakdown": disease_summaries
        }, f, indent=2)
        
    # 2. HRF Evaluation
    hrf_results = run_hrf_evaluation(raw_model, restored_model)
    hrf_overall_summary = compute_summary_statistics(hrf_results, "HRF_Overall")
    hrf_exposed = [r for r in hrf_results if not r["is_unexposed"]]
    hrf_unexposed = [r for r in hrf_results if r["is_unexposed"]]
    
    hrf_exposed_summary = compute_summary_statistics(hrf_exposed, "HRF_Exposed40")
    hrf_unexposed_summary = compute_summary_statistics(hrf_unexposed, "HRF_Unexposed5")
    
    with open(os.path.join(out_dir, "hrf_results.json"), 'w') as f:
        json.dump({
            "results": hrf_results,
            "summary_overall_45": hrf_overall_summary,
            "summary_exposed_40": hrf_exposed_summary,
            "summary_unexposed_5": hrf_unexposed_summary
        }, f, indent=2)
        
    # Final checkpoint verification
    print("\n[Post-Evaluation Audit] Re-verifying checkpoint integrity post-run...")
    verify_checkpoints()
    print(f"\n[DONE] Milestone 3 Evaluation completed in {time.time()-t_start:.1f}s!")

if __name__ == '__main__':
    main()

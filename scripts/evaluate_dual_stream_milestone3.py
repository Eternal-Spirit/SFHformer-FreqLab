"""
scripts/evaluate_dual_stream_milestone3.py
Frozen Milestone 3 Generalization Evaluation for the Dual-Stream Hybrid U-Net.

Evaluates:
- FIVES 200 untouched test images (paired raw + SFHformer-restored)
- HRF 45 external images (separated into 40 exposed vs 5 zero-shot unexposed)

Computes exact paired statistical comparisons (Wilcoxon signed-rank p-value and
10,000-sample paired bootstrap 95% CI) against both frozen single-stream baselines:
  1. Raw U-Net (checkpoints/unet_raw/best.pth)
  2. Restored U-Net (checkpoints/unet_restored/best.pth)

Outputs:
- results/milestone3/dual_stream_fives_results.json
- results/milestone3/dual_stream_hrf_results.json
"""

import os
import sys
import time
import json
import glob
import hashlib
import argparse
import numpy as np
from PIL import Image
import torch
from scipy import stats

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.dual_stream_unet import get_dual_stream_unet
from scripts.evaluate_milestone3 import (
    CKPTS as BASELINE_CKPTS,
    HRF_UNEXPOSED,
    sha256_file,
    get_hamming_window,
    analyze_topology_and_branches,
    compute_pixel_metrics,
    bootstrap_ci,
)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.65)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DUAL_CKPT_PATH = os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream", "best.pth")
CALIB_PATH = os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream", "validation_calibration.json")
RESULTS_DIR = os.path.join(REPO_DIR, "results", "milestone3")


def predict_dual_sliding_window(model, raw_arr, rest_arr, coords, tile_size=512, chunk_sz=4, window_2d=None):
    H, W = raw_arr.shape[:2]
    accum_prob = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
    accum_weight = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)

    raw_tiles, rest_tiles = [], []
    for c in coords:
        t, l = c["top"], c["left"]
        raw_tiles.append(torch.from_numpy(raw_arr[t:t + tile_size, l:l + tile_size]).permute(2, 0, 1))
        rest_tiles.append(torch.from_numpy(rest_arr[t:t + tile_size, l:l + tile_size]).permute(2, 0, 1))

    x_raw = torch.stack(raw_tiles, dim=0).to(DEVICE)
    x_rest = torch.stack(rest_tiles, dim=0).to(DEVICE)

    out_probs = []
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else torch.no_grad():
            for ch in range(0, len(x_raw), chunk_sz):
                logits = model(x_raw[ch:ch + chunk_sz], x_rest[ch:ch + chunk_sz])
                probs = torch.sigmoid(logits).squeeze(1).float()
                out_probs.append(probs)

    full_probs = torch.cat(out_probs, dim=0)
    for k, c in enumerate(coords):
        t, l = c["top"], c["left"]
        accum_prob[t:t + tile_size, l:l + tile_size] += full_probs[k] * window_2d
        accum_weight[t:t + tile_size, l:l + tile_size] += window_2d

    return (accum_prob / accum_weight).cpu().numpy()


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni step-down multiple-comparison adjustment."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def compute_paired_deltas(dual_records, baseline_records, baseline_key, metrics=("dice", "prec", "thin_skel_sens", "cldice", "branch_recall", "num_pred_cc")):
    base_by_file = {r["filename"]: r[baseline_key] for r in baseline_records}
    out = {}
    raw_wil_ps = []
    raw_t_ps = []
    for m in metrics:
        d_vals = np.array([r["dual_stream"][m] for r in dual_records], dtype=np.float64)
        b_vals = np.array([base_by_file[r["filename"]][m] for r in dual_records], dtype=np.float64)
        deltas = d_vals - b_vals
        ci_lo, ci_hi = bootstrap_ci(deltas)
        try:
            _, p_wil = stats.wilcoxon(d_vals, b_vals)
        except ValueError:
            p_wil = 1.0
        _, p_t = stats.ttest_rel(d_vals, b_vals)
        raw_wil_ps.append(float(p_wil))
        raw_t_ps.append(float(p_t))
        out[m] = {
            "dual_mean": round(float(np.mean(d_vals)), 4),
            "baseline_mean": round(float(np.mean(b_vals)), 4),
            "delta_mean": round(float(np.mean(deltas)), 4),
            "ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "wilcoxon_p": float(p_wil),
            "paired_t_p": float(p_t),
        }
    adj_wil = holm_bonferroni(raw_wil_ps)
    adj_t = holm_bonferroni(raw_t_ps)
    for idx, m in enumerate(metrics):
        out[m]["wilcoxon_p_holm"] = float(adj_wil[idx])
        out[m]["paired_t_p_holm"] = float(adj_t[idx])
    return out


def summarize_dual_records(records):
    keys = ["dice", "iou", "sens", "spec", "prec", "thin_skel_sens", "med_skel_sens", "thick_skel_sens", "thin_area_sens", "branch_recall", "thin_branch_recall", "cldice", "num_pred_cc", "endpoints_pred", "break_rate"]
    return {k: round(float(np.mean([r["dual_stream"][k] for r in records])), 4) for k in keys}


def evaluate_fives(model, threshold):
    print("\n" + "=" * 80)
    print(f" [STAGE 1/2] FIVES Test Set Evaluation (N=200 Untouched, tau={threshold:.2f}) ")
    print("=" * 80)

    with open(os.path.join(REPO_DIR, "data", "fives", "dataset_split.json"), "r") as f:
        split = json.load(f)
    test_meta = {x["filename"]: x["disease"] for x in split["test_images"]}

    raw_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "images")
    rest_dir = os.path.join(REPO_DIR, "data", "fives", "restored_test", "images")
    mask_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "masks")
    fov_dir = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "fov")

    files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    assert len(files) == 200, f"Expected 200 test images, got {len(files)}"

    offsets = [0, 384, 768, 1152, 1536]
    coords = [{"top": t, "left": l} for t in offsets for l in offsets]
    window_2d = get_hamming_window(512)

    records = []
    t0 = time.time()
    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        disease = test_meta.get(fname, "Unknown")

        raw_np = np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        rest_np = np.array(Image.open(os.path.join(rest_dir, fname)).convert("RGB"), dtype=np.float32) / 255.0
        gt_mask = np.array(Image.open(os.path.join(mask_dir, fname)).convert("L")) > 127
        fov_p = os.path.join(fov_dir, fname)
        fov = np.array(Image.open(fov_p).convert("L")) > 127 if os.path.exists(fov_p) else (raw_np.mean(axis=2) > (10.0 / 255.0))

        prob = predict_dual_sliding_window(model, raw_np, rest_np, coords, 512, 4, window_2d)
        pred = (prob >= threshold) & fov
        pix = compute_pixel_metrics(pred, gt_mask, fov)
        topo = analyze_topology_and_branches(pred, gt_mask, fov)

        records.append({
            "filename": fname,
            "disease": disease,
            "dual_stream": {**pix, **topo},
        })
        if (idx + 1) % 25 == 0:
            print(f"  -> FIVES [{idx + 1:3d}/200] elapsed={time.time() - t0:.1f}s")

    # Pair against frozen Milestone 3 baseline results
    with open(os.path.join(RESULTS_DIR, "fives_test_results.json"), "r") as f:
        base_fives = json.load(f)
    base_records = base_fives["results"]

    # Duplicate-clean N=199 subset (excluding duplicate pair test_48_A.png)
    records_dedup = [r for r in records if r["filename"] != "test_48_A.png"]
    base_dedup = [r for r in base_records if r["filename"] != "test_48_A.png"]

    diseases = ["N", "A", "D", "G"]
    disease_labels = {"N": "Normal", "A": "AMD", "D": "DR", "G": "Glaucoma"}
    disease_summary = {}
    for d in diseases:
        sub = [r for r in records if r["disease"] == d]
        b_sub = [r for r in base_records if r["disease"] == d]
        disease_summary[disease_labels[d]] = {
            "code": d,
            "n": len(sub),
            "dual_stream": summarize_dual_records(sub),
            "vs_raw_unet": compute_paired_deltas(sub, b_sub, "raw_unet"),
            "vs_restored_unet": compute_paired_deltas(sub, b_sub, "restored_unet"),
        }

    return {
        "n_images": len(records),
        "threshold": threshold,
        "overall": summarize_dual_records(records),
        "vs_raw_unet_n200": compute_paired_deltas(records, base_records, "raw_unet"),
        "vs_restored_unet_n200": compute_paired_deltas(records, base_records, "restored_unet"),
        "vs_raw_unet_dedup_n199": compute_paired_deltas(records_dedup, base_dedup, "raw_unet"),
        "vs_restored_unet_dedup_n199": compute_paired_deltas(records_dedup, base_dedup, "restored_unet"),
        "by_disease": disease_summary,
        "per_image_results": records,
    }


def evaluate_hrf(model, threshold):
    print("\n" + "=" * 80)
    print(f" [STAGE 2/2] HRF External Evaluation (N=45, 40 Exposed / 5 Unexposed, tau={threshold:.2f}) ")
    print("=" * 80)

    raw_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "images")
    rest_dir = os.path.join(REPO_DIR, "data", "hrf", "restored", "images")
    mask_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "masks")
    fov_dir = os.path.join(REPO_DIR, "data", "hrf", "raw", "fov")

    files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    assert len(files) == 45, f"Expected 45 HRF images, got {len(files)}"

    window_2d = get_hamming_window(512)
    records = []
    t0 = time.time()

    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        is_unexposed = fname in HRF_UNEXPOSED

        raw_np = np.array(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        rest_np = np.array(Image.open(os.path.join(rest_dir, fname)).convert("RGB"), dtype=np.float32) / 255.0
        gt_mask = np.array(Image.open(os.path.join(mask_dir, fname)).convert("L")) > 127
        fov_p = os.path.join(fov_dir, fname)
        fov = np.array(Image.open(fov_p).convert("L")) > 127 if os.path.exists(fov_p) else (raw_np.mean(axis=2) > (10.0 / 255.0))

        H, W = raw_np.shape[:2]
        stride = 384
        tops = sorted(list(set(list(range(0, H - 512, stride)) + [H - 512])))
        lefts = sorted(list(set(list(range(0, W - 512, stride)) + [W - 512])))
        coords = [{"top": t, "left": l} for t in tops for l in lefts]

        prob = predict_dual_sliding_window(model, raw_np, rest_np, coords, 512, 4, window_2d)
        pred = (prob >= threshold) & fov
        pix = compute_pixel_metrics(pred, gt_mask, fov)
        topo = analyze_topology_and_branches(pred, gt_mask, fov)

        records.append({
            "filename": fname,
            "is_unexposed": is_unexposed,
            "dual_stream": {**pix, **topo},
        })
        if (idx + 1) % 15 == 0:
            print(f"  -> HRF [{idx + 1:2d}/45] elapsed={time.time() - t0:.1f}s")

    with open(os.path.join(RESULTS_DIR, "hrf_results.json"), "r") as f:
        base_hrf = json.load(f)
    base_records = base_hrf["results"]

    exposed = [r for r in records if not r["is_unexposed"]]
    unexposed = [r for r in records if r["is_unexposed"]]
    base_exp = [r for r in base_records if not r["is_unexposed"]]
    base_unexp = [r for r in base_records if r["is_unexposed"]]

    assert len(exposed) == 40 and len(unexposed) == 5

    return {
        "n_images": len(records),
        "threshold": threshold,
        "overall_n45": summarize_dual_records(records),
        "exposed_n40": summarize_dual_records(exposed),
        "unexposed_n5_exploratory": {
            "statistical_status": "exploratory_n5_insufficient_power_min_wilcoxon_p_0.0625",
            "summary": summarize_dual_records(unexposed),
            "per_image": unexposed,
        },
        "overall_vs_raw": compute_paired_deltas(records, base_records, "raw_unet"),
        "overall_vs_restored": compute_paired_deltas(records, base_records, "restored_unet"),
        "exposed_vs_raw": compute_paired_deltas(exposed, base_exp, "raw_unet"),
        "exposed_vs_restored": compute_paired_deltas(exposed, base_exp, "restored_unet"),
        "unexposed_vs_raw_exploratory": compute_paired_deltas(unexposed, base_unexp, "raw_unet"),
        "unexposed_vs_restored_exploratory": compute_paired_deltas(unexposed, base_unexp, "restored_unet"),
        "per_image_results": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None, help="Override frozen threshold; defaults to validation_calibration.json")
    args = parser.parse_args()

    # Verify frozen baseline hashes first
    print("[Audit] Verifying baseline & Dual-Stream checkpoints...")
    for k, v in BASELINE_CKPTS.items():
        curr = sha256_file(v["path"])
        assert curr == v["sha"], f"Baseline integrity mismatch on {k}"
        print(f" -> {k.upper()} SHA-256 verified: {curr[:16]}...")

    dual_sha_pre = sha256_file(DUAL_CKPT_PATH)
    print(f" -> DUAL_STREAM SHA-256 (pre): {dual_sha_pre[:16]}...")

    if args.threshold is not None:
        threshold = args.threshold
    elif os.path.exists(CALIB_PATH):
        with open(CALIB_PATH, "r") as f:
            threshold = float(json.load(f)["selected_frozen_threshold"])
    else:
        threshold = 0.50
    print(f"[Protocol] Frozen Evaluation Threshold = {threshold:.2f}")

    model = get_dual_stream_unet(in_channels=3, out_channels=1, base_channels=32).to(DEVICE)
    ck = torch.load(DUAL_CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    fives_out = evaluate_fives(model, threshold)
    fives_out["dual_checkpoint_sha256"] = dual_sha_pre
    with open(os.path.join(RESULTS_DIR, "dual_stream_fives_results.json"), "w") as f:
        json.dump(fives_out, f, indent=2)

    hrf_out = evaluate_hrf(model, threshold)
    hrf_out["dual_checkpoint_sha256"] = dual_sha_pre
    with open(os.path.join(RESULTS_DIR, "dual_stream_hrf_results.json"), "w") as f:
        json.dump(hrf_out, f, indent=2)

    dual_sha_post = sha256_file(DUAL_CKPT_PATH)
    assert dual_sha_pre == dual_sha_post, "Checkpoint modified during evaluation!"
    print(f"\n[Audit Passed] DUAL_STREAM SHA-256 unchanged: {dual_sha_post[:16]}...")

    print("\n" + "=" * 80)
    print(" DUAL-STREAM HYBRID SUMMARY ")
    print("=" * 80)
    print("FIVES N=200 vs Raw U-Net:", json.dumps(fives_out["vs_raw_unet_n200"], indent=2))
    print("FIVES N=200 vs Restored U-Net:", json.dumps(fives_out["vs_restored_unet_n200"], indent=2))
    print("HRF Unexposed N=5 (Exploratory) vs Raw U-Net:", json.dumps(hrf_out["unexposed_vs_raw_exploratory"], indent=2))


if __name__ == "__main__":
    main()

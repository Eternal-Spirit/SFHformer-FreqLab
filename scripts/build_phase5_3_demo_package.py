"""
scripts/build_phase5_3_demo_package.py
Assembles the Phase 5.3 Demonstration Package & Visual Evidence under D:\SFHformer-FreqLab\demo_package\:
  1. Qualitative full-resolution FIVES test comparison across Normal, AMD, DR, and Glaucoma
  2. Qualitative external HRF unexposed comparison (14_h, 15_dr)
  3. Exact Milestone 3 benchmark tables (CSV + Markdown) derived directly from frozen JSONs
  4. Reproducibility SNAPSHOT_MANIFEST.json (git tag v3.1-phase5.2-verified + SHA-256 hashes)
  5. DEMO_WALKTHROUGH.md presentation & technical evaluation guide
"""

import os
import csv
import json
import shutil
import hashlib
import subprocess
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.unet import get_unet
from models.dual_stream_unet import get_dual_stream_unet
from scripts.evaluate_milestone3 import get_hamming_window, predict_sliding_window
from scripts.evaluate_dual_stream_milestone3 import predict_dual_sliding_window

DEMO_DIR = os.path.join(REPO_DIR, "demo_package")
QUAL_DIR = os.path.join(DEMO_DIR, "qualitative_examples")
TABLE_DIR = os.path.join(DEMO_DIR, "benchmark_tables")
REPRO_DIR = os.path.join(DEMO_DIR, "reproducibility")
ART_DIR = r"C:\Users\ARAN\.gemini\antigravity\brain\73eae9fd-ef3d-4e16-bd31-bdb8624d8e09"

for d in (QUAL_DIR, TABLE_DIR, REPRO_DIR):
    os.makedirs(d, exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.65)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def make_overlay(img_rgb, mask_bool, color=(0, 235, 175), alpha=0.60):
    base = np.clip(img_rgb * 255.0 + 0.5, 0, 255).astype(np.float32)
    out = base.copy()
    c = np.array(color, dtype=np.float32)
    out[mask_bool] = base[mask_bool] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def find_best_crop(gt_mask, fov, crop_sz=512):
    """Locate a vessel-dense interior 512x512 region away from borders."""
    H, W = gt_mask.shape
    cy, cx = H // 2, W // 2
    # Search a 3x3 grid around center for max vessel density inside FOV
    best_score, best_tl = -1, (cy - crop_sz // 2, cx - crop_sz // 2)
    for dy in (-350, 0, 350):
        for dx in (-350, 0, 350):
            t = max(0, min(H - crop_sz, cy + dy - crop_sz // 2))
            l = max(0, min(W - crop_sz, cx + dx - crop_sz // 2))
            if fov[t:t+crop_sz, l:l+crop_sz].mean() < 0.90:
                continue
            sc = int(gt_mask[t:t+crop_sz, l:l+crop_sz].sum())
            if sc > best_score:
                best_score = sc
                best_tl = (t, l)
    return best_tl


def generate_qualitative_examples():
    print("[1/4] Loading frozen models for full-resolution qualitative visual evidence...")
    u_raw = get_unet().to(DEVICE)
    u_raw.load_state_dict(torch.load(os.path.join(REPO_DIR, "checkpoints", "unet_raw", "best.pth"), map_location=DEVICE, weights_only=False)["model"])
    u_raw.eval()

    u_rest = get_unet().to(DEVICE)
    u_rest.load_state_dict(torch.load(os.path.join(REPO_DIR, "checkpoints", "unet_restored", "best.pth"), map_location=DEVICE, weights_only=False)["model"])
    u_rest.eval()

    u_dual = get_dual_stream_unet().to(DEVICE)
    u_dual.load_state_dict(torch.load(os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream", "best.pth"), map_location=DEVICE, weights_only=False)["model"])
    u_dual.eval()

    with open(os.path.join(REPO_DIR, "results", "milestone3", "fives_test_results.json")) as f:
        fives_base = {r["filename"]: r for r in json.load(f)["results"]}
    with open(os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_fives_results.json")) as f:
        fives_dual = {r["filename"]: r["dual_stream"] for r in json.load(f)["per_image_results"]}
    with open(os.path.join(REPO_DIR, "results", "milestone3", "hrf_results.json")) as f:
        hrf_base = {r["filename"]: r for r in json.load(f)["results"]}
    with open(os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_hrf_results.json")) as f:
        hrf_dual = {r["filename"]: r["dual_stream"] for r in json.load(f)["per_image_results"]}

    window_2d = get_hamming_window(512)

    # A) FIVES Disease-Stratified Full-Res Qualitative Grid (4 Rows: Normal, AMD, DR, Glaucoma)
    fives_cases = [
        ("test_161_N.png", "FIVES Normal (test_161_N)"),
        ("test_37_A.png", "FIVES AMD (test_37_A)"),
        ("test_100_D.png", "FIVES DR (test_100_D)"),
        ("test_133_G.png", "FIVES Glaucoma (test_133_G)"),
    ]
    offsets = [0, 384, 768, 1152, 1536]
    coords_fives = [{"top": t, "left": l} for t in offsets for l in offsets]

    fig, axes = plt.subplots(4, 5, figsize=(21, 17), facecolor="#0F172A")
    col_titles = [
        "Input + Ground Truth (512x512 ROI)",
        "Raw U-Net (tau=0.50)",
        "Restored U-Net (tau=0.50)",
        "Dual-Stream Hybrid (tau=0.40)",
        "Diff: Dual-Stream vs Raw (Amber=Added)",
    ]

    for r_idx, (fname, label) in enumerate(fives_cases):
        raw_np = np.array(Image.open(os.path.join(REPO_DIR, "data", "fives", "raw", "test", "images", fname)).convert("RGB"), dtype=np.float32) / 255.0
        rest_np = np.array(Image.open(os.path.join(REPO_DIR, "data", "fives", "restored_test", "images", fname)).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.array(Image.open(os.path.join(REPO_DIR, "data", "fives", "raw", "test", "masks", fname)).convert("L")) > 127
        fov_p = os.path.join(REPO_DIR, "data", "fives", "raw", "test", "fov", fname)
        fov = np.array(Image.open(fov_p).convert("L")) > 127 if os.path.exists(fov_p) else (raw_np.mean(axis=2) > 10.0 / 255.0)

        p_raw = predict_sliding_window(u_raw, raw_np, coords_fives, 512, 5, window_2d) >= 0.50
        p_rest = predict_sliding_window(u_rest, rest_np, coords_fives, 512, 5, window_2d) >= 0.50
        p_dual = predict_dual_sliding_window(u_dual, raw_np, rest_np, coords_fives, 512, 4, window_2d) >= 0.40

        p_raw &= fov
        p_rest &= fov
        p_dual &= fov

        # For Glaucoma test_133_G, zoom around optic disc / center if vessel-rich; else best ROI
        t, l = find_best_crop(gt, fov, crop_sz=512)
        s_raw = raw_np[t:t+512, l:l+512]
        s_gt = gt[t:t+512, l:l+512]
        s_pr = p_raw[t:t+512, l:l+512]
        s_ps = p_rest[t:t+512, l:l+512]
        s_pd = p_dual[t:t+512, l:l+512]

        # Diff overlay: Shared Cyan + Dual-exclusive Amber
        diff_vis = make_overlay(s_raw, s_pd & s_pr, color=(0, 210, 255), alpha=0.58)
        diff_only = s_pd & ~s_pr
        diff_vis[diff_only] = np.clip(diff_vis[diff_only].astype(np.float32) * 0.40 + np.array([255, 180, 0]) * 0.60, 0, 255).astype(np.uint8)

        panels = [
            make_overlay(s_raw, s_gt, color=(255, 255, 255), alpha=0.55),
            make_overlay(s_raw, s_pr, color=(56, 189, 248), alpha=0.60),
            make_overlay(s_raw, s_ps, color=(52, 211, 153), alpha=0.60),
            make_overlay(s_raw, s_pd, color=(0, 240, 180), alpha=0.62),
            diff_vis,
        ]
        m_raw = fives_base[fname]["raw_unet"]
        m_rest = fives_base[fname]["restored_unet"]
        m_dual = fives_dual[fname]
        subs = [
            f"{label} (Full Image Metrics)",
            f"Dice {m_raw['dice']:.3f} | Prec {m_raw['prec']:.3f} | Thin {m_raw['thin_skel_sens']*100:.1f}%",
            f"Dice {m_rest['dice']:.3f} | Prec {m_rest['prec']:.3f} | Thin {m_rest['thin_skel_sens']*100:.1f}%",
            f"Dice {m_dual['dice']:.3f} | Prec {m_dual['prec']:.3f} | Thin {m_dual['thin_skel_sens']*100:.1f}%",
            f"clDice: {m_raw['cldice']:.3f} -> {m_dual['cldice']:.3f} | CC: {m_raw['num_pred_cc']} -> {m_dual['num_pred_cc']}",
        ]
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            ax.imshow(panels[c_idx])
            ax.axis("off")
            ttl = f"{col_titles[c_idx]}\n{subs[c_idx]}" if r_idx == 0 else subs[c_idx]
            ax.set_title(ttl, color="#F8FAFC" if r_idx == 0 else "#CBD5E1", fontsize=9.2, fontweight="bold" if r_idx == 0 else "normal", pad=6)

    plt.suptitle("Milestone 3 Full-Resolution Untouched FIVES Test Qualitative Evidence (4 Disease Categories, 512x512 Vascular ROIs)",
                 color="#38BDF8", fontsize=13.5, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    fives_out = os.path.join(QUAL_DIR, "fives_test_disease_stratified_demo.png")
    plt.savefig(fives_out, dpi=200, facecolor="#0F172A")
    plt.close()
    print("  Saved:", fives_out)

    # B) HRF External Unexposed Exploratory Qualitative Grid (14_h, 15_dr)
    hrf_cases = [
        ("14_h.png", "HRF Unexposed Healthy (14_h)"),
        ("15_dr.png", "HRF Unexposed DR (15_dr)"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(21, 9.8), facecolor="#0F172A")
    for r_idx, (fname, label) in enumerate(hrf_cases):
        raw_np = np.array(Image.open(os.path.join(REPO_DIR, "data", "hrf", "raw", "images", fname)).convert("RGB"), dtype=np.float32) / 255.0
        rest_np = np.array(Image.open(os.path.join(REPO_DIR, "data", "hrf", "restored", "images", fname)).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.array(Image.open(os.path.join(REPO_DIR, "data", "hrf", "raw", "masks", fname)).convert("L")) > 127
        fov = np.array(Image.open(os.path.join(REPO_DIR, "data", "hrf", "raw", "fov", fname)).convert("L")) > 127

        H, W = raw_np.shape[:2]
        stride = 384
        tops = sorted(list(set(list(range(0, H - 512, stride)) + [H - 512])))
        lefts = sorted(list(set(list(range(0, W - 512, stride)) + [W - 512])))
        coords_hrf = [{"top": t, "left": l} for t in tops for l in lefts]

        p_raw = (predict_sliding_window(u_raw, raw_np, coords_hrf, 512, 5, window_2d) >= 0.50) & fov
        p_rest = (predict_sliding_window(u_rest, rest_np, coords_hrf, 512, 5, window_2d) >= 0.50) & fov
        p_dual = (predict_dual_sliding_window(u_dual, raw_np, rest_np, coords_hrf, 512, 4, window_2d) >= 0.40) & fov

        t, l = find_best_crop(gt, fov, crop_sz=512)
        s_raw = raw_np[t:t+512, l:l+512]
        s_gt = gt[t:t+512, l:l+512]
        s_pr = p_raw[t:t+512, l:l+512]
        s_ps = p_rest[t:t+512, l:l+512]
        s_pd = p_dual[t:t+512, l:l+512]

        diff_vis = make_overlay(s_raw, s_pd & s_pr, color=(0, 210, 255), alpha=0.58)
        diff_only = s_pd & ~s_pr
        diff_vis[diff_only] = np.clip(diff_vis[diff_only].astype(np.float32) * 0.40 + np.array([255, 180, 0]) * 0.60, 0, 255).astype(np.uint8)

        panels = [
            make_overlay(s_raw, s_gt, color=(255, 255, 255), alpha=0.55),
            make_overlay(s_raw, s_pr, color=(56, 189, 248), alpha=0.60),
            make_overlay(s_raw, s_ps, color=(52, 211, 153), alpha=0.60),
            make_overlay(s_raw, s_pd, color=(0, 240, 180), alpha=0.62),
            diff_vis,
        ]
        m_raw = hrf_base[fname]["raw_unet"]
        m_rest = hrf_base[fname]["restored_unet"]
        m_dual = hrf_dual[fname]
        subs = [
            f"{label} (Exploratory N=5)",
            f"Dice {m_raw['dice']:.3f} | Prec {m_raw['prec']:.3f} | Thin {m_raw['thin_skel_sens']*100:.1f}%",
            f"Dice {m_rest['dice']:.3f} | Prec {m_rest['prec']:.3f} | Thin {m_rest['thin_skel_sens']*100:.1f}%",
            f"Dice {m_dual['dice']:.3f} | Prec {m_dual['prec']:.3f} | Thin {m_dual['thin_skel_sens']*100:.1f}%",
            f"clDice: {m_raw['cldice']:.3f} -> {m_dual['cldice']:.3f} | CC: {m_raw['num_pred_cc']} -> {m_dual['num_pred_cc']}",
        ]
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            ax.imshow(panels[c_idx])
            ax.axis("off")
            ttl = f"{col_titles[c_idx]}\n{subs[c_idx]}" if r_idx == 0 else subs[c_idx]
            ax.set_title(ttl, color="#F8FAFC" if r_idx == 0 else "#CBD5E1", fontsize=9.2, fontweight="bold" if r_idx == 0 else "normal", pad=6)

    plt.suptitle("HRF Unexposed Exploratory Subset (3504x2336 -> 512x512 ROI): Mixed Domain Transfer Illustration",
                 color="#FBBF24", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.2)
    hrf_out = os.path.join(QUAL_DIR, "hrf_transfer_unexposed_demo.png")
    plt.savefig(hrf_out, dpi=200, facecolor="#0F172A")
    plt.close()
    print("  Saved:", hrf_out)

    # Copy Phase 5.1 & 5.2 screenshots into qualitative_examples/ and Phase 5.3 assets to artifact dir
    for fname in ("phase5_1_qualitative_multimodel_demo.png", "phase5_2_ui_tab1_fusion.png", "phase5_2_ui_tab2_benchmark.png"):
        src = os.path.join(ART_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(QUAL_DIR, fname))
    for dfile in os.listdir(os.path.join(DEMO_DIR, "diagrams")):
        if dfile.endswith(".png"):
            shutil.copy2(os.path.join(DEMO_DIR, "diagrams", dfile), os.path.join(ART_DIR, dfile))
    for qfile in ("fives_test_disease_stratified_demo.png", "hrf_transfer_unexposed_demo.png"):
        shutil.copy2(os.path.join(QUAL_DIR, qfile), os.path.join(ART_DIR, qfile))


def build_benchmark_tables():
    print("[2/4] Generating exact Benchmark Summary CSV & Markdown from frozen JSONs...")
    with open(os.path.join(REPO_DIR, "results", "milestone3", "fives_test_results.json")) as f:
        f_base = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_fives_results.json")) as f:
        f_dual = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "milestone3", "hrf_results.json")) as f:
        h_base = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_hrf_results.json")) as f:
        h_dual = json.load(f)

    rows = []
    def add_base_row(ds_label, sys_label, tau, d):
        rows.append({
            "Dataset": ds_label,
            "System": sys_label,
            "Threshold": tau,
            "Dice": round(d["dice"]["mean"], 4),
            "Precision": round(d["prec"]["mean"], 4),
            "Sensitivity": round(d["sens"]["mean"], 4),
            "Thin_Skel_Sens": round(d["thin_skel_sens"]["mean"], 4),
            "Thin_Branch_Recall": round(d["thin_branch_recall"]["mean"], 4),
            "clDice": round(d["cldice"]["mean"], 4),
            "Mean_CC": round(d["num_pred_cc"]["mean"], 2),
        })

    def add_dual_row(ds_label, tau, d):
        rows.append({
            "Dataset": ds_label,
            "System": "Dual-Stream Hybrid U-Net",
            "Threshold": tau,
            "Dice": d["dice"],
            "Precision": d["prec"],
            "Sensitivity": d["sens"],
            "Thin_Skel_Sens": d["thin_skel_sens"],
            "Thin_Branch_Recall": d["thin_branch_recall"],
            "clDice": d["cldice"],
            "Mean_CC": d["num_pred_cc"],
        })

    add_base_row("FIVES Test (N=200, Primary In-Domain)", "Raw U-Net", "0.50", f_base["summary_overall_200"]["per_system"]["raw_unet"])
    add_base_row("FIVES Test (N=200, Primary In-Domain)", "Restored U-Net", "0.50", f_base["summary_overall_200"]["per_system"]["restored_unet"])
    add_dual_row("FIVES Test (N=200, Primary In-Domain)", "0.40", f_dual["overall"])
    add_base_row("FIVES Test (N=200, Primary In-Domain)", "Raw Frangi", "0.0010", f_base["summary_overall_200"]["per_system"]["raw_frangi"])
    add_base_row("FIVES Test (N=200, Primary In-Domain)", "Restored Frangi", "0.0010", f_base["summary_overall_200"]["per_system"]["restored_frangi"])

    add_base_row("HRF External (N=45, 40 Exposed / 5 Unexposed)", "Raw U-Net", "0.50", h_base["summary_overall_45"]["per_system"]["raw_unet"])
    add_base_row("HRF External (N=45, 40 Exposed / 5 Unexposed)", "Restored U-Net", "0.50", h_base["summary_overall_45"]["per_system"]["restored_unet"])
    add_dual_row("HRF External (N=45, 40 Exposed / 5 Unexposed)", "0.40", h_dual["overall_n45"])

    add_base_row("HRF Unexposed (N=5, Exploratory)", "Raw U-Net", "0.50", h_base["summary_unexposed_5"]["per_system"]["raw_unet"])
    add_base_row("HRF Unexposed (N=5, Exploratory)", "Restored U-Net", "0.50", h_base["summary_unexposed_5"]["per_system"]["restored_unet"])
    add_dual_row("HRF Unexposed (N=5, Exploratory)", "0.40", h_dual["unexposed_n5_exploratory"]["summary"])

    csv_p = os.path.join(TABLE_DIR, "milestone3_benchmark_summary.csv")
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_p = os.path.join(TABLE_DIR, "milestone3_benchmark_summary.md")
    with open(md_p, "w") as f:
        f.write("# Milestone 3 Benchmark Summary (Frozen Experimental Records)\n\n")
        f.write("| Dataset | System | Threshold | Dice | Precision | Sensitivity | Thin Skel Sens | Thin Branch Recall | clDice | Mean CC |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows:
            f.write(f"| {r['Dataset']} | {r['System']} | `{r['Threshold']}` | `{r['Dice']:.4f}` | `{r['Precision']:.4f}` | `{r['Sensitivity']:.4f}` | `{r['Thin_Skel_Sens']*100:.2f}%` | `{r['Thin_Branch_Recall']*100:.2f}%` | `{r['clDice']:.4f}` | `{r['Mean_CC']}` |\n")
        f.write("\n## FIVES Disease Stratification (Dual-Stream Hybrid @ tau=0.40 vs Single-Stream @ tau=0.50)\n\n")
        f.write("| Pathology (n=50 each) | Raw Dice / Prec | Restored Dice / Prec | Dual-Stream Dice / Prec | Dual ThinSkel | Dual clDice | Dual CC |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for label, code in [("Normal", "N"), ("AMD", "A"), ("DR", "D"), ("Glaucoma", "G")]:
            rb = f_base["disease_breakdown"][code]["per_system"]["raw_unet"]
            sb = f_base["disease_breakdown"][code]["per_system"]["restored_unet"]
            db = f_dual["by_disease"][label]["dual_stream"]
            f.write(f"| **{label}** | `{rb['dice']['mean']:.4f}` / `{rb['prec']['mean']:.4f}` | `{sb['dice']['mean']:.4f}` / `{sb['prec']['mean']:.4f}` | **`{db['dice']:.4f}` / `{db['prec']:.4f}`** | `{db['thin_skel_sens']*100:.2f}%` | `{db['cldice']:.4f}` | `{db['num_pred_cc']}` |\n")
    print("  Saved:", csv_p, "and", md_p)


def build_reproducibility_manifest():
    print("[3/4] Writing cryptographic SNAPSHOT_MANIFEST.json...")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR).decode().strip()
    tags = subprocess.check_output(["git", "tag", "--points-at", "HEAD"], cwd=REPO_DIR).decode().strip().splitlines()
    files = [
        "checkpoints/unet_raw/best.pth",
        "checkpoints/unet_restored/best.pth",
        "checkpoints/unet_dual_stream/best.pth",
        "checkpoints/unet_dual_stream/validation_calibration.json",
        "checkpoints/fives_retina_finetune/best.pth",
        "results/milestone3/fives_test_results.json",
        "results/milestone3/hrf_results.json",
        "results/milestone3/dual_stream_fives_results.json",
        "results/milestone3/dual_stream_hrf_results.json",
        "models/SFHformer.py",
        "models/unet.py",
        "models/dual_stream_unet.py",
        "utils/losses.py",
        "scripts/serve_restore.py",
        "frontend/src/pages/RetinaStudio.jsx",
    ]
    manifest = {
        "snapshot_tag": "v3.1-phase5.2-verified",
        "git_commit": commit,
        "git_tags_at_head": tags,
        "research_disclaimer": "RetinaScan Studio is a research prototype for retinal vessel segmentation, not a certified clinical diagnostic system.",
        "sha256_inventory": {p: sha256_file(os.path.join(REPO_DIR, p)) for p in files},
    }
    out_m = os.path.join(REPRO_DIR, "SNAPSHOT_MANIFEST.json")
    with open(out_m, "w") as f:
        json.dump(manifest, f, indent=2)
    print("  Saved:", out_m)


def build_demo_walkthrough():
    print("[4/4] Writing DEMO_WALKTHROUGH.md...")
    md = """# Phase 5.3 Demonstration Package & Visual Evidence Guide

> **System Status**: Research Prototype (`v3.1-phase5.2-verified`, commit `42ac0565d1d1517ef5bad9fdc1f392608272db1b`)  
> **Scope Notice**: RetinaScan Studio is a **research prototype for retinal vessel segmentation and frequency restoration analysis**, not a certified clinical diagnostic system.

---

## 1. Architectural Diagrams (300 DPI, Neat & In-Depth)
1. **[diag_1_sfhformer_macro.png](../diagrams/diag_1_sfhformer_macro.png)**: 5-Stage SFHformer-m macro-architecture (`14.18M` params, `PixelShuffle(2)` upsampling, global identity residual, `31.92 dB` FIVES SOTA).
2. **[diag_2_sfh_block_detail.png](../diagrams/diag_2_sfh_block_detail.png)**: Internal SFH Block showing dual-branch token mixing (`TokenMixer_For_Local` dilated DW-Convs + `TokenMixer_For_Gloal` 2D Real FFT `FourierUnit` + optional 16-band `RadialBandGate`) and SE channel attention.
3. **[diag_3_single_unet.png](../diagrams/diag_3_single_unet.png)**: 4-Stage clinical single-stream U-Net with residual `DoubleConv` sub-blocks and bilinear upsampling (`unet_raw` vs `unet_restored`).
4. **[diag_4_dual_stream_hybrid_unet.png](../diagrams/diag_4_dual_stream_hybrid_unet.png)**: Novel Dual-Stream Hybrid U-Net (`16.41M` params) combining warm-started Raw and Restored encoders through 5 `CrossStreamFusionBlock` per-pixel spatial gates (`w_raw + w_rest = 1`) + `HybridTopologyLoss` (`0.4 BCE + 0.4 Dice + 0.2 FP32 Soft-clDice`).
5. **[diag_5_combined_retinascan_system.png](../diagrams/diag_5_combined_retinascan_system.png)**: Complete end-to-end system connecting ingestion, SFHformer restoration, 3-model segmentation core, thread-safe Flask serving (`:5001`, 6-stage reconciled telemetry), and React RetinaScan Studio UI (`:3000`).

---

## 2. Qualitative Multi-Model Visual Evidence
* **[fives_test_disease_stratified_demo.png](../qualitative_examples/fives_test_disease_stratified_demo.png)**: Full-resolution untouched FIVES test comparison across Normal (`test_161_N`), AMD (`test_37_A`), DR (`test_100_D`), and Glaucoma (`test_133_G`), highlighting Dual-Stream capillary recovery and Glaucoma cup-rim precision preservation (`89.98%` vs `57.59%` single-stream Restored U-Net).
* **[hrf_transfer_unexposed_demo.png](../qualitative_examples/hrf_transfer_unexposed_demo.png)**: Exploratory external HRF unexposed comparison (`14_h`, `15_dr`) demonstrating both thin-vessel recovery and mixed external-domain precision trade-offs.
* **[phase5_1_qualitative_multimodel_demo.png](../qualitative_examples/phase5_1_qualitative_multimodel_demo.png)**: Studio-resolution comparison strip showing Raw, Restored, Dual-Stream, and the `scale_0` spatial gate map ($w_{\\text{rest}}$).
* **[phase5_2_ui_tab1_fusion.png](../qualitative_examples/phase5_2_ui_tab1_fusion.png)** & **[phase5_2_ui_tab2_benchmark.png](../qualitative_examples/phase5_2_ui_tab2_benchmark.png)**: Live rendered Chromium (`Edg/157.0.4341.60`) screenshots of RetinaScan Studio Tab 1 (with optional `w_rest` heatmap legend) and Tab 2 (Milestone 3 comparative matrix).

---

## 3. Live Demonstration Script (3-Minute Walkthrough)
1. **Start Services**: Flask backend (`scripts/serve_restore.py` on `:5001`) + Vite frontend (`npm run dev` on `:3000`).
2. **Default State Verification (Tab 1)**: Navigate to `http://localhost:3000/retina`. Highlight the **Research Prototype** badge, **Dual-Stream Hybrid (Default)** active, and **Frozen Validation-Calibrated Threshold (`0.40`)**.
3. **Compare All Three Models**: Switch across `Raw U-Net (Baseline)` $\to$ `Restored U-Net (Baseline)` $\to$ `Dual-Stream Hybrid (Default)` on `hrf_unexposed_14h` and `fives_normal_37db` (also toggle `3-Model Comparative Grid`).
4. **Threshold Discipline & Spatial Gate Demo**: Click `τ = 0.50 (Baseline Match)` to show threshold-matched comparison, move the exploratory slider to `0.65` to show exploratory isolation, reset to `0.40`, and toggle **Optional Spatial Gate Map (`w_rest`)** on to inspect the `0.0 -> 1.0` spatial routing legend.
5. **Scientific Transparency (Tab 2)**: Open **Milestone 3 Clinical Benchmark** to contrast primary in-domain FIVES (`0.8474 Dice`, `+10.20 pp ThinSkel`, `41.22 CCs`) against external HRF mixed trade-offs (`40` pretrain-exposed + `5` exploratory).
"""
    with open(os.path.join(DEMO_DIR, "DEMO_WALKTHROUGH.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("  Saved:", os.path.join(DEMO_DIR, "DEMO_WALKTHROUGH.md"))


if __name__ == "__main__":
    generate_qualitative_examples()
    build_benchmark_tables()
    build_reproducibility_manifest()
    build_demo_walkthrough()
    print("\nPhase 5.3 Demonstration Package built successfully at:", DEMO_DIR)

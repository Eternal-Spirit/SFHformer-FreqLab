"""
scripts/train_dual_stream_unet.py - Dual-Stream Hybrid U-Net Training & Clean Validation.

Architecture:
- Warm-starts Stream A (Raw) from checkpoints/unet_raw/best.pth
- Warm-starts Stream B (Restored) from checkpoints/unet_restored/best.pth
- Warm-starts Shared Decoder from checkpoints/unet_raw/best.pth
- Stage 1 (Warmup): Freezes encoders, trains CrossStreamFusionBlock + decoder
- Stage 2 (Joint): Unfreezes encoders with differential LR (encoders 1e-4, fusion/decoder 3e-4)
- Loss: HybridTopologyLoss (0.4 BCE + 0.4 Soft-Dice + 0.2 Soft-clDice)
- Validation: Strictly on N=59 clean FIVES validation images (full 2048x2048 sliding window)
  plus post-training threshold calibration on N=59 clean val only. Zero test set access.
"""

import os
import sys
import time
import json
import hashlib
import argparse
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.dual_stream_unet import get_dual_stream_unet
from utils.losses import (
    HybridTopologyLoss,
    compute_segmentation_metrics,
    compute_thin_vessel_sensitivity,
    compute_vascular_continuity_ratio,
)

DATA_DIR = os.path.join(REPO_DIR, "data", "fives")
PATCHES_DIR = os.path.join(DATA_DIR, "patches_512")
VAL_MANIFEST_PATH = os.path.join(PATCHES_DIR, "val_full_image_tile_manifest.json")
TRAIN_MANIFEST_PATH = os.path.join(PATCHES_DIR, "train_patch_manifest_512.json")

RAW_CKPT_DEFAULT = os.path.join(REPO_DIR, "checkpoints", "unet_raw", "best.pth")
REST_CKPT_DEFAULT = os.path.join(REPO_DIR, "checkpoints", "unet_restored", "best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(fname):
    h = hashlib.sha256()
    with open(fname, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_clDice_np(v_p, v_l):
    s_p = skeletonize(v_p).astype(bool)
    s_l = skeletonize(v_l).astype(bool)
    tprec = (s_p & v_l).sum() / max(s_p.sum(), 1)
    tsens = (s_l & v_p).sum() / max(s_l.sum(), 1)
    if tprec + tsens == 0:
        return 0.0
    return float(2.0 * tprec * tsens / (tprec + tsens))


class DualStreamRetinalPatchDataset(Dataset):
    """Synchronized paired (raw, restored, mask) 512x512 training dataset."""
    def __init__(self, manifest_path=TRAIN_MANIFEST_PATH):
        self.raw_dir = os.path.join(PATCHES_DIR, "raw", "images")
        self.rest_dir = os.path.join(PATCHES_DIR, "restored", "images")
        self.mask_dir = os.path.join(PATCHES_DIR, "masks")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.patch_ids = [p["patch_id"] for p in self.manifest]

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        im_raw = Image.open(os.path.join(self.raw_dir, f"{pid}.png")).convert("RGB")
        im_rest = Image.open(os.path.join(self.rest_dir, f"{pid}.png")).convert("RGB")
        mk = Image.open(os.path.join(self.mask_dir, f"{pid}.png")).convert("L")

        # Synchronized geometric augmentations
        if np.random.rand() > 0.5:
            im_raw = im_raw.transpose(Image.FLIP_LEFT_RIGHT)
            im_rest = im_rest.transpose(Image.FLIP_LEFT_RIGHT)
            mk = mk.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() > 0.5:
            im_raw = im_raw.transpose(Image.FLIP_TOP_BOTTOM)
            im_rest = im_rest.transpose(Image.FLIP_TOP_BOTTOM)
            mk = mk.transpose(Image.FLIP_TOP_BOTTOM)
        k_rot = np.random.randint(0, 4)
        if k_rot > 0:
            rot_op = [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][k_rot]
            im_raw = im_raw.transpose(rot_op)
            im_rest = im_rest.transpose(rot_op)
            mk = mk.transpose(rot_op)

        raw_arr = np.array(im_raw, dtype=np.float32) / 255.0
        rest_arr = np.array(im_rest, dtype=np.float32) / 255.0
        mk_arr = (np.array(mk, dtype=np.float32) > 127).astype(np.float32)

        return (
            torch.from_numpy(raw_arr).permute(2, 0, 1),
            torch.from_numpy(rest_arr).permute(2, 0, 1),
            torch.from_numpy(mk_arr).unsqueeze(0),
        )


def predict_dual_sliding_window(model, raw_arr, rest_arr, coords, tile_size, window_2d, chunk_sz=4):
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

    prob_chunks = []
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else torch.no_grad():
            for ch in range(0, len(x_raw), chunk_sz):
                logits = model(x_raw[ch:ch + chunk_sz], x_rest[ch:ch + chunk_sz])
                probs = torch.sigmoid(logits).squeeze(1).float()
                prob_chunks.append(probs)

    pred_tiles = torch.cat(prob_chunks, dim=0)
    for k, c in enumerate(coords):
        t, l = c["top"], c["left"]
        accum_prob[t:t + tile_size, l:l + tile_size] += pred_tiles[k] * window_2d
        accum_weight[t:t + tile_size, l:l + tile_size] += window_2d

    return (accum_prob / accum_weight).cpu().numpy()


def validate_dual_full_images(model, thresholds=(0.50,), return_prob_cache=False):
    """
    Evaluates full 2048x2048 FIVES validation images (N=59 clean, N=60 total)
    using 25-tile sliding window with 2D Hamming blending.
    """
    model.eval()
    with open(VAL_MANIFEST_PATH, "r") as f:
        records = json.load(f)

    raw_val_dir = os.path.join(DATA_DIR, "raw", "val", "images")
    rest_val_dir = os.path.join(DATA_DIR, "restored_val", "images")
    val_mask_dir = os.path.join(DATA_DIR, "raw", "val", "masks")
    val_fov_dir = os.path.join(DATA_DIR, "raw", "val", "fov")

    tile_size = records[0]["tile_size"]
    coords = records[0]["tile_coordinates"]

    wy = np.hamming(tile_size).astype(np.float32)
    wx = np.hamming(tile_size).astype(np.float32)
    window_2d = torch.from_numpy(np.maximum(np.outer(wy, wx), 1e-4)).to(DEVICE)

    # Precompute probability maps once per validation image
    cached_outputs = []
    for rec in records:
        img_id = rec.get("image_id") or rec.get("source_image_id")
        is_clean = rec.get("is_clean_val", True)

        raw_arr = np.array(Image.open(os.path.join(raw_val_dir, f"{img_id}.png")).convert("RGB"), dtype=np.float32) / 255.0
        rest_arr = np.array(Image.open(os.path.join(rest_val_dir, f"{img_id}.png")).convert("RGB"), dtype=np.float32) / 255.0
        mk = np.array(Image.open(os.path.join(val_mask_dir, f"{img_id}.png")).convert("L")) > 0
        fov_p = os.path.join(val_fov_dir, f"{img_id}.png")
        fov = np.array(Image.open(fov_p).convert("L")) > 0 if os.path.exists(fov_p) else (raw_arr.mean(axis=2) > 0.05)

        full_prob = predict_dual_sliding_window(model, raw_arr, rest_arr, coords, tile_size, window_2d)
        cached_outputs.append((img_id, is_clean, full_prob, mk, fov))

    results_by_thresh = {}
    for thr in thresholds:
        clean_stats = {"dice": [], "iou": [], "sensitivity": [], "specificity": [], "precision": [], "thin_sens": [], "thin_r35": [], "cldice": [], "cont": [], "cc": []}
        full_stats = {"dice": [], "iou": [], "sensitivity": [], "specificity": [], "precision": [], "thin_sens": [], "thin_r35": [], "cldice": [], "cont": [], "cc": []}

        for _, is_clean, full_prob, mk, fov in cached_outputs:
            pred_bin = (full_prob >= thr) & fov
            m = compute_segmentation_metrics(pred_bin, mk, fov_mask=fov)
            ts = compute_thin_vessel_sensitivity(pred_bin, mk, fov_mask=fov)
            # Also track radius <= 3.5px skeleton sensitivity matching evaluate_milestone3.py
            import scipy.ndimage as ndi
            skel_gt = skeletonize(mk & fov)
            dist_gt = ndi.distance_transform_edt((mk & fov) > 0)
            skel_r35 = skel_gt & (dist_gt <= 3.5)
            ts_r35 = float((pred_bin & skel_r35).sum() / max(skel_r35.sum(), 1))
            cld = compute_clDice_np(pred_bin, mk & fov)
            cont, n_pred_cc, _ = compute_vascular_continuity_ratio(pred_bin, mk, fov_mask=fov)

            for bucket in ([full_stats, clean_stats] if is_clean else [full_stats]):
                bucket["dice"].append(m["dice"])
                bucket["iou"].append(m["iou"])
                bucket["sensitivity"].append(m["sensitivity"])
                bucket["specificity"].append(m["specificity"])
                bucket["precision"].append(m["precision"])
                bucket["thin_sens"].append(ts)
                bucket["thin_r35"].append(ts_r35)
                bucket["cldice"].append(cld)
                bucket["cont"].append(cont)
                bucket["cc"].append(n_pred_cc)

        def summarize(b):
            dice_m = float(np.mean(b["dice"]))
            cld_m = float(np.mean(b["cldice"]))
            thin_m = float(np.mean(b["thin_sens"]))
            comp = 0.40 * dice_m + 0.30 * cld_m + 0.30 * thin_m
            return {
                "dice": round(dice_m, 4),
                "iou": round(float(np.mean(b["iou"])), 4),
                "sensitivity": round(float(np.mean(b["sensitivity"])), 4),
                "specificity": round(float(np.mean(b["specificity"])), 4),
                "precision": round(float(np.mean(b["precision"])), 4),
                "thin_vessel_sensitivity": round(thin_m, 4),
                "thin_skel_sens_r35": round(float(np.mean(b["thin_r35"])), 4),
                "cldice": round(cld_m, 4),
                "vascular_continuity_ratio": round(float(np.mean(b["cont"])), 2),
                "mean_pred_cc": round(float(np.mean(b["cc"])), 1),
                "composite_score": round(comp, 4),
            }

        results_by_thresh[thr] = {
            "clean_n59": summarize(clean_stats),
            "full_n60": summarize(full_stats),
        }

    if len(thresholds) == 1 and not return_prob_cache:
        return results_by_thresh[thresholds[0]]
    return results_by_thresh


def train_dual_stream(epochs=12, warmup_epochs=4, batch_size=2, accum_steps=2, lr_warmup=5e-4, lr_enc=1e-4, lr_head=3e-4):
    print("=" * 80)
    print(" TRAINING DUAL-STREAM HYBRID U-NET (RAW MACRO + RESTORED CAPILLARY) ")
    print(f" Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")
    print(f" Precision: BF16 AMP | Batch Size: {batch_size} (Accum={accum_steps}, Eff={batch_size*accum_steps})")
    print(f" Schedule: {warmup_epochs} Warmup Epochs (Frozen Encoders) + {epochs - warmup_epochs} Joint Epochs")
    print("=" * 80)

    if DEVICE.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.65, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    out_dir = os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream")
    os.makedirs(out_dir, exist_ok=True)

    dataset = DualStreamRetinalPatchDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

    model = get_dual_stream_unet(in_channels=3, out_channels=1, base_channels=32).to(DEVICE)
    n_loaded, n_missing = model.load_pretrained_streams(
        RAW_CKPT_DEFAULT, REST_CKPT_DEFAULT, init_decoder_from="raw", map_location="cpu"
    )
    print(f"[Init] Loaded {n_loaded} pretrained tensors (Raw + Restored encoders, Raw decoder); {n_missing} fusion tensors initialized.")

    criterion = HybridTopologyLoss(bce_weight=0.4, dice_weight=0.4, cldice_weight=0.2, skel_iter=12).to(DEVICE)

    # Stage 1: Warmup optimizer (fusion + decoder only)
    model.freeze_encoders()
    fusion_and_dec_params = [
        p for n, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(fusion_and_dec_params, lr=lr_warmup, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(warmup_epochs, 1), eta_min=1e-4)

    best_composite = -1.0
    best_clean_dice = -1.0
    history = []
    t_start = time.time()

    for ep in range(1, epochs + 1):
        if ep == warmup_epochs + 1:
            print(f"\n[Stage 2 Transition @ Epoch {ep}] Unfreezing Dual Encoders (lr_enc={lr_enc}, lr_head={lr_head})...")
            model.unfreeze_encoders()
            enc_params = []
            head_params = []
            for n, p in model.named_parameters():
                if n.startswith("raw_downs.") or n.startswith("raw_bottleneck.") or n.startswith("rest_downs.") or n.startswith("rest_bottleneck."):
                    enc_params.append(p)
                else:
                    head_params.append(p)
            optimizer = torch.optim.AdamW(
                [
                    {"params": enc_params, "lr": lr_enc},
                    {"params": head_params, "lr": lr_head},
                ],
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-5
            )

        model.train()
        ep_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)

        t_ep = time.time()
        for step, (raw_im, rest_im, masks) in enumerate(loader):
            raw_im = raw_im.to(DEVICE, non_blocking=True)
            rest_im = rest_im.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE.type == "cuda" else torch.no_grad():
                logits = model(raw_im, rest_im)
                loss = criterion(logits, masks) / accum_steps

            loss.backward()
            ep_loss += loss.item() * accum_steps
            step_count += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        ep_loss_avg = ep_loss / max(step_count, 1)
        ep_time = time.time() - t_ep

        if ep % 2 == 0 or ep == warmup_epochs or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} [Loss: {ep_loss_avg:.4f}, Time: {ep_time:.1f}s] -> Validating on N=59 Clean / N=60 Full...")
            val_res = validate_dual_full_images(model, thresholds=(0.50,))
            c59 = val_res["clean_n59"]
            print(
                f"  * Clean Val (N=59): Comp={c59['composite_score']:.4f} | Dice={c59['dice']:.4f} | "
                f"Prec={c59['precision']:.4f} | ThinSens={c59['thin_vessel_sensitivity']:.4f} | "
                f"clDice={c59['cldice']:.4f} | CC={c59['mean_pred_cc']}"
            )
            history.append({"epoch": ep, "train_loss": round(ep_loss_avg, 4), "val_clean_n59": c59, "val_full_n60": val_res["full_n60"]})

            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "val_metrics": val_res,
            }, os.path.join(out_dir, "last.pth"))

            if c59["composite_score"] > best_composite:
                best_composite = c59["composite_score"]
                best_clean_dice = c59["dice"]
                torch.save({
                    "epoch": ep,
                    "model": model.state_dict(),
                    "best_composite": best_composite,
                    "best_clean_dice": best_clean_dice,
                    "val_metrics": val_res,
                }, os.path.join(out_dir, "best.pth"))
                print(f"  >>> New Best Checkpoint Saved! (Clean Composite: {best_composite:.4f}, Dice: {best_clean_dice:.4f})")
        else:
            print(f"Epoch {ep:02d}/{epochs:02d} [Loss: {ep_loss_avg:.4f}, Time: {ep_time:.1f}s]")

    total_mins = (time.time() - t_start) / 60.0
    print(f"\n[Training Complete] Best Clean Composite={best_composite:.4f} (Dice={best_clean_dice:.4f}) in {total_mins:.2f} mins.")

    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Clean-Validation-Only Threshold Calibration
    print("\n[Calibration] Sweeping threshold strictly on N=59 clean validation set...")
    best_ck = torch.load(os.path.join(out_dir, "best.pth"), map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ck["model"])
    cand_thresholds = (0.40, 0.45, 0.50, 0.55, 0.60)
    sweep_res = validate_dual_full_images(model, thresholds=cand_thresholds)

    best_thr = 0.50
    best_thr_comp = -1.0
    for thr in cand_thresholds:
        m = sweep_res[thr]["clean_n59"]
        print(f"  tau={thr:.2f} -> Comp={m['composite_score']:.4f} | Dice={m['dice']:.4f} | ThinSens={m['thin_vessel_sensitivity']:.4f} | clDice={m['cldice']:.4f} | Prec={m['precision']:.4f}")
        if m["composite_score"] > best_thr_comp:
            best_thr_comp = m["composite_score"]
            best_thr = thr

    calib = {
        "best_checkpoint_epoch": best_ck["epoch"],
        "best_checkpoint_sha256": sha256_file(os.path.join(out_dir, "best.pth")),
        "selected_frozen_threshold": best_thr,
        "validation_sweep_clean_n59": {str(k): v["clean_n59"] for k, v in sweep_res.items()},
    }
    with open(os.path.join(out_dir, "validation_calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"[Calibration Frozen] Optimal Clean Val Threshold = {best_thr:.2f} | SHA-256: {calib['best_checkpoint_sha256'][:16]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    args = parser.parse_args()

    train_dual_stream(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
    )


if __name__ == "__main__":
    main()

"""
scripts/train_unet.py - Milestone 2 U-Net Training & Validation Pipeline.
Implements:
- Experiment A: Raw fundus patches -> U-Net
- Experiment B: SFHFormer-restored patches -> U-Net
- Hardware safety: RTX 5060, BF16 AMP, 5.57 GB hard allocator cap (70%), batch size 2, accum 2.
- Clean validation benchmark on 59 clean images (N=59) + 60-image sensitivity check.
- Metrics: Dice, IoU, Sensitivity, Specificity, Precision, Thin-Vessel Sensitivity (<= 3.5 px), Vascular Continuity Ratio.
"""

import os
import sys
import time
import json
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.unet import get_unet
from utils.losses import BCEDiceLoss, compute_segmentation_metrics, compute_thin_vessel_sensitivity, compute_vascular_continuity_ratio

DATA_DIR = os.path.join(REPO_DIR, "data", "fives")
PATCHES_DIR = os.path.join(DATA_DIR, "patches_512")
VAL_MANIFEST_PATH = os.path.join(PATCHES_DIR, "val_full_image_tile_manifest.json")
TRAIN_MANIFEST_PATH = os.path.join(PATCHES_DIR, "train_patch_manifest_512.json")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RetinalPatchDataset(Dataset):
    def __init__(self, mode="raw"):
        self.mode = mode
        self.img_dir = os.path.join(PATCHES_DIR, mode, "images")
        self.mask_dir = os.path.join(PATCHES_DIR, "masks")
        
        with open(TRAIN_MANIFEST_PATH, 'r') as f:
            self.manifest = json.load(f)
            
        self.patch_ids = [p["patch_id"] for p in self.manifest]
        
    def __len__(self):
        return len(self.patch_ids)
        
    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        img_p = os.path.join(self.img_dir, f"{pid}.png")
        mk_p = os.path.join(self.mask_dir, f"{pid}.png")
        
        im = Image.open(img_p).convert('RGB')
        mk = Image.open(mk_p).convert('L')
        
        # Simple data augmentation for training
        if np.random.rand() > 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            mk = mk.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() > 0.5:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            mk = mk.transpose(Image.FLIP_TOP_BOTTOM)
            
        im_arr = np.array(im, dtype=np.float32) / 255.0
        mk_arr = (np.array(mk, dtype=np.float32) > 127).astype(np.float32)
        
        img_tensor = torch.from_numpy(im_arr).permute(2, 0, 1) # (3, H, W)
        mask_tensor = torch.from_numpy(mk_arr).unsqueeze(0)   # (1, H, W)
        return img_tensor, mask_tensor

def validate_full_images(model, mode="raw", threshold=0.5):
    """
    Evaluates full 2048x2048 validation images using 25-tile sliding window with 2D Hamming blending.
    Reports clean validation metrics (N=59) and full sensitivity metrics (N=60).
    """
    model.eval()
    with open(VAL_MANIFEST_PATH, 'r') as f:
        records = json.load(f)
        
    val_img_dir = os.path.join(DATA_DIR, "raw", "val", "images") if mode == "raw" else os.path.join(DATA_DIR, "restored_val", "images")
    val_mask_dir = os.path.join(DATA_DIR, "raw", "val", "masks")
    val_fov_dir = os.path.join(DATA_DIR, "raw", "val", "fov")
    
    tile_size = records[0]["tile_size"]
    coords = records[0]["tile_coordinates"]
    
    wy = np.hamming(tile_size)
    wx = np.hamming(tile_size)
    window_2d = torch.from_numpy(np.outer(wy, wx).astype(np.float32)).to(DEVICE)
    
    clean_dices, full_dices = [], []
    clean_ious, full_ious = [], []
    clean_sens, full_sens = [], []
    clean_spec, full_spec = [], []
    clean_prec, full_prec = [], []
    clean_thin_sens, full_thin_sens = [], []
    clean_cont, full_cont = [], []
    
    with torch.no_grad():
        for rec in records:
            img_id = rec.get("image_id") or rec.get("source_image_id")
            is_clean = rec.get("is_clean_val", True)
            
            img_p = os.path.join(val_img_dir, f"{img_id}.png")
            mk_p = os.path.join(val_mask_dir, f"{img_id}.png")
            fov_p = os.path.join(val_fov_dir, f"{img_id}.png")
            
            im_arr = np.array(Image.open(img_p).convert('RGB'), dtype=np.float32) / 255.0
            mk = np.array(Image.open(mk_p).convert('L')) > 0
            fov = np.array(Image.open(fov_p).convert('L')) > 0 if os.path.exists(fov_p) else (im_arr.mean(axis=2) > 0.05)
            
            H, W = im_arr.shape[:2]
            accum_prob = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
            accum_weight = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
            
            # Tile extraction
            tile_tensors = []
            for c in coords:
                t, l = c["top"], c["left"]
                tile = im_arr[t:t+tile_size, l:l+tile_size]
                tile_tensors.append(torch.from_numpy(tile).permute(2, 0, 1))
            x_tiles = torch.stack(tile_tensors, dim=0).to(DEVICE)
            
            # Predict in chunks
            chunk_sz = 5
            prob_chunks = []
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for ch in range(0, len(x_tiles), chunk_sz):
                    logits = model(x_tiles[ch:ch+chunk_sz])
                    probs = torch.sigmoid(logits).squeeze(1).float()
                    prob_chunks.append(probs)
            pred_tiles = torch.cat(prob_chunks, dim=0) # (25, 512, 512)
            
            # Blend
            for k, c in enumerate(coords):
                t, l = c["top"], c["left"]
                accum_prob[t:t+tile_size, l:l+tile_size] += pred_tiles[k] * window_2d
                accum_weight[t:t+tile_size, l:l+tile_size] += window_2d
                
            full_prob = (accum_prob / accum_weight).cpu().numpy()
            pred_binary = (full_prob >= threshold) & fov
            
            m = compute_segmentation_metrics(pred_binary, mk, fov_mask=fov)
            ts = compute_thin_vessel_sensitivity(pred_binary, mk, fov_mask=fov)
            cont, _, _ = compute_vascular_continuity_ratio(pred_binary, mk, fov_mask=fov)
            
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
                
    return {
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

def train_unet(mode="raw", epochs=15, batch_size=2, accum_steps=2, lr=1e-3):
    print("=" * 80)
    print(f" TRAINING U-NET: EXPERIMENT {'A (RAW)' if mode=='raw' else 'B (SFHFORMER-RESTORED)'} ")
    print(f" Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'CPU'})")
    print(f" Precision: BF16 AMP | Batch Size: {batch_size} (Accum={accum_steps}, Effective={batch_size*accum_steps})")
    print("=" * 80)
    
    if DEVICE.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.70, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    out_dir = os.path.join(REPO_DIR, "checkpoints", f"unet_{mode}")
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = RetinalPatchDataset(mode=mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    
    model = get_unet(in_channels=3, out_channels=1, base_channels=32).to(DEVICE)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_clean_dice = -1.0
    history = []
    
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        step_count = 0
        optimizer.zero_grad(set_to_none=True)
        
        t_ep = time.time()
        for step, (images, masks) in enumerate(loader):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
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
        
        # Validation every 2 epochs or on final epoch
        if ep % 2 == 0 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs:02d} [Loss: {ep_loss_avg:.4f}, Time: {ep_time:.1f}s] -> Running 60-image Full Validation...")
            val_res = validate_full_images(model, mode=mode, threshold=0.5)
            clean_dice = val_res["clean_n59"]["dice"]
            clean_thin = val_res["clean_n59"]["thin_vessel_sensitivity"]
            clean_cont = val_res["clean_n59"]["vascular_continuity_ratio"]
            print(f"  * Clean Val (N=59): Dice = {clean_dice:.4f} | Thin Sens = {clean_thin:.4f} | Cont Ratio = {clean_cont:.2f}")
            print(f"  * Full Val  (N=60): Dice = {val_res['full_n60']['dice']:.4f} | Thin Sens = {val_res['full_n60']['thin_vessel_sensitivity']:.4f} | Cont Ratio = {val_res['full_n60']['vascular_continuity_ratio']:.2f}")
            
            entry = {
                "epoch": ep,
                "train_loss": round(ep_loss_avg, 4),
                "val_clean_n59": val_res["clean_n59"],
                "val_full_n60": val_res["full_n60"]
            }
            history.append(entry)
            
            # Always save latest
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_metrics": val_res
            }, os.path.join(out_dir, "last.pth"))
            
            if clean_dice > best_clean_dice:
                best_clean_dice = clean_dice
                ckpt_p = os.path.join(out_dir, "best.pth")
                torch.save({
                    "epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_clean_dice": best_clean_dice,
                    "val_metrics": val_res
                }, ckpt_p)
                print(f"  >>> New Best Checkpoint Saved! (Clean Dice: {best_clean_dice:.4f})")
        else:
            print(f"Epoch {ep:02d}/{epochs:02d} [Loss: {ep_loss_avg:.4f}, Time: {ep_time:.1f}s]")
            
    total_time = time.time() - t_start
    print(f"\n[Finished {mode.upper()}] Best Clean Validation Dice: {best_clean_dice:.4f} in {total_time/60.0:.2f} mins.")
    
    # Save history
    with open(os.path.join(out_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    return best_clean_dice, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["raw", "restored", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=2)
    args = parser.parse_args()
    
    if args.mode in ["raw", "both"]:
        train_unet(mode="raw", epochs=args.epochs, batch_size=args.batch_size, accum_steps=args.accum_steps)
    if args.mode in ["restored", "both"]:
        train_unet(mode="restored", epochs=args.epochs, batch_size=args.batch_size, accum_steps=args.accum_steps)

if __name__ == '__main__':
    main()

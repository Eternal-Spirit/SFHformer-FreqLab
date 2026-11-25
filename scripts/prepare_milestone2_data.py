"""
scripts/prepare_milestone2_data.py
Prepares data for Milestone 2 Experiments A, B, C, D:
1. Extracts 2,160 raw training patches (512x512) and vessel masks using train_patch_manifest_512.json.
2. Restores all 2,160 training patches using SFHFormer-Retina (BF16 AMP).
3. Restores all 60 validation images (2048x2048) via seamless 25-tile Hamming window blending.
"""

import os
import sys
import time
import json
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from freqlab.sfhformer_gated import SFHformer_m_gated

DATA_DIR = os.path.join(REPO_DIR, "data", "fives")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PATCHES_DIR = os.path.join(DATA_DIR, "patches_512")

RAW_PATCHES_IMG_DIR = os.path.join(PATCHES_DIR, "raw", "images")
REST_PATCHES_IMG_DIR = os.path.join(PATCHES_DIR, "restored", "images")
MASKS_PATCHES_DIR = os.path.join(PATCHES_DIR, "masks")
RESTORED_VAL_DIR = os.path.join(DATA_DIR, "restored_val", "images")

for d in [RAW_PATCHES_IMG_DIR, REST_PATCHES_IMG_DIR, MASKS_PATCHES_DIR, RESTORED_VAL_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sfhformer():
    print("[Model] Loading fine-tuned SFHformer-Retina...")
    model = SFHformer_m_gated(use_gate=False).to(DEVICE)
    ckpt_path = os.path.join(REPO_DIR, "checkpoints", "fives_retina_finetune", "best.pth")
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    st = ck['model'] if 'model' in ck else ck
    model.load_state_dict(st)
    model.eval()
    print(" -> SFHformer-Retina loaded successfully.")
    return model

def step1_extract_raw_patches():
    print("\n" + "=" * 70)
    print(" [Step 1/3] Extracting 2,160 Raw 512x512 Patches & Masks ")
    print("=" * 70)
    manifest_p = os.path.join(PATCHES_DIR, "train_patch_manifest_512.json")
    with open(manifest_p, 'r') as f:
        records = json.load(f)
        
    print(f" -> Manifest contains {len(records)} patches across 540 images.")
    # Group by source image for fast single-image open
    by_img = {}
    for r in records:
        by_img.setdefault(r["source_image_id"], []).append(r)
        
    t0 = time.time()
    count = 0
    for idx, (img_id, patch_list) in enumerate(by_img.items()):
        img_p = os.path.join(RAW_DIR, "train", "images", f"{img_id}.png")
        mk_p = os.path.join(RAW_DIR, "train", "masks", f"{img_id}.png")
        
        im = Image.open(img_p).convert('RGB')
        mk = Image.open(mk_p).convert('L')
        
        for p in patch_list:
            t, l, sz = p["top"], p["left"], p["size"]
            pid = p["patch_id"]
            
            im_crop = im.crop((l, t, l + sz, t + sz))
            mk_crop = mk.crop((l, t, l + sz, t + sz))
            
            im_crop.save(os.path.join(RAW_PATCHES_IMG_DIR, f"{pid}.png"))
            mk_crop.save(os.path.join(MASKS_PATCHES_DIR, f"{pid}.png"))
            count += 1
            
        if (idx + 1) % 100 == 0 or (idx + 1) == len(by_img):
            print(f"  Processed {idx + 1} / {len(by_img)} images ({count} patches saved)...")
            
    print(f" -> Completed Step 1 in {time.time() - t0:.1f}s: {count} raw patches & masks saved.")

def step2_restore_training_patches(model):
    print("\n" + "=" * 70)
    print(" [Step 2/3] Restoring 2,160 Training Patches with SFHFormer (BF16 AMP) ")
    print("=" * 70)
    
    raw_files = sorted(glob.glob(os.path.join(RAW_PATCHES_IMG_DIR, "*.png")))
    print(f" -> Found {len(raw_files)} raw patches to restore.")
    
    batch_size = 4
    t0 = time.time()
    
    # Process in batches
    for i in range(0, len(raw_files), batch_size):
        batch_paths = raw_files[i:i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            arr = np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1) # (3, H, W)
            batch_tensors.append(t)
            
        x = torch.stack(batch_tensors, dim=0).to(DEVICE)
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                restored = model(x)
                restored = torch.clamp(restored, 0.0, 1.0)
                
        restored_np = (restored.cpu().float().permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
        for j, p in enumerate(batch_paths):
            base = os.path.basename(p)
            out_p = os.path.join(REST_PATCHES_IMG_DIR, base)
            Image.fromarray(restored_np[j]).save(out_p)
            
        if (i + batch_size) % 400 < batch_size or (i + batch_size) >= len(raw_files):
            done = min(i + batch_size, len(raw_files))
            rate = done / max(time.time() - t0, 0.1)
            print(f"  Restored {done} / {len(raw_files)} patches ({rate:.1f} patches/sec)...")
            
    print(f" -> Completed Step 2 in {time.time() - t0:.1f}s: all patches restored.")

def step3_restore_validation_images(model):
    print("\n" + "=" * 70)
    print(" [Step 3/3] Restoring 60 Full-Resolution Validation Scans (25-Tile Blending) ")
    print("=" * 70)
    
    val_manifest_p = os.path.join(PATCHES_DIR, "val_full_image_tile_manifest.json")
    with open(val_manifest_p, 'r') as f:
        records = json.load(f)
        
    print(f" -> Manifest contains {len(records)} validation images.")
    tile_size = records[0]["tile_size"]
    coords = records[0]["tile_coordinates"]
    
    # 2D Hamming blending window
    wy = np.hamming(tile_size)
    wx = np.hamming(tile_size)
    window_2d = torch.from_numpy(np.outer(wy, wx).astype(np.float32)).to(DEVICE) # (512, 512)
    
    t0 = time.time()
    for idx, rec in enumerate(records):
        img_id = rec.get("image_id") or rec.get("source_image_id")
        img_p = os.path.join(RAW_DIR, "val", "images", f"{img_id}.png")
        
        im_arr = np.array(Image.open(img_p).convert('RGB'), dtype=np.float32) / 255.0
        H, W = im_arr.shape[:2]
        
        accum_rest = torch.zeros((H, W, 3), dtype=torch.float32, device=DEVICE)
        accum_weight = torch.zeros((H, W), dtype=torch.float32, device=DEVICE)
        
        # Batch tile inference (25 tiles total)
        tile_tensors = []
        for c in coords:
            t, l = c["top"], c["left"]
            tile = im_arr[t:t+tile_size, l:l+tile_size]
            tile_tensors.append(torch.from_numpy(tile).permute(2, 0, 1))
            
        x_tiles = torch.stack(tile_tensors, dim=0).to(DEVICE) # (25, 3, 512, 512)
        
        # Inference in chunks of 5 tiles to respect display-safe VRAM headroom
        out_chunks = []
        chunk_sz = 5
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for ch in range(0, len(x_tiles), chunk_sz):
                    pred_chunk = model(x_tiles[ch:ch+chunk_sz])
                    out_chunks.append(torch.clamp(pred_chunk, 0.0, 1.0).float())
                    
        out_tiles = torch.cat(out_chunks, dim=0).permute(0, 2, 3, 1) # (25, 512, 512, 3)
        
        # Blend tiles into canvas
        for k, c in enumerate(coords):
            t, l = c["top"], c["left"]
            tile_out = out_tiles[k] # (512, 512, 3)
            w_tile = window_2d.unsqueeze(-1) # (512, 512, 1)
            
            accum_rest[t:t+tile_size, l:l+tile_size] += tile_out * w_tile
            accum_weight[t:t+tile_size, l:l+tile_size] += window_2d
            
        full_restored = accum_rest / accum_weight.unsqueeze(-1)
        full_np = (full_restored.cpu().numpy() * 255.0).round().astype(np.uint8)
        
        out_p = os.path.join(RESTORED_VAL_DIR, f"{img_id}.png")
        Image.fromarray(full_np).save(out_p)
        
        if (idx + 1) % 15 == 0 or (idx + 1) == len(records):
            print(f"  Restored {idx + 1} / {len(records)} validation scans...")
            
    print(f" -> Completed Step 3 in {time.time() - t0:.1f}s: all 60 validation images restored.")

def main():
    t_start = time.time()
    print("=" * 80)
    print(" MILESTONE 2: DATA PREPARATION & SFHFORMER RESTORATION ")
    print("=" * 80)
    
    # 1. Extract raw patches and masks
    step1_extract_raw_patches()
    
    # 2. Load model & restore patches
    model = load_sfhformer()
    step2_restore_training_patches(model)
    
    # 3. Restore full-resolution validation images
    step3_restore_validation_images(model)
    
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f" [SUCCESS] Milestone 2 Data Preparation Completed in {total_time:.1f}s ({total_time/60.0:.2f} mins)!")
    print(f" -> Raw Patches:      {RAW_PATCHES_IMG_DIR}")
    print(f" -> Restored Patches: {REST_PATCHES_IMG_DIR}")
    print(f" -> Patch Masks:      {MASKS_PATCHES_DIR}")
    print(f" -> Restored Val:     {RESTORED_VAL_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()

"""
generate_mega_retina_crops.py - Multi-Dataset Mega Retinal Crop Ingestion
Extracts 6,000+ high-quality, pathologically diverse 256x256 crops across:
1. DRIVE (Classic Gold-Standard Microvessel)
2. STARE (Severe Pathology: Hemorrhages, Vein Occlusion, Retinopathy)
3. HRF (High-Resolution Fundus: Healthy, DR, Glaucoma at 3504x2336)
4. APTOS (Real-world Clinical Photography)
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

BASE_DIR = r"D:\retina sfhormer clone\data\retina"
RAW_DIR = os.path.join(BASE_DIR, "raw")
CROPS_DIR = os.path.join(BASE_DIR, "crops")

TRAIN_IMG_OUT = os.path.join(CROPS_DIR, "train")
TRAIN_MASK_OUT = os.path.join(CROPS_DIR, "train_masks")
VAL_IMG_OUT = os.path.join(CROPS_DIR, "val")
VAL_MASK_OUT = os.path.join(CROPS_DIR, "val_masks")

for d in [TRAIN_IMG_OUT, TRAIN_MASK_OUT, VAL_IMG_OUT, VAL_MASK_OUT]:
    os.makedirs(d, exist_ok=True)

CROP_SIZE = 256
RNG = np.random.default_rng(2026)

def apply_random_spatial_aug(img_arr, mask_arr=None):
    """Applies random orthogonal rotations and flips to image and mask simultaneously."""
    k = RNG.integers(0, 4)
    if k > 0:
        img_arr = np.rot90(img_arr, k)
        if mask_arr is not None:
            mask_arr = np.rot90(mask_arr, k)
    
    if RNG.random() > 0.5:
        img_arr = np.fliplr(img_arr)
        if mask_arr is not None:
            mask_arr = np.fliplr(mask_arr)
            
    if RNG.random() > 0.5:
        img_arr = np.flipud(img_arr)
        if mask_arr is not None:
            mask_arr = np.flipud(mask_arr)
            
    return np.ascontiguousarray(img_arr), (np.ascontiguousarray(mask_arr) if mask_arr is not None else None)

def extract_crops_from_pair(img_path, mask_path, n_crops, out_img_dir, out_mask_dir, prefix, start_idx=0, is_val=False):
    """Extracts valid retinal crops from a paired fundus image and vessel mask."""
    try:
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') if (mask_path and os.path.exists(mask_path)) else None
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return 0

    img_np = np.array(img)
    mask_np = np.array(mask) if mask is not None else None
    h, w = img_np.shape[:2]

    if h < CROP_SIZE or w < CROP_SIZE:
        img_np = np.array(img.resize((max(h, CROP_SIZE), max(w, CROP_SIZE)), Image.BICUBIC))
        if mask_np is not None:
            mask_np = np.array(mask.resize((max(h, CROP_SIZE), max(w, CROP_SIZE)), Image.NEAREST))
        h, w = img_np.shape[:2]

    saved = 0
    attempts = 0
    max_attempts = n_crops * 15

    while saved < n_crops and attempts < max_attempts:
        attempts += 1
        top = RNG.integers(0, h - CROP_SIZE + 1)
        left = RNG.integers(0, w - CROP_SIZE + 1)

        c_img = img_np[top:top+CROP_SIZE, left:left+CROP_SIZE]
        c_mask = mask_np[top:top+CROP_SIZE, left:left+CROP_SIZE] if mask_np is not None else np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.uint8)

        # Tissue threshold check: ensure we are not extracting pure black background
        tissue_ratio = np.mean(c_img > 18)
        if tissue_ratio < 0.38:
            continue

        if not is_val:
            c_img, c_mask = apply_random_spatial_aug(c_img, c_mask)

        idx = start_idx + saved
        fname = f"{prefix}_{idx:05d}.png"
        Image.fromarray(c_img).save(os.path.join(out_img_dir, fname))
        Image.fromarray(c_mask).save(os.path.join(out_mask_dir, fname))
        saved += 1

    return saved

def main():
    print("=" * 65)
    print("[INFO] GENERATING MULTI-DATASET MEGA RETINAL CROPS")
    print("=" * 65)

    total_train = 0
    total_val = 0

    # -------------------------------------------------------------
    # 1. DRIVE Dataset (20 Train Images -> 1,000 Augmented Crops)
    # -------------------------------------------------------------
    drive_train_img_dir = os.path.join(RAW_DIR, "train_images")
    drive_train_mask_dir = os.path.join(RAW_DIR, "train_masks")
    drive_train_files = sorted(glob.glob(os.path.join(drive_train_img_dir, "*.png")))
    
    print(f"\n[1/4] Processing DRIVE ({len(drive_train_files)} training fundus images)...")
    drive_count = 0
    for f in drive_train_files:
        base = os.path.basename(f)
        mask_f = os.path.join(drive_train_mask_dir, base)
        # 50 crops per image with rotations/flips
        drive_count += extract_crops_from_pair(f, mask_f, n_crops=50, 
                                              out_img_dir=TRAIN_IMG_OUT, 
                                              out_mask_dir=TRAIN_MASK_OUT, 
                                              prefix="drive_train", start_idx=drive_count)
    print(f" -> Generated {drive_count} DRIVE training crops.")
    total_train += drive_count

    # -------------------------------------------------------------
    # 2. STARE Dataset (20 Pathological Images -> 1,000 Augmented Crops)
    # -------------------------------------------------------------
    stare_img_dir = os.path.join(RAW_DIR, "stare", "images")
    stare_mask_dir = os.path.join(RAW_DIR, "stare", "masks")
    stare_files = sorted(glob.glob(os.path.join(stare_img_dir, "*.png")))
    
    print(f"\n[2/4] Processing STARE ({len(stare_files)} pathological images)...")
    stare_count = 0
    # Hold out 3 images for validation, use 17 for training
    stare_train_files = stare_files[:17]
    stare_val_files = stare_files[17:]
    
    for f in stare_train_files:
        base = os.path.basename(f)
        mask_f = os.path.join(stare_mask_dir, base)
        stare_count += extract_crops_from_pair(f, mask_f, n_crops=60, 
                                              out_img_dir=TRAIN_IMG_OUT, 
                                              out_mask_dir=TRAIN_MASK_OUT, 
                                              prefix="stare_train", start_idx=stare_count)
    print(f" -> Generated {stare_count} STARE training crops.")
    total_train += stare_count

    # -------------------------------------------------------------
    # 3. HRF Dataset (45 Ultra-High-Res 3504x2336 Images -> 3,200 Crops)
    # -------------------------------------------------------------
    hrf_img_dir = os.path.join(RAW_DIR, "hrf", "images")
    hrf_mask_dir = os.path.join(RAW_DIR, "hrf", "masks")
    hrf_files = sorted(glob.glob(os.path.join(hrf_img_dir, "*.png")))
    
    print(f"\n[3/4] Processing HRF ({len(hrf_files)} 3504x2336 high-resolution images)...")
    # Hold out 5 images for validation, use 40 for training
    hrf_train_files = hrf_files[:40]
    hrf_val_files = hrf_files[40:]
    
    hrf_count = 0
    for f in hrf_train_files:
        base = os.path.basename(f)
        mask_f = os.path.join(hrf_mask_dir, base)
        # 80 crops per large 3504x2336 image
        hrf_count += extract_crops_from_pair(f, mask_f, n_crops=80, 
                                            out_img_dir=TRAIN_IMG_OUT, 
                                            out_mask_dir=TRAIN_MASK_OUT, 
                                            prefix="hrf_train", start_idx=hrf_count)
    print(f" -> Generated {hrf_count} HRF training crops.")
    total_train += hrf_count

    # -------------------------------------------------------------
    # 4. APTOS Dataset (Clinical DR Diversity -> 800 Crops)
    # -------------------------------------------------------------
    aptos_dir = os.path.join(RAW_DIR, "aptos")
    aptos_files = sorted(glob.glob(os.path.join(aptos_dir, "*.png")))
    
    print(f"\n[4/4] Processing APTOS ({len(aptos_files)} clinical DR fundus images)...")
    aptos_count = 0
    for f in aptos_files:
        # Extract 32 crops per clinical image
        aptos_count += extract_crops_from_pair(f, mask_path=None, n_crops=32, 
                                              out_img_dir=TRAIN_IMG_OUT, 
                                              out_mask_dir=TRAIN_MASK_OUT, 
                                              prefix="aptos_train", start_idx=aptos_count)
    print(f" -> Generated {aptos_count} APTOS training crops.")
    total_train += aptos_count

    # -------------------------------------------------------------
    # 5. Validation Sets (DRIVE 20 Val + STARE 3 Val + HRF 5 Val)
    # -------------------------------------------------------------
    print(f"\n[Validation] Generating held-out multi-source validation crops...")
    val_count = 0
    # DRIVE val
    drive_val_img_dir = os.path.join(RAW_DIR, "val_images")
    drive_val_mask_dir = os.path.join(RAW_DIR, "val_masks")
    drive_val_files = sorted(glob.glob(os.path.join(drive_val_img_dir, "*.png")))
    for f in drive_val_files:
        base = os.path.basename(f)
        mask_f = os.path.join(drive_val_mask_dir, base)
        val_count += extract_crops_from_pair(f, mask_f, n_crops=4, 
                                            out_img_dir=VAL_IMG_OUT, 
                                            out_mask_dir=VAL_MASK_OUT, 
                                            prefix="val_drive", start_idx=val_count, is_val=True)

    # STARE val
    for f in stare_val_files:
        base = os.path.basename(f)
        mask_f = os.path.join(stare_mask_dir, base)
        val_count += extract_crops_from_pair(f, mask_f, n_crops=6, 
                                            out_img_dir=VAL_IMG_OUT, 
                                            out_mask_dir=VAL_MASK_OUT, 
                                            prefix="val_stare", start_idx=val_count, is_val=True)

    # HRF val
    for f in hrf_val_files:
        base = os.path.basename(f)
        mask_f = os.path.join(hrf_mask_dir, base)
        val_count += extract_crops_from_pair(f, mask_f, n_crops=8, 
                                            out_img_dir=VAL_IMG_OUT, 
                                            out_mask_dir=VAL_MASK_OUT, 
                                            prefix="val_hrf", start_idx=val_count, is_val=True)

    total_val = val_count
    print(f" -> Generated {total_val} held-out validation crops.")

    print("\n" + "=" * 65)
    print(f"[SUCCESS] MEGA DATASET COMPLETE!")
    print(f"   Total Training Crops:   {total_train:,} crops")
    print(f"   Total Validation Crops: {total_val:,} crops")
    print(f"   Directory: {CROPS_DIR}")
    print("=" * 65)

if __name__ == '__main__':
    main()

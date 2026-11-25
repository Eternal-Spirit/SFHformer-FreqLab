"""
prepare_retina_data.py - Ingest DRIVE retinal fundus dataset, generate matched
256x256 crops and vessel ground-truth masks, and prepare sample gallery images.
"""

import os
import sys
import io
import time
import urllib.request
import numpy as np
from PIL import Image

BASE_DIR = r"D:\SFHformer-FreqLab\data\retina"
RAW_DIR = os.path.join(BASE_DIR, "raw")
CROPS_DIR = os.path.join(BASE_DIR, "crops")

TRAIN_IMG_DIR = os.path.join(RAW_DIR, "train_images")
TRAIN_MASK_DIR = os.path.join(RAW_DIR, "train_masks")
VAL_IMG_DIR = os.path.join(RAW_DIR, "val_images")
VAL_MASK_DIR = os.path.join(RAW_DIR, "val_masks")

CROP_TRAIN = os.path.join(CROPS_DIR, "train")
CROP_VAL = os.path.join(CROPS_DIR, "val")
CROP_TRAIN_MASKS = os.path.join(CROPS_DIR, "train_masks")
CROP_VAL_MASKS = os.path.join(CROPS_DIR, "val_masks")

PUBLIC_SAMPLES_DIR = r"C:\Users\ARAN\.gemini\antigravity-ide\scratch\freqlab-ui\public\retina_samples"

for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR,
          CROP_TRAIN, CROP_VAL, CROP_TRAIN_MASKS, CROP_VAL_MASKS, PUBLIC_SAMPLES_DIR]:
    os.makedirs(d, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets/Zomba/DRIVE-digital-retinal-images-for-vessel-extraction/resolve/main"

def fetch_url(url, retries=3):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(1.5)

def download_drive():
    print("=== Downloading DRIVE Dataset from Hugging Face Mirror ===")
    
    # 20 Training Images (21 to 40)
    train_ids = [f"{i:02d}" for i in range(21, 41)]
    for fid in train_ids:
        img_out = os.path.join(TRAIN_IMG_DIR, f"{fid}.png")
        mask_out = os.path.join(TRAIN_MASK_DIR, f"{fid}.png")
        
        if not os.path.exists(img_out):
            url_img = f"{HF_BASE}/train/input/{fid}.tif"
            data = fetch_url(url_img)
            img = Image.open(io.BytesIO(data)).convert('RGB')
            img.save(img_out, format='PNG')
            print(f" [Train] Saved image {fid}.png ({img.size})")
            
        if not os.path.exists(mask_out):
            url_mask = f"{HF_BASE}/train/label/{fid}.png"
            data_mask = fetch_url(url_mask)
            mask = Image.open(io.BytesIO(data_mask)).convert('L')
            mask.save(mask_out, format='PNG')
            print(f" [Train] Saved mask {fid}.png ({mask.size})")

    # 20 Validation/Test Images (01 to 20)
    val_ids = [f"{i:02d}" for i in range(1, 21)]
    for fid in val_ids:
        img_out = os.path.join(VAL_IMG_DIR, f"{fid}.png")
        mask_out = os.path.join(VAL_MASK_DIR, f"{fid}.png")
        
        if not os.path.exists(img_out):
            url_img = f"{HF_BASE}/val/input/{fid}.tif"
            data = fetch_url(url_img)
            img = Image.open(io.BytesIO(data)).convert('RGB')
            img.save(img_out, format='PNG')
            print(f" [Val] Saved image {fid}.png ({img.size})")
            
        if not os.path.exists(mask_out):
            url_mask = f"{HF_BASE}/val/label/{fid}_manual1.png"
            data_mask = fetch_url(url_mask)
            mask = Image.open(io.BytesIO(data_mask)).convert('L')
            mask.save(mask_out, format='PNG')
            print(f" [Val] Saved mask {fid}.png ({mask.size})")
            
    print("All DRIVE raw images and vessel masks downloaded successfully!")

def generate_crops(crop_size=256, n_train_per_img=30):
    print(f"\n=== Generating {crop_size}x{crop_size} Matched Crops ===")
    
    # 1. Training crops (random augmented within FOV)
    train_ids = [f"{i:02d}" for i in range(21, 41)]
    train_crop_idx = 0
    rng = np.random.default_rng(42)
    
    for fid in train_ids:
        img = Image.open(os.path.join(TRAIN_IMG_DIR, f"{fid}.png"))
        mask = Image.open(os.path.join(TRAIN_MASK_DIR, f"{fid}.png"))
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        h, w = img_np.shape[:2]
        
        # Extract crops that have at least some vessel content and retinal tissue (not pure black border)
        extracted = 0
        attempts = 0
        while extracted < n_train_per_img and attempts < 200:
            attempts += 1
            top = rng.integers(0, h - crop_size)
            left = rng.integers(0, w - crop_size)
            
            crop_im = img_np[top:top+crop_size, left:left+crop_size]
            crop_m = mask_np[top:top+crop_size, left:left+crop_size]
            
            # Check tissue coverage: mean intensity > 20 and at least 0.5% vessel pixels
            tissue_ratio = np.mean(crop_im > 20)
            vessel_ratio = np.mean(crop_m > 128)
            
            if tissue_ratio > 0.40:
                # Save crop
                c_img_name = f"retina_train_{train_crop_idx:04d}.png"
                Image.fromarray(crop_im).save(os.path.join(CROP_TRAIN, c_img_name))
                Image.fromarray(crop_m).save(os.path.join(CROP_TRAIN_MASKS, c_img_name))
                train_crop_idx += 1
                extracted += 1

    print(f"Generated {train_crop_idx} Training Crops in {CROP_TRAIN}")
    
    # 2. Validation crops (deterministic center & quadrant crops)
    val_ids = [f"{i:02d}" for i in range(1, 21)]
    val_crop_idx = 0
    
    for fid in val_ids:
        img = Image.open(os.path.join(VAL_IMG_DIR, f"{fid}.png"))
        mask = Image.open(os.path.join(VAL_MASK_DIR, f"{fid}.png"))
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        h, w = img_np.shape[:2]
        
        # Center crop + 4 quadrant-offset crops
        offsets = [
            (h // 2 - crop_size // 2, w // 2 - crop_size // 2),  # Center (macula / fovea)
            (h // 2 - crop_size // 2 - 50, w // 2 - crop_size // 2 - 50), # Upper left
            (h // 2 - crop_size // 2 + 50, w // 2 - crop_size // 2 + 50), # Lower right
            (h // 2 - crop_size // 2, w // 2 - crop_size // 2 + 80),      # Optic disc region
        ]
        
        for top, left in offsets:
            top = max(0, min(h - crop_size, top))
            left = max(0, min(w - crop_size, left))
            crop_im = img_np[top:top+crop_size, left:left+crop_size]
            crop_m = mask_np[top:top+crop_size, left:left+crop_size]
            
            c_img_name = f"retina_val_{val_crop_idx:04d}.png"
            Image.fromarray(crop_im).save(os.path.join(CROP_VAL, c_img_name))
            Image.fromarray(crop_m).save(os.path.join(CROP_VAL_MASKS, c_img_name))
            val_crop_idx += 1
            
    print(f"Generated {val_crop_idx} Validation Crops in {CROP_VAL}")

def copy_frontend_samples():
    print("\n=== Copying Showcase Samples to Frontend ===")
    # Copy 3 key crops for the frontend gallery
    # 1. Normal Healthy Retina (macular fovea + arcuate vessels)
    # 2. Micro-Vessel Dense Retina (macular capillary tree)
    # 3. Optic Disc & Vascular Emergence (glaucoma assessment)
    sample_mappings = [
        ("retina_val_0000.png", "normal_fundus.png", "normal_fundus_mask.png"),
        ("retina_val_0003.png", "optic_disc_fundus.png", "optic_disc_mask.png"),
        ("retina_val_0008.png", "microvessel_fundus.png", "microvessel_mask.png"),
    ]
    
    for src_name, dst_img, dst_mask in sample_mappings:
        src_img_p = os.path.join(CROP_VAL, src_name)
        src_mask_p = os.path.join(CROP_VAL_MASKS, src_name)
        
        if os.path.exists(src_img_p):
            Image.open(src_img_p).save(os.path.join(PUBLIC_SAMPLES_DIR, dst_img))
            Image.open(src_mask_p).save(os.path.join(PUBLIC_SAMPLES_DIR, dst_mask))
            print(f" Copied {src_name} -> {dst_img} & {dst_mask}")
            
    print("Frontend samples ready in public/retina_samples/")

if __name__ == '__main__':
    t0 = time.time()
    download_drive()
    generate_crops(crop_size=256, n_train_per_img=30)
    copy_frontend_samples()
    print(f"\nAll Data Preparation Complete in {time.time() - t0:.1f}s!")

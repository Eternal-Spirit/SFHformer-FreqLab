"""
prepare_fives_dataset.py - Ingest FIVES dataset, create patient-level stratified split (540 train / 60 val / 200 test),
and save full-resolution 2048x2048 PNGs and metadata manifest.
"""

import os
import io
import json
import time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE_DIR = r"D:\SFHformer-FreqLab\data\fives"
PARQUET_DIR = os.path.join(BASE_DIR, "parquet")
RAW_DIR = os.path.join(BASE_DIR, "raw")

TRAIN_IMG_DIR = os.path.join(RAW_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(RAW_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(RAW_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(RAW_DIR, "val", "masks")
TEST_IMG_DIR = os.path.join(RAW_DIR, "test", "images")
TEST_MASK_DIR = os.path.join(RAW_DIR, "test", "masks")

for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR]:
    os.makedirs(d, exist_ok=True)

RNG = np.random.default_rng(2026)

def extract_and_split():
    print("=" * 65)
    print(" [FIVES Ingestion] Stratified Patient-Level Split & Image Extraction ")
    print("=" * 65)
    t0 = time.time()

    # 1. Read all 3 training parquets
    train_files = [
        os.path.join(PARQUET_DIR, "train-00000-of-00003.parquet"),
        os.path.join(PARQUET_DIR, "train-00001-of-00003.parquet"),
        os.path.join(PARQUET_DIR, "train-00002-of-00003.parquet")
    ]
    
    print("\n[Step 1/3] Reading 600 official training images from parquets...")
    train_records = []
    for tf in train_files:
        table = pq.read_table(tf)
        for i in range(len(table)):
            img_dict = table['image'][i].as_py()
            mask_dict = table['mask'][i].as_py()
            img_id = table['image_id'][i].as_py()
            disease = img_id.split('_')[-1]
            train_records.append({
                "image_id": img_id,
                "disease": disease,
                "img_bytes": img_dict['bytes'],
                "mask_bytes": mask_dict['bytes']
            })
    print(f" -> Loaded {len(train_records)} training records.")

    # 2. Stratify by disease (150 per class -> 135 train, 15 val)
    by_disease = {}
    for r in train_records:
        by_disease.setdefault(r["disease"], []).append(r)

    train_final = []
    val_final = []

    for d_name, records in sorted(by_disease.items()):
        # Shuffle deterministically
        indices = RNG.permutation(len(records))
        train_idx = indices[:135]
        val_idx = indices[135:150]
        
        train_final.extend([records[i] for i in train_idx])
        val_final.extend([records[i] for i in val_idx])
        print(f" Disease {d_name}: {len(train_idx)} train, {len(val_idx)} val")

    print(f"\nFinal Split: {len(train_final)} Train, {len(val_final)} Validation.")

    # Save Train Images & Masks
    print("\n[Step 2/3] Writing 540 Train & 60 Val 2048x2048 images to disk...")
    for idx, r in enumerate(train_final):
        im = Image.open(io.BytesIO(r["img_bytes"]))
        mk = Image.open(io.BytesIO(r["mask_bytes"]))
        im.save(os.path.join(TRAIN_IMG_DIR, f"{r['image_id']}.png"), format="PNG")
        mk.save(os.path.join(TRAIN_MASK_DIR, f"{r['image_id']}.png"), format="PNG")
        if (idx + 1) % 100 == 0:
            print(f"  Saved {idx + 1}/540 train images...")

    for r in val_final:
        im = Image.open(io.BytesIO(r["img_bytes"]))
        mk = Image.open(io.BytesIO(r["mask_bytes"]))
        im.save(os.path.join(VAL_IMG_DIR, f"{r['image_id']}.png"), format="PNG")
        mk.save(os.path.join(VAL_MASK_DIR, f"{r['image_id']}.png"), format="PNG")

    print(" -> Train & Val images saved successfully.")

    # 3. Read & Save all 200 official held-out test images untouched
    print("\n[Step 3/3] Writing 200 official held-out Test images (untouched)...")
    test_file = os.path.join(PARQUET_DIR, "test-00000-of-00001.parquet")
    test_table = pq.read_table(test_file)
    test_records_meta = []
    
    for i in range(len(test_table)):
        img_dict = test_table['image'][i].as_py()
        mask_dict = test_table['mask'][i].as_py()
        img_id = test_table['image_id'][i].as_py()
        disease = img_id.split('_')[-1]
        
        im = Image.open(io.BytesIO(img_dict['bytes']))
        mk = Image.open(io.BytesIO(mask_dict['bytes']))
        im.save(os.path.join(TEST_IMG_DIR, f"{img_id}.png"), format="PNG")
        mk.save(os.path.join(TEST_MASK_DIR, f"{img_id}.png"), format="PNG")
        
        test_records_meta.append({
            "image_id": img_id,
            "disease": disease,
            "filename": f"{img_id}.png"
        })

    # Save dataset manifest
    manifest = {
        "dataset": "FIVES (Fundus Image Vessel Extraction)",
        "source": "https://huggingface.co/datasets/tyluan/FIVES",
        "resolution": "2048x2048",
        "split_summary": {
            "total_images": 800,
            "train_count": len(train_final),
            "val_count": len(val_final),
            "test_count": len(test_records_meta)
        },
        "disease_classes": {
            "A": "Age-related Macular Degeneration (AMD)",
            "D": "Diabetic Retinopathy (DR)",
            "G": "Glaucoma (GLA)",
            "N": "Normal (NOR)"
        },
        "train_images": [{"image_id": r["image_id"], "disease": r["disease"]} for r in train_final],
        "val_images": [{"image_id": r["image_id"], "disease": r["disease"]} for r in val_final],
        "test_images": test_records_meta
    }

    manifest_path = os.path.join(BASE_DIR, "dataset_split.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved to: {manifest_path}")
    print(f"All 800 images extracted and organized in {time.time() - t0:.1f}s!")

if __name__ == '__main__':
    extract_and_split()

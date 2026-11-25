"""
scripts/make_crops.py - Prepare train/val/test crops from raw images.

Splits by source image BEFORE cropping to prevent near-duplicate leakage.

Train: random 320x320 crops per image as PNG.
Val: 1 center 256x256 crop per image.
Test: 1 center 256x256 crop per image.

Usage:
    python scripts/make_crops.py --src data/raw --dst data/crops
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def center_crop(img, size):
    """Center-crop a PIL image to size x size."""
    w, h = img.size
    left = max(0, (w - size) // 2)
    top = max(0, (h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def random_crops(img, size, n, rng):
    """Extract n random crops of given size from a PIL image."""
    w, h = img.size
    crops = []
    for _ in range(n):
        if w > size and h > size:
            left = rng.integers(0, w - size)
            top = rng.integers(0, h - size)
        else:
            left = max(0, (w - size) // 2)
            top = max(0, (h - size) // 2)
        crops.append(img.crop((left, top, left + size, top + size)))
    return crops


def main():
    parser = argparse.ArgumentParser(description='Prepare crops from raw images')
    parser.add_argument('--src', type=str, required=True, help='Source directory of raw images')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory for crops')
    parser.add_argument('--train-crop-size', type=int, default=320, help='Training crop size')
    parser.add_argument('--eval-crop-size', type=int, default=256, help='Val/test crop size')
    parser.add_argument('--train-crops-per-image', type=int, default=8)
    parser.add_argument('--val-count', type=int, default=20,
                        help='Number of images for validation')
    parser.add_argument('--test-count', type=int, default=50,
                        help='Number of fixed test images')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Discover images
    src_path = Path(args.src)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    all_images = sorted([f for f in src_path.rglob('*') if f.suffix.lower() in extensions])

    if len(all_images) == 0:
        print(f"ERROR: No images found in {args.src}")
        sys.exit(1)

    print(f"Found {len(all_images)} images in {args.src}")

    # Split by source image
    indices = rng.permutation(len(all_images))

    total = len(all_images)
    if total >= args.test_count + args.val_count + 5:
        n_test = args.test_count
        n_val = args.val_count
        n_train = total - n_test - n_val
    else:
        n_test = min(args.test_count, max(1, total // 4))
        n_val = min(args.val_count, max(1, total // 4))
        n_train = max(1, total - n_test - n_val)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    splits = {
        'train': [all_images[i] for i in train_idx],
        'val': [all_images[i] for i in val_idx],
        'test': [all_images[i] for i in test_idx],
    }

    print(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Verify no overlap
    split_names_set = {}
    for split_name, files in splits.items():
        names = set(f.name for f in files)
        for other_name, other_names in split_names_set.items():
            overlap = names & other_names
            assert len(overlap) == 0, f"LEAKAGE: {split_name} and {other_name} share {overlap}"
        split_names_set[split_name] = names

    # Create directories and process
    dst_path = Path(args.dst)
    splits_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split_name, files in splits.items():
        split_dir = dst_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_list = []
        crop_count = 0

        for img_path in tqdm(files, desc=f'{split_name}'):
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"WARNING: Could not open {img_path}: {e}")
                continue

            base = img_path.stem

            if split_name == 'train':
                crops = random_crops(img, args.train_crop_size,
                                     args.train_crops_per_image, rng)
                for j, crop in enumerate(crops):
                    fname = f"{base}_crop{j}.png"
                    crop.save(split_dir / fname, 'PNG')
                    split_list.append(fname)
                    crop_count += 1
            else:
                crop = center_crop(img, args.eval_crop_size)
                fname = f"{base}.png"
                crop.save(split_dir / fname, 'PNG')
                split_list.append(fname)
                crop_count += 1

        # Save split list
        split_file = splits_dir / f'{split_name}.txt'
        with open(split_file, 'w') as f:
            f.write('\n'.join(split_list))

        counts[split_name] = crop_count
        print(f"{split_name}: {len(files)} images -> {crop_count} crops")

    # Save fixed test50 list
    test_list = splits_dir / 'test50.txt'
    with open(splits_dir / 'test.txt', 'r') as f:
        test_names = f.read().strip().split('\n')
    with open(test_list, 'w') as f:
        f.write('\n'.join(test_names[:50]))

    # Save metadata
    meta = {
        'source': str(args.src),
        'seed': args.seed,
        'counts': counts,
        'train_crop_size': args.train_crop_size,
        'eval_crop_size': args.eval_crop_size,
        'train_crops_per_image': args.train_crops_per_image,
    }
    with open(splits_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSummary: {counts}")
    print(f"Split lists saved to {splits_dir}")
    print("Leakage check: PASSED (no overlap between splits)")


if __name__ == '__main__':
    main()

"""
Dataset class for frequency-selective robustness training.

Produces (noisy, clean) pairs with online sinusoidal corruption.
Workers are seeded independently via worker_init_fn to avoid
all workers drawing identical noise sequences.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from freqlab.corrupt import corrupt, sample_train_comps


def worker_init_fn(worker_id):
    """Seed each DataLoader worker independently using worker_id + base seed."""
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed  # set by DataLoader or manual_seed
    np.random.seed((base_seed + worker_id) % (2**32))


class FreqCorruptDataset(Dataset):
    """Dataset that applies online sinusoidal corruption to clean crops.
    
    Args:
        image_dir: Path to directory of clean PNG crops.
        crop_size: Random crop size extracted from each image (for train).
        train: If True, apply random crop and horizontal/vertical flips.
        k_lo: Lower bound of training frequency band.
        k_hi: Upper bound of training frequency band.
        amp_range: (min, max) amplitude for sinusoidal noise.
        seed: Base seed for reproducible corruption (val/test mode).
    """

    def __init__(self, image_dir, crop_size=256, train=True,
                 k_lo=8, k_hi=38, amp_range=(0.05, 0.25), seed=42):
        super().__init__()
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, '*.png')) +
            glob.glob(os.path.join(image_dir, '*.jpg'))
        )
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")
        self.crop_size = crop_size
        self.train = train
        self.k_lo = k_lo
        self.k_hi = k_hi
        self.amp_range = amp_range
        self.seed = seed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = np.array(img, dtype=np.float32) / 255.0  # HxWx3 in [0,1]

        h, w = img.shape[:2]

        if self.train:
            # Random crop
            if h > self.crop_size and w > self.crop_size:
                rng = np.random.default_rng()
                top = rng.integers(0, h - self.crop_size)
                left = rng.integers(0, w - self.crop_size)
                img = img[top:top + self.crop_size, left:left + self.crop_size]
            else:
                # Center crop if image is too small
                top = max(0, (h - self.crop_size) // 2)
                left = max(0, (w - self.crop_size) // 2)
                img = img[top:top + self.crop_size, left:left + self.crop_size]

            # Random flips
            rng = np.random.default_rng()
            if rng.random() > 0.5:
                img = img[:, ::-1].copy()  # horizontal flip
            if rng.random() > 0.5:
                img = img[::-1, :].copy()  # vertical flip

            # Fresh corruption per call (training)
            rng = np.random.default_rng()
            comps = sample_train_comps(rng, self.k_lo, self.k_hi, self.amp_range)
        else:
            # Center crop for val/test
            top = max(0, (h - self.crop_size) // 2)
            left = max(0, (w - self.crop_size) // 2)
            img = img[top:top + self.crop_size, left:left + self.crop_size]

            # Fixed corruption for val/test (deterministic per image)
            rng = np.random.default_rng(self.seed + idx)
            comps = sample_train_comps(rng, self.k_lo, self.k_hi, self.amp_range)

        clean = img.copy()
        noisy = corrupt(clean, comps)

        # Convert to tensors: CxHxW
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1))
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1))

        return noisy_t, clean_t


def get_loader(image_dir, batch_size=4, crop_size=256, train=True,
               num_workers=4, seed=42, **kwargs):
    """Convenience function to create a DataLoader.
    
    Args:
        image_dir: Path to clean image crops.
        batch_size: Batch size.
        crop_size: Crop size for training.
        train: Whether this is a training loader.
        num_workers: Number of data loading workers.
        seed: Seed for val/test reproducibility.
    
    Returns:
        DataLoader instance.
    """
    ds = FreqCorruptDataset(
        image_dir, crop_size=crop_size, train=train, seed=seed, **kwargs
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        worker_init_fn=worker_init_fn if train else None,
    )

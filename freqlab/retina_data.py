"""
retina_data.py - PyTorch Dataset & DataLoader for Retinal Image Restoration.
Loads paired (clean crop, vessel mask) and applies online realistic ophthalmic corruptions.
"""

import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from retina_corrupt import retinal_corrupt

def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed
    np.random.seed((base_seed + worker_id) % (2**32))

class RetinalDataset(Dataset):
    """
    Dataset for Retinal Restoration training and validation.
    
    Args:
        image_dir: Path to directory of clean retinal crops.
        mask_dir: Optional path to corresponding vessel ground truth masks.
        is_train: If True, apply geometric augmentations and random corruption.
        corruption_mode: 'mixed' | 'raster_banding' | 'laser_speckle' | 'sensor_noise'
        seed: Base seed for deterministic validation.
    """
    def __init__(self, image_dir, mask_dir=None, is_train=True, corruption_mode='mixed', seed=42):
        super().__init__()
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, '*.png')) +
            glob.glob(os.path.join(image_dir, '*.jpg'))
        )
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No retinal images found in {image_dir}")
            
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.corruption_mode = corruption_mode
        self.seed = seed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        clean = np.array(img, dtype=np.float32) / 255.0  # HxWx3 [0, 1]
        
        # Load vessel mask if available
        mask = None
        if self.mask_dir:
            base_name = os.path.basename(img_path)
            mask_path = os.path.join(self.mask_dir, base_name)
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                mask = (np.array(mask_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

        rng = np.random.default_rng() if self.is_train else np.random.default_rng(self.seed + idx)

        # Geometric augmentations for training
        if self.is_train:
            # Random horizontal flip
            if rng.random() > 0.5:
                clean = clean[:, ::-1, :].copy()
                if mask is not None:
                    mask = mask[:, ::-1].copy()
            # Random vertical flip
            if rng.random() > 0.5:
                clean = clean[::-1, :, :].copy()
                if mask is not None:
                    mask = mask[::-1, :].copy()
            # Random 90 deg rotation
            k_rot = rng.integers(0, 4)
            if k_rot > 0:
                clean = np.rot90(clean, k_rot).copy()
                if mask is not None:
                    mask = np.rot90(mask, k_rot).copy()

            severity = float(rng.uniform(0.75, 1.25))
            noisy, _ = retinal_corrupt(clean, mode=self.corruption_mode, severity=severity, rng=rng)
        else:
            # Deterministic corruption for validation
            noisy, _ = retinal_corrupt(clean, mode=self.corruption_mode, severity=1.0, rng=rng)

        # Convert to PyTorch tensors (CxHxW)
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1)).float()
        
        if mask is not None:
            mask_t = torch.from_numpy(mask).float().unsqueeze(0)
            return noisy_t, clean_t, mask_t
            
        return noisy_t, clean_t

def get_retina_loaders(crops_base_dir, batch_size=4, num_workers=0, corruption_mode='mixed'):
    train_img_dir = os.path.join(crops_base_dir, "train")
    train_mask_dir = os.path.join(crops_base_dir, "train_masks")
    val_img_dir = os.path.join(crops_base_dir, "val")
    val_mask_dir = os.path.join(crops_base_dir, "val_masks")
    
    train_ds = RetinalDataset(train_img_dir, train_mask_dir, is_train=True, corruption_mode=corruption_mode)
    val_ds = RetinalDataset(val_img_dir, val_mask_dir, is_train=False, corruption_mode=corruption_mode)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False
    )
    return train_loader, val_loader

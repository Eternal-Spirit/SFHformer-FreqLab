"""
train_fives_retina.py - Fine-Tuning SFHformer on FIVES 2048x2048 Retinal Dataset
Stratified across 540 Train / 60 Val images with 256x256 random crops & online degradation.
Runs on NVIDIA GeForce RTX 5060 Laptop GPU.
Physical batch size: 1, Gradient accumulation: 4 (effective batch size 4).
"""

import os
import sys
import time
import glob
import math
import argparse
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from freqlab.sfhformer_gated import SFHformer_m_gated
from freqlab.metrics import psnr_torch, psnr_np, ssim_np
from retina_corrupt import (
    apply_raster_banding,
    apply_laser_speckle,
    apply_optical_halo,
    apply_illumination_gradient,
    apply_sensor_noise
)

DATA_DIR = r"D:\SFHformer-FreqLab\data\fives"
RAW_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_IMG_DIR = os.path.join(RAW_DIR, "train", "images")
VAL_IMG_DIR = os.path.join(RAW_DIR, "val", "images")

CKPT_OUT_DIR = os.path.join(REPO_DIR, "checkpoints", "fives_retina_finetune")
os.makedirs(CKPT_OUT_DIR, exist_ok=True)

class FIVESOnlineDataset(Dataset):
    """
    Dataset that extracts random 256x256 crops from 2048x2048 FIVES fundus images
    and generates synthetic ophthalmic degradation online.
    """
    def __init__(self, img_paths, crop_size=256, is_train=True, seed=2026):
        self.img_paths = img_paths
        self.crop_size = crop_size
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.img_paths) * (50 if self.is_train else 2)

    def __getitem__(self, idx):
        img_idx = idx % len(self.img_paths)
        img_p = self.img_paths[img_idx]

        # Load clean 2048x2048 image
        with Image.open(img_p) as pil_im:
            img_arr = np.array(pil_im.convert('RGB'), dtype=np.float32) / 255.0

        h, w = img_arr.shape[:2]
        cs = self.crop_size

        if self.is_train:
            # Random valid tissue crop
            attempts = 0
            while attempts < 30:
                attempts += 1
                top = self.rng.integers(0, h - cs + 1)
                left = self.rng.integers(0, w - cs + 1)
                crop_clean = img_arr[top:top+cs, left:left+cs]
                # Tissue check: at least 35% non-background
                if np.mean(crop_clean > 0.08) > 0.35:
                    break

            # Random spatial augmentations (rot90, flips)
            k_rot = self.rng.integers(0, 4)
            if k_rot > 0:
                crop_clean = np.rot90(crop_clean, k_rot)
            if self.rng.random() > 0.5:
                crop_clean = np.fliplr(crop_clean)
            if self.rng.random() > 0.5:
                crop_clean = np.flipud(crop_clean)
            crop_clean = np.ascontiguousarray(crop_clean)

            # Online Multi-Modal Degradation Simulation
            corrupted = crop_clean.copy()
            # 1. Periodic raster stripes
            k_freq = int(self.rng.integers(8, 40))
            stripe_amp = float(self.rng.uniform(0.08, 0.16))
            angle = float(self.rng.choice([0.0, np.pi/2, np.pi/4, np.pi/6]))
            corrupted = apply_raster_banding(corrupted, k=k_freq, amp=stripe_amp, angle=angle)

            # 2. Gaussian sensor shot noise
            sigma = float(self.rng.uniform(0.02, 0.05))
            corrupted = apply_sensor_noise(corrupted, sigma=sigma, rng=self.rng)

            # 3. Multiplicative laser speckle
            speckle_int = float(self.rng.uniform(0.06, 0.14))
            corrupted = apply_laser_speckle(corrupted, intensity=speckle_int, rng=self.rng)

            # 4. Illumination gradient
            if self.rng.random() > 0.4:
                falloff = float(self.rng.uniform(0.15, 0.30))
                corrupted = apply_illumination_gradient(corrupted, falloff=falloff, rng=self.rng)

            corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)

        else:
            # Deterministic center & paracentral validation crops
            det_rng = np.random.default_rng(idx * 7919)
            sub_crop = idx // len(self.img_paths)
            if sub_crop == 0:
                top = h // 2 - cs // 2
                left = w // 2 - cs // 2
            else:
                top = min(max(0, h // 2 - cs // 2 + 80), h - cs)
                left = min(max(0, w // 2 - cs // 2 + 80), w - cs)

            crop_clean = np.ascontiguousarray(img_arr[top:top+cs, left:left+cs])

            # Deterministic synthetic degradation
            corrupted = apply_raster_banding(crop_clean.copy(), k=22, amp=0.12, angle=0.0)
            corrupted = apply_sensor_noise(corrupted, sigma=0.03, rng=det_rng)
            corrupted = apply_laser_speckle(corrupted, intensity=0.10, rng=det_rng)
            corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)

        # To PyTorch Tensor (C, H, W)
        x_noisy = torch.from_numpy(corrupted.transpose(2, 0, 1)).float()
        y_clean = torch.from_numpy(crop_clean.transpose(2, 0, 1)).float()

        return x_noisy, y_clean

def validate(model, val_loader, device):
    """Computes exact validation metrics: PSNR, Green-channel PSNR, and L1 loss."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_green_psnr = 0.0
    count = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy).clamp(0.0, 1.0)
            loss = F.l1_loss(pred, clean)

            p = psnr_torch(pred, clean)
            p_green = psnr_torch(pred[:, 1:2, :, :], clean[:, 1:2, :, :])

            b = noisy.size(0)
            total_loss += loss.item() * b
            total_psnr += p * b
            total_green_psnr += p_green * b
            count += b

    model.train()
    c = max(count, 1)
    return total_loss / c, total_psnr / c, total_green_psnr / c

def run_calibration_benchmark(model, train_loader, optimizer, device, accum_steps=2, use_amp=True, n_iters=100):
    """
    Requirement 9: Benchmark iterations before determining the training budget.
    Measures exact ms/step, peak VRAM, and gradient accumulation timing.
    """
    print("\n" + "=" * 65)
    print(f" [BENCHMARK] Executing {n_iters}-Iteration Hardware Calibration (AMP={use_amp})")
    print("=" * 65)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    t0 = time.time()
    data_iter = iter(train_loader)
    losses = []

    for step in range(1, n_iters + 1):
        try:
            noisy, clean = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            noisy, clean = next(data_iter)

        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(noisy)
                l_spatial = F.l1_loss(pred, clean)
                pred_fft = torch.fft.rfft2(pred.float(), norm="backward")
                clean_fft = torch.fft.rfft2(clean.float(), norm="backward")
                l_freq = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft)) * 0.1
                loss = (l_spatial + l_freq) / accum_steps
        else:
            pred = model(noisy)
            l_spatial = F.l1_loss(pred, clean)
            pred_fft = torch.fft.rfft2(pred, norm="backward")
            clean_fft = torch.fft.rfft2(clean, norm="backward")
            l_freq = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft)) * 0.1
            loss = (l_spatial + l_freq) / accum_steps

        loss.backward()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        losses.append((l_spatial + l_freq).item())

    torch.cuda.synchronize()
    total_elapsed = time.time() - t0
    ms_per_step = (total_elapsed / n_iters) * 1000.0
    ms_per_opt_iter = ms_per_step * accum_steps
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024**3)

    summary = {
        "n_benchmark_iters": n_iters,
        "total_elapsed_s": round(total_elapsed, 2),
        "ms_per_forward_backward": round(ms_per_step, 1),
        "ms_per_effective_optimizer_step": round(ms_per_opt_iter, 1),
        "effective_steps_per_minute": round(60000.0 / ms_per_opt_iter, 1),
        "projected_steps_in_100_mins": int(100 * (60000.0 / ms_per_opt_iter)),
        "peak_vram_gb": round(peak_vram, 2),
        "initial_loss": round(losses[0], 5),
        "final_benchmark_loss": round(losses[-1], 5),
        "loss_reduction": round(losses[0] - losses[-1], 5)
    }

    print(f" -> Total Benchmark Time:     {total_elapsed:.2f} s")
    print(f" -> Forward/Backward Latency: {ms_per_step:.1f} ms/step")
    print(f" -> Effective Optimizer Step: {ms_per_opt_iter:.1f} ms/iter (Accum=4)")
    print(f" -> Peak VRAM Allocated:      {peak_vram:.2f} GB (Display safe)")
    print(f" -> Loss Trend:               {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f" -> Projected 100-Min Budget: {summary['projected_steps_in_100_mins']} iterations")
    print("=" * 65)

    bench_json_path = os.path.join(CKPT_OUT_DIR, "benchmark_100_results.json")
    with open(bench_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true", help="Run 100 iters calibration benchmark and exit")
    parser.add_argument("--iters", type=int, default=8000, help="Total training iterations")
    parser.add_argument("--batch-size", type=int, default=2, help="Physical batch size (2 is optimal within 6GB VRAM)")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps (effective batch=4)")
    parser.add_argument("--no-amp", action="store_true", help="Disable BF16 mixed precision")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--val-interval", type=int, default=500, help="Validation interval")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint interval")
    args = parser.parse_args()

    use_amp = not args.no_amp and torch.cuda.is_bf16_supported()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 65)
    print(" SFHformer-Retina: FIVES 2048x2048 Fine-Tuning Engine ")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f" Mixed Precision (AMP BF16): {use_amp} | Batch: {args.batch_size} (Accum={args.accum_steps})")
    print("=" * 65)

    if device.type == 'cuda':
        # HARD SAFETY CEILING: 5.57 GB VRAM cap (70% of 7.96 GB) to guarantee OS/DWM headroom
        torch.cuda.set_per_process_memory_fraction(0.70, device=0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # 1. Image path collections
    train_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
    val_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.png")))

    print(f"Found {len(train_paths)} Train images and {len(val_paths)} Val images.")
    if len(train_paths) == 0:
        print("[ERROR] Train images not found yet. Ensure prepare_fives_dataset.py finishes extraction.")
        sys.exit(1)

    # 2. Datasets & Loaders
    train_dataset = FIVESOnlineDataset(train_paths, crop_size=256, is_train=True)
    val_dataset = FIVESOnlineDataset(val_paths, crop_size=256, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # 3. Model instantiation & Checkpoint loading
    print("\n[Model] Instantiating SFHformer_m (ungated baseline backbone)...")
    model = SFHformer_m_gated(use_gate=False).to(device)

    # Load starting checkpoint (retina best.pth)
    base_ckpt_path = os.path.join(REPO_DIR, "checkpoints", "retina", "best.pth")
    if os.path.exists(base_ckpt_path):
        print(f" -> Loading fine-tuning starting weights from: {base_ckpt_path}")
        ck = torch.load(base_ckpt_path, map_location=device, weights_only=False)
        st = ck['model'] if 'model' in ck else ck
        model.load_state_dict(st, strict=True)
        print(" -> Initial weights successfully loaded into SFHformer.")
    else:
        print("[WARNING] Baseline retina checkpoint not found; starting from r1_baseline.")
        r1_p = os.path.join(REPO_DIR, "checkpoints", "r1_baseline", "best.pth")
        ck = torch.load(r1_p, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=True)

    # 4. Optimizer & Cosine LR Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters, eta_min=1e-6)

    # 5. Run Calibration Benchmark (Requirement 9)
    bench_results = run_calibration_benchmark(model, train_loader, optimizer, device, accum_steps=args.accum_steps, use_amp=use_amp, n_iters=100)
    if args.benchmark_only:
        print("\n[INFO] Benchmark complete. Exiting as requested by --benchmark-only flag.")
        return

    # Calibrate iters to safety window if needed
    total_iters = args.iters
    print(f"\n[Training] Commencing Fine-Tuning for {total_iters} iterations...")
    print(f" Physical Batch: {args.batch_size} | Gradient Accumulation: {args.accum_steps} | Effective Batch: {args.batch_size * args.accum_steps}")
    print(f" Initial LR: {args.lr} -> Min LR: 1e-6 (Cosine Decay)")

    train_log_file = os.path.join(CKPT_OUT_DIR, "train_log.csv")
    val_log_file = os.path.join(CKPT_OUT_DIR, "val_log.csv")

    with open(train_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "loss", "l_spatial", "l_freq", "lr", "elapsed_s", "peak_vram_gb"])

    with open(val_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "val_loss", "val_psnr", "val_green_psnr", "elapsed_s"])

    # Initial Validation zero-point
    print("\nRunning Baseline Zero-Point Validation on held-out 60 FIVES validation cases...")
    v_loss_0, v_psnr_0, v_gpsnr_0 = validate(model, val_loader, device)
    print(f" [Iter 0 Baseline] Val Loss: {v_loss_0:.5f} | Val PSNR: {v_psnr_0:.2f} dB | Green PSNR: {v_gpsnr_0:.2f} dB")
    with open(val_log_file, "a", newline="") as f:
        csv.writer(f).writerow([0, round(v_loss_0, 5), round(v_psnr_0, 2), round(v_gpsnr_0, 2), 0.0])

    best_val_psnr = v_psnr_0
    train_iter = iter(train_loader)
    accum_steps = args.accum_steps
    optimizer.zero_grad(set_to_none=True)
    t_start = time.time()

    running_loss = 0.0
    running_spatial = 0.0
    running_freq = 0.0

    for step in range(1, total_iters + 1):
        try:
            noisy, clean = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            noisy, clean = next(train_iter)

        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(noisy)
                l_spatial = F.l1_loss(pred, clean)
                pred_fft = torch.fft.rfft2(pred.float(), norm="backward")
                clean_fft = torch.fft.rfft2(clean.float(), norm="backward")
                l_freq = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft)) * 0.1
                loss = (l_spatial + l_freq) / accum_steps
        else:
            pred = model(noisy)
            l_spatial = F.l1_loss(pred, clean)
            pred_fft = torch.fft.rfft2(pred, norm="backward")
            clean_fft = torch.fft.rfft2(clean, norm="backward")
            l_freq = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft)) * 0.1
            loss = (l_spatial + l_freq) / accum_steps

        loss.backward()

        running_loss += (l_spatial + l_freq).item()
        running_spatial += l_spatial.item()
        running_freq += l_freq.item()

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0:
            avg_l = running_loss / 50.0
            avg_sp = running_spatial / 50.0
            avg_fr = running_freq / 50.0
            elapsed = time.time() - t_start
            cur_lr = scheduler.get_last_lr()[0]
            vram = torch.cuda.max_memory_allocated(device) / (1024**3)

            print(f" Iter [{step:5d}/{total_iters}] | Loss: {avg_l:.4f} (S:{avg_sp:.4f}, F:{avg_fr:.4f}) | LR: {cur_lr:.2e} | Elapsed: {elapsed/60.0:.1f}m | VRAM: {vram:.2f}G")

            with open(train_log_file, "a", newline="") as f:
                csv.writer(f).writerow([step, round(avg_l, 5), round(avg_sp, 5), round(avg_fr, 5), f"{cur_lr:.3e}", round(elapsed, 1), round(vram, 2)])

            running_loss = 0.0
            running_spatial = 0.0
            running_freq = 0.0

        # Periodic Validation & Best Model Checkpoint
        if step % args.val_interval == 0:
            v_loss, v_psnr, v_gpsnr = validate(model, val_loader, device)
            elapsed = time.time() - t_start
            print(f"\n >>> [Validation @ Iter {step}] Loss: {v_loss:.5f} | PSNR: {v_psnr:.2f} dB (Green: {v_gpsnr:.2f} dB) | Best: {best_val_psnr:.2f} dB")

            with open(val_log_file, "a", newline="") as f:
                csv.writer(f).writerow([step, round(v_loss, 5), round(v_psnr, 2), round(v_gpsnr, 2), round(elapsed, 1)])

            if v_psnr > best_val_psnr:
                best_val_psnr = v_psnr
                torch.save({
                    "iteration": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_psnr": v_psnr,
                    "val_green_psnr": v_gpsnr,
                    "best_val_psnr": best_val_psnr
                }, os.path.join(CKPT_OUT_DIR, "best.pth"))
                print(f" *** NEW BEST CHECKPOINT SAVED @ {v_psnr:.2f} dB ***\n")

        # Save last checkpoint
        if step % args.save_interval == 0:
            torch.save({
                "iteration": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_psnr": best_val_psnr
            }, os.path.join(CKPT_OUT_DIR, "last.pth"))

    print("\n" + "=" * 65)
    print(f" [SUCCESS] FIVES Retinal Fine-Tuning Complete in {(time.time() - t_start)/60.0:.1f} mins!")
    print(f" Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f" Checkpoint saved to: {os.path.join(CKPT_OUT_DIR, 'best.pth')}")
    print("=" * 65)

if __name__ == '__main__':
    main()

"""
train_mega_retina.py - Transfer Learning & Fine-Tuning SFHformer on Mega Retinal Dataset (DRIVE + STARE + HRF + APTOS).
Runs safely on NVIDIA GeForce RTX 5060 Laptop GPU with display-safe execution.
"""

import os
import sys
import time
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_DIR = r"D:\retina sfhormer clone"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
FREQLAB_SCRATCH = r"C:\Users\ARAN\.gemini\antigravity-ide\scratch\freqlab-ui"
if FREQLAB_SCRATCH not in sys.path:
    sys.path.insert(0, FREQLAB_SCRATCH)

from models import SFHformer_m
from freqlab.metrics import psnr_torch
from retina_data import get_retina_loaders

def validate(model, val_loader, criterion, device):
    """Validation: computes overall PSNR, Green-channel PSNR, and L1 loss."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_green_psnr = 0.0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            noisy = batch[0].to(device, non_blocking=True)
            clean = batch[1].to(device, non_blocking=True)
            pred = model(noisy).clamp(0.0, 1.0)
            loss = criterion(pred, clean)
            
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

def main():
    print("=" * 65, flush=True)
    print(" SFHformer-Retina: Mega Multi-Dataset Fine-Tuning Pipeline ", flush=True)
    print("=" * 65, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True

    # Checkpoint output directory
    ckpt_dir = Path(r"D:\retina sfhormer clone\checkpoints\retina")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "retina_mega_train_log.csv"
    val_csv_path = ckpt_dir / "retina_mega_val_log.csv"

    # Instantiate model
    print("\nInstantiating SFHformer_m...", flush=True)
    net = SFHformer_m().to(device)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Model parameters: {n_params:.2f}M", flush=True)

    # Load baseline weights
    pretrained_path = Path(r"D:\retina sfhormer clone\checkpoints\retina\best.pth")
    if not pretrained_path.exists():
        pretrained_path = Path(r"D:\retina sfhormer clone\checkpoints\r1_baseline\best.pth")
        
    if pretrained_path.exists():
        print(f"Loading weights from: {pretrained_path}", flush=True)
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        net.load_state_dict(state_dict)
        print(" -> Weights successfully loaded.", flush=True)
    else:
        print("Warning: Pre-trained weights not found, starting from random initialization.", flush=True)

    # Optimization config
    ITERS = 4000
    BATCH_SIZE = 1
    GRAD_ACCUM = 4
    LR = 4e-5
    LOG_EVERY = 50
    VAL_EVERY = 500

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ITERS, eta_min=1e-6)

    # Data loaders
    crops_base = r"D:\retina sfhormer clone\data\retina\crops"
    print(f"\nLoading retinal crops from: {crops_base}", flush=True)
    train_loader, val_loader = get_retina_loaders(crops_base, batch_size=BATCH_SIZE, num_workers=0)
    print(f"Train crops: {len(train_loader)}, Val crops: {len(val_loader)}", flush=True)

    # CSV headers
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'loss', 'lr', 'elapsed_s', 'peak_vram_gb'])
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration', 'val_loss', 'val_psnr', 'val_green_psnr', 'elapsed_s'])

    # Baseline validation before training
    print("\nRunning initial baseline validation on multi-source test crops...", flush=True)
    init_loss, init_psnr, init_green_psnr = validate(net, val_loader, criterion, device)
    print(f"Initial Multi-Source Performance: Loss={init_loss:.5f}, PSNR={init_psnr:.2f}dB, Green-PSNR={init_green_psnr:.2f}dB", flush=True)
    
    with open(val_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([0, f'{init_loss:.5f}', f'{init_psnr:.2f}', f'{init_green_psnr:.2f}', '0.0'])

    best_psnr = init_psnr
    net.train()
    it = 0
    t_start = time.time()
    running_loss = 0.0
    loss_count = 0
    accum_step = 0

    print(f"\nStarting {ITERS} iterations on {len(train_loader)} multi-source crops...", flush=True)
    print("-" * 65, flush=True)

    optimizer.zero_grad(set_to_none=True)

    while it < ITERS:
        for batch in train_loader:
            if it >= ITERS:
                break

            noisy = batch[0].to(device, non_blocking=False)
            clean = batch[1].to(device, non_blocking=False)

            pred = net(noisy)
            loss = criterion(pred, clean) / GRAD_ACCUM
            loss.backward()

            accum_step += 1
            if accum_step % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Cooperative OS yield (prevents Windows Desktop TDR watchdog timeout)
            time.sleep(0.002)

            it += 1
            running_loss += (loss.item() * GRAD_ACCUM)
            loss_count += 1

            if it % LOG_EVERY == 0:
                avg_l = running_loss / max(loss_count, 1)
                elapsed = time.time() - t_start
                it_per_sec = it / max(elapsed, 0.01)
                lr_curr = scheduler.get_last_lr()[0]
                peak_vram = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0.0
                print(f"Iter [{it:5d}/{ITERS}] | Loss: {avg_l:.5f} | LR: {lr_curr:.2e} | Speed: {it_per_sec:.1f} it/s | Peak VRAM: {peak_vram:.2f}GB", flush=True)

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([it, f'{avg_l:.6f}', f'{lr_curr:.3e}', f'{elapsed:.1f}', f'{peak_vram:.2f}'])

                running_loss = 0.0
                loss_count = 0

            if it % VAL_EVERY == 0 or it == ITERS:
                elapsed = time.time() - t_start
                vl, v_psnr, v_green_psnr = validate(net, val_loader, criterion, device)
                print(f"\n>>> [VAL @ {it:5d}] Val Loss: {vl:.5f} | Val PSNR: {v_psnr:.2f}dB | Green-PSNR: {v_green_psnr:.2f}dB (Best: {best_psnr:.2f}dB)", flush=True)

                with open(val_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([it, f'{vl:.5f}', f'{v_psnr:.2f}', f'{v_green_psnr:.2f}', f'{elapsed:.1f}'])

                # Save latest checkpoint
                torch.save({'model': net.state_dict(), 'iteration': it, 'psnr': v_psnr, 'green_psnr': v_green_psnr}, ckpt_dir / "last.pth")

                # Save best checkpoint
                if v_psnr > best_psnr:
                    best_psnr = v_psnr
                    torch.save({'model': net.state_dict(), 'iteration': it, 'psnr': v_psnr, 'green_psnr': v_green_psnr}, ckpt_dir / "best.pth")
                    print(f"    * New Best Checkpoint Saved! (PSNR: {best_psnr:.2f}dB)\n", flush=True)
                else:
                    print("", flush=True)

    total_time = time.time() - t_start
    print("=" * 65, flush=True)
    print(f"Training Complete in {total_time/60:.1f} min! Best Val PSNR: {best_psnr:.2f}dB", flush=True)
    print(f"Model saved to: {ckpt_dir / 'best.pth'}", flush=True)
    print("=" * 65, flush=True)

if __name__ == '__main__':
    main()

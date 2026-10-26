"""
scripts/train_freq.py — Iteration-based training script for frequency robustness experiments.

Single-GPU, no DataParallel. Supports resume from checkpoint.
Uses L1 loss (matching upstream pattern) with AdamW + CosineAnnealing.

Usage:
    python scripts/train_freq.py --model SFHformer_m --iters 40000 --seed 0 --run r1_baseline
    python scripts/train_freq.py --model SFHformer_m --iters 40000 --seed 0 --run r2_gated --gate
    python scripts/train_freq.py --resume --run r1_baseline
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freqlab.data import get_loader
from freqlab.metrics import psnr_torch


def parse_args():
    parser = argparse.ArgumentParser(description='Frequency robustness training')
    parser.add_argument('--model', default='SFHformer_m', type=str)
    parser.add_argument('--gate', action='store_true', help='Use gated model variant')
    parser.add_argument('--gate-n-bands', type=int, default=16)
    parser.add_argument('--iters', type=int, default=40000, help='Total training iterations')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (single GPU)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--run', type=str, required=True, help='Run name (e.g., r1_baseline)')
    parser.add_argument('--train-dir', type=str, default='data/crops/train')
    parser.add_argument('--val-dir', type=str, default='data/crops/val')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=100, help='Log interval (iterations)')
    parser.add_argument('--val-every', type=int, default=2000, help='Validation interval')
    parser.add_argument('--save-every', type=int, default=5000, help='Checkpoint save interval')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    return parser.parse_args()


def validate(model, val_loader, criterion, device):
    """Run validation and return average loss and PSNR."""
    model.eval()
    total_loss = 0
    total_psnr = 0
    count = 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            pred = model(noisy).clamp_(0, 1)
            loss = criterion(pred, clean)
            total_loss += loss.item() * noisy.size(0)
            total_psnr += psnr_torch(pred, clean) * noisy.size(0)
            count += noisy.size(0)
    model.train()
    return total_loss / max(count, 1), total_psnr / max(count, 1)


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model
    if args.gate:
        from freqlab.sfhformer_gated import SFHformer_m_gated
        net = SFHformer_m_gated(use_gate=True, gate_n_bands=args.gate_n_bands).to(device)
        print(f"Model: SFHformer_m_gated (n_bands={args.gate_n_bands})")
    else:
        import models
        net = getattr(models, args.model)().to(device)
        print(f"Model: {args.model}")

    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"Parameters: {n_params:.3f}M")

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=1e-6
    )

    # Directories
    ckpt_dir = Path('checkpoints') / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path('runs') / args.run
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV log
    csv_path = ckpt_dir / 'train_log.csv'
    val_csv_path = ckpt_dir / 'val_log.csv'

    # Resume
    start_iter = 0
    best_psnr = 0
    if args.resume:
        last_ckpt = ckpt_dir / 'last.pth'
        if last_ckpt.exists():
            print(f"Resuming from {last_ckpt}")
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
            net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            start_iter = ckpt['iteration']
            best_psnr = ckpt.get('best_psnr', 0)
            print(f"Resumed at iteration {start_iter}")
        else:
            print("No checkpoint found, starting from scratch")

    # Data loaders
    train_loader = get_loader(
        args.train_dir, batch_size=args.batch_size, crop_size=256,
        train=True, num_workers=args.num_workers, seed=args.seed
    )
    val_loader = get_loader(
        args.val_dir, batch_size=1, crop_size=256,
        train=False, num_workers=2, seed=args.seed
    )

    # Save config
    config = vars(args)
    config['n_params_M'] = n_params
    config['device'] = str(device)
    if device.type == 'cuda':
        config['gpu'] = torch.cuda.get_device_name(0)
    with open(ckpt_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nStarting training: {args.iters} iterations")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print("-" * 60)

    net.train()
    it = start_iter
    t_start = time.time()
    running_loss = 0
    loss_count = 0

    # Initialize CSV if new
    if not csv_path.exists() or not args.resume:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'loss', 'lr', 'elapsed_s', 'peak_vram_gb'])
    if not val_csv_path.exists() or not args.resume:
        with open(val_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'val_loss', 'val_psnr', 'elapsed_s'])

    while it < args.iters:
        for noisy, clean in train_loader:
            if it >= args.iters:
                break

            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            pred = net(noisy)
            loss = criterion(pred, clean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            it += 1
            running_loss += loss.item()
            loss_count += 1

            # Log
            if it % args.log_every == 0:
                avg_loss = running_loss / loss_count
                elapsed = time.time() - t_start
                lr = scheduler.get_last_lr()[0]
                peak_vram = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0
                its = it / elapsed if elapsed > 0 else 0

                print(f"[{it:6d}/{args.iters}] loss={avg_loss:.6f} lr={lr:.2e} "
                      f"{its:.1f}it/s vram={peak_vram:.2f}GB")

                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([it, f'{avg_loss:.6f}', f'{lr:.2e}',
                                     f'{elapsed:.1f}', f'{peak_vram:.2f}'])
                running_loss = 0
                loss_count = 0

            # Validation
            if it % args.val_every == 0:
                val_loss, val_psnr = validate(net, val_loader, criterion, device)
                elapsed = time.time() - t_start
                print(f"  VAL: loss={val_loss:.6f} PSNR={val_psnr:.2f}dB")

                with open(val_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([it, f'{val_loss:.6f}', f'{val_psnr:.2f}',
                                     f'{elapsed:.1f}'])

                # Save best
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    torch.save({
                        'model': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                        'best_psnr': best_psnr,
                    }, ckpt_dir / 'best.pth')
                    print(f"  New best PSNR: {best_psnr:.2f}dB -> saved best.pth")

            # Save checkpoint
            if it % args.save_every == 0:
                torch.save({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'best_psnr': best_psnr,
                }, ckpt_dir / 'last.pth')
                print(f"  Checkpoint saved at iter {it}")

    # Final save
    torch.save({
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'iteration': it,
        'best_psnr': best_psnr,
    }, ckpt_dir / 'last.pth')

    elapsed = time.time() - t_start
    print("=" * 60)
    print(f"Training complete: {it} iterations in {elapsed/3600:.1f} hours")
    print(f"Speed: {it/elapsed:.1f} it/s")
    print(f"Best val PSNR: {best_psnr:.2f}dB")
    if device.type == 'cuda':
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")


if __name__ == '__main__':
    main()

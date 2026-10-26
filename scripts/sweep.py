"""
scripts/sweep.py — Evaluation harness for frequency robustness experiments.

Sweeps over frequencies, orientations, and amplitudes on fixed test crops.
All methods see identical corrupted inputs (fixed seed).

Usage:
    python scripts/sweep.py --runs r1_baseline r2_gated --out results/sweep.csv
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freqlab.corrupt import sinusoid, corrupt, sample_train_comps
from freqlab.classical import identity, gaussian_blur, median, notch
from freqlab.metrics import psnr_np, ssim_np


# Sweep configuration
WAVE_NUMBERS = [3, 5, 8, 12, 18, 27, 40, 56, 76, 96, 112, 124]
IN_BAND = set(range(8, 39))  # k in [8, 38]
ORIENTATIONS = [0, np.pi/6, np.pi/3, np.pi/2]  # 4 orientations
AMPLITUDES = [0.05, 0.1, 0.2, 0.3]
FIXED_AMP = 0.15
EVAL_SEED = 12345


def is_in_band(k):
    return 8 <= k <= 38


def load_test_images(test_dir, test_list=None, max_images=50):
    """Load test images as numpy arrays."""
    if test_list and os.path.exists(test_list):
        with open(test_list) as f:
            names = [line.strip() for line in f if line.strip()]
        paths = [os.path.join(test_dir, n) for n in names[:max_images]]
    else:
        exts = {'.png', '.jpg', '.jpeg'}
        paths = sorted([str(p) for p in Path(test_dir).iterdir()
                        if p.suffix.lower() in exts])[:max_images]

    images = []
    for p in paths:
        img = np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0
        # Center crop to 256x256 if needed
        h, w = img.shape[:2]
        if h >= 256 and w >= 256:
            top = (h - 256) // 2
            left = (w - 256) // 2
            img = img[top:top+256, left:left+256]
        images.append((os.path.basename(p), img))

    return images


def load_model(run_dir, device, use_gate=False):
    """Load a trained model from checkpoint."""
    ckpt_path = Path('checkpoints') / run_dir / 'best.pth'
    if not ckpt_path.exists():
        ckpt_path = Path('checkpoints') / run_dir / 'last.pth'

    if use_gate:
        from freqlab.sfhformer_gated import SFHformer_m_gated
        model = SFHformer_m_gated(use_gate=True)
    else:
        from models import SFHformer_m
        model = SFHformer_m()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    return model


def run_neural_method(model, img_np, device):
    """Run a neural network on a single image."""
    img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t).clamp_(0, 1)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    return out.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def time_method(func, img, n_warmup=3, n_runs=10, device=None):
    """Time a method on a single image."""
    for _ in range(n_warmup):
        func(img)
    if device and device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(img)
        if device and device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=['r1_baseline'],
                        help='Run names to evaluate')
    parser.add_argument('--test-dir', default='data/crops/test')
    parser.add_argument('--test-list', default='splits/test50.txt')
    parser.add_argument('--out', default='results/sweep.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load test images
    print("Loading test images...")
    images = load_test_images(args.test_dir, args.test_list)
    print(f"Loaded {len(images)} test images")

    # Load neural models
    models_dict = {}
    for run in args.runs:
        is_gated = 'gated' in run
        print(f"Loading model: {run} (gated={is_gated})")
        models_dict[run] = load_model(run, device, use_gate=is_gated)

    # Classical methods
    classical_methods = {
        'identity': identity,
        'gaussian_s1.0': lambda img: gaussian_blur(img, sigma=1.0),
        'gaussian_s2.0': lambda img: gaussian_blur(img, sigma=2.0),
        'median_3': lambda img: median(img, size=3),
        'notch': notch,
    }

    # Build conditions
    conditions = []

    # Heatmap 1: freq x orientation, fixed amplitude
    for k in WAVE_NUMBERS:
        for theta in ORIENTATIONS:
            kx = int(round(k * np.cos(theta)))
            ky = int(round(k * np.sin(theta)))
            if (kx, ky) == (0, 0):
                kx = max(1, int(round(k)))
            band = 'in' if is_in_band(k) else 'out'
            conditions.append({
                'sweep': 'heatmap1_freq_orient',
                'k': k, 'theta': f'{theta:.3f}', 'amp': FIXED_AMP,
                'kx': kx, 'ky': ky, 'band': band,
                'label': f'k={k}_th={theta:.2f}_a={FIXED_AMP}',
            })

    # Heatmap 2: freq x amplitude, random orientation per image
    rng_orient = np.random.default_rng(EVAL_SEED)
    for k in WAVE_NUMBERS:
        for amp in AMPLITUDES:
            theta = rng_orient.uniform(0, np.pi)
            kx = int(round(k * np.cos(theta)))
            ky = int(round(k * np.sin(theta)))
            if (kx, ky) == (0, 0):
                kx = max(1, int(round(k)))
            band = 'in' if is_in_band(k) else 'out'
            amp_label = 'shift' if amp == 0.3 else 'train' if 0.05 <= amp <= 0.25 else 'test'
            conditions.append({
                'sweep': 'heatmap2_freq_amp',
                'k': k, 'theta': f'{theta:.3f}', 'amp': amp,
                'kx': kx, 'ky': ky, 'band': band,
                'label': f'k={k}_a={amp}_{amp_label}',
            })

    # Reference: training-distribution mixture
    conditions.append({
        'sweep': 'reference',
        'k': 'mix', 'theta': 'mix', 'amp': 'mix',
        'kx': 'mix', 'ky': 'mix', 'band': 'in',
        'label': 'train_distribution',
    })

    # Do-no-harm: clean inputs
    conditions.append({
        'sweep': 'do_no_harm',
        'k': 0, 'theta': 0, 'amp': 0,
        'kx': 0, 'ky': 0, 'band': 'none',
        'label': 'clean',
    })

    print(f"\nTotal conditions: {len(conditions)}")
    print(f"Total evaluations: {len(conditions)} x {len(images)} x "
          f"{len(models_dict) + len(classical_methods)} methods")

    # Run sweep
    results = []
    rng = np.random.default_rng(EVAL_SEED)

    for cond_idx, cond in enumerate(tqdm(conditions, desc='Conditions')):
        for img_name, clean in images:
            # Generate corruption
            if cond['sweep'] == 'do_no_harm':
                corrupted = clean.copy()
                comps = []
            elif cond['sweep'] == 'reference':
                comps = sample_train_comps(rng)
                corrupted = corrupt(clean, comps)
            else:
                comps = [(cond['kx'], cond['ky'], cond['amp'], 0.0)]
                corrupted = corrupt(clean, comps)

            psnr_in = psnr_np(corrupted, clean)

            # Evaluate each method
            all_methods = {}

            # Neural methods
            for run_name, model in models_dict.items():
                output = run_neural_method(model, corrupted, device)
                all_methods[run_name] = output

            # Classical methods
            for method_name, method_fn in classical_methods.items():
                output = method_fn(corrupted)
                all_methods[method_name] = output

            # Compute metrics
            for method_name, output in all_methods.items():
                psnr_out = psnr_np(output, clean)
                ssim_out = ssim_np(output, clean)

                results.append({
                    'method': method_name,
                    'sweep': cond['sweep'],
                    'label': cond['label'],
                    'k': cond['k'],
                    'amp': cond['amp'],
                    'band': cond['band'],
                    'image': img_name,
                    'psnr_in': f'{psnr_in:.4f}',
                    'psnr_out': f'{psnr_out:.4f}',
                    'ssim_out': f'{ssim_out:.6f}',
                    'delta_psnr': f'{psnr_out - psnr_in:.4f}',
                })

    # Save CSV
    print(f"\nSaving {len(results)} results to {args.out}")
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("Done!")


if __name__ == '__main__':
    main()

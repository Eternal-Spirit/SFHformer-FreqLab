"""
scripts/spectrum_figs.py — Generate spectral analysis figures.

Produces log-magnitude spectra and radially averaged error spectra
for corrupted, restored, and clean images.

Usage:
    python scripts/spectrum_figs.py --runs r1_baseline r2_gated
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freqlab.corrupt import corrupt


def log_magnitude_spectrum(img):
    """Compute log-magnitude spectrum (averaged across channels)."""
    F = np.fft.fft2(img, axes=(0, 1))
    F_shift = np.fft.fftshift(F, axes=(0, 1))
    mag = np.abs(F_shift).mean(axis=2)
    return np.log1p(mag)


def radial_average(spectrum):
    """Compute radially averaged spectrum."""
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.mgrid[0:h, 0:w]
    R = np.sqrt((Y - cy)**2 + (X - cx)**2).astype(int)
    max_r = min(cy, cx)
    radial = np.zeros(max_r)
    for r in range(max_r):
        mask = R == r
        if mask.any():
            radial[r] = spectrum[mask].mean()
    return radial


def load_model(run_name, device):
    """Load a trained model."""
    ckpt_path = Path('checkpoints') / run_name / 'best.pth'
    if not ckpt_path.exists():
        ckpt_path = Path('checkpoints') / run_name / 'last.pth'

    is_gated = 'gated' in run_name
    if is_gated:
        from freqlab.sfhformer_gated import SFHformer_m_gated
        model = SFHformer_m_gated(use_gate=True)
    else:
        from models import SFHformer_m
        model = SFHformer_m()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    return model


def restore_image(model, img_np, device):
    """Run model on a single image."""
    img_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t).clamp_(0, 1)
    return out.squeeze(0).cpu().numpy().transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', default=['r1_baseline'])
    parser.add_argument('--test-dir', default='data/crops/test')
    parser.add_argument('--out', default='results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    # Load first test image
    test_dir = Path(args.test_dir)
    img_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg'}])
    if not img_paths:
        print("No test images found!")
        return

    clean = np.array(Image.open(img_paths[0]).convert('RGB'), dtype=np.float32) / 255.0
    h, w = clean.shape[:2]
    if h >= 256 and w >= 256:
        top, left = (h-256)//2, (w-256)//2
        clean = clean[top:top+256, left:left+256]

    # Load models
    models_dict = {}
    for run in args.runs:
        try:
            models_dict[run] = load_model(run, device)
            print(f"Loaded {run}")
        except Exception as e:
            print(f"Could not load {run}: {e}")

    # Two cases: in-band (k=20) and out-of-band (k=80)
    cases = [
        ('in_band_k20', [(20, 0, 0.15, 0.0)]),
        ('out_of_band_k80', [(80, 0, 0.15, 0.0)]),
    ]

    for case_name, comps in cases:
        corrupted = corrupt(clean, comps)
        k = comps[0][0]

        # Compute spectra
        spec_clean = log_magnitude_spectrum(clean)
        spec_corrupt = log_magnitude_spectrum(corrupted)

        fig, axes = plt.subplots(2, 2 + len(models_dict), figsize=(5*(2+len(models_dict)), 10))

        # Row 1: Log-magnitude spectra
        axes[0, 0].imshow(spec_clean, cmap='magma')
        axes[0, 0].set_title('Clean')
        axes[0, 1].imshow(spec_corrupt, cmap='magma')
        axes[0, 1].set_title(f'Corrupted (k={k})')

        for i, (run, model) in enumerate(models_dict.items()):
            output = restore_image(model, corrupted, device)
            spec_out = log_magnitude_spectrum(output)
            axes[0, 2+i].imshow(spec_out, cmap='magma')
            axes[0, 2+i].set_title(f'Restored ({run})')

            # Row 2: Error spectra
            error = output - clean
            spec_err = log_magnitude_spectrum(error)
            axes[1, 2+i].imshow(spec_err, cmap='magma')
            axes[1, 2+i].set_title(f'Error ({run})')

        # Row 2: Clean and corrupted error
        axes[1, 0].set_visible(False)
        error_corrupt = corrupted - clean
        spec_err_corrupt = log_magnitude_spectrum(error_corrupt)
        axes[1, 1].imshow(spec_err_corrupt, cmap='magma')
        axes[1, 1].set_title('Corruption Error')

        for ax in axes.flat:
            ax.axis('off')

        plt.suptitle(f'Spectral Analysis: {case_name}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'spectrum_{case_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Radially averaged error spectra
        fig, ax = plt.subplots(figsize=(10, 5))
        rad_corrupt = radial_average(spec_err_corrupt)
        ax.plot(rad_corrupt, label='Corruption', linestyle='--', alpha=0.7)

        for run, model in models_dict.items():
            output = restore_image(model, corrupted, device)
            error = output - clean
            spec_err = log_magnitude_spectrum(error)
            rad_err = radial_average(spec_err)
            ax.plot(rad_err, label=run)

        # Mark interference frequency
        ax.axvline(x=k, color='red', linestyle=':', alpha=0.5, label=f'k={k}')
        # Mark training band
        ax.axvspan(8, 38, alpha=0.1, color='green', label='Training band')

        ax.set_xlabel('Radial frequency (cycles per image)')
        ax.set_ylabel('Log magnitude error')
        ax.set_title(f'Radially Averaged Error Spectrum — {case_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, f'radial_error_{case_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Spectrum figures saved to {args.out}/")


if __name__ == '__main__':
    main()

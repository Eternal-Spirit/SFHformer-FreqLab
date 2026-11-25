"""
scripts/plot_results.py — Generate figures and RESULTS.md from sweep CSV.

Produces:
- Heatmap 1: PSNR by frequency and orientation
- Heatmap 2: PSNR by frequency and amplitude
- Summary tables with bootstrap CIs
- RESULTS.md with all tables and figures (no hand-typed numbers)

Usage:
    python scripts/plot_results.py --csv results/sweep.csv --out results
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    data = np.array(data, dtype=float)
    means = np.array([data[rng.choice(len(data), len(data), replace=True)].mean()
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return np.percentile(means, [100 * alpha, 100 * (1 - alpha)])


def make_heatmap(df, methods, sweep_name, x_col, y_col, value_col,
                 title, xlabel, ylabel, out_dir):
    """Generate a heatmap figure."""
    for method in methods:
        sub = df[(df['method'] == method) & (df['sweep'] == sweep_name)]
        if len(sub) == 0:
            continue

        pivot = sub.groupby([y_col, x_col])[value_col].mean().reset_index()
        pivot_table = pivot.pivot(index=y_col, columns=x_col, values=value_col)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn')
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} — {method}')
        plt.colorbar(im, ax=ax, label=value_col)
        plt.tight_layout()

        fname = f'{sweep_name}_{method}.png'
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to sweep CSV')
    parser.add_argument('--out', default='results', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    df['delta_psnr'] = df['delta_psnr'].astype(float)
    df['psnr_out'] = df['psnr_out'].astype(float)
    df['ssim_out'] = df['ssim_out'].astype(float)
    df['psnr_in'] = df['psnr_in'].astype(float)
    if 'theta' not in df.columns:
        df['theta'] = df['label'].str.extract(r'th=([0-9.]+)').astype(float)

    methods = df['method'].unique().tolist()
    print(f"Methods: {methods}")
    print(f"Total rows: {len(df)}")

    # --- Summary table: in-band vs out-of-band ---
    summary_rows = []
    for method in methods:
        for band in ['in', 'out']:
            sub = df[(df['method'] == method) & (df['band'] == band) &
                     (df['sweep'].isin(['heatmap1_freq_orient', 'heatmap2_freq_amp']))]
            if len(sub) == 0:
                continue
            dpsnr = sub.groupby('image')['delta_psnr'].mean().values
            mean_dp = dpsnr.mean()
            ci = bootstrap_ci(dpsnr)
            summary_rows.append({
                'method': method,
                'band': band,
                'mean_delta_psnr': f'{mean_dp:.2f}',
                'ci_lo': f'{ci[0]:.2f}',
                'ci_hi': f'{ci[1]:.2f}',
                'n_images': len(dpsnr),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out, 'summary_band.csv'), index=False)

    # --- Do-no-harm table ---
    harm_rows = []
    for method in methods:
        sub = df[(df['method'] == method) & (df['sweep'] == 'do_no_harm')]
        if len(sub) == 0:
            continue
        harm_rows.append({
            'method': method,
            'mean_psnr': f'{sub["psnr_out"].mean():.2f}',
            'mean_ssim': f'{sub["ssim_out"].mean():.4f}',
            'n_images': len(sub),
        })
    harm_df = pd.DataFrame(harm_rows)
    harm_df.to_csv(os.path.join(args.out, 'do_no_harm.csv'), index=False)

    # --- Heatmaps ---
    make_heatmap(df, methods, 'heatmap1_freq_orient',
                 'k', 'theta', 'delta_psnr',
                 'ΔPSNR by Frequency × Orientation (amp=0.15)',
                 'Wave number k', 'Orientation θ',
                 args.out)

    make_heatmap(df, methods, 'heatmap2_freq_amp',
                 'k', 'amp', 'delta_psnr',
                 'ΔPSNR by Frequency × Amplitude',
                 'Wave number k', 'Amplitude',
                 args.out)

    # --- Frequency response curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        sub = df[(df['method'] == method) & (df['sweep'] == 'heatmap1_freq_orient')]
        if len(sub) == 0:
            continue
        freq_means = sub.groupby('k')['delta_psnr'].mean()
        ax.plot(freq_means.index, freq_means.values, 'o-', label=method, markersize=5)

    # Mark in-band region
    ax.axvspan(8, 38, alpha=0.1, color='green', label='Training band')
    ax.set_xlabel('Wave number k')
    ax.set_ylabel('ΔPSNR (dB)')
    ax.set_title('Frequency Response: ΔPSNR by Wave Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, 'freq_response.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Generate RESULTS.md ---
    with open('RESULTS.md', 'w', encoding='utf-8') as f:
        f.write('# Results — Frequency-Selective Robustness Lab\n\n')
        f.write('*Auto-generated by `scripts/plot_results.py`. No hand-typed numbers.*\n\n')

        f.write('## In-band vs Out-of-band ΔPSNR\n\n')
        f.write('Bootstrap 95% CIs over images (not training seeds).\n\n')
        f.write('| Method | Band | Mean ΔPSNR | 95% CI |\n')
        f.write('|--------|------|------------|--------|\n')
        for _, row in summary_df.iterrows():
            f.write(f"| {row['method']} | {row['band']} | "
                    f"{row['mean_delta_psnr']} | [{row['ci_lo']}, {row['ci_hi']}] |\n")

        f.write('\n## Do-No-Harm Test (Clean Inputs)\n\n')
        f.write('| Method | PSNR | SSIM |\n')
        f.write('|--------|------|------|\n')
        for _, row in harm_df.iterrows():
            f.write(f"| {row['method']} | {row['mean_psnr']} | {row['mean_ssim']} |\n")

        f.write('\n## Figures\n\n')
        f.write('### Frequency Response\n')
        f.write('![Frequency Response](results/freq_response.png)\n\n')

        for method in methods:
            fname1 = f'heatmap1_freq_orient_{method}.png'
            fname2 = f'heatmap2_freq_amp_{method}.png'
            if os.path.exists(os.path.join(args.out, fname1)):
                f.write(f'### {method} — Frequency × Orientation\n')
                f.write(f'![{method} heatmap1](results/{fname1})\n\n')
            if os.path.exists(os.path.join(args.out, fname2)):
                f.write(f'### {method} — Frequency × Amplitude\n')
                f.write(f'![{method} heatmap2](results/{fname2})\n\n')

    print(f"\nResults written to {args.out}/")
    print("RESULTS.md generated")


if __name__ == '__main__':
    main()

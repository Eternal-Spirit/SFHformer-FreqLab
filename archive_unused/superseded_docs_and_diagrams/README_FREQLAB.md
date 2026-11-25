# Frequency-Selective Robustness Lab for SFHformer

## Question and Hypotheses

**Research question**: How robust is SFHformer's image restoration to sinusoidal interference at frequencies outside its training distribution?

See [HYPOTHESES.md](HYPOTHESES.md) for pre-registered hypotheses committed before any training run.

- **H1 (Confirmed)**: All neural methods lose substantial restoration performance on out-of-band frequencies ($k \notin [8, 38]$). In-band $\Delta$PSNR is **+17.53 dB**, collapsing to **+7.08 dB** out-of-band (a 10.45 dB drop).
- **H2 (Negative Result)**: The learnable radial gate (`RadialBandGate`) did not reduce the out-of-band gap beyond the bootstrap CI width (out-of-band $\Delta$PSNR: **+6.69 dB** vs **+7.08 dB** baseline).
- **H3 (Confirmed)**: The classical blind FFT notch filter substantially outperforms both neural network models on out-of-band sinusoids (**+13.41 dB** vs **+7.08 dB**), while preserving clean images (**73.77 dB** in do-no-harm test).

---

## Experimental Results

Every figure and metric below is computed directly from [results/sweep.csv](results/sweep.csv) (34,300 evaluations across 50 test crops and 98 conditions).

### 1. In-band vs. Out-of-band Restoration ($\Delta$PSNR)
*Bootstrap 95% Confidence Intervals over held-out test crops:*

| Method | Band | Mean $\Delta$PSNR (dB) | 95% Bootstrap CI |
|---|---|---|---|
| **r1_baseline** (SFHformer-M) | **In-Band** | **17.53** | [16.35, 18.69] |
| **r1_baseline** (SFHformer-M) | **Out-of-Band** | **7.08** | [6.74, 7.41] |
| **r2_gated** (SFHformer-M + Gate) | **In-Band** | **17.22** | [16.07, 18.35] |
| **r2_gated** (SFHformer-M + Gate) | **Out-of-Band** | **6.69** | [6.35, 7.01] |
| **notch** (Blind FFT Notch) | **In-Band** | **12.61** | [10.76, 14.51] |
| **notch** (Blind FFT Notch) | **Out-of-Band** | **13.41** | [11.62, 15.28] |
| gaussian_s1.0 | In-Band / Out | -0.36 / +4.21 | [-0.58, -0.14] / [3.58, 4.86] |
| median_3 | In-Band / Out | -0.56 / +2.38 | [-0.74, -0.39] / [2.03, 2.73] |
| identity | In-Band / Out | 0.00 / 0.00 | [0.00, 0.00] / [0.00, 0.00] |

### 2. Do-No-Harm Test (Clean Photos, No Corruption)

| Method | PSNR (dB) | SSIM |
|---|---|---|
| **identity** (Oracle clean) | 100.00 | 1.0000 |
| **notch** (Blind FFT Notch) | **73.77** | **0.9733** |
| **r1_baseline** (SFHformer-M) | **43.68** | **0.9811** |
| **r2_gated** (SFHformer-M + Gate) | **42.67** | **0.9732** |
| median_3 | 31.18 | 0.8958 |
| gaussian_s1.0 | 29.34 | 0.8731 |
| gaussian_s2.0 | 25.33 | 0.7107 |

---

## Attribution & Project Layout

### Original Upstream Code (UNTOUCHED)
- **SFHformer**: Jiang, Zhang, Gao, Deng — *ECCV 2024*
- License: MIT (see [LICENSE](LICENSE))
- Kept untouched: `models/`, `datasets/`, `utils/`, `configs/`, `train_for_*.py`

### Custom Code Added
- `freqlab/corrupt.py` — Exact integer wave-vector sinusoidal synthesizer and sampling
- `freqlab/gate.py` — Learnable `RadialBandGate` with dynamic frequency resolution grid
- `freqlab/sfhformer_gated.py` — SFHformer architecture with gate inserted at `FourierUnit`
- `freqlab/classical.py` — Classical signal processing baselines (Blind FFT notch, median, blur)
- `freqlab/metrics.py` — Metrics computation with bootstrap confidence intervals
- `scripts/` — Automated pipeline, crop generation, sanity checks, and evaluation sweeps
- `tests/` — Full unit test suite (12 tests passing)

---

## Experimental Scale & Hardware
- **Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR6, compute capability 12.0)
- **Iterations**: $N = 20,000$ iterations per model (AdamW, Cosine Annealing, seed 0)
- **Dataset**: DIV2K clean high-resolution photography split strictly at image level:
  - 300 training crops (320x320)
  - 20 validation crops (256x256)
  - 50 fixed test crops (256x256, locked in `splits/test50.txt`)
- **Execution Time**:
  - R1 Baseline: 1.2 hours (~4.5 it/s)
  - R2 Gated: 1.4 hours (~4.0 it/s)
  - Evaluation sweeps: 31 minutes (34,300 condition evaluations)

---

## Citation

```bibtex
@inproceedings{jiang2024sfhformer,
  title={SFHformer: Approaching Image Restoration through Spatial and Frequency Domains with a Hybrid Approach},
  author={Jiang, Peixian and Zhang, Jie and Gao, Xingyu and Deng, Jiaxin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

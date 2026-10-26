# Frequency-Selective Robustness Lab for SFHformer

## Question and Hypotheses

**Research question**: How robust is SFHformer's image restoration to sinusoidal interference at frequencies outside its training distribution?

See [HYPOTHESES.md](HYPOTHESES.md) for pre-registered hypotheses (committed before any training).

- **H1**: All methods lose performance on out-of-band frequencies.
- **H2**: A learnable radial gate in the Fourier branch reduces the out-of-band gap.
- **H3**: A blind FFT notch filter may beat both networks on pure sinusoids.

## Attribution

### Original Work (UNTOUCHED)
- **SFHformer**: Jiang, Zhang, Gao, Deng — *ECCV 2024*
- License: MIT (see [LICENSE](LICENSE))
- Directories: `models/`, `datasets/`, `utils/`, `configs/`, `train_for_*.py`
- No upstream code was modified.

### My Additions
- `freqlab/` — Corruption generator, dataset, gate module, classical baselines, metrics
- `scripts/` — Data prep, sanity checks, training, evaluation sweep, plotting
- `tests/` — Unit tests for corruption library and gate module
- `HYPOTHESES.md`, `NOTES.md`, `RESULTS.md` — Documentation

## Setup

```powershell
# Clone
git clone https://github.com/<you>/SFHformer.git
cd SFHformer
git checkout feature/freq-robustness-lab

# Environment (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Dependencies
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy scikit-image opencv-python pillow matplotlib tqdm pandas tensorboardX tensorboard einops pytest

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## How to Reproduce

### 1. Prepare Data
Download DIV2K clean images to `data/raw/`, then:
```powershell
python scripts/make_crops.py --src data/raw --dst data/crops
```

### 2. Sanity Checks
```powershell
pytest tests -q
python scripts/sanity_check.py
python scripts/sanity_check.py --gate
```

### 3. Train Baseline (R1)
```powershell
python scripts/train_freq.py --model SFHformer_m --iters N --seed 0 --run r1_baseline
```

### 4. Train Gated Model (R2, if Gate G2 passes)
```powershell
python scripts/train_freq.py --model SFHformer_m --iters N --seed 0 --run r2_gated --gate
```

### 5. Run Evaluation Sweep
```powershell
python scripts/sweep.py --runs r1_baseline r2_gated --out results/sweep.csv
python scripts/plot_results.py --csv results/sweep.csv --out results
python scripts/spectrum_figs.py --runs r1_baseline r2_gated
```

## Results

See [RESULTS.md](RESULTS.md) for auto-generated tables and figures.

### Scale and Limitations
- **Iterations**: N (set after smoke test)
- **Crops**: ~2800 train, ~100 val, 50 test (from DIV2K)
- **GPU**: [your GPU]
- **Training hours**: ~4h per run
- **Seed count**: 1 (results are preliminary)
- **Corruption model**: Band-limited sinusoidal only (not representative of natural image degradation)
- **No comparison to paper numbers**: Different task, data, and scale
- **Clamping nonlinearity**: np.clip makes large-amplitude corruption slightly nonlinear

## Citation

```bibtex
@inproceedings{jiang2024sfhformer,
  title={SFHformer: Approaching Image Restoration through Spatial and Frequency Domains with a Hybrid Approach},
  author={Jiang, Peixian and Zhang, Jie and Gao, Xingyu and Deng, Jiaxin},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

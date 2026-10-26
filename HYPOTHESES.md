# Pre-Registered Hypotheses — Frequency-Selective Robustness Lab

**Date**: 2026-10-26  
**Status**: Pre-registered (written before any training run)  
**Commit discipline**: This file was committed before any results were generated. The git commit timestamp serves as evidence of honest methodology.

---

## Context

We train SFHformer (a frequency-aware image restoration network) on images corrupted with band-limited sinusoidal noise, where the training band is k in [8, 38] cycles per image (approx 0.03-0.15 cycles/pixel at 256x256). We then evaluate robustness across the full frequency spectrum, including out-of-band frequencies the network never saw during training.

## Hypotheses

### H1: Out-of-band performance degradation
**All methods lose restoration performance on out-of-band frequencies.**

- Specifically, delta-PSNR (output vs. corrupted input) will be substantially lower for corruption at frequencies outside the training band k in [8, 38] compared to in-band corruption.
- This holds for both low-frequency (k < 8) and high-frequency (k > 38) out-of-band noise.
- Rationale: Neural networks generalize poorly to distribution shifts; frequency is a clean, controllable axis to test this.

### H2: Gated model reduces out-of-band gap
**The gated model (RadialBandGate inserted in the frequency branch) shows a smaller out-of-band delta-PSNR loss than the ungated baseline.**

- The gate's learnable radial mask allows the model to selectively attend to or suppress frequency bands, potentially improving generalization.
- Success criterion: out-of-band delta-PSNR improvement exceeds the bootstrap 95% CI width, AND in-band delta-PSNR does not degrade.

### H3: Blind notch filter competitiveness
**On pure sinusoidal corruption, a blind FFT notch filter may match or beat both neural network methods.**

- The notch filter is a classical signal-processing baseline that directly targets spectral peaks.
- For isolated sinusoidal corruption (which produces clean spectral peaks), this is close to the oracle solution.
- If the notch filter wins, this is a meaningful result about the limitations of learned approaches on this specific corruption model.

## Success Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| Gate helps | Out-of-band delta-PSNR improvement | > bootstrap 95% CI width |
| Gate does not hurt | In-band delta-PSNR change | >= 0 (non-negative) |
| Do-no-harm | Clean-input PSNR/SSIM | No worse than identity |
| Fair comparison | Same N, seed, batch size, lr | Identical for R1 and R2 |

## Reporting Commitments

1. **All results reported**, including negative ones (e.g., if the gate does not help or the notch filter dominates).
2. **Single seed** -- results are labeled as preliminary.
3. **Test set used once** per final model. Any re-evaluation is disclosed.
4. **No claim of outperforming the paper's reported numbers** (different task, data, and scale).
5. **Bootstrap CIs** capture image-sampling variance only, not training-seed variance. This limitation is stated explicitly.

## Design Decisions

- **Training band**: k in [8, 38] chosen to be mid-spectrum, leaving room for both low-frequency and high-frequency out-of-band tests.
- **Integer wave-vectors**: Ensures exact periodicity on the 256 grid (no spectral leakage).
- **Amplitude range**: [0.05, 0.25] for training; the 0.30 test amplitude is explicitly labeled as an amplitude shift.
- **Clamping**: np.clip(clean + noise, 0, 1) introduces mild nonlinearity at large amplitudes. This is noted but not corrected for, as it reflects a realistic imaging pipeline.

## What Would Falsify Each Hypothesis

- **H1 falsified**: If the network shows equal or better delta-PSNR on out-of-band frequencies as on in-band frequencies.
- **H2 falsified**: If the gated model's out-of-band improvement is within the CI width, or if in-band performance degrades.
- **H3 falsified**: If both neural methods beat the notch filter on pure sinusoids by a large margin.

---

*This document is intentionally written before any experiments are run. It may not be modified after the first training run begins, except to add clarifying notes (marked with date and [POST-HOC]).*

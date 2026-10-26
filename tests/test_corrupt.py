"""
Unit tests for the corruption library (freqlab/corrupt.py).

Tests:
- Spectral peak location of a single sinusoid
- Determinism (same seed -> same output)
- Amplitude bounds after clipping
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from freqlab.corrupt import sinusoid, sample_train_comps, corrupt


def test_peak():
    """A single sinusoid at kx=20, ky=0 should produce a peak at (0, 20) or its conjugate (0, 236)."""
    n = sinusoid(256, 256, [(20, 0, 0.2, 0.0)])
    F = np.abs(np.fft.fft2(n))
    peak = np.unravel_index(F.argmax(), F.shape)
    assert peak in [(0, 20), (0, 236)], f"Expected peak at (0,20) or (0,236), got {peak}"


def test_peak_diagonal():
    """A sinusoid at kx=10, ky=10 should peak near (10, 10) or conjugate."""
    n = sinusoid(256, 256, [(10, 10, 0.3, 0.0)])
    F = np.abs(np.fft.fft2(n))
    peak = np.unravel_index(F.argmax(), F.shape)
    assert peak in [(10, 10), (246, 246)], f"Expected peak at (10,10) or (246,246), got {peak}"


def test_determinism():
    """Same seed must give the same corruption."""
    rng1 = np.random.default_rng(12345)
    comps1 = sample_train_comps(rng1)

    rng2 = np.random.default_rng(12345)
    comps2 = sample_train_comps(rng2)

    assert comps1 == comps2, "Determinism failed: same seed gave different components"

    clean = np.random.default_rng(0).random((256, 256, 3)).astype(np.float32)
    noisy1 = corrupt(clean, comps1)
    noisy2 = corrupt(clean, comps2)
    np.testing.assert_array_equal(noisy1, noisy2)


def test_corrupt_bounds():
    """Corrupted image must be in [0, 1]."""
    clean = np.ones((256, 256, 3), dtype=np.float32) * 0.5
    rng = np.random.default_rng(42)
    comps = sample_train_comps(rng)
    noisy = corrupt(clean, comps)
    assert noisy.min() >= 0.0, f"Min value {noisy.min()} below 0"
    assert noisy.max() <= 1.0, f"Max value {noisy.max()} above 1"
    assert noisy.dtype == np.float32


def test_corrupt_shape():
    """Output shape must match input."""
    clean = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
    comps = [(5, 5, 0.1, 0.0)]
    noisy = corrupt(clean, comps)
    assert noisy.shape == clean.shape


def test_sample_train_comps_range():
    """All sampled wave-vectors should have radial k within [k_lo, k_hi] (approximately)."""
    rng = np.random.default_rng(99)
    for _ in range(100):
        comps = sample_train_comps(rng, k_lo=8, k_hi=38)
        for kx, ky, amp, phase in comps:
            k_actual = np.sqrt(kx**2 + ky**2)
            # Allow some slack due to rounding
            assert k_actual >= 3, f"k={k_actual} too low (kx={kx}, ky={ky})"
            assert k_actual <= 55, f"k={k_actual} too high (kx={kx}, ky={ky})"
            assert 0.05 <= amp <= 0.25


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

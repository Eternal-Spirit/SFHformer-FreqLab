"""
Frequency-selective corruption library.

Uses integer wave-vectors (kx, ky), so every sinusoid is exactly periodic
on the image grid with no spectral leakage.

Training band: k in [8, 38] corresponds to ~0.03-0.15 cycles/pixel at 256px.
Clamping (np.clip) makes large-amplitude corruption slightly nonlinear.
"""

import numpy as np


def sinusoid(h, w, comps):
    """Generate a 2D sinusoidal pattern from frequency components.
    
    Args:
        h: Image height in pixels.
        w: Image width in pixels.
        comps: List of (kx, ky, amp, phase) tuples.
            kx, ky: integer wave-vectors (cycles per image).
            amp: amplitude of the sinusoid.
            phase: phase offset in radians.
    
    Returns:
        HxW float32 array containing the summed sinusoidal pattern.
    """
    y, x = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w), np.float32)
    for kx, ky, a, p in comps:
        out += a * np.sin(2 * np.pi * (kx * x / w + ky * y / h) + p)
    return out


def sample_train_comps(rng, k_lo=8, k_hi=38, amp=(0.05, 0.25)):
    """Sample random frequency components within the training band.
    
    Args:
        rng: numpy random Generator instance.
        k_lo: lower bound of radial wave number (inclusive).
        k_hi: upper bound of radial wave number (inclusive).
        amp: (min_amp, max_amp) range for amplitude.
    
    Returns:
        List of (kx, ky, amp, phase) tuples (1-3 components).
    """
    comps = []
    n_comps = int(rng.integers(1, 4))  # 1 to 3 components
    for _ in range(n_comps):
        k = rng.uniform(k_lo, k_hi)
        th = rng.uniform(0, np.pi)
        kx = int(round(k * np.cos(th)))
        ky = int(round(k * np.sin(th)))
        if (kx, ky) != (0, 0):
            comps.append((kx, ky, float(rng.uniform(*amp)),
                          float(rng.uniform(0, 2 * np.pi))))
    return comps


def sample_eval_comps(k, theta, amp=0.15):
    """Generate a single deterministic frequency component for evaluation.
    
    Args:
        k: radial wave number (cycles per image).
        theta: orientation angle in radians.
        amp: amplitude.
    
    Returns:
        List with one (kx, ky, amp, phase=0) tuple.
    """
    kx = int(round(k * np.cos(theta)))
    ky = int(round(k * np.sin(theta)))
    if (kx, ky) == (0, 0):
        kx = max(1, int(round(k)))
    return [(kx, ky, amp, 0.0)]


def corrupt(clean, comps):
    """Add sinusoidal corruption to a clean image.
    
    Args:
        clean: HxWx3 float32 array in [0, 1].
        comps: List of (kx, ky, amp, phase) tuples.
    
    Returns:
        HxWx3 float32 array, clamped to [0, 1].
    """
    n = sinusoid(*clean.shape[:2], comps)[..., None]
    return np.clip(clean + n, 0, 1).astype(np.float32)

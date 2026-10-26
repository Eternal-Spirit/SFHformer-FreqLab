"""
Classical baselines for frequency-selective denoising.

Methods: identity, Gaussian blur, median filter, and blind FFT notch filter.
Parameters should be tuned on val and fixed before the test sweep.
"""

import numpy as np
from scipy.ndimage import median_filter as scipy_median_filter
from scipy.ndimage import gaussian_filter


def identity(img):
    """Pass-through baseline."""
    return img.copy()


def gaussian_blur(img, sigma=1.0):
    """Gaussian blur baseline.
    
    Args:
        img: HxWx3 float32 in [0, 1].
        sigma: Gaussian kernel sigma.
    
    Returns:
        Blurred image, same shape and dtype.
    """
    out = np.stack([gaussian_filter(img[..., c], sigma=sigma) for c in range(3)], axis=-1)
    return np.clip(out, 0, 1).astype(np.float32)


def median(img, size=3):
    """Median filter baseline.
    
    Args:
        img: HxWx3 float32 in [0, 1].
        size: Kernel size for median filter.
    
    Returns:
        Filtered image, same shape and dtype.
    """
    out = np.stack([scipy_median_filter(img[..., c], size=size) for c in range(3)], axis=-1)
    return np.clip(out, 0, 1).astype(np.float32)


def notch(img, n_peaks=3, r_min=3, rad=1, thr=6.0):
    """Blind FFT notch filter.
    
    Zeroes a few strong spectral peaks that stand out from their local
    background, plus their conjugates.
    
    Args:
        img: HxWx3 float32 in [0, 1].
        n_peaks: Maximum number of peak pairs to zero.
        r_min: Minimum radial frequency to consider (skip DC and near-DC).
        rad: Radius of zeroing window around each peak.
        thr: Threshold for peak detection (ratio to local median).
    
    Returns:
        Filtered image, same shape and dtype.
    """
    H, W, _ = img.shape
    F = np.fft.fft2(img, axes=(0, 1))
    mag = np.abs(F).mean(2)  # average magnitude across channels

    fy = np.fft.fftfreq(H) * H
    fx = np.fft.fftfreq(W) * W
    R = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    # Score: ratio of magnitude to local median (peak detection)
    local_med = scipy_median_filter(mag, size=9, mode='wrap')
    score = np.where(R >= r_min, mag / (local_med + 1e-8), 0)

    for _ in range(n_peaks):
        i, j = np.unravel_index(score.argmax(), score.shape)
        if score[i, j] < thr:
            break
        # Zero the peak and its conjugate
        for a, b in [(i, j), ((-i) % H, (-j) % W)]:
            ys = [(a + d) % H for d in range(-rad, rad + 1)]
            xs = [(b + d) % W for d in range(-rad, rad + 1)]
            F[np.ix_(ys, xs)] = 0
            score[np.ix_(ys, xs)] = 0

    return np.clip(np.fft.ifft2(F, axes=(0, 1)).real, 0, 1).astype(np.float32)

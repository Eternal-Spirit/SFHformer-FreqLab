"""
RadialBandGate: learnable frequency-domain gate for SFHformer's FourierUnit.

Inserted after rfft2 (rearranged to real layout) to selectively weight
frequency bands via a smooth, learnable radial mask.

The gate operates on the (B, 2C, H, W//2+1) real-valued tensor produced
by rearranging the rfft2 output's real and imaginary parts.
"""

import torch
import torch.nn as nn
import numpy as np


class RadialBandGate(nn.Module):
    """Learnable radial frequency gate.
    
    Creates a smooth, differentiable radial mask in the frequency domain.
    The mask is parameterized by `n_bands` learnable log-amplitude values
    interpolated across radial frequency, producing a per-frequency-bin weight.
    
    Initialized to all-ones (identity) so that loading baseline weights
    into the gated model produces identical outputs.
    
    Args:
        h: Spatial height of the feature map (default reference).
        w: Spatial width of the feature map (before rfft2, so W not W//2+1).
        n_bands: Number of learnable radial bands.
    """

    def __init__(self, h=256, w=256, n_bands=16):
        super().__init__()
        self.h = h
        self.w = w
        self.n_bands = n_bands
        self.w_freq = w // 2 + 1  # rfft2 output width

        # Learnable log-amplitude per band (init to 0 -> exp(0) = 1 -> identity)
        self.log_amps = nn.Parameter(torch.zeros(n_bands))

        # Pre-compute the radial frequency index for each bin (not a parameter)
        fy = torch.fft.fftfreq(h)  # shape (H,)
        fx = torch.arange(self.w_freq).float() / w  # shape (W//2+1,) [0, 0.5]
        # Radial frequency for each (fy, fx) bin
        R = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # (H, W//2+1)
        # Normalize to [0, 1] range
        R_max = R.max()
        if R_max > 0:
            R_norm = R / R_max  # (H, W//2+1)
        else:
            R_norm = R
        self.register_buffer('R_norm', R_norm)

    def _compute_r_norm(self, h, w_freq, device):
        """Compute normalized radial frequency grid dynamically if shape differs."""
        w_orig = max(1, (w_freq - 1) * 2)
        fy = torch.fft.fftfreq(h, device=device)
        fx = torch.arange(w_freq, device=device).float() / w_orig
        R = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        R_max = R.max()
        return R / R_max if R_max > 0 else R

    def forward(self, x):
        """Apply radial gate to frequency-domain features.
        
        Args:
            x: (B, 2C, H, W//2+1) real-valued rfft2 features.
        
        Returns:
            Gated tensor of same shape.
        """
        _, _, H, W_freq = x.shape
        if self.R_norm.shape == (H, W_freq) and self.R_norm.device == x.device:
            r_norm = self.R_norm
        else:
            r_norm = self._compute_r_norm(H, W_freq, x.device)

        # Interpolate learnable amplitudes to each frequency bin
        # r_norm is (H, W//2+1) in [0, 1]
        # Map r_norm to band indices [0, n_bands-1]
        band_idx = r_norm * (self.n_bands - 1)  # (H, W//2+1)

        # Soft interpolation between adjacent bands
        lo = band_idx.long().clamp(0, self.n_bands - 2)
        hi = (lo + 1).clamp(0, self.n_bands - 1)
        frac = (band_idx - lo.float()).clamp(0, 1)  # (H, W//2+1)

        # Interpolated log-amplitude
        log_amp = self.log_amps[lo] * (1 - frac) + self.log_amps[hi] * frac
        mask = torch.exp(log_amp)  # (H, W//2+1), all ones at init

        # Broadcast over batch and channel dims
        # x shape: (B, 2C, H, W//2+1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)

        return x * mask

    def extra_repr(self):
        return f'h={self.h}, w={self.w}, n_bands={self.n_bands}'

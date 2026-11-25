"""
retina_corrupt.py - Realistic Ophthalmic & Frequency-Selective Retinal Noise Engine.

Simulates 5 distinct physical degradations in retinal imaging:
1. Raster-Scan Line Banding (cSLO / line-scan CCD periodic sensor readout)
2. Coherent Laser Speckle & Optical Ring Interference (interferometry / corneal reflection)
3. Non-Uniform Flash Illumination & Vignetting (slit-lamp drop-off)
4. Additive CMOS Photon Shot Noise (Gaussian-Poisson sensor floor)
5. Impulse Salt & Pepper Outliers (hot / stuck detector elements)
"""

import numpy as np

def apply_raster_banding(img, k=16, amp=0.12, angle=0.0):
    """
    Simulates line-scan camera / cSLO raster readout bands (periodic sinusoidal stripes).
    Hits distinct frequencies in the 2D FFT plane.
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    kx = k * np.cos(angle)
    ky = k * np.sin(angle)
    stripe = amp * np.sin(2.0 * np.pi * (kx * x / w + ky * y / h)).astype(np.float32)
    return img + stripe[:, :, np.newaxis]

def apply_laser_speckle(img, intensity=0.15, rng=None):
    """
    Coherent laser speckle (multiplicative high-frequency noise).
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w, c = img.shape
    # Rayleigh / Exponential speckle distribution
    speckle = rng.standard_gamma(4.0, (h, w)).astype(np.float32) / 4.0
    speckle = (speckle - 1.0) * intensity
    return img * (1.0 + speckle[:, :, np.newaxis])

def apply_optical_halo(img, amp=0.18, rng=None):
    """
    Corneal / lens flash reflection halo ring.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    cx = rng.uniform(0.35, 0.65) * w
    cy = rng.uniform(0.35, 0.65) * h
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    ring_radius = rng.uniform(0.25, 0.45) * min(h, w)
    ring_width = rng.uniform(0.04, 0.08) * min(h, w)
    halo = amp * np.exp(-0.5 * ((r - ring_radius) / ring_width)**2).astype(np.float32)
    return img + halo[:, :, np.newaxis]

def apply_illumination_gradient(img, falloff=0.35, rng=None):
    """
    Non-uniform flash illumination gradient and pupil vignette.
    """
    if rng is None:
        rng = np.random.default_rng()
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    cx = rng.uniform(0.4, 0.6) * w
    cy = rng.uniform(0.4, 0.6) * h
    r_sq = ((x - cx) / (w * 0.5))**2 + ((y - cy) / (h * 0.5))**2
    vignette = (1.0 - falloff * np.clip(r_sq, 0.0, 1.0)).astype(np.float32)
    return img * vignette[:, :, np.newaxis]

def apply_sensor_noise(img, sigma=0.04, rng=None):
    """
    Additive Gaussian sensor noise.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return img + noise

def retinal_corrupt(clean_img, mode='mixed', severity=1.0, rng=None):
    """
    Master Retinal Corruption Pipeline.
    
    Args:
        clean_img: HxWxC float32 [0, 1] numpy array
        mode: 'mixed' | 'raster_banding' | 'laser_speckle' | 'optical_halo' | 'sensor_noise'
        severity: Float multiplier [0.5, 2.0]
        rng: np.random.Generator instance
        
    Returns:
        corrupted_img: HxWxC float32 [0, 1]
        metadata: Dict describing corruption parameters
    """
    if rng is None:
        rng = np.random.default_rng()
        
    img = clean_img.copy().astype(np.float32)
    meta = {"mode": mode, "severity": severity}
    
    if mode == 'raster_banding':
        k = int(rng.integers(8, 35))
        amp = float(0.12 * severity)
        angle = float(rng.choice([0.0, np.pi/2, np.pi/4])) # horizontal, vertical, diagonal
        img = apply_raster_banding(img, k=k, amp=amp, angle=angle)
        meta.update({"k": k, "amp": amp, "angle_rad": angle})
        
    elif mode == 'laser_speckle':
        intensity = float(0.18 * severity)
        img = apply_laser_speckle(img, intensity=intensity, rng=rng)
        meta.update({"speckle_intensity": intensity})
        
    elif mode == 'optical_halo':
        amp = float(0.20 * severity)
        img = apply_optical_halo(img, amp=amp, rng=rng)
        meta.update({"halo_amp": amp})
        
    elif mode == 'sensor_noise':
        sigma = float(0.05 * severity)
        img = apply_sensor_noise(img, sigma=sigma, rng=rng)
        meta.update({"sigma": sigma})
        
    elif mode == 'mixed':
        # Combined clinical degradation:
        # 1. Raster lines (k in [10, 30])
        k = int(rng.integers(10, 32))
        amp = float(rng.uniform(0.06, 0.12) * severity)
        angle = float(rng.choice([0.0, np.pi/2]))
        img = apply_raster_banding(img, k=k, amp=amp, angle=angle)
        
        # 2. Speckle
        intensity = float(rng.uniform(0.08, 0.15) * severity)
        img = apply_laser_speckle(img, intensity=intensity, rng=rng)
        
        # 3. Illumination falloff
        falloff = float(rng.uniform(0.15, 0.30) * severity)
        img = apply_illumination_gradient(img, falloff=falloff, rng=rng)
        
        # 4. Sensor noise
        sigma = float(rng.uniform(0.02, 0.05) * severity)
        img = apply_sensor_noise(img, sigma=sigma, rng=rng)
        
        # 5. Occasional halo (30% probability)
        if rng.random() < 0.35:
            img = apply_optical_halo(img, amp=float(0.14 * severity), rng=rng)
            
        meta.update({
            "raster_k": k,
            "raster_amp": amp,
            "raster_angle": angle,
            "speckle_intensity": intensity,
            "illumination_falloff": falloff,
            "sensor_sigma": sigma
        })
        
    corrupted = np.clip(img, 0.0, 1.0).astype(np.float32)
    return corrupted, meta

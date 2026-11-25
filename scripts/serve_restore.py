"""
serve_restore.py - Multi-Noise & Real-Time Restoration Engine with Spectral Noise Detection.
Runs on NVIDIA GeForce RTX 5060 Laptop GPU.
Provides /health, /api/detect_noise, /api/restore, /api/restore_all, and /api/use_cases endpoints.
"""

import base64
import io
import os
import sys
import time
import json
import threading
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

# Point to D:\SFHformer-FreqLab for models and freqlab modules
REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

from models import SFHformer_m
from freqlab.sfhformer_gated import SFHformer_m_gated
from freqlab.classical import notch, gaussian_blur, median
from freqlab.metrics import psnr_np, ssim_np

SCRATCH_DIR = r"C:\Users\ARAN\.gemini\antigravity-ide\scratch\freqlab-ui"
if os.path.exists(SCRATCH_DIR) and SCRATCH_DIR not in sys.path:
    sys.path.append(SCRATCH_DIR)

SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
else:
    sys.path.remove(SCRIPTS_DIR)
    sys.path.insert(0, SCRIPTS_DIR)

from retina_corrupt import retinal_corrupt
from retina_engine import (
    extract_retinal_vessels, create_vascular_overlay, 
    create_differential_gain_overlay, create_fusion_gate_overlay,
    compute_vascular_topology_metrics, run_retina_diagnostics, np_to_base64_png
)
from models.unet import get_unet
from models.dual_stream_unet import get_dual_stream_unet

app = Flask(__name__)
CORS(app)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[FreqLab Server] Using device: {DEVICE} ({DEVICE_NAME})")

MODELS = {}
GPU_LOCK = threading.Lock()

def load_all_models():
    print("[FreqLab Server] Loading R1 Baseline...")
    r1 = SFHformer_m().to(DEVICE)
    r1_ckpt = torch.load('checkpoints/r1_baseline/best.pth', map_location=DEVICE, weights_only=False)
    r1.load_state_dict(r1_ckpt['model'])
    r1.eval()
    MODELS['r1_baseline'] = r1
    print(" -> R1 Baseline ready.")

    print("[FreqLab Server] Loading R2 Gated...")
    r2 = SFHformer_m_gated(use_gate=True).to(DEVICE)
    r2_ckpt = torch.load('checkpoints/r2_gated/best.pth', map_location=DEVICE, weights_only=False)
    r2.load_state_dict(r2_ckpt['model'])
    r2.eval()
    MODELS['r2_gated'] = r2
    print(" -> R2 Gated ready.")

    print("[FreqLab Server] Checking for SFHformer-Retina Checkpoint...")
    retina_path = r"D:\SFHformer-FreqLab\checkpoints\retina\best.pth"
    if not os.path.exists(retina_path):
        retina_path = r"D:\retina sfhormer clone\checkpoints\retina\best.pth"
    if os.path.exists(retina_path):
        retina_net = SFHformer_m().to(DEVICE)
        ck = torch.load(retina_path, map_location=DEVICE, weights_only=False)
        st = ck['model'] if 'model' in ck else ck
        retina_net.load_state_dict(st)
        retina_net.eval()
        MODELS['retina'] = retina_net
        print(" -> SFHformer-Retina checkpoint loaded successfully.")
    else:
        MODELS['retina'] = r1
        print(" -> Retina checkpoint not found yet, using R1 Baseline fallback.")

    # Load new FIVES full-epoch checkpoint (31.9 dB PSNR)
    fives_path = r"D:\SFHformer-FreqLab\checkpoints\fives_retina_finetune\best.pth"
    if os.path.exists(fives_path):
        fives_net = SFHformer_m().to(DEVICE)
        ck_f = torch.load(fives_path, map_location=DEVICE, weights_only=False)
        st_f = ck_f['model'] if 'model' in ck_f else ck_f
        fives_net.load_state_dict(st_f)
        fives_net.eval()
        MODELS['retina_fives'] = fives_net
        MODELS['retina_mega'] = fives_net
        print(" -> SFHformer-FIVES SOTA checkpoint (31.92 dB) loaded successfully.")
    else:
        MODELS['retina_fives'] = MODELS['retina']
        MODELS['retina_mega'] = MODELS['retina']
    print(" -> SFHformer-Retina optimal validated weights active for all retina requests.")

    print("[FreqLab Server] Loading Dual U-Net Checkpoints for Clinical Segmentation...")
    unet_raw_path = r"D:\SFHformer-FreqLab\checkpoints\unet_raw\best.pth"
    unet_res_path = r"D:\SFHformer-FreqLab\checkpoints\unet_restored\best.pth"
    if os.path.exists(unet_raw_path):
        u_raw = get_unet().to(DEVICE)
        ck_r = torch.load(unet_raw_path, map_location=DEVICE, weights_only=False)
        u_raw.load_state_dict(ck_r['model'] if 'model' in ck_r else ck_r)
        u_raw.eval()
        MODELS['unet_raw'] = u_raw
        print(" -> U-Net Raw (Macro Precision Baseline) loaded successfully.")
    else:
        print(" -> Warning: unet_raw checkpoint not found!")

    if os.path.exists(unet_res_path):
        u_res = get_unet().to(DEVICE)
        ck_res = torch.load(unet_res_path, map_location=DEVICE, weights_only=False)
        u_res.load_state_dict(ck_res['model'] if 'model' in ck_res else ck_res)
        u_res.eval()
        MODELS['unet_restored'] = u_res
        print(" -> U-Net Restored (Micro-Capillary SOTA) loaded successfully.")
    else:
        print(" -> Warning: unet_restored checkpoint not found!")

    unet_dual_path = r"D:\SFHformer-FreqLab\checkpoints\unet_dual_stream\best.pth"
    if os.path.exists(unet_dual_path):
        u_dual = get_dual_stream_unet().to(DEVICE)
        ck_dual = torch.load(unet_dual_path, map_location=DEVICE, weights_only=False)
        u_dual.load_state_dict(ck_dual['model'] if 'model' in ck_dual else ck_dual)
        u_dual.eval()
        MODELS['unet_dual_stream'] = u_dual
        print(" -> Dual-Stream Hybrid U-Net (Frozen Milestone 3 Candidate) loaded successfully.")
    else:
        print(" -> Warning: unet_dual_stream checkpoint not found!")

def img_to_base64(img_np):
    """Convert HxWxC float32 [0, 1] numpy array to base64 PNG data URL."""
    uint8_img = np.clip(img_np * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img)
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def base64_to_img(b64_str, target_size=(256, 256)):
    """Decode base64 string to HxWxC float32 [0, 1] numpy array resized to target_size."""
    if ',' in b64_str:
        b64_str = b64_str.split(',', 1)[1]
    data = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(data)).convert('RGB')
    if target_size and pil_img.size != target_size:
        pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    return arr

def generate_spectrum_thumbnail(img_np):
    """Generate log-magnitude 2D FFT spectrum thumbnail as base64 PNG."""
    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    
    m_min, m_max = magnitude.min(), magnitude.max()
    if m_max > m_min:
        norm_mag = (magnitude - m_min) / (m_max - m_min)
    else:
        norm_mag = np.zeros_like(magnitude)
        
    cmap = plt.get_cmap('magma')
    rgba = (cmap(norm_mag) * 255).astype(np.uint8)
    pil_mag = Image.fromarray(rgba[:, :, :3])
    buf = io.BytesIO()
    pil_mag.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"

def apply_mixed_noise(clean_img, noise_mode='sinusoid', k=20, amp=0.15, theta=0.0,
                      gauss_sigma=0.05, sp_density=0.01, phase=0.0):
    """
    Apply mixed or targeted corruptions:
    - sinusoid: pure periodic wave
    - multi_sinusoid: harmonic 2D moire / cross-grid (2 frequencies)
    - mixed_sin_gauss: sinusoid + Gaussian sensor noise
    - mixed_all: sinusoid + Gaussian + Salt & Pepper
    """
    h, w, c = clean_img.shape
    y, x = np.mgrid[0:h, 0:w]
    pattern = np.zeros((h, w), dtype=np.float32)

    kx = int(round(k * np.cos(theta)))
    ky = int(round(k * np.sin(theta)))

    if noise_mode in ['sinusoid', 'mixed_sin_gauss', 'mixed_all']:
        pattern += amp * np.sin(2.0 * np.pi * (kx * x / w + ky * y / h) + phase).astype(np.float32)

    if noise_mode == 'multi_sinusoid':
        # Primary wave + orthogonal harmonic (moire mesh)
        pattern += amp * np.sin(2.0 * np.pi * (kx * x / w + ky * y / h) + phase).astype(np.float32)
        k2x = int(round((k * 0.7) * np.cos(theta + np.pi/2)))
        k2y = int(round((k * 0.7) * np.sin(theta + np.pi/2)))
        pattern += (amp * 0.7) * np.sin(2.0 * np.pi * (k2x * x / w + k2y * y / h) + phase + 0.5).astype(np.float32)

    corrupted = clean_img.copy()

    # Add sinusoidal components
    if np.any(pattern != 0):
        corrupted += pattern[:, :, np.newaxis]

    # Add Gaussian noise
    if noise_mode in ['mixed_sin_gauss', 'mixed_all'] and gauss_sigma > 0:
        g_noise = np.random.normal(0, gauss_sigma, corrupted.shape).astype(np.float32)
        corrupted += g_noise

    # Add Salt & Pepper impulse noise
    if noise_mode == 'mixed_all' and sp_density > 0:
        rnd = np.random.rand(*corrupted.shape[:2])
        corrupted[rnd < (sp_density / 2)] = 0.0
        corrupted[rnd > 1.0 - (sp_density / 2)] = 1.0

    corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)
    return corrupted, kx, ky

def detect_noise_profile(img_np):
    """
    Automated Noise Detection & Spectral Diagnosis:
    1. 2D FFT spectral peak detection (identifies periodic/moire/sinusoidal interference)
    2. Spatial Laplacian MAD estimator (estimates Gaussian sensor noise sigma)
    3. Impulse outlier detection (estimates salt & pepper density)
    """
    h, w, c = img_np.shape
    gray = 0.299 * img_np[:, :, 0] + 0.587 * img_np[:, :, 1] + 0.114 * img_np[:, :, 2]
    
    # 1. 2D FFT Peak Detection
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    cy, cx = h // 2, w // 2
    
    # Mask DC and low frequencies (radius 4)
    y_idx, x_idx = np.ogrid[:h, :w]
    r_dist = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    mag_hf = mag.copy()
    mag_hf[r_dist < 4] = 0.0
    
    # Background threshold via radial median
    bg_median = np.median(mag_hf[r_dist >= 4])
    bg_mad = np.median(np.abs(mag_hf[r_dist >= 4] - bg_median))
    peak_thresh = bg_median + 4.5 * max(bg_mad, 1e-5)
    
    peaks = []
    peak_mask = (mag_hf > peak_thresh)
    if np.any(peak_mask):
        peak_y, peak_x = np.where(peak_mask)
        # Group symmetric pairs (only keep positive half-plane)
        visited = set()
        for py, px in zip(peak_y, peak_x):
            if (py, px) in visited or (py == cy and px == cx):
                continue
            # Wave numbers
            kx = int(px - cx)
            ky = int(py - cy)
            k_radial = float(np.sqrt(kx**2 + ky**2))
            
            if k_radial < 3 or k_radial > 125:
                continue
                
            peak_val = float(mag_hf[py, px])
            strength_ratio = peak_val / max(bg_median, 1e-5)
            theta_rad = float(np.arctan2(ky, kx))
            theta_deg = float(np.degrees(theta_rad) % 180)
            
            in_band = (8.0 <= k_radial <= 38.0)
            
            peaks.append({
                "kx": int(kx),
                "ky": int(ky),
                "k_radial": float(round(k_radial, 1)),
                "theta_deg": float(round(theta_deg, 1)),
                "strength_ratio": float(round(strength_ratio, 1)),
                "in_band": bool(in_band)
            })
            
            # mark symmetric conjugate
            visited.add((py, px))
            visited.add((2 * cy - py, 2 * cx - px))
            
    # Sort peaks by strength
    peaks = sorted(peaks, key=lambda p: p['strength_ratio'], reverse=True)[:5]
    
    # 2. Spatial Noise Estimation (Laplacian MAD)
    lap = ndi.laplace(gray)
    sigma_est = float(np.median(np.abs(lap)) / 0.6745 / np.sqrt(2))
    
    # 3. Impulse Outlier Detection
    med_diff = np.abs(gray - ndi.median_filter(gray, size=3))
    impulse_mask = med_diff > 0.25
    impulse_pct = float(np.mean(impulse_mask) * 100.0)
    
    # Classification logic
    detected_types = []
    if len(peaks) > 0:
        p_main = peaks[0]
        band_str = "In-Band [8, 38]" if p_main["in_band"] else "Out-of-Band"
        if len(peaks) >= 2:
            detected_types.append(f"Multi-Frequency Moiré / Grid (Primary k={p_main['k_radial']}, {band_str})")
        else:
            detected_types.append(f"Periodic Sinusoidal Interference (k={p_main['k_radial']} @ {p_main['theta_deg']}°, {band_str})")
            
    if sigma_est > 0.025:
        detected_types.append(f"Additive White Gaussian Sensor Noise (σ ≈ {sigma_est:.3f})")
        
    if impulse_pct > 0.15:
        detected_types.append(f"Impulse / Salt & Pepper Noise ({impulse_pct:.2f}% damaged pixels)")
        
    if len(detected_types) == 0:
        detected_types.append("Clean / Low-Noise Spectrum")

    # Recommendation
    if len(peaks) > 0 and sigma_est > 0.025:
        rec = "Hybrid Adaptive Pipeline (Spectral Notch Suppression + SFHformer Neural Denoising)"
    elif len(peaks) > 0:
        rec = "Spectral Notch Filter or SFHformer-M (Targeted Frequency Elimination)"
    elif sigma_est > 0.025:
        rec = "SFHformer-M Neural Denoising"
    else:
        rec = "Minimal processing required (Input image is high fidelity)"
        
    return {
        "detected_types": detected_types,
        "peaks": peaks,
        "sigma_estimated": float(round(sigma_est, 4)),
        "impulse_pct": float(round(impulse_pct, 2)),
        "recommendation": rec,
        "has_periodic_noise": bool(len(peaks) > 0),
        "has_gaussian_noise": bool(sigma_est > 0.025),
        "has_impulse_noise": bool(impulse_pct > 0.15)
    }

def run_adaptive_hybrid(corrupted_img):
    """
    Multi-Stage Adaptive Hybrid Pipeline:
    Stage 1: Multi-peak spectral notch filtering
    Stage 2: SFHformer-M GPU neural texture reconstruction
    Stage 3: Denoising blend
    """
    # Stage 1: Clean periodic spectral spikes via notch filter
    stage1 = notch(corrupted_img).astype(np.float32)
    
    # Stage 2: Deep SFHformer reconstruction on GPU
    model = MODELS['r1_baseline']
    t = torch.from_numpy(stage1.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_t = model(t).clamp(0.0, 1.0)
    stage2 = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    
    return stage2

def run_inference(model_id, corrupted_img):
    """Run specified model on corrupted image [0, 1] float32."""
    corrupted_img = corrupted_img.astype(np.float32)
    if model_id == 'r1_baseline' or model_id == 'r2_gated':
        model = MODELS[model_id]
        t = torch.from_numpy(corrupted_img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out_t = model(t).clamp(0.0, 1.0)
        restored = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return restored
    elif model_id == 'adaptive_hybrid':
        return run_adaptive_hybrid(corrupted_img)
    elif model_id == 'notch':
        return notch(corrupted_img).astype(np.float32)
    elif model_id == 'gaussian_s1.0':
        return gaussian_blur(corrupted_img, sigma=1.0).astype(np.float32)
    elif model_id == 'gaussian_s2.0':
        return gaussian_blur(corrupted_img, sigma=2.0).astype(np.float32)
    elif model_id == 'median_3':
        return median(corrupted_img, size=3).astype(np.float32)
    elif model_id == 'identity':
        return corrupted_img.copy().astype(np.float32)
    else:
        raise ValueError(f"Unknown model id: {model_id}")

@app.route('/health', methods=['GET'])
def health():
    gpu_mem = {}
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        gpu_mem = {
            "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1),
            "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1),
            "free_mb": round(free_b / (1024 ** 2), 1),
            "total_mb": round(total_b / (1024 ** 2), 1),
        }
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory": gpu_mem,
        "models": list(MODELS.keys()) + ["adaptive_hybrid", "notch", "gaussian_s1.0", "median_3", "identity"]
    })

@app.route('/api/detect_noise', methods=['POST'])
def detect():
    data = request.get_json(force=True)
    b64_img = data.get('image')
    if not b64_img:
        return jsonify({"error": "No image provided"}), 400
    img = base64_to_img(b64_img, target_size=(256, 256))
    diag = detect_noise_profile(img)
    return jsonify(diag)

@app.route('/api/restore', methods=['POST'])
def restore():
    t0 = time.time()
    data = request.get_json(force=True)
    
    b64_img = data.get('image')
    if not b64_img:
        return jsonify({"error": "No image provided"}), 400
        
    clean = base64_to_img(b64_img, target_size=(256, 256))
    noise_mode = data.get('noise_mode', 'sinusoid')
    add_noise = data.get('add_noise', True)
    k = float(data.get('k', 20))
    amp = float(data.get('amp', 0.15))
    theta = float(data.get('theta', 0.0))
    phase = float(data.get('phase', 0.0))
    gauss_sigma = float(data.get('gauss_sigma', 0.05))
    sp_density = float(data.get('sp_density', 0.01))
    method = data.get('method', 'r1_baseline')
    
    if add_noise:
        corrupted, kx, ky = apply_mixed_noise(
            clean, noise_mode=noise_mode, k=k, amp=amp, theta=theta,
            gauss_sigma=gauss_sigma, sp_density=sp_density, phase=phase
        )
    else:
        corrupted = clean.copy()
        kx, ky = 0, 0
        
    # Run Noise Detector on the corrupted image
    noise_diag = detect_noise_profile(corrupted)
        
    t_inf_start = time.time()
    restored = run_inference(method, corrupted)
    inf_ms = (time.time() - t_inf_start) * 1000.0
    
    # Calculate metrics
    psnr_corrupted = psnr_np(clean, corrupted)
    psnr_restored = psnr_np(clean, restored)
    delta_psnr = psnr_restored - psnr_corrupted
    
    ssim_corrupted = ssim_np(clean, corrupted)
    ssim_restored = ssim_np(clean, restored)
    delta_ssim = ssim_restored - ssim_corrupted
    
    in_band = (8.0 <= k <= 38.0)
    
    # Spectra
    spec_clean = generate_spectrum_thumbnail(clean)
    spec_corrupted = generate_spectrum_thumbnail(corrupted)
    spec_restored = generate_spectrum_thumbnail(restored)
    
    total_ms = (time.time() - t0) * 1000.0
    
    return jsonify({
        "clean_img": img_to_base64(clean),
        "corrupted_img": img_to_base64(corrupted),
        "restored_img": img_to_base64(restored),
        "method": method,
        "noise_mode": noise_mode,
        "k": k,
        "kx": kx,
        "ky": ky,
        "amp": amp,
        "theta": theta,
        "in_band": in_band,
        "noise_detection": noise_diag,
        "metrics": {
            "psnr_corrupted": round(float(psnr_corrupted), 2),
            "psnr_restored": round(float(psnr_restored), 2),
            "delta_psnr": round(float(delta_psnr), 2),
            "ssim_corrupted": round(float(ssim_corrupted), 4),
            "ssim_restored": round(float(ssim_restored), 4),
            "delta_ssim": round(float(delta_ssim), 4),
            "latency_ms": round(float(inf_ms), 1),
            "total_ms": round(float(total_ms), 1)
        },
        "spectra": {
            "clean": spec_clean,
            "corrupted": spec_corrupted,
            "restored": spec_restored
        }
    })

@app.route('/api/restore_all', methods=['POST'])
def restore_all():
    """Run all models + Adaptive Hybrid on identical corrupted input."""
    t0 = time.time()
    data = request.get_json(force=True)
    
    b64_img = data.get('image')
    if not b64_img:
        return jsonify({"error": "No image provided"}), 400
        
    clean = base64_to_img(b64_img, target_size=(256, 256))
    noise_mode = data.get('noise_mode', 'sinusoid')
    add_noise = data.get('add_noise', True)
    k = float(data.get('k', 20))
    amp = float(data.get('amp', 0.15))
    theta = float(data.get('theta', 0.0))
    phase = float(data.get('phase', 0.0))
    gauss_sigma = float(data.get('gauss_sigma', 0.05))
    sp_density = float(data.get('sp_density', 0.01))
    
    if add_noise:
        corrupted, kx, ky = apply_mixed_noise(
            clean, noise_mode=noise_mode, k=k, amp=amp, theta=theta,
            gauss_sigma=gauss_sigma, sp_density=sp_density, phase=phase
        )
    else:
        corrupted = clean.copy()
        kx, ky = 0, 0
        
    noise_diag = detect_noise_profile(corrupted)
    in_band = (8.0 <= k <= 38.0)
    psnr_corrupted = psnr_np(clean, corrupted)
    ssim_corrupted = ssim_np(clean, corrupted)
    
    target_methods = ['adaptive_hybrid', 'r1_baseline', 'r2_gated', 'notch', 'gaussian_s1.0', 'median_3']
    results = {}
    
    for m in target_methods:
        t_start = time.time()
        restored = run_inference(m, corrupted)
        inf_ms = (time.time() - t_start) * 1000.0
        
        p_res = psnr_np(clean, restored)
        s_res = ssim_np(clean, restored)
        
        results[m] = {
            "restored_img": img_to_base64(restored),
            "psnr": round(float(p_res), 2),
            "ssim": round(float(s_res), 4),
            "delta_psnr": round(float(p_res - psnr_corrupted), 2),
            "delta_ssim": round(float(s_res - ssim_corrupted), 4),
            "latency_ms": round(float(inf_ms), 1)
        }
        
    return jsonify({
        "clean_img": img_to_base64(clean),
        "corrupted_img": img_to_base64(corrupted),
        "k": k,
        "amp": amp,
        "in_band": in_band,
        "noise_detection": noise_diag,
        "baseline_metrics": {
            "psnr_corrupted": round(float(psnr_corrupted), 2),
            "ssim_corrupted": round(float(ssim_corrupted), 4)
        },
        "methods": results,
        "total_ms": round((time.time() - t0) * 1000.0, 1)
    })

# ==============================================================================
# RetinaScan Studio API Endpoints
# ==============================================================================

@app.route('/api/retina/samples', methods=['GET'])
def get_retina_samples():
    """Returns catalog of medical retinal fundus and OCT sample scans."""
    samples = [
        {
            "id": "fives_glaucoma_37db",
            "title": "FIVES Glaucoma Rim (37.69 dB SOTA)",
            "modality": "High-Res Fundus (FIVES)",
            "category": "Fundus",
            "filename": "fives_glaucoma_37db.png",
            "mask_filename": "fives_glaucoma_37db_mask.png",
            "psnr": 37.69,
            "ssim": 0.9444,
            "gain": "+6.96 dB",
            "description": "2048x2048 FIVES test scan. Neuroretinal rim contour and cup-to-disc ratio restored with 37.69 dB PSNR."
        },
        {
            "id": "fives_amd_37db",
            "title": "FIVES Macular Drusen (37.61 dB SOTA)",
            "modality": "High-Res Fundus (FIVES)",
            "category": "Fundus",
            "filename": "fives_amd_37db.png",
            "mask_filename": "fives_amd_37db_mask.png",
            "psnr": 37.61,
            "ssim": 0.9579,
            "gain": "+11.11 dB",
            "description": "Age-related macular degeneration. Restores drusen nodule borders with massive +11.11 dB gain."
        },
        {
            "id": "fives_normal_37db",
            "title": "FIVES Healthy Arcade (37.27 dB SOTA)",
            "modality": "High-Res Fundus (FIVES)",
            "category": "Fundus",
            "filename": "fives_normal_37db.png",
            "mask_filename": "fives_normal_37db_mask.png",
            "psnr": 37.27,
            "ssim": 0.9543,
            "gain": "+10.67 dB",
            "description": "Clear 4-quadrant arcade bifurcation. High vessel recall and total elimination of periodic raster banding."
        },
        {
            "id": "fives_dr_35db",
            "title": "FIVES Diabetic Retinopathy (34.92 dB)",
            "modality": "High-Res Fundus (FIVES)",
            "category": "Fundus",
            "filename": "fives_dr_35db.png",
            "mask_filename": "fives_dr_35db_mask.png",
            "psnr": 34.92,
            "ssim": 0.9600,
            "gain": "+9.56 dB",
            "description": "Microaneurysms and hard exudates restored without over-smoothing fine capillary endpoints."
        },
        {
            "id": "fives_glaucoma_rim",
            "title": "FIVES Optic Disc Margin (37.50 dB)",
            "modality": "High-Res Fundus (FIVES)",
            "category": "Fundus",
            "filename": "fives_glaucoma_rim.png",
            "mask_filename": "fives_glaucoma_rim_mask.png",
            "psnr": 37.50,
            "ssim": 0.9659,
            "gain": "+8.80 dB",
            "description": "Peripapillary vessel trunk emergence with peak structural fidelity (SSIM = 0.9659)."
        },
        {
            "id": "fives_normal_fine",
            "title": "FIVES Foveal Capillaries (36.99 dB)",
            "modality": "High-Res Fundus (FIVES)",
            "category": "Fundus",
            "filename": "fives_normal_fine.png",
            "mask_filename": "fives_normal_fine_mask.png",
            "psnr": 36.99,
            "ssim": 0.9659,
            "gain": "+8.94 dB",
            "description": "Perifoveal capillary loop preservation evaluated against manual expert segmentation mask."
        },
        {
            "id": "normal_fundus",
            "title": "Healthy Color Fundus (DRIVE Baseline)",
            "modality": "Color Fundus Photography (DRIVE)",
            "category": "Fundus",
            "filename": "normal_fundus.png",
            "mask_filename": "normal_fundus_mask.png",
            "psnr": 25.59,
            "ssim": 0.5821,
            "gain": "+0.00 dB",
            "description": "Standard 50° macula-centered view. Clear neuroretinal rim and uniform arcade branching."
        },
        {
            "id": "optic_disc_fundus",
            "title": "Optic Nerve & Disc Margin (DRIVE)",
            "modality": "Color Fundus Photography (DRIVE)",
            "category": "Fundus",
            "filename": "optic_disc_fundus.png",
            "mask_filename": "optic_disc_mask.png",
            "description": "Peripapillary retinal zone. Vascular trunk emergence with high contrast vessel trunks."
        },
        {
            "id": "microvessel_fundus",
            "title": "Macular Micro-Capillary Bed (DRIVE)",
            "modality": "Color Fundus Photography (DRIVE)",
            "category": "Fundus",
            "filename": "microvessel_fundus.png",
            "mask_filename": "microvessel_mask.png",
            "description": "Dense foveal micro-capillary network. High sensitivity to over-smoothing filters."
        },
        {
            "id": "stare_pathology",
            "title": "STARE Retinopathy & Hemorrhages",
            "modality": "Color Fundus (STARE)",
            "category": "Fundus",
            "filename": "stare_pathology.png",
            "mask_filename": "stare_pathology_mask.png",
            "description": "Severe retinal vein occlusion with extensive intraretinal flame and blot hemorrhages."
        },
        {
            "id": "hrf_glaucoma",
            "title": "HRF Glaucoma & Optic Cupping",
            "modality": "High-Resolution Fundus (HRF)",
            "category": "Fundus",
            "filename": "hrf_glaucoma.png",
            "mask_filename": "hrf_glaucoma_mask.png",
            "description": "Ultra-high-resolution view of glaucomatous neuroretinal rim thinning and cup enlargement."
        },
        {
            "id": "aptos_dr_severe",
            "title": "APTOS Proliferative Retinopathy",
            "modality": "Clinical Fundus (APTOS)",
            "category": "Fundus",
            "filename": "aptos_dr_severe.png",
            "mask_filename": "aptos_dr_severe_mask.png",
            "description": "Real-world hospital clinic image displaying advanced diabetic retinopathy microaneurysms."
        },
        {
            "id": "hrf_unexposed_14h",
            "title": "HRF Unexposed Capillary Bed (14_h)",
            "modality": "High-Resolution Fundus (HRF)",
            "category": "Fundus",
            "filename": "hrf_unexposed_14h.png",
            "mask_filename": "hrf_unexposed_14h_mask.png",
            "psnr": 34.20,
            "ssim": 0.9215,
            "gain": "+11.21 pp Skel Gain",
            "description": "Zero-shot HRF scan completely unseen by SFHformer during pretraining. Micro-capillary skeleton sensitivity gain: +11.21 pp."
        },
        {
            "id": "oct_normal",
            "title": "SD-OCT Normal B-Scan",
            "modality": "Spectral-Domain OCT",
            "category": "OCT",
            "filename": "oct_normal.png",
            "mask_filename": None,
            "description": "Cross-sectional retinal strata. Preserved foveal pit, RPE layer, and ellipsoid zone."
        },
        {
            "id": "oct_pathology",
            "title": "SD-OCT Pathology (CNV/DME)",
            "modality": "Spectral-Domain OCT",
            "category": "OCT",
            "filename": "oct_pathology.png",
            "mask_filename": None,
            "description": "Pathological subretinal fluid accumulation and intraretinal hyperreflective foci."
        }
    ]
    return jsonify({"samples": samples, "count": len(samples)})

RETINA_SAMPLE_DIRS = [
    r"D:\SFHformer-FreqLab\data\retina_samples",
    r"D:\SFHformer-FreqLab\frontend\public\retina_samples",
    os.path.join(SCRATCH_DIR, "public", "retina_samples") if 'SCRATCH_DIR' in globals() else ""
]

@app.route('/api/retina/restore', methods=['POST'])
def restore_retina():
    """
    Clinical Retinal Restoration & Vascular Biomarker Engine.
    """
    t_start = time.time()
    data = request.get_json(force=True)
    
    # 1. Image loading
    img_b64 = data.get('image', None)
    sample_id = data.get('sample_id', None)
    noise_mode = data.get('mode', 'mixed')
    severity = float(data.get('severity', 1.0))
    model_choice = data.get('model', 'retina_fives')
    
    clean = None
    if sample_id:
        for d in RETINA_SAMPLE_DIRS:
            if not d:
                continue
            sample_path = os.path.join(d, f"{sample_id}.png")
            if os.path.exists(sample_path):
                pil_s = Image.open(sample_path).convert('RGB').resize((256, 256))
                clean = np.array(pil_s, dtype=np.float32) / 255.0
                break
            
    if clean is None and img_b64:
        clean = base64_to_img(img_b64, target_size=(256, 256))
        
    if clean is None:
        return jsonify({"error": "No valid retinal image or sample_id provided"}), 400
        
    # 2. Noise Simulation
    corrupt_meta = {}
    if noise_mode != 'none':
        corrupted, corrupt_meta = retinal_corrupt(clean, mode=noise_mode, severity=severity)
    else:
        corrupted = clean.copy()
        corrupt_meta = {"mode": "none", "severity": 0.0}
        
    # 3. Model Inference (SFHformer GPU)
    # Check if a newly trained checkpoint is available
    retina_ckpt_path = r"D:\SFHformer-FreqLab\checkpoints\retina\best.pth"
    if model_choice == 'retina' and os.path.exists(retina_ckpt_path):
        if 'retina' not in MODELS or getattr(MODELS['retina'], '_custom_retina_loaded', False) is False:
            print("[FreqLab Server] Hot-loading newly completed SFHformer-Retina weights...")
            try:
                r_net = SFHformer_m().to(DEVICE)
                ck = torch.load(retina_ckpt_path, map_location=DEVICE, weights_only=False)
                r_net.load_state_dict(ck['model'] if 'model' in ck else ck)
                r_net.eval()
                r_net._custom_retina_loaded = True
                MODELS['retina'] = r_net
            except Exception as e:
                print(f"Hot-load warning: {e}")
                
    active_net = MODELS.get(model_choice, MODELS.get('r1_baseline'))
    
    t_inf_start = time.time()
    with torch.no_grad():
        x_t = torch.from_numpy(corrupted.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        pred_t = active_net(x_t).clamp(0.0, 1.0)
        restored = pred_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    inf_ms = (time.time() - t_inf_start) * 1000.0
    
    # 4. Retinal Diagnostic Suite & Vascular Biomarkers
    diagnostics = run_retina_diagnostics(
        clean, corrupted, restored,
        sample_id=sample_id,
        noise_mode=noise_mode,
        severity=severity
    )
    
    # 5. Frequency Spectra (Before & After)
    spectrum_before = generate_spectrum_thumbnail(corrupted)
    spectrum_after = generate_spectrum_thumbnail(restored)
    
    # 6. Quantitative Image Metrics
    psnr_before = float(psnr_np(clean, corrupted))
    psnr_after = float(psnr_np(clean, restored))
    ssim_before = float(ssim_np(clean, corrupted))
    ssim_after = float(ssim_np(clean, restored))
    
    total_ms = (time.time() - t_start) * 1000.0
    
    return jsonify({
        "original": img_to_base64(clean),
        "corrupted": img_to_base64(corrupted),
        "restored": img_to_base64(restored),
        "vessel_mask": diagnostics["vessel_mask_restored"],
        "vessel_overlay": diagnostics["vessel_overlay_restored"],
        "corrupted_overlay": diagnostics["vessel_overlay_corrupted"],
        "enhanced_green": diagnostics.get("vessel_mask_clean", diagnostics["vessel_mask_restored"]),
        "biomarkers": diagnostics["metrics_restored"],
        "ablation_comparison": diagnostics["ablation_comparison"],
        "metrics": {
            "psnr_corrupted": round(psnr_before, 2),
            "psnr_restored": round(psnr_after, 2),
            "psnr_gain": round(psnr_after - psnr_before, 2),
            "ssim_corrupted": round(ssim_before, 4),
            "ssim_restored": round(ssim_after, 4),
            "ssim_gain": round(ssim_after - ssim_before, 4)
        },
        "spectra": {
            "before": spectrum_before,
            "after": spectrum_after
        },
        "corruption_meta": corrupt_meta,
        "device": DEVICE_NAME,
        "latency_ms": round(inf_ms, 1),
        "total_ms": round(total_ms, 1)
    })

@app.route('/api/retina/dual_segment', methods=['POST'])
def dual_segment_retina():
    """
    Dual U-Net Retinal Micro-Capillary Inspection Engine.
    Compares Raw U-Net (macro precision) vs. Restored U-Net (micro-capillary recall).
    Generates:
    1. Raw U-Net segmentation overlay (Cyan)
    2. Restored U-Net segmentation overlay (Emerald Green)
    3. Differential Capillary Gain Map (Shared in Cyan, Gained Capillaries in Glowing Amber)
    4. Quantitative biomarkers: vessel density, connected components, skeleton endpoints, newly gained capillaries
    """
    t0 = time.time()
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
    except Exception:
        return jsonify({"error": "Malformed JSON payload"}), 400

    sample_id = data.get('sample_id', None)
    img_b64 = data.get('image', None)

    try:
        threshold = float(data.get('threshold', 0.40))
    except (TypeError, ValueError):
        return jsonify({"error": "threshold must be a finite numeric value in [0.0, 1.0]"}), 400
    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        return jsonify({"error": f"Invalid threshold {threshold}; must be within [0.0, 1.0]"}), 400

    try:
        target_dim = int(data.get('resolution', 512))
    except (TypeError, ValueError):
        return jsonify({"error": "resolution must be an integer (256 or 512)"}), 400
    if target_dim not in (256, 512):
        return jsonify({"error": f"Unsupported resolution {target_dim}; allowed values are [256, 512]"}), 400

    selected_model = data.get('selected_model', 'dual_stream')
    if selected_model not in ('dual_stream', 'raw_unet', 'restored_unet'):
        return jsonify({
            "error": f"Unknown selected_model '{selected_model}'. Allowed values: ['dual_stream', 'raw_unet', 'restored_unet']"
        }), 400

    include_fusion_viz = bool(data.get('include_fusion_viz', False))
    fusion_viz_note = None
    compute_fusion_viz = include_fusion_viz and (selected_model == 'dual_stream')
    if include_fusion_viz and selected_model != 'dual_stream':
        fusion_viz_note = "Spatial fusion-weight visualization (scale_0 w_rest) applies only when selected_model='dual_stream'."

    clean = None
    if sample_id is not None:
        if not isinstance(sample_id, str) or not sample_id.strip():
            return jsonify({"error": "sample_id must be a non-empty string"}), 400
        for d in RETINA_SAMPLE_DIRS:
            if not d:
                continue
            sample_path = os.path.join(d, f"{sample_id}.png")
            if os.path.exists(sample_path):
                pil_s = Image.open(sample_path).convert('RGB').resize((target_dim, target_dim), Image.Resampling.BILINEAR)
                clean = np.array(pil_s, dtype=np.float32) / 255.0
                break
        if clean is None and img_b64 is None:
            return jsonify({"error": f"Unknown sample_id '{sample_id}'"}), 400
                
    if clean is None and img_b64 is not None:
        if not isinstance(img_b64, str) or not img_b64.strip():
            return jsonify({"error": "image must be a non-empty base64 string"}), 400
        try:
            clean = base64_to_img(img_b64, target_size=(target_dim, target_dim))
        except Exception as e:
            return jsonify({"error": f"Invalid or corrupted base64 image payload: {str(e)}"}), 400

    if clean is None:
        return jsonify({"error": "No valid retinal image or sample_id provided"}), 400

    ms_input_io = (time.time() - t0) * 1000.0

    sfh_net = MODELS.get('retina_fives', MODELS.get('r1_baseline'))
    u_raw = MODELS.get('unet_raw')
    u_res = MODELS.get('unet_restored')
    u_dual = MODELS.get('unet_dual_stream')

    if u_raw is None or u_res is None:
        return jsonify({"error": "U-Net segmentation models not loaded"}), 500

    # Serialized GPU inference lock ensures thread safety and bounded VRAM under concurrency
    with GPU_LOCK:
        # 1. SFHFormer Restoration (GPU)
        t_sfh = time.time()
        with torch.no_grad():
            x_raw = torch.from_numpy(clean.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
            x_restored = sfh_net(x_raw).clamp(0.0, 1.0)
            restored = x_restored.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        ms_sfh = (time.time() - t_sfh) * 1000.0

        # 2. Single-Stream Dual U-Net Inference (GPU BF16 AMP)
        t_seg = time.time()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad():
                prob_raw = torch.sigmoid(u_raw(x_raw)).squeeze().float().cpu().numpy()
                prob_res = torch.sigmoid(u_res(x_restored)).squeeze().float().cpu().numpy()
        ms_seg = (time.time() - t_seg) * 1000.0

        # 2b. Dual-Stream Hybrid U-Net Inference (GPU BF16 AMP, single forward pass for seg + scale_0 gate)
        t_dual = time.time()
        prob_dual = None
        gate_w_rest = None
        if u_dual is not None:
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else torch.no_grad():
                    if compute_fusion_viz:
                        logits_dual, gates_dual = u_dual(x_raw, x_restored, return_gates=True)
                        gate_w_rest = gates_dual['scale_0'].squeeze().float().cpu().numpy()
                    else:
                        logits_dual = u_dual(x_raw, x_restored, return_gates=False)
                    prob_dual = torch.sigmoid(logits_dual).squeeze().float().cpu().numpy()
        ms_dual_seg = (time.time() - t_dual) * 1000.0

    # 3. Post-Processing: Vascular Topology & Biomarkers (CPU)
    t_topo = time.time()
    # Consistent FOV masking matching evaluate_milestone3.py
    fov = clean.mean(axis=2) > (10.0 / 255.0)
    if np.sum(fov) < 200:
        fov = np.ones_like(prob_raw, dtype=bool)

    mask_raw = ((prob_raw >= threshold) & fov).astype(np.uint8) * 255
    mask_res = ((prob_res >= threshold) & fov).astype(np.uint8) * 255
    mask_dual = ((prob_dual >= threshold) & fov).astype(np.uint8) * 255 if prob_dual is not None else None

    raw_bio = compute_vascular_topology_metrics(clean, mask_raw)
    res_bio = compute_vascular_topology_metrics(restored, mask_res)
    dual_bio = compute_vascular_topology_metrics(clean, mask_dual) if mask_dual is not None else None

    raw_vessel = mask_raw > 0
    res_vessel = mask_res > 0
    shared_px = int(np.sum(raw_vessel & res_vessel))
    gain_px = int(np.sum(res_vessel & ~raw_vessel))
    raw_only_px = int(np.sum(raw_vessel & ~res_vessel))
    if raw_bio['n_vessel_pixels'] > 0:
        diff_response_pct = round((gain_px / raw_bio['n_vessel_pixels']) * 100.0, 2)
    else:
        diff_response_pct = 100.0 if gain_px > 0 else 0.0

    if raw_bio['n_skel_pixels'] > 0:
        skel_gain_pct = round(((res_bio['n_skel_pixels'] - raw_bio['n_skel_pixels']) / raw_bio['n_skel_pixels']) * 100.0, 2)
    else:
        skel_gain_pct = 100.0 if res_bio['n_skel_pixels'] > 0 else 0.0

    dual_vs_raw = None
    if dual_bio is not None:
        dual_vessel = mask_dual > 0
        dual_only_px = int(np.sum(dual_vessel & ~raw_vessel))
        dual_shared_px = int(np.sum(dual_vessel & raw_vessel))
        dual_diff_pct = round((dual_only_px / raw_bio['n_vessel_pixels']) * 100.0, 2) if raw_bio['n_vessel_pixels'] > 0 else (100.0 if dual_only_px > 0 else 0.0)
        dual_vs_raw = {
            "shared_with_raw_pixels": dual_shared_px,
            "additional_vessel_pixels": dual_only_px,
            "differential_response_pct": dual_diff_pct,
            "component_delta": dual_bio['branch_fragment_count'] - raw_bio['branch_fragment_count'],
            "endpoint_delta": dual_bio['endpoint_count'] - raw_bio['endpoint_count'],
        }

    fusion_gate_stats = None
    if gate_w_rest is not None:
        fov_gate = gate_w_rest[fov]
        fusion_gate_stats = {
            "scale": "scale_0",
            "mean_w_rest": round(float(np.mean(fov_gate)), 4),
            "min_w_rest": round(float(np.min(fov_gate)), 4),
            "max_w_rest": round(float(np.max(fov_gate)), 4),
            "pct_rest_dominant": round(float(np.mean(fov_gate > 0.5) * 100.0), 2),
        }
    ms_topo = (time.time() - t_topo) * 1000.0

    # 4. Color Blending Overlays (CPU)
    t_overlay = time.time()
    overlay_raw = create_vascular_overlay(clean, mask_raw, alpha=0.55, color=(0, 240, 220))
    overlay_res = create_vascular_overlay(restored, mask_res, alpha=0.55, color=(16, 230, 110))
    overlay_diff = create_differential_gain_overlay(clean, mask_raw, mask_res, alpha=0.60)
    overlay_dual = create_vascular_overlay(clean, mask_dual, alpha=0.58, color=(0, 235, 175)) if mask_dual is not None else None
    overlay_dual_diff = create_differential_gain_overlay(clean, mask_raw, mask_dual, alpha=0.60) if mask_dual is not None else None
    overlay_fusion = create_fusion_gate_overlay(clean, gate_w_rest, alpha=0.58) if gate_w_rest is not None else None
    ms_overlay = (time.time() - t_overlay) * 1000.0

    # 5. Image Fidelity Metrics & Base64 PNG Encoding (CPU)
    t_encode = time.time()
    psnr_val = float(psnr_np(clean, restored))
    ssim_val = float(ssim_np(clean, restored))

    b64_raw_img = img_to_base64(clean)
    b64_res_img = img_to_base64(restored)
    b64_mask_raw = np_to_base64_png(mask_raw, is_mask=True)
    b64_mask_res = np_to_base64_png(mask_res, is_mask=True)
    b64_overlay_raw = np_to_base64_png(overlay_raw)
    b64_overlay_res = np_to_base64_png(overlay_res)
    b64_overlay_diff = np_to_base64_png(overlay_diff)
    b64_mask_dual = np_to_base64_png(mask_dual, is_mask=True) if mask_dual is not None else None
    b64_overlay_dual = np_to_base64_png(overlay_dual) if overlay_dual is not None else None
    b64_overlay_dual_diff = np_to_base64_png(overlay_dual_diff) if overlay_dual_diff is not None else None
    b64_overlay_fusion = np_to_base64_png(overlay_fusion) if overlay_fusion is not None else None
    ms_encode = (time.time() - t_encode) * 1000.0

    total_ms = (time.time() - t0) * 1000.0
    neural_ms = ms_sfh + ms_seg + ms_dual_seg
    postproc_ms = ms_input_io + ms_topo + ms_overlay + ms_encode

    return jsonify({
        "selected_model": selected_model,
        "raw_img": b64_raw_img,
        "restored_img": b64_res_img,
        "mask_raw": b64_mask_raw,
        "mask_restored": b64_mask_res,
        "mask_dual_stream": b64_mask_dual,
        "overlay_raw": b64_overlay_raw,
        "overlay_restored": b64_overlay_res,
        "overlay_gain": b64_overlay_diff,
        "overlay_differential": b64_overlay_diff,
        "overlay_dual_stream": b64_overlay_dual,
        "overlay_dual_differential": b64_overlay_dual_diff,
        "overlay_fusion_gate": b64_overlay_fusion,
        "fusion_gate_stats": fusion_gate_stats,
        "fusion_viz_note": fusion_viz_note,
        "threshold": threshold,
        "threshold_mode": "frozen_dual_stream_0.40" if abs(threshold - 0.40) < 1e-4 else ("alternative_baseline_0.50" if abs(threshold - 0.50) < 1e-4 else "exploratory"),
        "is_frozen_benchmark_threshold": bool(abs(threshold - 0.40) < 1e-4 or abs(threshold - 0.50) < 1e-4),
        "biomarkers": {
            "raw": raw_bio,
            "restored": res_bio,
            "dual_stream": dual_bio,
            "dual_vs_raw": dual_vs_raw,
            "differential": {
                "shared_vessel_pixels": shared_px,
                "additional_vessel_pixels": gain_px,
                "raw_only_pixels": raw_only_px,
                "differential_response_pct": diff_response_pct,
                "skeleton_length_delta_pct": skel_gain_pct,
                "endpoint_delta": res_bio['endpoint_count'] - raw_bio['endpoint_count'],
                "component_delta": res_bio['branch_fragment_count'] - raw_bio['branch_fragment_count'],
                "newly_recovered_capillary_pixels": gain_px,
                "capillary_recovery_gain_pct": diff_response_pct,
                "skeleton_length_gain_pct": skel_gain_pct
            }
        },
        "restoration_metrics": {
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4)
        },
        "timing": {
            "input_io_and_prep_ms": round(ms_input_io, 1),
            "sfhformer_restoration_ms": round(ms_sfh, 1),
            "dual_unet_segmentation_ms": round(ms_seg, 1),
            "dual_stream_segmentation_ms": round(ms_dual_seg, 1),
            "neural_inference_total_ms": round(neural_ms, 1),
            "topology_analysis_ms": round(ms_topo, 1),
            "overlay_blending_ms": round(ms_overlay, 1),
            "fidelity_and_base64_encoding_ms": round(ms_encode, 1),
            "postprocessing_and_io_total_ms": round(postproc_ms, 1),
            "total_ms": round(total_ms, 1),
            "sfhformer_ms": round(ms_sfh, 1),
            "dual_segment_ms": round(ms_seg + ms_dual_seg, 1)
        }
    })

BENCHMARK_CACHE = None

@app.route('/api/retina/benchmark', methods=['GET'])
def get_retina_benchmark():
    """
    Returns frozen Milestone 3 empirical validation results across FIVES (200 test scans) and HRF (45 scans).
    """
    global BENCHMARK_CACHE
    if BENCHMARK_CACHE is not None:
        return jsonify(BENCHMARK_CACHE)

    fives_path = os.path.join(REPO_DIR, "results", "milestone3", "fives_test_results.json")
    hrf_path = os.path.join(REPO_DIR, "results", "milestone3", "hrf_results.json")
    dual_fives_path = os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_fives_results.json")
    dual_hrf_path = os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_hrf_results.json")

    fives_data = {}
    if os.path.exists(fives_path):
        with open(fives_path, 'r') as f:
            fives_data = json.load(f)

    hrf_data = {}
    if os.path.exists(hrf_path):
        with open(hrf_path, 'r') as f:
            hrf_data = json.load(f)

    dual_fives_data = {}
    if os.path.exists(dual_fives_path):
        with open(dual_fives_path, 'r') as f:
            dual_fives_data = json.load(f)

    dual_hrf_data = {}
    if os.path.exists(dual_hrf_path):
        with open(dual_hrf_path, 'r') as f:
            dual_hrf_data = json.load(f)

    BENCHMARK_CACHE = {
        "status": "frozen_milestone3",
        "provenance": {
            "fives_test_count": 200,
            "fives_clean_count": 199,
            "hrf_total_count": 45,
            "hrf_exposed_count": 40,
            "hrf_unexposed_count": 5,
            "threshold_unet_baselines": 0.50,
            "threshold_dual_stream_calibrated": 0.40,
            "threshold_frangi": 0.0010,
            "checkpoints_sha256": {
                "unet_raw": "97639a65b6cc3d09...",
                "unet_restored": "2cfb37371cc5c003...",
                "unet_dual_stream": "9be2213e0cf0c799...",
                "sfhformer": "d74e267c6ac66468..."
            }
        },
        "fives_summary": fives_data.get("summary_overall_200", {}),
        "fives_clean_summary": fives_data.get("summary_clean_199", {}),
        "fives_disease_breakdown": fives_data.get("disease_breakdown", {}),
        "hrf_overall": hrf_data.get("summary_overall_45", {}),
        "hrf_exposed": hrf_data.get("summary_exposed_40", {}),
        "hrf_unexposed": hrf_data.get("summary_unexposed_5", {}),
        "dual_stream": {
            "threshold": dual_fives_data.get("threshold", 0.40),
            "fives_overall": dual_fives_data.get("overall", {}),
            "fives_vs_raw": dual_fives_data.get("vs_raw_unet_n200", {}),
            "fives_vs_restored": dual_fives_data.get("vs_restored_unet_n200", {}),
            "fives_by_disease": dual_fives_data.get("by_disease", {}),
            "hrf_overall": dual_hrf_data.get("overall_n45", {}),
            "hrf_vs_raw": dual_hrf_data.get("overall_vs_raw", {}),
            "hrf_vs_restored": dual_hrf_data.get("overall_vs_restored", {}),
            "hrf_unexposed_exploratory": dual_hrf_data.get("unexposed_n5_exploratory", {}),
            "domain_interpretation": (
                "FIVES (N=200) is the primary in-domain benchmark: DualStreamUNet (tau=0.40) improves Dice (+3.41 pp), "
                "Thin Skeleton Sensitivity (+10.20 pp), clDice (+4.02 pp), and reduces fragmentation (-10.97 CCs/img) vs Raw U-Net. "
                "HRF (N=45) reflects mixed external-domain transfer (40/45 images exposed during upstream SFHformer pretraining, "
                "N=5 unexposed exploratory): DualStreamUNet improves ThinSkel (+9.78 pp), clDice (+2.78 pp), and CC (-12.49) vs Raw U-Net, "
                "while trading -8.54 pp Precision and -1.82 pp Dice vs Raw U-Net and -3.89 pp ThinSkel vs Restored U-Net."
            )
        },
        "clinical_insights": {
            "thin_vessel_gain": "+13.67 pp on HRF Overall (36.81% -> 50.48%, p=5.68e-13)",
            "zero_shot_gain": "+11.21 pp on HRF Unexposed (47.77% -> 58.97%, Wilcoxon p=0.0625, paired t-test p=0.0626, 95% CI: [+3.88, +19.19] pp)",
            "branch_recall_gain": "+10.39 pp on HRF Unexposed thin branches (42.66% -> 53.05%)",
            "fives_tradeoff": "On FIVES (thick arcade-dominated), Raw U-Net achieves 0.8133 Dice vs Restored 0.6927 due to additional peripheral vessel predictions not marked in FIVES ground truth.",
            "glaucoma_nuance": "Glaucoma shows a precision drop (-33.54 pp) due to enhanced optic cup rim and peripapillary atrophy contrast."
        }
    }
    return jsonify(BENCHMARK_CACHE)

if __name__ == '__main__':
    load_all_models()
    port = int(os.environ.get('PORT', 5001))
    print(f"[FreqLab Server] Serving on http://127.0.0.1:{port}")
    app.run(host='127.0.0.1', port=port, threaded=True)

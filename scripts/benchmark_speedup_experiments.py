"""
benchmark_speedup_experiments.py - Systematic Hardware Acceleration Benchmark
Compares:
1. Baseline: FP32, batch_size=1, accum=4
2. Experiment 1: AMP BF16, batch_size=1, accum=4
3. Experiment 2: AMP BF16, batch_size=2, accum=2
4. Experiment 3: AMP BF16, batch_size=2, accum=2 + compile (cudagraphs)
5. Experiment 4: AMP BF16, batch_size=2, accum=2 + channels_last

Runs 50 iterations per configuration (with 10 warmup steps excluded from timing)
and computes validation PSNR and SSIM on held-out validation cases.
"""

import os
import sys
import time
import json
import glob
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_DIR = r"D:\SFHformer-FreqLab"
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
SCRIPTS_DIR = os.path.join(REPO_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from freqlab.sfhformer_gated import SFHformer_m_gated
from freqlab.metrics import psnr_torch, psnr_np, ssim_np
from train_fives_retina import FIVESOnlineDataset, validate

DATA_DIR = r"D:\SFHformer-FreqLab\data\fives"
RAW_DIR = os.path.join(DATA_DIR, "raw")
TRAIN_IMG_DIR = os.path.join(RAW_DIR, "train", "images")
VAL_IMG_DIR = os.path.join(RAW_DIR, "val", "images")

FIVES_BEST_CKPT = os.path.join(REPO_DIR, "checkpoints", "fives_retina_finetune", "best.pth")
DRIVE_BEST_CKPT = os.path.join(REPO_DIR, "checkpoints", "retina", "best.pth")
STARTING_CKPT = FIVES_BEST_CKPT if os.path.exists(FIVES_BEST_CKPT) else DRIVE_BEST_CKPT

RESULTS_FILE = os.path.join(REPO_DIR, "results", "speedup_benchmark_results.json")
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

def load_fresh_model(device, channels_last=False):
    model = SFHformer_m_gated(use_gate=False).to(device)
    if os.path.exists(STARTING_CKPT):
        ck = torch.load(STARTING_CKPT, map_location=device, weights_only=False)
        st = ck['model'] if 'model' in ck else ck
        model.load_state_dict(st, strict=True)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model

def compute_val_metrics(model, val_loader, device, use_amp=False, channels_last=False, max_cases=30):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_green_psnr = 0.0
    ssim_list = []
    count = 0
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(val_loader):
            if i >= max_cases:
                break
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            if channels_last:
                noisy = noisy.contiguous(memory_format=torch.channels_last)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model(noisy).clamp(0.0, 1.0)
            else:
                pred = model(noisy).clamp(0.0, 1.0)
            loss = F.l1_loss(pred.float(), clean.float())
            p = psnr_torch(pred.float(), clean.float())
            p_green = psnr_torch(pred[:, 1:2, :, :].float(), clean[:, 1:2, :, :].float())
            b = noisy.size(0)
            total_loss += loss.item() * b
            total_psnr += p * b
            total_green_psnr += p_green * b
            count += b
            
            p_np = pred.squeeze(0).float().cpu().numpy().transpose(1, 2, 0)
            c_np = clean.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            ssim_list.append(float(ssim_np(c_np, p_np)))
    model.train()
    c = max(count, 1)
    return total_loss / c, total_psnr / c, total_green_psnr / c, float(np.mean(ssim_list))

def run_experiment(exp_name, config, train_paths, val_paths, device, n_iters=50, warmup=10):
    print("\n" + "=" * 70)
    print(f" [RUNNING] {exp_name}")
    print(f" Config: {config}")
    print("=" * 70)
    
    batch_size = config.get("batch_size", 1)
    accum_steps = config.get("accum_steps", 4)
    use_amp = config.get("use_amp", False)
    use_compile = config.get("use_compile", False)
    channels_last = config.get("channels_last", False)
    num_workers = config.get("num_workers", 2)
    
    # DataLoader
    train_dataset = FIVESOnlineDataset(train_paths, crop_size=256, is_train=True, seed=2026)
    val_dataset = FIVESOnlineDataset(val_paths, crop_size=256, is_train=False, seed=2026)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Model
    model = load_fresh_model(device, channels_last=channels_last)
    if use_compile:
        try:
            print(" -> Compiling model with backend='cudagraphs'...")
            model = torch.compile(model, backend="cudagraphs")
        except Exception as e:
            print(f" -> Compile error: {e}, falling back to uncompiled.")
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    
    # Reset GPU memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    model.train()
    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    
    step_times = []
    losses = []
    
    # Training Loop
    for step in range(1, n_iters + 1):
        try:
            noisy, clean = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            noisy, clean = next(data_iter)
            
        t0 = time.time()
        
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        
        if channels_last:
            noisy = noisy.contiguous(memory_format=torch.channels_last)
            clean = clean.contiguous(memory_format=torch.channels_last)
            
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(noisy)
                l_spatial = F.l1_loss(pred, clean)
                # Compute FFT on float32 for numerical stability in frequency loss
                pred_fft = torch.fft.rfft2(pred.float(), norm="backward")
                clean_fft = torch.fft.rfft2(clean.float(), norm="backward")
                l_freq = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft)) * 0.1
                total_loss = (l_spatial + l_freq) / accum_steps
        else:
            pred = model(noisy)
            l_spatial = F.l1_loss(pred, clean)
            pred_fft = torch.fft.rfft2(pred, norm="backward")
            clean_fft = torch.fft.rfft2(clean, norm="backward")
            l_freq = F.l1_loss(torch.abs(pred_fft), torch.abs(clean_fft)) * 0.1
            total_loss = (l_spatial + l_freq) / accum_steps
            
        total_loss.backward()
        
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
        torch.cuda.synchronize(device)
        step_elapsed = (time.time() - t0) * 1000.0 # ms
        
        if step > warmup:
            step_times.append(step_elapsed)
            
        losses.append((l_spatial + l_freq).item())
        
        if step % 10 == 0:
            print(f"  Step [{step:2d}/{n_iters}] | Latency: {step_elapsed:5.1f} ms | Loss: {losses[-1]:.4f}", flush=True)
            
    # Compute hardware metrics
    avg_ms_per_step = float(np.mean(step_times))
    # Effective optimizer step latency (samples processed per optimizer update = 4)
    effective_step_ms = avg_ms_per_step * accum_steps
    samples_per_sec = (batch_size * 1000.0) / avg_ms_per_step
    peak_vram_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
    
    # Compute validation metrics
    print(" -> Computing validation metrics...")
    v_loss, v_psnr, v_green_psnr, v_ssim = compute_val_metrics(
        model, val_loader, device, use_amp=use_amp, channels_last=channels_last, max_cases=30
    )
    
    result = {
        "experiment": exp_name,
        "config": config,
        "avg_ms_per_step": round(avg_ms_per_step, 1),
        "effective_step_ms": round(effective_step_ms, 1),
        "samples_per_sec": round(samples_per_sec, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "val_psnr": round(float(v_psnr), 2),
        "val_green_psnr": round(float(v_green_psnr), 2),
        "val_ssim": round(v_ssim, 4)
    }
    
    print(f" [RESULT] {exp_name}:")
    print(f"  -> Forward/Backward:      {avg_ms_per_step:.1f} ms/step")
    print(f"  -> Effective Optimizer:   {effective_step_ms:.1f} ms/iter (Accum={accum_steps})")
    print(f"  -> Throughput:            {samples_per_sec:.2f} samples/sec")
    print(f"  -> Peak VRAM:             {peak_vram_gb:.2f} GB")
    print(f"  -> Val PSNR:              {result['val_psnr']} dB (SSIM: {result['val_ssim']})")
    
    return result

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(" SYSTEMATIC HARDWARE ACCELERATION BENCHMARK SUITE ")
    print(f" Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f" PyTorch: {torch.__version__} | BF16 Supported: {torch.cuda.is_bf16_supported()}")
    
    # HARD SAFETY LIMIT: 5.57 GB (70% of 7.96 GB VRAM)
    # Guaranteed headroom for Windows DWM and OS to prevent BSOD (0x101 Watchdog Timeout)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.70, device=0)
        print(" [SAFETY LOCK] PyTorch VRAM allocated capped at 5.57 GB (max 6.0 GB rule active)")
    print("=" * 70)
    
    train_paths = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
    val_paths = sorted(glob.glob(os.path.join(VAL_IMG_DIR, "*.png")))
    
    # Note on torch.compile & Batch Size 4:
    # 1. Batch Size 4 requires ~8.5 GB VRAM, exceeding the 6.0 GB safety threshold.
    # 2. torch.compile on Windows fails: Inductor lacks Triton, and cudagraphs causes
    #    driver stream deadlocks with Windows DWM resulting in 0x101 CLOCK_WATCHDOG_TIMEOUT.
    # Therefore, the valid hardware acceleration configurations within the 6 GB budget are:
    experiments = [
        ("Baseline (FP32, Batch 1)", {
            "use_amp": False, "batch_size": 1, "accum_steps": 4, "use_compile": False, "channels_last": False
        }),
        ("Experiment 1 (AMP BF16, Batch 1)", {
            "use_amp": True, "batch_size": 1, "accum_steps": 4, "use_compile": False, "channels_last": False
        }),
        ("Experiment 2 (AMP BF16, Batch 2)", {
            "use_amp": True, "batch_size": 2, "accum_steps": 2, "use_compile": False, "channels_last": False
        }),
        ("Experiment 4 (AMP BF16, Batch 2 + Channels Last)", {
            "use_amp": True, "batch_size": 2, "accum_steps": 2, "use_compile": False, "channels_last": True
        })
    ]
    
    all_results = []
    baseline_throughput = None
    
    for exp_name, cfg in experiments:
        res = run_experiment(exp_name, cfg, train_paths, val_paths, device, n_iters=35, warmup=5)
        if baseline_throughput is None:
            baseline_throughput = res["samples_per_sec"]
            res["speedup_factor"] = 1.00
        else:
            res["speedup_factor"] = round(res["samples_per_sec"] / baseline_throughput, 2)
        all_results.append(res)
        
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "=" * 70)
    print(" SUMMARY TABLE OF ACCELERATION EXPERIMENTS ")
    print("=" * 70)
    print(f"{'Experiment':<44} | {'Step (ms)':<9} | {'Thpt (img/s)':<12} | {'Speedup':<8} | {'VRAM':<7} | {'Val PSNR':<8} | {'SSIM':<6}")
    print("-" * 105)
    for r in all_results:
        print(f"{r['experiment']:<44} | {r['avg_ms_per_step']:<9.1f} | {r['samples_per_sec']:<12.2f} | {r['speedup_factor']:<7.2f}x | {r['peak_vram_gb']:<5.2f}GB | {r['val_psnr']:<8.2f} | {r['val_ssim']:<6.4f}")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_FILE}")

if __name__ == '__main__':
    main()

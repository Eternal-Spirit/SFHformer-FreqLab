"""
scripts/sanity_check.py — Pre-training sanity checks for SFHformer.

Checks (in order):
1. Forward pass on (2,3,256,256) returns the same shape.
2. Backward pass leaves no parameter with grad is None.
3. Overfit test: train on 8 fixed pairs for ~200 iterations. Loss must fall clearly.

Usage:
    python scripts/sanity_check.py [--gate]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freqlab.corrupt import corrupt, sample_train_comps


def check_forward(model, device='cpu'):
    """Check 1: Forward pass shape."""
    print("=" * 60)
    print("CHECK 1: Forward pass shape")
    x = torch.randn(2, 3, 256, 256, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape, f"Shape mismatch: input {x.shape}, output {y.shape}"
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print("  PASSED")
    return True


def check_backward(model, device='cpu'):
    """Check 2: Backward pass — no None gradients."""
    print("=" * 60)
    print("CHECK 2: Backward pass — gradient check")
    x = torch.randn(2, 3, 256, 256, device=device, requires_grad=False)
    y = model(x)
    loss = y.sum()
    loss.backward()

    none_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            none_grads.append(name)

    if none_grads:
        print(f"  FAILED: {len(none_grads)} parameters have grad=None:")
        for n in none_grads[:10]:
            print(f"    {n}")
        return False

    total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  All {total} trainable parameters have gradients")
    print("  PASSED")
    model.zero_grad()
    return True


def check_overfit(model, device='cpu', n_iters=200, n_pairs=8):
    """Check 3: Overfit on a small fixed dataset."""
    print("=" * 60)
    print(f"CHECK 3: Overfit test ({n_pairs} pairs, {n_iters} iterations)")

    # Generate fixed pairs
    rng = np.random.default_rng(42)
    pairs = []
    for i in range(n_pairs):
        clean = rng.random((256, 256, 3)).astype(np.float32)
        comps = sample_train_comps(rng)
        noisy = corrupt(clean, comps)
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1)).unsqueeze(0).to(device)
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).to(device)
        pairs.append((noisy_t, clean_t))

    crit = nn.L1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    losses = []
    model.train()
    t0 = time.time()
    for it in range(n_iters):
        noisy, clean = pairs[it % n_pairs]
        pred = model(noisy)
        loss = crit(pred, clean)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (it + 1) % 50 == 0:
            print(f"  iter {it+1:4d}: loss = {loss.item():.6f}")
    elapsed = time.time() - t0

    first_20 = np.mean(losses[:20])
    last_20 = np.mean(losses[-20:])
    ratio = last_20 / first_20

    print(f"  First 20 avg: {first_20:.6f}")
    print(f"  Last 20 avg:  {last_20:.6f}")
    print(f"  Ratio:        {ratio:.4f}")
    print(f"  Time:         {elapsed:.1f}s ({n_iters/elapsed:.1f} it/s)")

    if ratio < 0.5:
        print("  PASSED (loss decreased by >50%)")
        return True
    else:
        print("  FAILED (loss did not decrease enough)")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', action='store_true', help='Use gated model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    if args.gate:
        from freqlab.sfhformer_gated import SFHformer_m_gated
        model = SFHformer_m_gated(use_gate=True).to(args.device)
        print("Model: SFHformer_m_gated")
    else:
        from models import SFHformer_m
        model = SFHformer_m().to(args.device)
        print("Model: SFHformer_m (baseline)")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.3f}M")

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Capability: {torch.cuda.get_device_capability(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True

    passed = 0
    total = 3

    if check_forward(model, args.device):
        passed += 1
    if check_backward(model, args.device):
        passed += 1
    if check_overfit(model, args.device):
        passed += 1

    print("=" * 60)
    print(f"RESULTS: {passed}/{total} checks passed")

    if args.device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak VRAM: {peak_mem:.2f} GB")

    if passed < total:
        print("GATE G1: FAILED — do not proceed to training")
        sys.exit(1)
    else:
        print("GATE G1: PASSED — safe to proceed to training")


if __name__ == '__main__':
    main()

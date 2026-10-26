"""
Unit tests for RadialBandGate and gated SFHformer.

Tests:
- Identity test: gated model with default (all-ones) gate matches baseline
- Gradient test: gradients reach gate parameters
- Parameter count and overhead
"""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from freqlab.gate import RadialBandGate


def test_gate_identity():
    """Gate initialized to zeros (exp(0)=1) should act as identity."""
    gate = RadialBandGate(h=64, w=64, n_bands=16)
    x = torch.randn(2, 32, 64, 33)  # (B, 2C, H, W//2+1)
    y = gate(x)
    torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def test_gate_gradient():
    """Gradients must reach the gate's log_amps parameter."""
    gate = RadialBandGate(h=64, w=64, n_bands=16)
    x = torch.randn(2, 32, 64, 33, requires_grad=True)
    y = gate(x)
    loss = y.sum()
    loss.backward()
    assert gate.log_amps.grad is not None, "No gradient on log_amps"
    assert gate.log_amps.grad.abs().sum() > 0, "Zero gradient on log_amps"


def test_gate_shape():
    """Gate output shape must match input shape."""
    gate = RadialBandGate(h=128, w=128, n_bands=8)
    x = torch.randn(1, 16, 128, 65)  # W//2+1 = 65
    y = gate(x)
    assert y.shape == x.shape


def test_gate_modulation():
    """Non-zero log_amps should change the output."""
    gate = RadialBandGate(h=64, w=64, n_bands=16)
    gate.log_amps.data = torch.randn(16) * 0.5
    x = torch.randn(2, 32, 64, 33)
    y = gate(x)
    assert not torch.allclose(x, y, atol=1e-4), "Gate with non-zero amps should modulate signal"


def test_gated_model_identity():
    """Gated SFHformer with identity gate should match baseline output.
    
    Load baseline weights into gated model with strict=False.
    The gate's log_amps are zero by default -> exp(0)=1 -> identity.
    """
    try:
        from models import SFHformer_m
        from freqlab.sfhformer_gated import SFHformer_m_gated
    except ImportError:
        pytest.skip("Cannot import models (einops or torch version issue)")

    torch.manual_seed(42)
    baseline = SFHformer_m()
    baseline.eval()

    gated = SFHformer_m_gated(use_gate=True)
    gated.eval()

    # Load baseline weights into gated model
    missing, unexpected = gated.load_state_dict(baseline.state_dict(), strict=False)
    # Only gate parameters should be missing
    for key in missing:
        assert 'gate' in key, f"Unexpected missing key: {key}"

    x = torch.randn(1, 3, 64, 64)  # Small input for speed
    with torch.no_grad():
        y_base = baseline(x)
        y_gated = gated(x)

    # Should be very close since gate is identity
    torch.testing.assert_close(y_base, y_gated, atol=1e-4, rtol=1e-4)


def test_gated_model_param_count():
    """Report parameter counts for baseline and gated models."""
    try:
        from models import SFHformer_m
        from freqlab.sfhformer_gated import SFHformer_m_gated
    except ImportError:
        pytest.skip("Cannot import models (einops or torch version issue)")

    baseline = SFHformer_m()
    gated = SFHformer_m_gated(use_gate=True)

    n_base = sum(p.numel() for p in baseline.parameters())
    n_gated = sum(p.numel() for p in gated.parameters())
    n_gate_only = sum(p.numel() for n, p in gated.named_parameters() if 'gate' in n)

    print(f"\nBaseline params: {n_base/1e6:.3f}M")
    print(f"Gated params:   {n_gated/1e6:.3f}M")
    print(f"Gate overhead:   {n_gate_only} params ({n_gate_only/n_base*100:.4f}%)")

    assert n_gated > n_base, "Gated model should have more parameters"
    assert n_gate_only < 10000, "Gate should have very few parameters"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

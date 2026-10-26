# NOTES.md — Repo Inspection Findings (Phase 3)

**Date**: 2026-10-26

## Model Architecture: SFHformer_m

### Parameter Count
- `SFHformer_m()`: embed_dim=[32, 64, 128, 64, 32], depth=[4, 4, 8, 4, 4]
- Needs measurement (see sanity_check.py)

### FFT Usage in FourierUnit (models/SFHformer.py, lines ~144-180)

The `FourierUnit` class is the sole FFT-based component:

```python
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        self.fdc = nn.Conv2d(in_channels*2, out_channels*2*groups, kernel_size=1, ...)
        self.weight = nn.Sequential(Conv2d(in_channels*2, groups, 1), Softmax(dim=1))
        self.fpe = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=3, padding=1, groups=in_channels*2)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')             # -> (B, C, H, W//2+1) complex
        # Split to real/imag, stack, rearrange to (B, 2C, H, W//2+1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w')  # d=2 (real,imag)
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, groups, 2*c, h, -1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2)
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')  # -> (B, C, H, W)
        return output
```

### Tensor shapes through FourierUnit
- **Input**: (B, C, H, W) where C = dim*2 (e.g., 64 at first stage)
- **After rfft2**: (B, C, H, W//2+1) complex
- **After rearrange**: (B, 2C, H, W//2+1) real
- **After processing**: same shape
- **After irfft2**: (B, C, H, W) real
- **Output**: same as input

### Gate Insertion Point
**The gate should be inserted AFTER `torch.fft.rfft2` and BEFORE the `self.bn(ffted)` line.**

Specifically, after the rearrange that produces (B, 2C, H, W//2+1), the gate operates on the real-valued frequency representation. Since the data is in rfft2 layout (half-spectrum), the RadialBandGate must work with shape (B, 2C, H, W//2+1) where W//2+1 is the frequency axis.

Alternative: insert after `self.fpe(ffted) + ffted` and before the weight computation, to gate the frequency features after positional encoding.

### Call Chain
```
Backbone.forward(x)
  -> layer1/2/3/8/9 = Stage
    -> Block
      -> Mixer
        -> TokenMixer_For_Gloal
          -> FourierUnit.forward(x)  # THIS is where rfft2/irfft2 happen
        -> TokenMixer_For_Local (spatial convs only, no FFT)
```

### Model Output
The Backbone adds a residual: `x = copy0 + x` (line in Backbone.forward)
So the network IS a residual network: output = input + learned_residual.
**CORRECTION**: The upstream training script compares `network(source_img)` directly to `target_img` with L1, but the network itself has a residual add internally. The output is the restored image.

### Training Config (from configs/rain/SFHformer_m.json)
- **Optimizer**: AdamW
- **Learning rate**: 1e-3
- **Epochs**: 1000
- **Batch size**: 12
- **Patch size**: 256
- **Scheduler**: CosineAnnealingLR with T_max=epochs, eta_min=1e-6
- **Loss**: L1 + 0.1 * FFT_L1 (spatial + frequency domain L1 loss)

### Pretrained Loading
- No pretrained weight loading in the training scripts (trains from scratch)
- All training scripts use `nn.DataParallel` with 4 GPUs

### Additional Notes
- The upstream uses a combined L1 + 0.1*FFT_L1 loss. Our simplified script uses just L1 initially.
- The `valid_mode: "test"` in config means the upstream selects checkpoints on the test set (bad practice). Our script uses a held-out val set.
- The repo has a misspelled `requirments.txt`.
- `einops` is a required dependency (used for rearrange in FourierUnit).

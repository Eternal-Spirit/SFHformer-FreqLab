"""
models/dual_stream_unet.py - Dual-Stream Hybrid U-Net for Retinal Vessel Segmentation.

Combines:
- Stream A (Raw Encoder): Captures Raw U-Net's macro-vessel precision, topological
  continuity, and resistance to optic-cup rim / contrast artifacts.
- Stream B (Restored Encoder): Captures SFHformer-restored U-Net's enhanced fine
  capillary and low-contrast peripheral vessel sensitivity.
- CrossStreamFusionBlock: Multi-scale per-pixel spatial gated fusion weighting
  Stream A vs. Stream B at every skip resolution (32, 64, 128, 256) and bottleneck (512),
  augmented with an SE-weighted 1x1 cross-stream residual projection.
- Shared Bilinear Decoder: Reconstructs unified vessel probability map without
  checkerboard artifacts.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import DoubleConv


class SEBlock(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class CrossStreamFusionBlock(nn.Module):
    """
    Per-pixel spatial gated fusion between Raw (macro-precision) and Restored
    (fine-capillary sensitivity) feature streams at a given spatial scale.
    Produces normalized spatial weights (w_raw + w_rest = 1 per pixel) plus an
    SE-weighted 1x1 cross-stream residual projection.

    Note: Returned `w_rest` is a 2D spatial fusion-weight map for visualization,
    not an independent clinical proof of vessel recovery.
    """
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 2, 16)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 2, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, 2, kernel_size=1, bias=True),
        )
        self.cross_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.se = SEBlock(channels)

    def forward(self, f_raw, f_rest):
        cat = torch.cat([f_raw, f_rest], dim=1)
        weights = torch.softmax(self.spatial_gate(cat), dim=1)
        w_raw = weights[:, 0:1]
        w_rest = weights[:, 1:2]

        gated = w_raw * f_raw + w_rest * f_rest
        residual = self.se(self.cross_proj(cat))
        return gated + residual, w_rest


class DualStreamUNet(nn.Module):
    """
    Two-stream hybrid U-Net processing paired (raw, restored) retinal images.
    """
    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256), bottleneck_dropout=0.2):
        super().__init__()
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stream A: Raw fundus encoder
        self.raw_downs = nn.ModuleList()
        cur_in = in_channels
        for feature in features:
            self.raw_downs.append(DoubleConv(cur_in, feature))
            cur_in = feature
        bottleneck_features = features[-1] * 2
        self.raw_bottleneck = DoubleConv(features[-1], bottleneck_features, dropout_p=bottleneck_dropout)

        # Stream B: SFHformer-restored fundus encoder
        self.rest_downs = nn.ModuleList()
        cur_in = in_channels
        for feature in features:
            self.rest_downs.append(DoubleConv(cur_in, feature))
            cur_in = feature
        self.rest_bottleneck = DoubleConv(features[-1], bottleneck_features, dropout_p=bottleneck_dropout)

        # Multi-scale cross-stream fusion gates
        self.fusions = nn.ModuleList([CrossStreamFusionBlock(f) for f in features])
        self.bottleneck_fusion = CrossStreamFusionBlock(bottleneck_features)

        # Shared Bilinear Decoder
        self.ups = nn.ModuleList()
        reversed_features = list(reversed(features))
        cur_features = bottleneck_features
        for feature in reversed_features:
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(cur_features, feature, kernel_size=1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.GELU(),
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            cur_features = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def load_pretrained_streams(self, raw_ckpt_path, rest_ckpt_path, init_decoder_from="raw", map_location="cpu"):
        """
        Warm-start Stream A from unet_raw, Stream B from unet_restored,
        and optionally the shared decoder from either baseline checkpoint.
        """
        raw_ck = torch.load(raw_ckpt_path, map_location=map_location, weights_only=False)
        raw_st = raw_ck["model"] if "model" in raw_ck else raw_ck

        rest_ck = torch.load(rest_ckpt_path, map_location=map_location, weights_only=False)
        rest_st = rest_ck["model"] if "model" in rest_ck else rest_ck

        mapped = {}
        for k, v in raw_st.items():
            if k.startswith("downs."):
                mapped[f"raw_{k}"] = v
            elif k.startswith("bottleneck."):
                mapped[f"raw_{k}"] = v
            elif init_decoder_from == "raw" and (k.startswith("ups.") or k.startswith("final_conv.")):
                mapped[k] = v

        for k, v in rest_st.items():
            if k.startswith("downs."):
                mapped[f"rest_{k}"] = v
            elif k.startswith("bottleneck."):
                mapped[f"rest_{k}"] = v
            elif init_decoder_from == "restored" and (k.startswith("ups.") or k.startswith("final_conv.")):
                mapped[k] = v

        missing, unexpected = self.load_state_dict(mapped, strict=False)
        # Only fusion blocks should be missing when warm-starting
        non_fusion_missing = [m for m in missing if not ("fusion" in m)]
        if non_fusion_missing or unexpected:
            raise RuntimeError(
                f"Unexpected warm-start mismatch: missing={non_fusion_missing}, unexpected={unexpected}"
            )
        return len(mapped), len(missing)

    def freeze_encoders(self):
        """Freeze both encoder streams and bottlenecks during gate warmup."""
        for mod in (self.raw_downs, self.raw_bottleneck, self.rest_downs, self.rest_bottleneck):
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze both encoder streams for joint end-to-end refinement."""
        for mod in (self.raw_downs, self.raw_bottleneck, self.rest_downs, self.rest_bottleneck):
            for p in mod.parameters():
                p.requires_grad = True

    def forward(self, x_raw, x_rest, return_gates=False):
        raw_skips = []
        r = x_raw
        for down in self.raw_downs:
            r = down(r)
            raw_skips.append(r)
            r = self.pool(r)
        r = self.raw_bottleneck(r)

        rest_skips = []
        s = x_rest
        for down in self.rest_downs:
            s = down(s)
            rest_skips.append(s)
            s = self.pool(s)
        s = self.rest_bottleneck(s)

        fused_skips = []
        gates = {}
        for idx, (f_r, f_s, fuse) in enumerate(zip(raw_skips, rest_skips, self.fusions)):
            fused, w_rest = fuse(f_r, f_s)
            fused_skips.append(fused)
            if return_gates:
                gates[f"scale_{idx}"] = w_rest

        x, w_bn = self.bottleneck_fusion(r, s)
        if return_gates:
            gates["bottleneck"] = w_bn

        fused_skips = fused_skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = fused_skips[idx // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        logits = self.final_conv(x)
        if return_gates:
            return logits, gates
        return logits


def get_dual_stream_unet(in_channels=3, out_channels=1, base_channels=32, bottleneck_dropout=0.2):
    """Factory function for DualStreamUNet with configurable base channels."""
    features = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
    return DualStreamUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        bottleneck_dropout=bottleneck_dropout,
    )


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing DualStreamUNet on device: {device}")

    model = get_dual_stream_unet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    raw_ckpt = r"D:\SFHformer-FreqLab\checkpoints\unet_raw\best.pth"
    rest_ckpt = r"D:\SFHformer-FreqLab\checkpoints\unet_restored\best.pth"
    if os.path.exists(raw_ckpt) and os.path.exists(rest_ckpt):
        n_loaded, n_missing = model.load_pretrained_streams(raw_ckpt, rest_ckpt, map_location="cpu")
        print(f"Warm-start verified: {n_loaded} tensors loaded, {n_missing} fusion tensors randomly initialized.")

    x_raw = torch.randn(2, 3, 512, 512, device=device)
    x_rest = torch.randn(2, 3, 512, 512, device=device)
    target = torch.randint(0, 2, (2, 1, 512, 512), device=device, dtype=torch.float32)

    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.65, device=0)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, gates = model(x_raw, x_rest, return_gates=True)
            loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Output shape: {out.shape} | Gate keys: {list(gates.keys())}")
        print(f"BF16 AMP Peak VRAM (Batch 2, 512x512): {peak_mb:.2f} MB")
    else:
        out, gates = model(x_raw, x_rest, return_gates=True)
        print(f"Output shape: {out.shape} | Gate keys: {list(gates.keys())}")

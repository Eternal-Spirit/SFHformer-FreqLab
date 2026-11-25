"""
models/unet.py - Modular U-Net for Retinal Vessel Segmentation.
Supports BF16 AMP, gradient checkpointing (optional), and bilinear upsampling to prevent checkerboard artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two Conv-BatchNorm-GELU layers with optional residual connection."""
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.residual(x)

class UNet(nn.Module):
    """
    Standard 4-stage U-Net architecture for medical image segmentation.
    Features: 32 -> 64 -> 128 -> 256 -> 512 (Bottleneck).
    Decoder uses Bilinear interpolation + Conv to avoid checkerboard artifacts.
    """
    def __init__(self, in_channels=3, out_channels=1, features=(32, 64, 128, 256), bottleneck_dropout=0.2):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        cur_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(cur_in, feature))
            cur_in = feature

        # Bottleneck
        bottleneck_features = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_features, dropout_p=bottleneck_dropout)

        # Decoder (Upsampling)
        reversed_features = list(reversed(features))
        cur_features = bottleneck_features
        for feature in reversed_features:
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(cur_features, feature, kernel_size=1, bias=False),
                    nn.BatchNorm2d(feature),
                    nn.GELU()
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            cur_features = feature

        # Final classification head
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downward path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upward path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # In case input dimensions are not perfectly divisible by 2^N
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            concat_x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        return self.final_conv(x)

def get_unet(in_channels=3, out_channels=1, base_channels=32):
    """Factory function for U-Net with configurable base channels."""
    features = (base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
    return UNet(in_channels=in_channels, out_channels=out_channels, features=features)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing UNet on device: {device}")
    
    model = get_unet(in_channels=3, out_channels=1, base_channels=32).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Simulate forward + backward pass under BF16 AMP
    x = torch.randn(2, 3, 512, 512, device=device)
    target = torch.randint(0, 2, (2, 1, 512, 512), device=device, dtype=torch.float32)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, target)
        loss.backward()
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Output Shape: {out.shape}")
        print(f"BF16 AMP Peak VRAM (Batch 2, 512x512): {peak_vram:.2f} MB")
        print("Forward & Backward Pass Passed Successfully!")
    else:
        out = model(x)
        print(f"Output Shape: {out.shape}")

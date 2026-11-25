"""
scripts/generate_phase5_3_diagrams.py
Generates 5 neat, well-ordered, publication-grade architectural diagrams (300 DPI)
covering every architecture in SFHformer-FreqLab before compiling the Phase 5.3 demonstration package:
  1. diag_1_sfhformer_macro.png
  2. diag_2_sfh_block_detail.png
  3. diag_3_single_unet.png
  4. diag_4_dual_stream_hybrid_unet.png
  5. diag_5_combined_retinascan_system.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = r"D:\SFHformer-FreqLab\demo_package\diagrams"
os.makedirs(OUT_DIR, exist_ok=True)

# Clean Light Scientific Palette
BG_PAGE   = "#F8FAFC"
BG_WHITE  = "#FFFFFF"
BDR_SLATE = "#CBD5E1"
NAVY_DARK = "#0F172A"
TEXT_BODY = "#334155"
TEXT_MUTE = "#64748B"

C_BLUE    = "#1D4ED8"
BG_BLUE   = "#EFF6FF"
BDR_BLUE  = "#60A5FA"

C_INDIGO  = "#4338CA"
BG_INDIGO = "#EEF2FF"
BDR_IND   = "#818CF8"

C_PURPLE  = "#7E22CE"
BG_PURPLE = "#FAF5FF"
BDR_PURP  = "#C084FC"

C_TEAL    = "#0F766E"
BG_TEAL   = "#F0FDFA"
BDR_TEAL  = "#2DD4BF"

C_GREEN   = "#15803D"
BG_GREEN  = "#F0FDF4"
BDR_GRN   = "#4ADE80"

C_AMBER   = "#B45309"
BG_AMBER  = "#FFFBEB"
BDR_AMB   = "#FBBF24"

C_CYAN    = "#0369A1"
BG_CYAN   = "#F0F9FF"
BDR_CYAN  = "#38BDF8"


def card(ax, x, y, w, h, bg=BG_WHITE, bdr=BDR_SLATE, lw=1.2, r=0.18, z=1, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=bg, edgecolor=bdr, linewidth=lw, linestyle=ls, zorder=z)
    ax.add_patch(p)


def box(ax, x, y, w, h, title, sub="", bg=BG_WHITE, bdr=BDR_SLATE, lw=1.1, r=0.14,
        tc=NAVY_DARK, sc=TEXT_MUTE, ts=9.5, ss=7.8, z=3):
    card(ax, x, y, w, h, bg=bg, bdr=bdr, lw=lw, r=r, z=z)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=tc, zorder=z + 1)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=ss, color=sc, linespacing=1.28, zorder=z + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color=tc, zorder=z + 1)


def arrow(ax, x1, y1, x2, y2, col="#2563EB", lw=1.6, ls="-", rad=0.0, lbl="", lx=None, ly=None, z=5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35",
                                color=col, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0),
                zorder=z)
    if lbl:
        mx = lx if lx is not None else (x1 + x2) / 2
        my = ly if ly is not None else (y1 + y2) / 2
        ax.text(mx, my, lbl, ha="center", va="center", fontsize=7.5, fontweight="bold",
                color=NAVY_DARK, bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=BDR_SLATE, lw=0.7),
                zorder=z + 1)


def header_banner(ax, W, H, title, subtitle):
    card(ax, 0.4, H - 1.15, W - 0.8, 0.85, bg=NAVY_DARK, bdr="#1E293B", lw=1.4, r=0.15, z=2)
    ax.text(W / 2, H - 0.60, title, ha="center", va="center", fontsize=14, fontweight="bold", color="#F8FAFC", zorder=3)
    ax.text(W / 2, H - 0.94, subtitle, ha="center", va="center", fontsize=9.0, color="#94A3B8", zorder=3)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1: SFHFORMER-M MACRO ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def make_diag_1():
    W, H = 22, 12
    fig = plt.figure(figsize=(W, H), facecolor=BG_PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    header_banner(ax, W, H,
                  "1. SFHformer-m Macro-Architecture (Spatial-Frequency Hybrid Restoration Transformer)",
                  "models/SFHformer.py  |  14.18M Parameters  |  Depths [4, 4, 8, 4, 4]  |  Channels [32, 64, 128, 64, 32]  |  Checkpoint: fives_retina_finetune (31.92 dB)")

    # Top Global Residual Track (Y = 9.5 .. 10.5)
    box(ax, 0.6, 7.6, 2.3, 1.5, "Input Fundus", "(B, 3, H, W)\nNormalized [0, 1]", bg=BG_CYAN, bdr=BDR_CYAN, ts=10, ss=8.5)
    box(ax, 3.5, 7.8, 2.1, 1.1, "PatchEmbed", "Conv 3x3, s=1, p=1\n3 -> 32 channels", bg=BG_BLUE, bdr=BDR_BLUE, ts=9.5, ss=8)
    arrow(ax, 2.9, 8.35, 3.5, 8.35, col=C_BLUE)

    # Encoder Stage 1
    card(ax, 6.2, 6.8, 3.2, 2.8, bg=BG_BLUE, bdr=C_BLUE, lw=1.3)
    ax.text(7.8, 9.3, "ENCODER STAGE 1", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_BLUE)
    box(ax, 6.45, 7.85, 2.7, 1.1, "4x SFH Blocks", "dim = 32  |  H x W\nLocal + Fourier Mixing", bg=BG_WHITE, bdr=BDR_BLUE, ts=9, ss=7.8)
    box(ax, 6.45, 7.0, 2.7, 0.65, "Save Skip Feature (copy1)", "(B, 32, H, W)", bg=BG_AMBER, bdr=BDR_AMB, ts=8, ss=7.2)
    arrow(ax, 5.6, 8.35, 6.45, 8.35, col=C_BLUE)

    # DownSample 1
    box(ax, 6.7, 5.4, 2.2, 0.85, "DownSample 1", "Conv 3x3, s=2 (32 -> 64)", bg=BG_INDIGO, bdr=BDR_IND, ts=8.5, ss=7.5)
    arrow(ax, 7.8, 6.8, 7.8, 6.25, col=C_INDIGO)

    # Encoder Stage 2
    card(ax, 6.2, 2.1, 3.2, 2.8, bg=BG_INDIGO, bdr=C_INDIGO, lw=1.3)
    ax.text(7.8, 4.6, "ENCODER STAGE 2", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_INDIGO)
    box(ax, 6.45, 3.15, 2.7, 1.1, "4x SFH Blocks", "dim = 64  |  H/2 x W/2\nMid-Scale Spectral Filter", bg=BG_WHITE, bdr=BDR_IND, ts=9, ss=7.8)
    box(ax, 6.45, 2.3, 2.7, 0.65, "Save Skip Feature (copy2)", "(B, 64, H/2, W/2)", bg=BG_AMBER, bdr=BDR_AMB, ts=8, ss=7.2)
    arrow(ax, 7.8, 5.4, 7.8, 4.9, col=C_INDIGO)

    # DownSample 2
    box(ax, 9.9, 3.25, 2.1, 0.9, "DownSample 2", "Conv 3x3, s=2\n64 -> 128 dims", bg=BG_INDIGO, bdr=BDR_IND, ts=8.5, ss=7.5)
    arrow(ax, 9.4, 3.7, 9.9, 3.7, col=C_INDIGO)

    # Bottleneck Stage 3
    card(ax, 9.7, 0.5, 4.0, 2.1, bg=BG_PURPLE, bdr=C_PURPLE, lw=1.5)
    ax.text(11.7, 2.3, "STAGE 3: DEEP BOTTLENECK", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_PURPLE)
    box(ax, 10.0, 0.75, 3.4, 1.25, "8x SFH Blocks (dim = 128)", "Spatial Size: H/4 x W/4\nGlobal Illumination, Halo & Destriping", bg=BG_WHITE, bdr=BDR_PURP, ts=9, ss=7.8)
    arrow(ax, 10.95, 3.25, 10.95, 2.6, col=C_PURPLE)

    # Upsample 3
    box(ax, 14.3, 1.15, 2.2, 0.95, "Upsample 3", "Conv 1x1 + PixelShuffle(2)\n128 -> 64 dims", bg=BG_GREEN, bdr=BDR_GRN, ts=8.5, ss=7.5)
    arrow(ax, 13.7, 1.62, 14.3, 1.62, col=C_GREEN)

    # Decoder Stage 4
    card(ax, 14.0, 2.6, 3.4, 2.8, bg=BG_GREEN, bdr=C_GREEN, lw=1.3)
    ax.text(15.7, 5.1, "DECODER STAGE 4", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_GREEN)
    box(ax, 14.25, 4.05, 2.9, 0.72, "Skip2 Fusion: Conv 1x1", "Concat(x, copy2): 128 -> 64", bg=BG_AMBER, bdr=BDR_AMB, ts=8, ss=7.2)
    box(ax, 14.25, 2.85, 2.9, 1.0, "4x SFH Blocks", "dim = 64  |  H/2 x W/2\nVessel Branch Continuity", bg=BG_WHITE, bdr=BDR_GRN, ts=9, ss=7.8)
    arrow(ax, 15.4, 2.1, 15.4, 2.6, col=C_GREEN)

    # Skip 2 Lateral Arrow (routed cleanly below DownSample 2)
    ax.plot([9.15, 11.7, 13.7, 14.25], [2.62, 2.62, 4.4, 4.4], color=C_AMBER, lw=1.5, ls="--", zorder=4)
    arrow(ax, 13.95, 4.4, 14.25, 4.4, col=C_AMBER, lw=1.5, ls="--")
    ax.text(11.6, 2.82, "Skip 2 (copy2)", ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=NAVY_DARK, bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=BDR_SLATE, lw=0.7), zorder=6)

    # Upsample 4
    box(ax, 14.6, 5.75, 2.2, 0.85, "Upsample 4", "Conv 1x1 + PixelShuffle(2)\n64 -> 32 dims", bg=BG_GREEN, bdr=BDR_GRN, ts=8.5, ss=7.5)
    arrow(ax, 15.7, 5.4, 15.7, 5.75, col=C_GREEN)

    # Decoder Stage 5
    card(ax, 14.0, 6.9, 3.4, 2.7, bg=BG_GREEN, bdr=C_GREEN, lw=1.3)
    ax.text(15.7, 9.3, "DECODER STAGE 5", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_GREEN)
    box(ax, 14.25, 8.25, 2.9, 0.72, "Skip1 Fusion: Conv 1x1", "Concat(x, copy1): 64 -> 32", bg=BG_AMBER, bdr=BDR_AMB, ts=8, ss=7.2)
    box(ax, 14.25, 7.1, 2.9, 0.95, "4x SFH Blocks", "dim = 32  |  H x W\nCapillary Edge Sharpening", bg=BG_WHITE, bdr=BDR_GRN, ts=9, ss=7.8)
    arrow(ax, 15.7, 6.6, 15.7, 6.9, col=C_GREEN)

    # Skip 1 Lateral Arrow
    arrow(ax, 9.15, 7.32, 14.25, 8.6, col=C_AMBER, lw=1.5, ls="--", lbl="Skip 1 (copy1)", lx=11.7, ly=8.2)

    # PatchUnEmbed & Global Residual Output
    box(ax, 18.0, 7.8, 2.1, 1.1, "PatchUnEmbed", "Conv 3x3, s=1, p=1\n32 -> 3 channels", bg=BG_BLUE, bdr=BDR_BLUE, ts=9.5, ss=8)
    arrow(ax, 17.4, 8.35, 18.0, 8.35, col=C_GREEN)

    box(ax, 18.2, 5.5, 1.7, 1.0, "Global Sum (+)", "x_in + Residual", bg=BG_AMBER, bdr=C_AMBER, ts=9, ss=7.8)
    arrow(ax, 19.05, 7.8, 19.05, 6.5, col=C_BLUE)

    # Long Global Identity Wire across Top
    ax.plot([1.75, 1.75, 20.6, 20.6, 19.9], [9.1, 10.35, 10.35, 6.0, 6.0], color=C_CYAN, lw=1.8, linestyle="-", zorder=4)
    arrow(ax, 20.1, 6.0, 19.9, 6.0, col=C_CYAN, lw=1.8)
    ax.text(11.0, 10.52, "Global Identity Residual Connection (Network learns pure degradation residual: Output = copy0 + Backbone(copy0))",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_CYAN,
            bbox=dict(boxstyle="round,pad=0.22", fc=BG_CYAN, ec=BDR_CYAN))

    box(ax, 17.9, 3.6, 2.4, 1.25, "Restored Fundus", "(B, 3, H, W)  |  [0, 1]\nFIVES SOTA: 31.92 dB", bg=BG_TEAL, bdr=C_TEAL, lw=1.5, ts=10, ss=8.2)
    arrow(ax, 19.05, 5.5, 19.05, 4.85, col=C_TEAL, lw=1.8)

    # Training Info Footer Card
    card(ax, 0.6, 0.5, 8.5, 1.3, bg=BG_WHITE, bdr=BDR_SLATE)
    ax.text(4.85, 1.48, "SFHformer Training & Restoration Specifications", ha="center", va="center", fontsize=9.2, fontweight="bold", color=NAVY_DARK)
    ax.text(4.85, 0.92, "• Loss: L1 Pixel Reconstruction   • Optimizer: AdamW + CosineAnnealingLR\n"
                        "• Data: 540 FIVES Training Scans (27,000 crops)   • SHA-256: d74e267c6ac66468...",
            ha="center", va="center", fontsize=8.0, color=TEXT_BODY, linespacing=1.4)

    out = os.path.join(OUT_DIR, "diag_1_sfhformer_macro.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2: SFH BLOCK IN-DEPTH ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def make_diag_2():
    W, H = 22, 12
    fig = plt.figure(figsize=(W, H), facecolor=BG_PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    header_banner(ax, W, H,
                  "2. SFH Block: Spatial-Frequency Hybrid Token Mixer & Feed-Forward Network",
                  "models/SFHformer.py (Block, Mixer, TokenMixer_For_Local, TokenMixer_For_Gloal, FourierUnit, FFN)  &  freqlab/gate.py (RadialBandGate)")

    # Outer Block Frame
    card(ax, 0.4, 0.5, 21.2, 10.1, bg=BG_WHITE, bdr=BDR_SLATE, lw=1.3)

    # Left Input & Norm 1
    box(ax, 0.7, 5.1, 1.8, 1.3, "Block Input\nTensor X", "(B, C, H, W)", bg=BG_CYAN, bdr=BDR_CYAN, ts=9.5, ss=8)
    box(ax, 2.8, 5.25, 1.3, 1.0, "norm1", "BatchNorm2d\n(C dims)", bg="#F1F5F9", bdr=BDR_SLATE, ts=9, ss=7.5)
    arrow(ax, 2.5, 5.75, 2.8, 5.75, col=C_BLUE)

    # ── MIXER MODULE ──
    card(ax, 4.4, 1.2, 11.4, 9.1, bg="#F8FAFC", bdr=C_INDIGO, lw=1.5)
    ax.text(10.1, 9.95, "HYBRID TOKEN MIXER (Parallel Local Spatial + Global Spectral Mixing)",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_INDIGO)

    box(ax, 4.65, 5.25, 1.3, 1.0, "conv_init", "Conv 1x1\nC -> 2C", bg=BG_INDIGO, bdr=BDR_IND, ts=8.5, ss=7.5)
    arrow(ax, 4.1, 5.75, 4.65, 5.75, col=C_INDIGO)

    box(ax, 6.2, 5.25, 1.0, 1.0, "Split", "x[0]: C\nx[1]: C", bg=BG_AMBER, bdr=BDR_AMB, ts=8.5, ss=7.5)
    arrow(ax, 5.95, 5.75, 6.2, 5.75, col=C_INDIGO)

    # Branch A: Local Spatial Mixer (Top)
    card(ax, 7.5, 6.6, 5.8, 2.9, bg=BG_BLUE, bdr=C_BLUE, lw=1.3)
    ax.text(10.4, 9.2, "LOCAL SPATIAL BRANCH: TokenMixer_For_Local", ha="center", va="center", fontsize=9, fontweight="bold", color=C_BLUE)
    box(ax, 7.75, 7.75, 1.2, 1.0, "Chunk(2)", "x1: C/2\nx2: C/2", bg=BG_WHITE, bdr=BDR_BLUE, ts=8, ss=7.2)
    box(ax, 9.3, 8.15, 2.1, 0.75, "CDilated_1", "DW-Conv 3x3, d=1, g=C/2", bg=BG_WHITE, bdr=BDR_BLUE, ts=8, ss=7.0)
    box(ax, 9.3, 7.1, 2.1, 0.75, "CDilated_2", "DW-Conv 3x3, d=2, g=C/2", bg=BG_WHITE, bdr=BDR_BLUE, ts=8, ss=7.0)
    box(ax, 11.75, 7.6, 1.3, 0.9, "Concat", "[cd1, cd2]\n(B, C, H, W)", bg=BG_WHITE, bdr=BDR_BLUE, ts=8, ss=7.0)
    arrow(ax, 6.7, 6.25, 7.75, 8.25, col=C_BLUE)
    arrow(ax, 8.95, 8.4, 9.3, 8.5, col=C_BLUE)
    arrow(ax, 8.95, 8.0, 9.3, 7.45, col=C_BLUE)
    arrow(ax, 11.4, 8.5, 11.75, 8.2, col=C_BLUE)
    arrow(ax, 11.4, 7.45, 11.75, 7.9, col=C_BLUE)

    # Branch B: Global Fourier Mixer (Bottom - strictly left-to-right flow, zero crossings)
    card(ax, 7.5, 1.4, 6.2, 4.9, bg=BG_PURPLE, bdr=C_PURPLE, lw=1.3)
    ax.text(10.6, 6.02, "GLOBAL SPECTRAL BRANCH: TokenMixer_For_Gloal + FourierUnit", ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_PURPLE)

    box(ax, 7.68, 3.3, 1.15, 1.2, "conv_init", "Conv 1x1\n+ GELU\nC->2C (x0)", bg=BG_WHITE, bdr=BDR_PURP, ts=7.6, ss=6.4)
    arrow(ax, 6.7, 5.25, 7.68, 3.9, col=C_PURPLE)

    # FourierUnit subcard (neat 2x2 grid + bottom gate bar)
    card(ax, 9.05, 1.7, 3.2, 3.95, bg="#FFFFFF", bdr=C_PURPLE, lw=1.1, ls="--")
    ax.text(10.65, 5.45, "FourierUnit (2D Spectral)", ha="center", va="center", fontsize=7.9, fontweight="bold", color=C_PURPLE)
    box(ax, 9.18, 4.3, 1.4, 0.85, "2D rfft2", "ortho FFT\nCat(Re,Im)", bg=BG_PURPLE, bdr=BDR_PURP, ts=7.4, ss=6.2)
    box(ax, 10.72, 4.3, 1.4, 0.85, "BN + FPE", "BatchNorm\nDW 3x3+id", bg=BG_PURPLE, bdr=BDR_PURP, ts=7.4, ss=6.2)
    box(ax, 10.72, 3.05, 1.4, 0.85, "FDC+Einsum", "Grp 1x1 *\nSoftmax W", bg=BG_PURPLE, bdr=BDR_PURP, ts=7.2, ss=6.2)
    box(ax, 9.18, 3.05, 1.4, 0.85, "GELU+irfft2", "Complex ->\nirfft2(H,W)", bg=BG_PURPLE, bdr=BDR_PURP, ts=7.2, ss=6.2)
    box(ax, 9.18, 1.88, 2.94, 0.82, "Optional RadialBandGate (R2)", "16 learnable radial bands [0, 1]", bg=BG_AMBER, bdr=BDR_AMB, ts=7.2, ss=6.2)

    arrow(ax, 8.83, 4.72, 9.18, 4.72, col=C_PURPLE)
    arrow(ax, 10.58, 4.72, 10.72, 4.72, col=C_PURPLE)
    arrow(ax, 11.42, 4.3, 11.42, 3.9, col=C_PURPLE)
    arrow(ax, 10.72, 3.47, 10.58, 3.47, col=C_PURPLE)
    # Clean bottom skip wire x0 -> Add
    ax.plot([8.25, 8.25, 12.95, 12.95], [3.3, 1.55, 1.55, 2.85], color=C_PURPLE, lw=1.1, ls="--", zorder=4)

    box(ax, 12.42, 2.85, 1.12, 1.2, "Skip + Final", "FU + x0\nConv 1x1\n2C -> C", bg=BG_WHITE, bdr=BDR_PURP, ts=7.5, ss=6.3)
    arrow(ax, 9.88, 3.05, 9.88, 2.75, col=C_AMBER, lw=1.0)
    # Route GELU+irfft2 output cleanly along y=2.88 lane below FDC+Einsum into Skip + Final
    ax.plot([9.88, 9.88, 12.22, 12.42], [3.05, 2.88, 2.88, 3.25], color=C_PURPLE, lw=1.4, zorder=5)
    arrow(ax, 12.22, 2.88, 12.42, 3.25, col=C_PURPLE, lw=1.4)

    # Channel Attention & Fusion (pushed slightly right to clear subcard cleanly)
    box(ax, 14.0, 5.2, 1.65, 1.4, "Channel Fusion", "Concat(2C)+GELU\nSE Attn (Sigmoid)\nConv 1x1 (2C->C)", bg=BG_GREEN, bdr=BDR_GRN, ts=8.1, ss=6.8)
    arrow(ax, 13.05, 8.05, 14.0, 6.2, col=C_BLUE)
    arrow(ax, 13.54, 3.45, 14.0, 5.4, col=C_PURPLE)

    # Post-Mixer Scale Beta + Skip
    box(ax, 16.05, 5.3, 1.4, 1.15, "Residual Sum 1", "x * beta + copy\n(Learnable beta)", bg=BG_AMBER, bdr=BDR_AMB, ts=8.3, ss=7.2)
    arrow(ax, 15.65, 5.88, 16.05, 5.88, col=C_GREEN)

    # Long Skip over Mixer
    ax.plot([1.6, 1.6, 16.8, 16.8], [6.4, 10.1, 10.1, 6.45], color=TEXT_MUTE, lw=1.3, ls=":", zorder=4)
    arrow(ax, 16.8, 6.65, 16.8, 6.45, col=TEXT_MUTE, lw=1.3)

    # Norm 2 -> FFN -> Gamma Skip -> Out
    box(ax, 17.8, 5.4, 1.2, 0.95, "norm2", "BatchNorm2d", bg="#F1F5F9", bdr=BDR_SLATE, ts=8.5, ss=7.2)
    arrow(ax, 17.5, 5.88, 17.8, 5.88, col=NAVY_DARK)

    box(ax, 19.25, 5.05, 2.05, 1.65, "FFN Module", "Conv 1x1 (C -> 4C)\nDW-Conv 3x3 + GELU\nConv 1x1 (4C -> C)", bg="#FDF2F8", bdr="#EC4899", ts=8.8, ss=7.4)
    arrow(ax, 19.0, 5.88, 19.25, 5.88, col=NAVY_DARK)

    box(ax, 19.25, 2.6, 2.05, 1.25, "Residual Sum 2\n(Block Output)", "x * gamma + copy\n(B, C, H, W)", bg=BG_CYAN, bdr=BDR_CYAN, ts=8.8, ss=7.6)
    arrow(ax, 20.27, 5.05, 20.27, 3.85, col="#DB2777")
    ax.plot([16.8, 16.8, 19.25], [5.3, 3.22, 3.22], color=TEXT_MUTE, lw=1.3, ls=":", zorder=4)
    arrow(ax, 19.05, 3.22, 19.25, 3.22, col=TEXT_MUTE, lw=1.3)

    out = os.path.join(OUT_DIR, "diag_2_sfh_block_detail.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3: CLINICAL SINGLE-STREAM U-NET ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def make_diag_3():
    W, H = 22, 12
    fig = plt.figure(figsize=(W, H), facecolor=BG_PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    header_banner(ax, W, H,
                  "3. Clinical Single-Stream U-Net Segmentation Architecture",
                  "models/unet.py  |  4-Stage Residual Encoder-Decoder (32 -> 64 -> 128 -> 256 -> 512)  |  Bilinear Upsampling  |  Checkpoints: unet_raw & unet_restored")

    # Input
    box(ax, 0.5, 7.6, 2.1, 1.3, "Fundus Input", "(B, 3, 512, 512)\nRaw OR Restored", bg=BG_CYAN, bdr=BDR_CYAN, ts=9.5, ss=8)

    # Encoder Staircase
    box(ax, 3.1, 7.75, 2.2, 1.0, "Encoder Stage 1", "DoubleConv (3 -> 32)\n512 x 512", bg=BG_BLUE, bdr=BDR_BLUE, ts=9, ss=7.8)
    arrow(ax, 2.6, 8.25, 3.1, 8.25, col=C_BLUE)

    box(ax, 4.9, 6.0, 2.2, 1.0, "Encoder Stage 2", "DoubleConv (32 -> 64)\n256 x 256", bg=BG_BLUE, bdr=BDR_BLUE, ts=9, ss=7.8)
    arrow(ax, 4.2, 7.75, 4.9, 6.5, col=C_BLUE, lbl="MaxPool 2x", lx=4.1, ly=6.9)

    box(ax, 6.7, 4.25, 2.2, 1.0, "Encoder Stage 3", "DoubleConv (64 -> 128)\n128 x 128", bg=BG_BLUE, bdr=BDR_BLUE, ts=9, ss=7.8)
    arrow(ax, 6.0, 6.0, 6.7, 4.75, col=C_BLUE, lbl="MaxPool 2x", lx=5.9, ly=5.15)

    box(ax, 8.5, 2.5, 2.2, 1.0, "Encoder Stage 4", "DoubleConv (128 -> 256)\n64 x 64", bg=BG_BLUE, bdr=BDR_BLUE, ts=9, ss=7.8)
    arrow(ax, 7.8, 4.25, 8.5, 3.0, col=C_BLUE, lbl="MaxPool 2x", lx=7.7, ly=3.4)

    # Bottleneck
    box(ax, 9.8, 0.75, 3.1, 1.15, "Bottleneck (512)", "DoubleConv (256 -> 512)\nDropout2d(p=0.20)  |  32 x 32", bg=BG_PURPLE, bdr=BDR_PURP, lw=1.4, ts=9.2, ss=7.8)
    arrow(ax, 9.6, 2.5, 10.6, 1.9, col=C_PURPLE, lbl="MaxPool 2x", lx=9.7, ly=2.15)

    # Decoder Staircase
    box(ax, 12.2, 2.5, 2.7, 1.0, "Decoder Stage 4", "Bilinear 2x + Conv 1x1\nDoubleConv (512 -> 256)", bg=BG_GREEN, bdr=BDR_GRN, ts=9, ss=7.6)
    arrow(ax, 12.1, 1.9, 13.0, 2.5, col=C_GREEN)

    box(ax, 14.0, 4.25, 2.7, 1.0, "Decoder Stage 3", "Bilinear 2x + Conv 1x1\nDoubleConv (256 -> 128)", bg=BG_GREEN, bdr=BDR_GRN, ts=9, ss=7.6)
    arrow(ax, 13.8, 3.5, 14.8, 4.25, col=C_GREEN)

    box(ax, 15.8, 6.0, 2.7, 1.0, "Decoder Stage 2", "Bilinear 2x + Conv 1x1\nDoubleConv (128 -> 64)", bg=BG_GREEN, bdr=BDR_GRN, ts=9, ss=7.6)
    arrow(ax, 15.6, 5.25, 16.6, 6.0, col=C_GREEN)

    box(ax, 17.6, 7.75, 2.7, 1.0, "Decoder Stage 1", "Bilinear 2x + Conv 1x1\nDoubleConv (64 -> 32)", bg=BG_GREEN, bdr=BDR_GRN, ts=9, ss=7.6)
    arrow(ax, 17.4, 7.0, 18.4, 7.75, col=C_GREEN)

    # Lateral Skip Connections
    arrow(ax, 5.3, 8.25, 17.6, 8.25, col=C_AMBER, lw=1.4, ls="--", lbl="Skip 1 (32 ch, 512x512)", lx=11.45, ly=8.45)
    arrow(ax, 7.1, 6.5, 15.8, 6.5, col=C_AMBER, lw=1.4, ls="--", lbl="Skip 2 (64 ch, 256x256)", lx=11.45, ly=6.7)
    arrow(ax, 8.9, 4.75, 14.0, 4.75, col=C_AMBER, lw=1.4, ls="--", lbl="Skip 3 (128 ch, 128x128)", lx=11.45, ly=4.95)
    arrow(ax, 10.7, 3.0, 12.2, 3.0, col=C_AMBER, lw=1.4, ls="--", lbl="Skip 4 (256 ch)", lx=11.45, ly=3.2)

    # Output Head
    box(ax, 17.8, 9.4, 2.3, 1.05, "Segmentation Head", "Conv 1x1 (32 -> 1) + Sigmoid\nFrozen Baseline tau = 0.50", bg=BG_AMBER, bdr=BDR_AMB, ts=9, ss=7.6)
    arrow(ax, 18.95, 8.75, 18.95, 9.4, col=C_GREEN, lw=1.6)

    # Inset 1: DoubleConv Internals
    card(ax, 0.5, 0.75, 6.0, 3.2, bg=BG_WHITE, bdr=C_BLUE, lw=1.3)
    ax.text(3.5, 3.65, "Internal Structure: DoubleConv Residual Block", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_BLUE)
    ax.text(3.5, 2.2,
            "Main Path:\n"
            "  Conv2d(C_in -> C_out, 3x3, p=1) -> BatchNorm2d -> GELU\n"
            "  -> [Dropout2d(0.2) in Bottleneck only]\n"
            "  -> Conv2d(C_out -> C_out, 3x3, p=1) -> BatchNorm2d -> GELU\n"
            "Residual Path (+):\n"
            "  Conv2d(C_in -> C_out, 1x1) if C_in != C_out else Identity()",
            ha="center", va="center", fontsize=8.0, color=TEXT_BODY, linespacing=1.4)

    # Inset 2: Dual Single-Stream Baseline Roles
    card(ax, 14.0, 0.75, 7.4, 1.45, bg=BG_WHITE, bdr=C_TEAL, lw=1.3)
    ax.text(17.7, 1.92, "Dual Single-Stream Baselines (Trained Separately with BCEDiceLoss)", ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_TEAL)
    ax.text(17.7, 1.25,
            "• Raw U-Net (unet_raw, SHA: 97639a65): FIVES Test Dice 0.8133, Prec 0.9266, clDice 0.8134\n"
            "• Restored U-Net (unet_restored, SHA: 2cfb3737): HRF ThinSkel +13.67 pp, FIVES Dice 0.6927",
            ha="center", va="center", fontsize=7.8, color=TEXT_BODY, linespacing=1.35)

    out = os.path.join(OUT_DIR, "diag_3_single_unet.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 4: DUAL-STREAM HYBRID U-NET (NOVEL ARCHITECTURE + FUSION BLOCK)
# ─────────────────────────────────────────────────────────────────────────────
def make_diag_4():
    W, H = 24, 13
    fig = plt.figure(figsize=(W, H), facecolor=BG_PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    header_banner(ax, W, H,
                  "4. Dual-Stream Hybrid U-Net Architecture (DualStreamUNet + CrossStreamFusionBlock)",
                  "models/dual_stream_unet.py  |  16.41M Params  |  208 Warm-Started + 80 Fusion Tensors  |  Calibrated Frozen tau = 0.40  |  FIVES Dice: 0.8474")

    # Left Inputs
    box(ax, 0.5, 9.2, 2.3, 1.4, "Stream A Input\n(Raw Fundus)", "(B, 3, H, W)\nMacro Precision Source", bg=BG_CYAN, bdr=BDR_CYAN, ts=9.5, ss=8)
    box(ax, 0.5, 4.4, 2.3, 1.4, "Stream B Input\n(SFHformer Restored)", "(B, 3, H, W)\nFine Capillary Source", bg=BG_PURPLE, bdr=BDR_PURP, ts=9.5, ss=8)

    # Stream A Encoder Row (Top, Y = 8.9 .. 10.9)
    card(ax, 3.2, 8.6, 8.4, 2.7, bg=BG_BLUE, bdr=C_BLUE, lw=1.3)
    ax.text(7.4, 11.0, "STREAM A: RAW MACRO-PRECISION ENCODER (Init: checkpoints/unet_raw/best.pth)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_BLUE)
    s_labels = [("s0: 32", "512"), ("s1: 64", "256"), ("s2: 128", "128"), ("s3: 256", "64"), ("BN: 512", "32")]
    for i, (ch, res) in enumerate(s_labels):
        bx = 3.4 + i * 1.62
        box(ax, bx, 9.25, 1.4, 1.25, ch, f"{res}x{res}\nDoubleConv", bg=BG_WHITE, bdr=BDR_BLUE, ts=8.2, ss=7.0)
        if i > 0:
            arrow(ax, bx - 0.22, 9.87, bx, 9.87, col=C_BLUE, lw=1.3)
    arrow(ax, 2.8, 9.9, 3.4, 9.9, col=C_CYAN, lw=1.6)

    # Stream B Encoder Row (Bottom, Y = 3.8 .. 6.5)
    card(ax, 3.2, 3.8, 8.4, 2.7, bg=BG_PURPLE, bdr=C_PURPLE, lw=1.3)
    ax.text(7.4, 6.2, "STREAM B: RESTORED CAPILLARY-SENSITIVITY ENCODER (Init: checkpoints/unet_restored/best.pth)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_PURPLE)
    for i, (ch, res) in enumerate(s_labels):
        bx = 3.4 + i * 1.62
        box(ax, bx, 4.45, 1.4, 1.25, ch, f"{res}x{res}\nDoubleConv", bg=BG_WHITE, bdr=BDR_PURP, ts=8.2, ss=7.0)
        if i > 0:
            arrow(ax, bx - 0.22, 5.07, bx, 5.07, col=C_PURPLE, lw=1.3)
    arrow(ax, 2.8, 5.1, 3.4, 5.1, col=C_PURPLE, lw=1.6)

    # Center Fusion Spine (Y = 6.95 .. 8.15)
    fuse_names = ["Fusion s0\n(32 ch)", "Fusion s1\n(64 ch)", "Fusion s2\n(128 ch)", "Fusion s3\n(256 ch)", "Fusion BN\n(512 ch)"]
    for i, fn in enumerate(fuse_names):
        bx = 3.4 + i * 1.62
        box(ax, bx, 7.05, 1.4, 1.0, fn, "Spatial Gate\n+ SE Cross", bg=BG_AMBER, bdr=C_AMBER, lw=1.3, ts=8.0, ss=6.8)
        arrow(ax, bx + 0.7, 9.25, bx + 0.7, 8.05, col=C_BLUE, lw=1.2)
        arrow(ax, bx + 0.7, 5.7, bx + 0.7, 7.05, col=C_PURPLE, lw=1.2)

    # Optional Gate Export Callout from Fusion s0 (routed cleanly left of Stream B)
    ax.plot([3.4, 3.0, 3.0, 1.7], [7.55, 7.55, 3.25, 2.9], color=C_AMBER, lw=1.4, ls="--", zorder=5)
    arrow(ax, 2.1, 3.0, 1.7, 2.9, col=C_AMBER, lw=1.4, ls="--")
    box(ax, 0.5, 1.6, 2.4, 1.3, "Optional Gate Map\n(scale_0: w_rest)", "2D Spatial Routing [0, 1]\nVisualized in UI only", bg=BG_AMBER, bdr=BDR_AMB, ts=8.3, ss=7.2)

    # Shared Bilinear Decoder (Right Side X = 12.1 .. 17.5)
    card(ax, 12.1, 5.6, 5.4, 5.7, bg=BG_GREEN, bdr=C_GREEN, lw=1.4)
    ax.text(14.8, 10.95, "SHARED BILINEAR DECODER (Init: unet_raw decoder)", ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_GREEN)

    dec_specs = [
        ("Dec 4 (512 -> 256)", "Bilinear 2x + Concat(Fused s3)", 9.7),
        ("Dec 3 (256 -> 128)", "Bilinear 2x + Concat(Fused s2)", 8.4),
        ("Dec 2 (128 -> 64)", "Bilinear 2x + Concat(Fused s1)", 7.1),
        ("Dec 1 (64 -> 32)", "Bilinear 2x + Concat(Fused s0)", 5.85),
    ]
    for idx, (dt, ds, dy) in enumerate(dec_specs):
        box(ax, 12.45, dy, 4.7, 0.95, dt, ds, bg=BG_WHITE, bdr=BDR_GRN, ts=8.6, ss=7.5)
        if idx > 0:
            arrow(ax, 14.8, dec_specs[idx - 1][2], 14.8, dy + 0.95, col=C_GREEN, lw=1.4)

    # Fused Bottleneck + Multi-Scale Fused Skips -> Decoder
    arrow(ax, 11.28, 7.55, 12.45, 10.15, col=C_AMBER, lw=1.5, lbl="Fused BN (512)", lx=11.6, ly=9.05)
    # Show fused skip trunk from s0..s3 cleanly along y=6.72 corridor into Dec 1
    ax.plot([4.1, 9.0, 11.85, 12.45], [7.05, 6.72, 6.72, 6.35], color=C_AMBER, lw=1.1, ls=":", zorder=4)
    arrow(ax, 12.1, 6.5, 12.45, 6.35, col=C_AMBER, lw=1.2, ls=":")
    ax.text(8.1, 6.56, "Multi-scale fused skip connections (s0..s3 -> Dec 1..4)", ha="center", va="center", fontsize=6.8, color=C_AMBER, fontweight="bold")

    # Output Head & Loss
    box(ax, 12.55, 3.9, 4.5, 1.15, "Final Classification Head (Conv 1x1: 32 -> 1)", "Unified Vessel Logits  |  Frozen Calibrated tau = 0.40", bg=BG_TEAL, bdr=C_TEAL, lw=1.4, ts=8.8, ss=7.6)
    arrow(ax, 14.8, 5.85, 14.8, 5.05, col=C_GREEN, lw=1.6)

    box(ax, 12.55, 1.6, 4.5, 1.65, "HybridTopologyLoss (Training Only)", "0.4 BCE + 0.4 Soft-Dice + 0.2 FP32 Soft-clDice\nSoftSkeletonize(iter=12, border_ignore=2)", bg="#FDF2F8", bdr="#EC4899", ts=8.8, ss=7.6)
    arrow(ax, 14.8, 3.9, 14.8, 3.25, col="#DB2777", lw=1.4, ls="--")

    # Zoom Inset on Far Right: CrossStreamFusionBlock Detail (X = 18.0 .. 23.5)
    card(ax, 18.0, 1.6, 5.5, 9.7, bg=BG_WHITE, bdr=C_AMBER, lw=1.5)
    ax.text(20.75, 10.9, "CrossStreamFusionBlock(C)", ha="center", va="center", fontsize=10.5, fontweight="bold", color=C_AMBER)
    ax.text(20.75, 10.45, "Per-Pixel Spatial Gate + SE Residual", ha="center", va="center", fontsize=8.2, color=TEXT_MUTE)

    box(ax, 18.5, 9.2, 4.5, 0.9, "Inputs: F_raw, F_rest", "Concat -> [F_raw, F_rest] (B, 2C, H_s, W_s)", bg=BG_CYAN, bdr=BDR_CYAN, ts=8.5, ss=7.4)

    # Left inside inset: Spatial Gate
    box(ax, 18.3, 6.7, 2.35, 1.8, "Spatial Gate", "Conv 3x3 (2C -> mid)\nBN + GELU -> Conv 1x1\nSoftmax(dim=1) -> 2 ch\nw_raw + w_rest = 1", bg=BG_AMBER, bdr=BDR_AMB, ts=8.2, ss=7.0)
    # Right inside inset: Cross Residual
    box(ax, 20.9, 6.7, 2.35, 1.8, "Cross Residual", "Conv 1x1 (2C -> C)\nBN + GELU\n-> SEBlock(C)\n(GAP + Sigmoid)", bg=BG_GREEN, bdr=BDR_GRN, ts=8.2, ss=7.0)

    arrow(ax, 20.0, 9.2, 19.47, 8.5, col=C_AMBER)
    arrow(ax, 21.5, 9.2, 22.07, 8.5, col=C_GREEN)

    box(ax, 18.5, 4.8, 4.5, 1.15, "Weighted Sum + Cross Residual", "F_fused = w_raw * F_raw + w_rest * F_rest\n          + SE(cross_proj(cat))", bg=BG_TEAL, bdr=C_TEAL, ts=8.5, ss=7.4)
    arrow(ax, 19.47, 6.7, 19.8, 5.95, col=C_AMBER)
    arrow(ax, 22.07, 6.7, 21.7, 5.95, col=C_GREEN)

    box(ax, 18.5, 2.2, 4.5, 2.0, "2-Stage Fine-Tuning & Provenance",
        "• Stage 1 (Ep 1-4): Frozen Encoders (lr=5e-4)\n"
        "• Stage 2 (Ep 5-12): Joint (enc 1e-4, head 3e-4)\n"
        "• Peak VRAM: 2,948.8 MB (RTX 5060 BF16)\n"
        "• Best Ckpt Ep 10 (SHA: 9be2213e...)",
        bg="#F8FAFC", bdr=BDR_SLATE, ts=8.3, ss=7.2)

    # Bottom Summary Ribbon
    card(ax, 3.2, 0.4, 20.3, 0.9, bg=BG_TEAL, bdr=C_TEAL, lw=1.2)
    ax.text(13.35, 0.85,
            "Untouched FIVES Test (N=200): Dual-Stream Dice = 0.8474 (+3.41 pp vs Raw) | ThinSkel = 43.52% (+10.20 pp vs Raw) | clDice = 0.8537 | Glaucoma Prec = 89.98% (+32.39 pp vs Restored)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color=C_TEAL)

    out = os.path.join(OUT_DIR, "diag_4_dual_stream_hybrid_unet.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 5: COMBINED END-TO-END RETINASCAN STUDIO SYSTEM ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def make_diag_5():
    W, H = 24, 13
    fig = plt.figure(figsize=(W, H), facecolor=BG_PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    header_banner(ax, W, H,
                  "5. Combined End-to-End RetinaScan Studio System Architecture",
                  "Research Prototype Integration: Ingestion -> SFHformer Restoration -> Dual-Stream & Baseline Models -> Morphology -> Flask (:5001) -> React UI (:3000)")

    # Column 1: Data Ingestion (X = 0.5 .. 4.3)
    card(ax, 0.5, 3.8, 3.8, 7.6, bg=BG_WHITE, bdr=BDR_CYAN, lw=1.3)
    ax.text(2.4, 11.0, "1. DATA & PREPROCESSING", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_CYAN)
    box(ax, 0.8, 8.8, 3.2, 1.7, "Input Fundus Image", "FIVES (2048x2048, N=200)\nHRF (3504x2336, N=45)\nInteractive Upload / 15 Samples", bg=BG_CYAN, bdr=BDR_CYAN, ts=8.8, ss=7.5)
    box(ax, 0.8, 6.4, 3.2, 1.8, "Standardization & FOV", "RGB Float32 [0, 1]\nBenchmark: 512 Hamming Tiles\nAPI Studio: 512/256 Bilinear", bg="#F1F5F9", bdr=BDR_SLATE, ts=8.8, ss=7.5)
    box(ax, 0.8, 4.2, 3.2, 1.6, "Tissue FOV Mask", "Intensity > 10/255\nExcludes camera aperture", bg="#F1F5F9", bdr=BDR_SLATE, ts=8.5, ss=7.5)
    arrow(ax, 2.4, 8.8, 2.4, 8.2, col=C_CYAN)
    arrow(ax, 2.4, 6.4, 2.4, 5.8, col=C_CYAN)

    # Column 2: Restoration (X = 4.8 .. 8.6)
    card(ax, 4.8, 3.8, 3.8, 7.6, bg=BG_WHITE, bdr=C_INDIGO, lw=1.3)
    ax.text(6.7, 11.0, "2. NEURAL RESTORATION", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_INDIGO)
    box(ax, 5.1, 7.6, 3.2, 2.6, "SFHformer-m Restorer", "5 Stages, 24 SFH Blocks\nLocal DW-Conv + 2D FFT\n31.92 dB FIVES SOTA\nSHA: d74e267c...", bg=BG_INDIGO, bdr=BDR_IND, ts=9.0, ss=7.6)
    box(ax, 5.1, 4.5, 3.2, 2.1, "Restored Tensor x_rest", "Denoised & Destriped\nPreserves low-contrast\ncapillary valleys", bg=BG_PURPLE, bdr=BDR_PURP, ts=8.8, ss=7.6)
    arrow(ax, 4.0, 7.3, 5.1, 8.7, col=C_INDIGO, lbl="x_raw", lx=4.55, ly=8.2)
    arrow(ax, 6.7, 7.6, 6.7, 6.6, col=C_PURPLE)

    # Column 3: Segmentation Core (X = 9.1 .. 13.7)
    card(ax, 9.1, 3.8, 4.6, 7.6, bg=BG_WHITE, bdr=C_TEAL, lw=1.4)
    ax.text(11.4, 11.0, "3. SEGMENTATION MODELS", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_TEAL)
    box(ax, 9.4, 8.75, 4.0, 1.75, "Dual-Stream Hybrid U-Net [Default]", "Inputs: (x_raw, x_rest) -> 5-Scale Gate\nCalibrated Frozen Threshold tau = 0.40\nSHA: 9be2213e...", bg=BG_TEAL, bdr=C_TEAL, lw=1.4, ts=8.8, ss=7.4)
    box(ax, 9.4, 6.75, 4.0, 1.55, "Raw U-Net [Macro Baseline]", "Input: x_raw  |  tau = 0.50\nSHA: 97639a65...", bg=BG_BLUE, bdr=BDR_BLUE, ts=8.6, ss=7.4)
    box(ax, 9.4, 4.95, 4.0, 1.4, "Restored U-Net [Capillary Baseline]", "Input: x_rest  |  tau = 0.50\nSHA: 2cfb3737...", bg=BG_PURPLE, bdr=BDR_PURP, ts=8.6, ss=7.4)
    box(ax, 9.4, 4.0, 4.0, 0.72, "Frangi Hessian Control (tau=0.0010)", "CLAHE Green -> Multiscale sigmas [1..5]", bg="#F1F5F9", bdr=BDR_SLATE, ts=7.8, ss=6.8)

    # Clean bypass & feedforward arrows into Models
    ax.plot([4.0, 4.55, 4.55, 8.85, 9.4], [9.65, 9.65, 11.6, 11.6, 9.9], color=C_CYAN, lw=1.5, zorder=5)
    arrow(ax, 9.0, 10.8, 9.4, 9.9, col=C_CYAN, lw=1.5)
    ax.text(6.7, 11.74, "x_raw direct stream (bypassing restorer)", ha="center", va="center", fontsize=7.5, fontweight="bold",
            color=C_CYAN, bbox=dict(boxstyle="round,pad=0.18", fc="#FFFFFF", ec=BDR_CYAN, lw=0.7), zorder=6)
    ax.plot([4.0, 4.55, 4.55, 8.85, 9.4], [7.3, 7.3, 3.55, 3.55, 7.3], color=C_CYAN, lw=1.3, ls="--", zorder=4)
    arrow(ax, 9.0, 6.2, 9.4, 7.3, col=C_CYAN, lw=1.3, ls="--")
    arrow(ax, 8.3, 5.55, 9.4, 9.2, col=C_PURPLE)
    arrow(ax, 8.3, 5.55, 9.4, 5.65, col=C_PURPLE)

    # Column 4: Backend Engine (X = 14.2 .. 18.8)
    card(ax, 14.2, 3.8, 4.6, 7.6, bg=BG_WHITE, bdr=C_AMBER, lw=1.3)
    ax.text(16.5, 11.0, "4. FLASK BACKEND (:5001)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_AMBER)
    box(ax, 14.5, 8.85, 4.0, 1.65, "GPU_LOCK & Validation", "Strict [0,1] tau & model guards\nThread-safe BF16 inference\n0.0 MB VRAM leak (20-run QA)", bg=BG_AMBER, bdr=BDR_AMB, ts=8.6, ss=7.3)
    box(ax, 14.5, 6.75, 4.0, 1.65, "Topology & Differential Engine", "Additional Vessel Predictions (Amber)\nConnected Components, Endpoints\nCaliber Stratification (r <= 3.5px)", bg="#FFF7ED", bdr=BDR_AMB, ts=8.5, ss=7.3)
    box(ax, 14.5, 4.2, 4.0, 2.05, "6-Stage Reconciled Telemetry", "I/O + SFH + UNet + Topo + Blend + Enc\nMedian: 1,300.8 ms | P95: 1,724.9 ms\nStage discrepancy <= 0.1 ms\nOptional scale_0 Gate Overlay", bg=BG_GREEN, bdr=BDR_GRN, ts=8.5, ss=7.2)

    arrow(ax, 13.4, 9.6, 14.5, 9.6, col=C_TEAL)
    arrow(ax, 13.4, 7.5, 14.5, 7.5, col=C_BLUE)
    arrow(ax, 13.4, 5.65, 14.5, 5.4, col=C_PURPLE)

    # Column 5: Frontend UI (X = 19.3 .. 23.5)
    card(ax, 19.3, 3.8, 4.2, 7.6, bg=BG_WHITE, bdr=C_GREEN, lw=1.4)
    ax.text(21.4, 11.0, "5. RETINASCAN UI (:3000)", ha="center", va="center", fontsize=9.5, fontweight="bold", color=C_GREEN)
    box(ax, 19.6, 8.5, 3.6, 2.0, "Tab 1: Vessel Inspection", "Model Selector (Dual-Stream Def)\nFrozen 0.40 vs Alt 0.50 Presets\nOptional w_rest Heatmap [0->1]\nExploratory Slider Isolation", bg=BG_TEAL, bdr=C_TEAL, ts=8.5, ss=7.2)
    box(ax, 19.6, 6.1, 3.6, 1.9, "Tab 2: Milestone 3 Matrix", "FIVES (Primary) vs HRF (Mixed)\n40 Exposed / 5 Exploratory Split\nHolm-Bonferroni P-Values\nDisease Breakdown (N/A/D/G)", bg=BG_BLUE, bdr=BDR_BLUE, ts=8.5, ss=7.2)
    box(ax, 19.6, 4.1, 3.6, 1.5, "Tab 3 & Prototype Guard", "Noise & Frangi Ablation Studio\nNotice: Research Prototype Only\n(Not for Clinical Diagnosis)", bg=BG_AMBER, bdr=BDR_AMB, ts=8.3, ss=7.2)

    arrow(ax, 18.5, 9.3, 19.6, 9.3, col=C_GREEN, lw=1.6)
    arrow(ax, 18.5, 7.1, 19.6, 7.1, col=C_GREEN, lw=1.6)

    # Bottom Benchmark Summary Banner
    card(ax, 0.5, 0.5, 23.0, 2.8, bg=BG_WHITE, bdr=BDR_SLATE, lw=1.3)
    ax.text(12.0, 2.95, "VERIFIED MILESTONE 3 EMPIRICAL EVIDENCE & HONEST DOMAIN TRANSFER SUMMARY",
            ha="center", va="center", fontsize=10, fontweight="bold", color=NAVY_DARK)

    box(ax, 0.8, 0.75, 7.2, 1.85, "FIVES Untouched Test (N=200, Primary In-Domain)",
        "• Dual-Stream (tau=0.40): Dice 0.8474 | ThinSkel 43.52% | clDice 0.8537 | CC 41.22\n"
        "• Beats Raw U-Net (+3.41 pp Dice, +10.20 pp ThinSkel, -10.97 CCs, Holm p < 10^-18)\n"
        "• Beats Restored U-Net (+15.46 pp Dice, +18.66 pp Prec) & fixes Glaucoma Prec (89.98%)",
        bg=BG_TEAL, bdr=C_TEAL, ts=8.8, ss=7.6)

    box(ax, 8.4, 0.75, 7.2, 1.85, "HRF External Transfer (N=45: 40 Exposed / 5 Unexposed)",
        "• Mixed external generalization: Dual-Stream improves ThinSkel (+9.78 pp), clDice (+2.78 pp),\n"
        "  and CC (-12.49) vs Raw U-Net, while trading -8.54 pp Precision & -1.82 pp Dice vs Raw\n"
        "• N=5 Unexposed (Exploratory, Holm p=0.375): Restored 58.97% | Dual-Stream 56.30% ThinSkel",
        bg=BG_BLUE, bdr=BDR_BLUE, ts=8.8, ss=7.6)

    box(ax, 16.0, 0.75, 7.2, 1.85, "QA & Reproducibility Snapshot (v3.1-phase5.2-verified)",
        "• Backend QA: 9/9 checks pass (Median 1,300.8 ms, P95 1,724.9 ms, 0.0 MB VRAM leak)\n"
        "• Frontend Live Edge CDP QA: 11/11 checks pass (0 JS/React errors, all overlays verified)\n"
        "• Frozen SHA-256 checkpoints & Phase 1-4 JSON benchmarks strictly preserved untouched",
        bg=BG_AMBER, bdr=BDR_AMB, ts=8.8, ss=7.6)

    out = os.path.join(OUT_DIR, "diag_5_combined_retinascan_system.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    make_diag_1()
    make_diag_2()
    make_diag_3()
    make_diag_4()
    make_diag_5()
    print("All 5 Phase 5.3 architectural diagrams generated cleanly!")

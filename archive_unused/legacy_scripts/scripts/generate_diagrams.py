"""
Generate publication-quality architecture diagrams for SFHformer-Retina project.
Outputs: diagrams.pdf with 3 pages (3 diagrams).
All facts verified from source code.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette ──
C_INPUT   = '#E3F2FD'  # light blue
C_ENCODER = '#BBDEFB'  # blue
C_BOTTLE  = '#90CAF9'  # darker blue
C_DECODER = '#C8E6C9'  # green
C_OUTPUT  = '#E8F5E9'  # light green
C_SKIP    = '#FFF9C4'  # yellow
C_SPATIAL = '#FFCCBC'  # orange/coral
C_FREQ    = '#D1C4E9'  # purple
C_FUSION  = '#F0F4C3'  # lime
C_FFT     = '#CE93D8'  # purple accent
C_DATA    = '#B2DFDB'  # teal
C_TRAIN   = '#FFE0B2'  # amber
C_INFER   = '#B3E5FC'  # light blue
C_DIAG    = '#F8BBD0'  # pink
C_UI      = '#DCEDC8'  # light green
C_ARROW   = '#455A64'
C_TEXT    = '#212121'

def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=False, border_color='#546E7A', lw=1.2, text_color='#212121'):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor=border_color, linewidth=lw)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=text_color, wrap=True,
            fontfamily='sans-serif')
    return (x + w/2, y, y + h)  # center_x, bottom, top

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, style='->', lw=1.2, connectionstyle="arc3,rad=0"):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle))


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 1: Overall SFHformer U-Net Architecture
# ═══════════════════════════════════════════════════════════════
def diagram1_overall_architecture(pdf):
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_aspect('equal')

    fig.suptitle('Diagram 1: SFHformer-M Overall U-Net Architecture',
                 fontsize=14, fontweight='bold', y=0.97, fontfamily='sans-serif')

    # ── Title subtitle ──
    ax.text(7, 8.5, 'Encoder-Decoder with Skip Connections  |  embed_dim = [32, 64, 128, 64, 32]  |  depths = [4, 4, 8, 4, 4]',
            ha='center', va='center', fontsize=8, style='italic', color='#616161')

    # ── Input ──
    bw, bh = 2.2, 0.55
    ix, iy = 0.3, 4.0
    draw_box(ax, ix, iy, bw, bh, 'Degraded Retinal Input\n(B × 3 × 256 × 256)', C_INPUT, 7.5, bold=True)

    # ── PatchEmbed ──
    px, py = 3.0, 4.0
    draw_box(ax, px, py, 1.8, bh, 'PatchEmbed\nConv2d(3→32, k=3)', C_ENCODER, 7)
    draw_arrow(ax, ix+bw, iy+bh/2, px, py+bh/2)

    # ── Encoder stages ──
    # Stage 1
    e1x, e1y = 5.3, 6.5
    draw_box(ax, e1x, e1y, 2.0, 0.6, 'Encoder Stage 1\n4× SFH Block (C=32)', C_ENCODER, 7, bold=True)
    draw_arrow(ax, px+1.8, py+bh/2, e1x, e1y+0.3, connectionstyle="arc3,rad=-0.2")

    # Downsample 1
    d1x, d1y = 5.6, 5.6
    draw_box(ax, d1x, d1y, 1.4, 0.45, 'DownSample\nConv2d(32→64, k=2, s=2)', C_SKIP, 6)
    draw_arrow(ax, e1x+1.0, e1y, d1x+0.7, d1y+0.45)

    # Stage 2
    e2x, e2y = 5.3, 4.6
    draw_box(ax, e2x, e2y, 2.0, 0.6, 'Encoder Stage 2\n4× SFH Block (C=64)', C_ENCODER, 7, bold=True)
    draw_arrow(ax, d1x+0.7, d1y, e2x+1.0, e2y+0.6)

    # Downsample 2
    d2x, d2y = 5.6, 3.7
    draw_box(ax, d2x, d2y, 1.4, 0.45, 'DownSample\nConv2d(64→128, k=2, s=2)', C_SKIP, 6)
    draw_arrow(ax, e2x+1.0, e2y, d2x+0.7, d2y+0.45)

    # Bottleneck
    bnx, bny = 5.0, 2.6
    draw_box(ax, bnx, bny, 2.6, 0.7, 'Bottleneck\n8× SFH Block (C=128)\n(Deepest Feature Extraction)', C_BOTTLE, 7.5, bold=True, border_color='#1565C0', lw=2)
    draw_arrow(ax, d2x+0.7, d2y, bnx+1.3, bny+0.7)

    # ── Decoder ──
    # Upsample 1
    u1x, u1y = 8.4, 3.7
    draw_box(ax, u1x, u1y, 1.5, 0.45, 'PixelShuffle ↑2\nConv2d(128→64)', C_SKIP, 6)
    draw_arrow(ax, bnx+2.6, bny+0.35, u1x, u1y+0.22, connectionstyle="arc3,rad=-0.15")

    # Decoder Stage 2
    de2x, de2y = 8.2, 4.6
    draw_box(ax, de2x, de2y, 2.0, 0.6, 'Decoder Stage 2\n4× SFH Block (C=64)', C_DECODER, 7, bold=True)
    draw_arrow(ax, u1x+0.75, u1y+0.45, de2x+1.0, de2y)

    # Skip connection 2 (encoder stage 2 → decoder stage 2)
    draw_arrow(ax, e2x+2.0, e2y+0.3, de2x, de2y+0.3, color='#EF6C00', style='->', lw=1.5, connectionstyle="arc3,rad=0")
    ax.text(7.65, 4.95, 'skip', fontsize=6, color='#EF6C00', style='italic')

    # Upsample 2
    u2x, u2y = 8.4, 5.6
    draw_box(ax, u2x, u2y, 1.5, 0.45, 'PixelShuffle ↑2\nConv2d(64→32)', C_SKIP, 6)
    draw_arrow(ax, de2x+1.0, de2y+0.6, u2x+0.75, u2y)

    # Decoder Stage 1
    de1x, de1y = 8.2, 6.5
    draw_box(ax, de1x, de1y, 2.0, 0.6, 'Decoder Stage 1\n4× SFH Block (C=32)', C_DECODER, 7, bold=True)
    draw_arrow(ax, u2x+0.75, u2y+0.45, de1x+1.0, de1y)

    # Skip connection 1 (encoder stage 1 → decoder stage 1)
    draw_arrow(ax, e1x+2.0, e1y+0.3, de1x, de1y+0.3, color='#EF6C00', style='->', lw=1.5, connectionstyle="arc3,rad=0")
    ax.text(7.65, 6.85, 'skip', fontsize=6, color='#EF6C00', style='italic')

    # ── Output ──
    pux, puy = 10.7, 6.5
    draw_box(ax, pux, puy, 1.6, 0.6, 'PatchUnEmbed\nConv2d(32→3, k=3)', C_OUTPUT, 7)
    draw_arrow(ax, de1x+2.0, de1y+0.3, pux, puy+0.3)

    # Residual add
    resx, resy = 12.6, 5.0
    draw_box(ax, resx, resy, 1.1, 0.5, 'Residual\nAdd (+)', C_OUTPUT, 7, bold=True, border_color='#2E7D32', lw=1.5)
    draw_arrow(ax, pux+0.8, puy, resx+0.55, resy+0.5, connectionstyle="arc3,rad=0.15")
    # Long residual from input
    draw_arrow(ax, ix+1.1, iy, resx, resy+0.25, color='#2E7D32', style='->', lw=1.5, connectionstyle="arc3,rad=-0.4")
    ax.text(5.5, 1.6, 'Global Residual Skip (Input)', fontsize=6.5, color='#2E7D32', style='italic')

    # Final output
    ox, oy = 12.0, 3.8
    draw_box(ax, ox, oy, 1.8, 0.55, 'Enhanced Retinal Output\n(B × 3 × 256 × 256)', C_OUTPUT, 7.5, bold=True, border_color='#2E7D32', lw=1.5)
    draw_arrow(ax, resx+0.55, resy, ox+0.9, oy+0.55)

    # ── Legend ──
    ax.text(0.5, 1.5, 'Model: SFHformer-M (ECCV 2024)', fontsize=7, fontweight='bold', color='#37474F')
    ax.text(0.5, 1.1, 'Total Stages: 5 (2 Encoder + 1 Bottleneck + 2 Decoder)', fontsize=6.5, color='#546E7A')
    ax.text(0.5, 0.8, 'Total SFH Blocks: 4+4+8+4+4 = 24 blocks', fontsize=6.5, color='#546E7A')
    ax.text(0.5, 0.5, 'Patch Size: 1 (no spatial downsampling in embedding)', fontsize=6.5, color='#546E7A')
    ax.text(0.5, 0.2, 'Each SFH Block = BatchNorm → Mixer (Local+Global) → FFN (Multi-Scale)', fontsize=6.5, color='#546E7A')

    legend_items = [
        mpatches.Patch(facecolor=C_ENCODER, edgecolor='#546E7A', label='Encoder'),
        mpatches.Patch(facecolor=C_BOTTLE, edgecolor='#1565C0', label='Bottleneck'),
        mpatches.Patch(facecolor=C_DECODER, edgecolor='#546E7A', label='Decoder'),
        mpatches.Patch(facecolor=C_SKIP, edgecolor='#546E7A', label='Resample / Skip'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=7, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 2: SFH Block Internal — Dual-Branch Mixer + FFN
# ═══════════════════════════════════════════════════════════════
def diagram2_sfh_block(pdf):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    fig.suptitle('Diagram 2: SFH Block Internal Architecture — Dual-Branch Mixer & Multi-Scale FFN',
                 fontsize=13, fontweight='bold', y=0.97, fontfamily='sans-serif')

    # ── Top: Block-level flow ──
    ax.text(7, 9.4, 'Each SFH Block:  Input → BatchNorm → Mixer → (β·out + Input) → BatchNorm → FFN → (γ·out + Skip)',
            ha='center', fontsize=7.5, style='italic', color='#616161')
    ax.text(7, 9.1, 'β and γ are learnable parameters initialized to zeros',
            ha='center', fontsize=6.5, color='#9E9E9E')

    # ═══ MIXER SECTION (top half) ═══
    ax.text(0.4, 8.6, 'MIXER (Dual-Branch: Local + Global)', fontsize=10, fontweight='bold', color='#37474F')

    # Input projection
    mpx, mpy = 0.5, 7.5
    draw_box(ax, mpx, mpy, 2.2, 0.55, 'Input Projection\nConv2d(C → 2C, k=1)', C_INPUT, 7)

    # Split arrow
    ax.text(3.1, 7.75, 'Split\n(C each)', fontsize=6.5, ha='center', color='#455A64')
    draw_arrow(ax, mpx+2.2, mpy+0.4, 3.6, 7.3+0.4, connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, mpx+2.2, mpy+0.15, 3.6, 5.5+0.3, connectionstyle="arc3,rad=0.15")

    # ── LOCAL BRANCH ──
    lbx, lby = 3.6, 7.0
    draw_box(ax, lbx, lby, 3.5, 0.75, '', C_SPATIAL, border_color='#E64A19', lw=1.5)
    ax.text(lbx+1.75, lby+0.6, 'LOCAL BRANCH (TokenMixer_For_Local)', fontsize=7, fontweight='bold',
            ha='center', color='#BF360C')

    # Two parallel dilated convolutions
    dc1x, dc1y = 3.8, 6.0
    draw_box(ax, dc1x, dc1y, 1.4, 0.55, 'DW-Conv2d\nk=3, dilation=1\ngroups=C', C_SPATIAL, 6)
    dc2x, dc2y = 5.5, 6.0
    draw_box(ax, dc2x, dc2y, 1.4, 0.55, 'DW-Conv2d\nk=3, dilation=2\ngroups=C', C_SPATIAL, 6)

    draw_arrow(ax, lbx+0.8, lby, dc1x+0.7, dc1y+0.55)
    draw_arrow(ax, lbx+2.7, lby, dc2x+0.7, dc2y+0.55)

    ax.text(5.35, 6.65, 'Parallel', fontsize=6, ha='center', color='#E64A19', style='italic')

    # Merge local
    lmx, lmy = 4.5, 5.2
    draw_box(ax, lmx, lmy, 1.5, 0.4, 'Element-wise Add', C_SPATIAL, 6.5)
    draw_arrow(ax, dc1x+0.7, dc1y, lmx+0.3, lmy+0.4)
    draw_arrow(ax, dc2x+0.7, dc2y, lmx+1.2, lmy+0.4)

    # ── GLOBAL BRANCH ──
    gbx, gby = 3.6, 4.1
    draw_box(ax, gbx, gby, 3.5, 0.65, '', C_FREQ, border_color='#6A1B9A', lw=1.5)
    ax.text(gbx+1.75, gby+0.5, 'GLOBAL BRANCH (TokenMixer_For_Gloal)', fontsize=7, fontweight='bold',
            ha='center', color='#4A148C')

    # Fourier Unit detail
    fu1x, fu1y = 3.7, 2.8
    draw_box(ax, fu1x, fu1y, 1.3, 0.55, 'Conv2d\n(C→2C, k=1)\n+ GELU', C_FREQ, 5.5)

    fu2x, fu2y = 5.3, 2.8
    draw_box(ax, fu2x, fu2y, 2.3, 0.75, 'FourierUnit\nrfft2 → BN → fpe\n→ FreqDynConv\n→ irfft2', C_FFT, 6, bold=True, border_color='#6A1B9A', lw=1.5)

    fu3x, fu3y = 7.9, 2.8
    draw_box(ax, fu3x, fu3y, 1.3, 0.55, 'Conv2d\n(2C→C, k=1)\n+ GELU', C_FREQ, 5.5)

    draw_arrow(ax, gbx+1.75, gby, fu1x+0.65, fu1y+0.75+0.2, connectionstyle="arc3,rad=0")
    draw_arrow(ax, fu1x+1.3, fu1y+0.35, fu2x, fu2y+0.35)
    draw_arrow(ax, fu2x+2.3, fu2y+0.35, fu3x, fu3y+0.35)

    # Residual on global branch
    ax.annotate('', xy=(fu3x+1.3, fu3y+0.35), xytext=(gbx+3.5, gby+0.3),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.0, connectionstyle="arc3,rad=0.3"))
    ax.text(8.3, 4.5, '+ residual', fontsize=5.5, color='#6A1B9A', style='italic')

    # ═══ FUSION ═══
    fux, fuy = 8.0, 5.8
    draw_box(ax, fux, fuy, 2.8, 0.9, 'Fusion\nConcat(Local, Global)\n→ GELU → SE Attention\n→ Conv2d(2C→C, k=1)', C_FUSION, 6.5, bold=True, border_color='#827717', lw=1.5)

    draw_arrow(ax, lmx+1.5, lmy+0.2, fux, fuy+0.6, connectionstyle="arc3,rad=-0.1")
    draw_arrow(ax, fu3x+1.3, fu3y+0.35, fux+1.4, fuy, connectionstyle="arc3,rad=-0.25")

    # SE detail
    ax.text(11.1, 6.65, 'SE Channel Attention:', fontsize=6.5, fontweight='bold', color='#546E7A')
    ax.text(11.1, 6.35, 'AvgPool → Conv(2C→C) → ReLU', fontsize=6, color='#78909C')
    ax.text(11.1, 6.1, '→ Conv(C→2C) → Sigmoid', fontsize=6, color='#78909C')

    # Mixer output
    mox, moy = 11.3, 5.8
    draw_box(ax, mox, moy, 1.5, 0.5, 'Mixer Output\n(B × C × H × W)', C_INPUT, 6.5)
    draw_arrow(ax, fux+2.8, fuy+0.45, mox, moy+0.25)

    # ═══ FFN SECTION (bottom) ═══
    ax.text(0.4, 1.9, 'FFN (Multi-Scale Feed-Forward Network)', fontsize=10, fontweight='bold', color='#37474F')

    ffn_ix, ffn_iy = 0.5, 0.7
    draw_box(ax, ffn_ix, ffn_iy, 1.6, 0.55, 'Expand\nConv2d(C→2C, k=1)', C_INPUT, 6.5)

    ax.text(2.5, 0.95, 'Split into\n4 parts', fontsize=6, ha='center', color='#455A64')

    # 4 parallel paths
    colors4 = ['#E8F5E9', '#C8E6C9', '#A5D6A7', '#81C784']
    labels4 = ['Identity\n(pass-through)', 'DW-Conv2d\nk=3', 'DW-Conv2d\nk=5', 'DW-Conv2d\nk=7']
    for i, (c, lbl) in enumerate(zip(colors4, labels4)):
        bx = 3.2 + i * 1.9
        draw_box(ax, bx, 0.65, 1.5, 0.55, lbl, c, 6)
        draw_arrow(ax, ffn_ix+1.6, ffn_iy+0.28, bx, 0.92)

    # Concat + output
    ffn_ox, ffn_oy = 10.8, 0.7
    draw_box(ax, ffn_ox, ffn_oy, 2.4, 0.55, 'Concat → GELU\n→ Conv2d(2C→C, k=1)', C_OUTPUT, 6.5)
    for i in range(4):
        bx = 3.2 + i * 1.9
        draw_arrow(ax, bx+1.5, 0.92, ffn_ox, ffn_oy+0.28)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# DIAGRAM 3: Full System Pipeline (Data → Training → Inference → Diagnostics → UI)
# ═══════════════════════════════════════════════════════════════
def diagram3_system_pipeline(pdf):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')

    fig.suptitle('Diagram 3: End-to-End System Pipeline — Data, Training, Inference & Diagnostics',
                 fontsize=13, fontweight='bold', y=0.97, fontfamily='sans-serif')

    # ═══ ROW 1: DATA INGESTION ═══
    ax.text(0.3, 9.3, '① DATA INGESTION & AUGMENTATION', fontsize=9, fontweight='bold', color='#004D40')

    datasets = [
        ('DRIVE\n20 imgs\n565×584', 0.3),
        ('STARE\n20 imgs\n700×605', 2.5),
        ('HRF\n45 imgs\n3504×2336', 4.7),
        ('APTOS 2019\nHospital\n1024×1024', 6.9),
    ]
    for label, x in datasets:
        draw_box(ax, x, 8.3, 1.8, 0.75, label, C_DATA, 6.5, bold=True, border_color='#00695C')

    # Arrow to crop pipeline
    for _, x in datasets:
        draw_arrow(ax, x+0.9, 8.3, 5.5, 7.65)

    # Crop pipeline
    draw_box(ax, 4.0, 7.1, 3.0, 0.55, 'Crop Pipeline (256×256)\nTissue Filter > 38%  |  8-way augment', C_DATA, 6.5, border_color='#00695C')

    # Output
    draw_box(ax, 8.0, 7.1, 2.4, 0.55, '~6,000+ Train Crops\n~138+ Val Crops', '#B2DFDB', 7, bold=True)
    draw_arrow(ax, 7.0, 7.37, 8.0, 7.37)

    # Degradation
    draw_box(ax, 8.0, 8.3, 4.5, 0.75, 'Synthetic Degradation Engine (retina_corrupt.py)\nRaster Banding | Laser Speckle | Optical Halo\nIllum. Gradient | Sensor Noise | Mixed Mode', '#FFECB3', 6, border_color='#F57F17')
    draw_arrow(ax, 10.25, 8.3, 10.25, 7.65)
    draw_arrow(ax, 9.2, 7.65, 9.2, 7.65)

    # ═══ ROW 2: TRAINING ═══
    ax.text(0.3, 6.5, '② TRAINING ON RTX 5060', fontsize=9, fontweight='bold', color='#E65100')

    # Baseline training
    draw_box(ax, 0.3, 5.3, 3.5, 0.9, 'Baseline Training (DRIVE only)\nSFHformer-M | 2,000 iters | BS=1 × GA=4\nAdamW (lr=5e-5, wd=1e-4) | L1 Loss\nCosineAnnealing → best.pth (25.94 dB)', C_TRAIN, 6, border_color='#E65100')

    # Mega fine-tune
    draw_box(ax, 4.5, 5.3, 3.8, 0.9, 'Mega Fine-Tuning (4 Datasets)\nSFHformer-M from best.pth | 4,000 iters\nAdamW (lr=4e-5) | L1 Loss | TF32 On\nPeak VRAM: 4.60 GB | ~29 min → last.pth', C_TRAIN, 6, bold=True, border_color='#E65100', lw=1.5)
    draw_arrow(ax, 3.8, 5.75, 4.5, 5.75)

    # Engineering notes
    draw_box(ax, 9.0, 5.3, 4.5, 0.9, 'Hardware Engineering\n• time.sleep(0.002) per iter → prevents TDR BSOD\n• Gradient Accumulation (eff. BS=4)\n• TF32 tensor cores enabled\n• Peak VRAM: 4.60 GB / 8.15 GB available', '#FFF3E0', 5.5, border_color='#E65100')

    # ═══ ROW 3: INFERENCE SERVER ═══
    ax.text(0.3, 4.6, '③ INFERENCE SERVER (Flask REST API — Port 5001)', fontsize=9, fontweight='bold', color='#1565C0')

    # Models
    draw_box(ax, 0.3, 3.3, 3.0, 0.9, 'Hot-Loaded GPU Models\n• retina (best.pth)\n• r1_baseline (DIV2K weights)\n• r2_gated (RadialBandGate)\n+ notch, gaussian, median', C_INFER, 6, border_color='#1565C0')

    # Endpoints
    draw_box(ax, 3.8, 3.3, 3.5, 0.9, 'REST Endpoints\nGET /health\nPOST /api/retina/restore\nPOST /api/restore  |  /api/restore_all\nGET /api/retina/samples (8 cases)', C_INFER, 6, border_color='#1565C0')

    draw_arrow(ax, 3.3, 3.75, 3.8, 3.75)

    # Noise detection
    draw_box(ax, 7.8, 3.3, 3.0, 0.9, 'Noise Detection Pipeline\n• FFT 2D Peak Detection\n• Gaussian: Laplacian MAD\n• Impulse: Median Filter Diff\n• Adaptive Hybrid: Notch+NN', '#E3F2FD', 6, border_color='#1565C0')
    draw_arrow(ax, 7.3, 3.75, 7.8, 3.75)

    # Adaptive
    draw_box(ax, 11.2, 3.3, 2.3, 0.9, 'Adaptive Hybrid\nStage 1: Blind\nFFT Notch Filter\nStage 2: SFHformer\nGPU Inference', '#BBDEFB', 6, bold=True, border_color='#0D47A1', lw=1.5)
    draw_arrow(ax, 10.8, 3.75, 11.2, 3.75)

    # ═══ ROW 4: DIAGNOSTIC ENGINE ═══
    ax.text(0.3, 2.65, '④ VASCULAR DIAGNOSTIC ENGINE (retina_engine.py)', fontsize=9, fontweight='bold', color='#880E4F')

    # Fundus pipeline
    draw_box(ax, 0.3, 1.2, 4.2, 1.0, 'Fundus Vascular Pipeline\nGreen Channel → CLAHE (clip=0.03)\n→ Multi-Scale Frangi Hessian\n   σ = [1.0, 1.6, 2.2, 2.8, 3.4]\n→ Otsu Threshold → Morphology Cleanup', C_DIAG, 5.5, border_color='#880E4F')

    # OCT pipeline
    draw_box(ax, 4.8, 1.2, 3.5, 1.0, 'OCT B-Scan Pipeline\nGrayscale → Gaussian σ=(1.5,2.5)\n→ Column-wise ILM Detection\n→ RPE Detection → 1D Smooth\n→ Layer Mask & Boundary Overlay', C_DIAG, 5.5, border_color='#880E4F')

    # Biomarkers
    draw_box(ax, 8.8, 1.2, 2.3, 1.0, 'Computed Biomarkers\n• Vessel Density %\n• CNR Ratio\n• Branch Count\n• Vessel Dice Score', '#FCE4EC', 6, bold=True, border_color='#880E4F')
    draw_arrow(ax, 4.5, 1.7, 4.8, 1.7)
    draw_arrow(ax, 8.3, 1.7, 8.8, 1.7)

    # Ablation
    draw_box(ax, 11.5, 1.2, 2.2, 1.0, 'Ablation Matrix\nvs. Gaussian σ=1.5\nvs. Median 3×3\nvs. Raw Corrupted\n→ Clinical Verdict', '#F8BBD0', 6, border_color='#880E4F')
    draw_arrow(ax, 11.1, 1.7, 11.5, 1.7)

    # ═══ ROW 5: FRONTEND ═══
    ax.text(0.3, 0.55, '⑤ FRONTEND: React + Vite (RetinaStudio) — 8 benchmark cases, model selector, degradation controls, triple-panel viewer, FFT spectrum',
            fontsize=7.5, fontweight='bold', color='#33691E')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# GENERATE PDF
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    output_path = r'd:\SFHformer-FreqLab\SFHformer_Architecture_Diagrams.pdf'
    with PdfPages(output_path) as pdf:
        diagram1_overall_architecture(pdf)
        diagram2_sfh_block(pdf)
        diagram3_system_pipeline(pdf)
    print(f'✅ Diagrams saved to: {output_path}')


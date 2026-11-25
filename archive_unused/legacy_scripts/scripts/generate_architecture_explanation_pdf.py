"""
scripts/generate_architecture_explanation_pdf.py
Generates the publication-grade document:
D:\SFHformer-FreqLab\ARCHITECTURE_AND_SYSTEM_DESIGN_EXPLAINED.pdf
Includes 4 detailed diagrams:
  1. The complete SFHformer macro-architecture
  2. The detailed SFH Block (Mixer, Local Dilated, Global FourierUnit, Channel Attention, FFN)
  3. The Clinical U-Net architecture (DoubleConv, residual skips, bilinear upsampling, dual specialization)
  4. The Combined End-to-End System Pipeline (Input -> Restoration -> Dual Segmentation -> Differential Engine -> Topology -> Serving & UI)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image
)
from reportlab.pdfgen import canvas

# Directory setup
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "diagrams")
os.makedirs(DOCS_DIR, exist_ok=True)
PDF_OUTPUT_PATH = r"D:\SFHformer-FreqLab\ARCHITECTURE_AND_SYSTEM_DESIGN_EXPLAINED.pdf"

# ── Color Palette ──
C_PAGE_BG    = '#F8FAFC'
C_CARD_WHITE = '#FFFFFF'
C_BORDER     = '#CBD5E1'
C_NAVY_DARK  = '#0F172A'
C_BLUE_MAIN  = '#1E3A8A'
C_BLUE_BRIGHT= '#2563EB'
C_CYAN_LIGHT = '#E0F2FE'
C_GREEN_LIGHT= '#DCFCE7'
C_AMBER_LIGHT= '#FEF3C7'
C_PURPLE_LGT = '#F3E8FF'
C_ROSE_LIGHT = '#FFE4E6'
C_SLATE_TEXT = '#334155'
C_MUTED_TEXT = '#64748B'

def draw_card(ax, x, y, w, h, bg_color=C_CARD_WHITE, bdr_color=C_BORDER, lw=1.2, r=0.15):
    """Draw a rounded rectangle card."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         facecolor=bg_color, edgecolor=bdr_color,
                         linewidth=lw, zorder=1)
    ax.add_patch(box)

def draw_box(ax, x, y, w, h, title, subtitle="", bg=C_CARD_WHITE, bdr=C_BORDER,
             lw=1.0, r=0.12, tcolor=C_NAVY_DARK, scolor=C_MUTED_TEXT, tsize=9, ssize=7.5, zorder=2):
    """Draw an annotated rounded component box."""
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={r}",
                         facecolor=bg, edgecolor=bdr,
                         linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w/2.0, y + h*0.60, title, ha='center', va='center',
                fontsize=tsize, fontweight='bold', color=tcolor, zorder=zorder+1)
        ax.text(x + w/2.0, y + h*0.28, subtitle, ha='center', va='center',
                fontsize=ssize, color=scolor, zorder=zorder+1)
    else:
        ax.text(x + w/2.0, y + h/2.0, title, ha='center', va='center',
                fontsize=tsize, fontweight='bold', color=tcolor, zorder=zorder+1)

def draw_flow_arrow(ax, x1, y1, x2, y2, color=C_BLUE_BRIGHT, lw=1.5, label="", lsize=7):
    """Draw an arrow with an optional center label."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35",
                                color=color, lw=lw, shrinkA=0, shrinkB=0),
                zorder=5)
    if label:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(mx, my + 0.12, label, ha='center', va='bottom',
                fontsize=lsize, color='#1E293B', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.15", fc='#FFFFFF', ec='#CBD5E1', lw=0.6),
                zorder=6)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1: THE SFHFORMER MACRO-ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def generate_diagram_1():
    png_path = os.path.join(DOCS_DIR, "arch_diag1_sfhformer_macro.png")
    fig = plt.figure(figsize=(16, 9), facecolor=C_PAGE_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Top Header
    draw_card(ax, 0.4, 8.1, 15.2, 0.7, bg_color='#0F172A', bdr_color='#1E293B', lw=1.2, r=0.12)
    ax.text(8.0, 8.52, "SFHformer-m: 5-Stage Spatial-Frequency Hierarchical Transformer Architecture",
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#F8FAFC')
    ax.text(8.0, 8.25, "14.18M Parameters  •  Global Input Residual  •  PixelShuffle 2x Upsampling  •  Checkpoints: fives_retina_finetune (31.92 dB PSNR)",
            ha='center', va='center', fontsize=8.0, color='#94A3B8')

    # Input Box
    draw_box(ax, 0.5, 4.6, 1.8, 1.4, "Degraded Input", "(B, 3, H, W)\nNormalized [0, 1]",
             bg=C_CYAN_LIGHT, bdr='#38BDF8', tsize=8.5, ssize=7.0)

    # Patch Embed
    draw_box(ax, 2.7, 4.8, 1.4, 1.0, "PatchEmbed", "Conv 3x3, s=1\n3 -> 32 dims",
             bg='#DBEAFE', bdr='#60A5FA', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 2.3, 5.3, 2.7, 5.3)

    # Stage 1 Encoder
    draw_card(ax, 4.4, 4.1, 2.0, 2.4, bg_color='#EFF6FF', bdr_color='#3B82F6', lw=1.2)
    ax.text(5.4, 6.25, "STAGE 1 (Encoder)", ha='center', va='center', fontsize=8, fontweight='bold', color='#1D4ED8')
    draw_box(ax, 4.6, 5.2, 1.6, 0.8, "4x SFH Blocks", "dim=32, HxW", bg='#FFFFFF', bdr='#93C5FD', tsize=7.5, ssize=6.5)
    draw_box(ax, 4.6, 4.3, 1.6, 0.65, "Conv 1x1 Skip", "Proj to Decoder", bg='#FEF3C7', bdr='#F59E0B', tsize=7, ssize=6)
    draw_flow_arrow(ax, 4.1, 5.3, 4.4, 5.3)

    # Downsample 1
    draw_box(ax, 4.8, 3.1, 1.2, 0.65, "DownSample 1", "Conv 3x3, s=2", bg='#E0E7FF', bdr='#6366F1', tsize=7, ssize=6)
    draw_flow_arrow(ax, 5.4, 4.1, 5.4, 3.75)

    # Stage 2 Encoder
    draw_card(ax, 4.4, 0.5, 2.0, 2.3, bg_color='#EEF2FF', bdr_color='#4F46E5', lw=1.2)
    ax.text(5.4, 2.55, "STAGE 2 (Encoder)", ha='center', va='center', fontsize=8, fontweight='bold', color='#4338CA')
    draw_box(ax, 4.6, 1.5, 1.6, 0.8, "4x SFH Blocks", "dim=64, H/2 x W/2", bg='#FFFFFF', bdr='#A5B4FC', tsize=7.5, ssize=6.5)
    draw_box(ax, 4.6, 0.7, 1.6, 0.6, "Conv 1x1 Skip", "Proj to Decoder", bg='#FEF3C7', bdr='#F59E0B', tsize=7, ssize=6)
    draw_flow_arrow(ax, 5.4, 3.1, 5.4, 2.8)

    # Downsample 2
    draw_box(ax, 6.8, 1.2, 1.1, 0.7, "DownSample 2", "Conv 3x3, s=2", bg='#E0E7FF', bdr='#6366F1', tsize=7, ssize=6)
    draw_flow_arrow(ax, 6.4, 1.55, 6.8, 1.55)

    # Stage 3 Bottleneck
    draw_card(ax, 8.2, 0.5, 2.2, 2.4, bg_color='#FAF5FF', bdr_color='#9333EA', lw=1.4)
    ax.text(9.3, 2.65, "STAGE 3 (Bottleneck)", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#7E22CE')
    draw_box(ax, 8.4, 1.45, 1.8, 1.0, "8x SFH Blocks", "dim=128\nH/4 x W/4 Resolution", bg='#FFFFFF', bdr='#C084FC', tsize=8, ssize=6.5)
    ax.text(9.3, 0.95, "Deepest Global Receptive Field\nMax Frequency Band Resolution", ha='center', va='center', fontsize=6.2, color='#6B21A8')
    draw_flow_arrow(ax, 7.9, 1.55, 8.2, 1.55)

    # Upsample 3
    draw_box(ax, 10.7, 1.2, 1.2, 0.7, "Upsample 3", "PixelShuffle 2x\n128 -> 64 dims", bg='#DCFCE7', bdr='#22C55E', tsize=7, ssize=6)
    draw_flow_arrow(ax, 10.4, 1.55, 10.7, 1.55)

    # Stage 4 Decoder
    draw_card(ax, 12.2, 0.5, 2.2, 2.4, bg_color='#F0FDF4', bdr_color='#16A34A', lw=1.2)
    ax.text(13.3, 2.65, "STAGE 4 (Decoder)", ha='center', va='center', fontsize=8, fontweight='bold', color='#15803D')
    draw_box(ax, 12.4, 1.6, 1.8, 0.65, "Concat + Skip 2", "dim=(64+64)->64", bg='#FEF3C7', bdr='#F59E0B', tsize=7, ssize=6)
    draw_box(ax, 12.4, 0.7, 1.8, 0.75, "4x SFH Blocks", "dim=64, H/2 x W/2", bg='#FFFFFF', bdr='#86EFAC', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 11.9, 1.55, 12.2, 1.55)

    # Lateral Skip 2 Arrow (Stage 2 to Stage 4)
    ax.annotate("", xy=(12.4, 1.9), xytext=(6.2, 1.0),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#D97706', lw=1.3, linestyle='--', connectionstyle="arc3,rad=-0.15"),
                zorder=4)
    ax.text(9.3, 0.25, "Lateral Skip Connection 2 (1x1 Conv)", ha='center', va='center', fontsize=6.5, color='#B45309')

    # Upsample 4
    draw_box(ax, 12.7, 3.2, 1.2, 0.7, "Upsample 4", "PixelShuffle 2x\n64 -> 32 dims", bg='#DCFCE7', bdr='#22C55E', tsize=7, ssize=6)
    draw_flow_arrow(ax, 13.3, 2.9, 13.3, 3.2)

    # Stage 5 Decoder
    draw_card(ax, 12.2, 4.1, 2.2, 2.5, bg_color='#F0FDF4', bdr_color='#16A34A', lw=1.2)
    ax.text(13.3, 6.35, "STAGE 5 (Decoder)", ha='center', va='center', fontsize=8, fontweight='bold', color='#15803D')
    draw_box(ax, 12.4, 5.3, 1.8, 0.65, "Concat + Skip 1", "dim=(32+32)->32", bg='#FEF3C7', bdr='#F59E0B', tsize=7, ssize=6)
    draw_box(ax, 12.4, 4.35, 1.8, 0.75, "4x SFH Blocks", "dim=32, H x W", bg='#FFFFFF', bdr='#86EFAC', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 13.3, 3.9, 13.3, 4.1)

    # Lateral Skip 1 Arrow (Stage 1 to Stage 5)
    ax.annotate("", xy=(12.4, 5.6), xytext=(6.2, 4.6),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#D97706', lw=1.3, linestyle='--', connectionstyle="arc3,rad=-0.1"),
                zorder=4)
    ax.text(9.3, 5.7, "Lateral Skip Connection 1 (1x1 Conv)", ha='center', va='center', fontsize=6.5, color='#B45309')

    # Patch UnEmbed
    draw_box(ax, 12.5, 7.0, 1.6, 0.75, "PatchUnEmbed", "Conv 3x3, s=1\n32 -> 3 dims",
             bg='#DBEAFE', bdr='#60A5FA', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 13.3, 6.6, 13.3, 7.0)

    # Global Residual Addition
    draw_box(ax, 10.3, 7.0, 1.3, 0.75, "Global Residual\nSum (+)", "x_in + x_out",
             bg='#FEF08A', bdr='#CA8A04', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 12.5, 7.37, 11.6, 7.37)

    # Global Long Identity Wire from Input
    ax.annotate("", xy=(10.95, 7.75), xytext=(1.4, 6.0),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35",
                                color='#0284C7', lw=1.5, connectionstyle="arc3,rad=-0.12"),
                zorder=5)
    ax.text(6.0, 7.6, "Global Identity Skip Connection: Output = Input + SFHformer(Input)",
            ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0369A1',
            bbox=dict(boxstyle="round,pad=0.2", fc='#F0F9FF', ec='#BAE6FD'))

    # Final Restored Output
    draw_box(ax, 8.2, 6.9, 1.6, 0.95, "Restored Output", "(B, 3, H, W)\n31.92 dB PSNR",
             bg='#DCFCE7', bdr='#15803D', tsize=8.5, ssize=7.0)
    draw_flow_arrow(ax, 10.3, 7.37, 9.8, 7.37)

    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Generated Diagram 1: {png_path}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2: THE SFH BLOCK (INTERNAL ARCHITECTURE)
# ─────────────────────────────────────────────────────────────────────────────
def generate_diagram_2():
    png_path = os.path.join(DOCS_DIR, "arch_diag2_sfh_block.png")
    fig = plt.figure(figsize=(16, 9), facecolor=C_PAGE_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Top Header
    draw_card(ax, 0.4, 8.1, 15.2, 0.7, bg_color='#0F172A', bdr_color='#1E293B', lw=1.2, r=0.12)
    ax.text(8.0, 8.52, "Internal Architecture of the Spatial-Frequency Hybrid Block (SFH Block)",
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#F8FAFC')
    ax.text(8.0, 8.25, "Dual Parallel Mixer: Local Dilated Spatial DW-Conv + Global 2D FourierUnit  •  Squeeze-and-Excitation Channel Attention  •  Depthwise FFN",
            ha='center', va='center', fontsize=8.0, color='#94A3B8')

    # Outer Block Frame
    draw_card(ax, 0.4, 0.4, 15.2, 7.5, bg_color=C_CARD_WHITE, bdr_color='#94A3B8', lw=1.4)
    ax.text(1.2, 7.6, "Block Structure", fontsize=9, fontweight='bold', color='#334155')

    # Input Tensor
    draw_box(ax, 0.6, 4.4, 1.4, 1.1, "Block Input X", "(B, C, H, W)", bg=C_CYAN_LIGHT, bdr='#38BDF8', tsize=8, ssize=7)

    # LayerNorm 1
    draw_box(ax, 2.3, 4.6, 1.0, 0.7, "LayerNorm", "BatchNorm2d", bg='#F1F5F9', bdr='#CBD5E1', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 2.0, 4.95, 2.3, 4.95)

    # ── THE MIXER CONTAINER ──
    draw_card(ax, 3.6, 2.0, 7.8, 5.6, bg_color='#F8FAFC', bdr_color='#6366F1', lw=1.4)
    ax.text(7.5, 7.35, "HYBRID TOKEN MIXER MODULE (Local + Global Parallel Fusion)",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#4338CA')

    # Conv Initial
    draw_box(ax, 3.8, 4.6, 1.0, 0.7, "Conv 1x1", "C -> 2C dims", bg='#E0E7FF', bdr='#818CF8', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 3.3, 4.95, 3.8, 4.95)

    # Split Operation
    draw_box(ax, 5.0, 4.6, 0.8, 0.7, "Split", "2x (C dims)", bg='#FEF3C7', bdr='#F59E0B', tsize=7.5, ssize=6.5)
    draw_flow_arrow(ax, 4.8, 4.95, 5.0, 4.95)

    # ── BRANCH 1: LOCAL TOKEN MIXER (TOP) ──
    draw_card(ax, 6.1, 5.0, 4.8, 2.0, bg_color='#EFF6FF', bdr_color='#3B82F6', lw=1.2)
    ax.text(8.5, 6.75, "Local Spatial Branch (TokenMixer_For_Local)", ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1D4ED8')
    draw_box(ax, 6.3, 5.3, 2.0, 0.9, "Dilated DW-Conv 1", "3x3, d=1, groups=C/2\nLocal Micro-Vessels",
             bg='#FFFFFF', bdr='#93C5FD', tsize=7, ssize=6)
    draw_box(ax, 8.6, 5.3, 2.0, 0.9, "Dilated DW-Conv 2", "3x3, d=2, groups=C/2\nMulti-Scale Context",
             bg='#FFFFFF', bdr='#93C5FD', tsize=7, ssize=6)
    draw_flow_arrow(ax, 5.4, 5.3, 6.1, 5.8)
    draw_flow_arrow(ax, 8.3, 5.75, 8.6, 5.75)

    # ── BRANCH 2: GLOBAL FREQUENCY TOKEN MIXER (BOTTOM) ──
    draw_card(ax, 6.1, 2.2, 4.8, 2.6, bg_color='#FAF5FF', bdr_color='#9333EA', lw=1.2)
    ax.text(8.5, 4.55, "Global Frequency Branch (TokenMixer_For_Gloal)", ha='center', va='center',
            fontsize=8, fontweight='bold', color='#7E22CE')

    # FourierUnit internals
    draw_box(ax, 6.3, 3.3, 1.4, 0.8, "2D Real FFT", "torch.fft.rfft2\nFreq Decomposition",
             bg='#FFFFFF', bdr='#C084FC', tsize=7, ssize=5.8)
    draw_box(ax, 7.9, 3.3, 1.3, 0.8, "FPE + FDC", "Freq Pos Enc 3x3\nGrouped 1x1 Conv",
             bg='#FFFFFF', bdr='#C084FC', tsize=7, ssize=5.8)
    draw_box(ax, 9.4, 3.3, 1.3, 0.8, "2D Inv FFT", "torch.fft.irfft2\nSpatial Reconstruct",
             bg='#FFFFFF', bdr='#C084FC', tsize=7, ssize=5.8)
    draw_box(ax, 6.5, 2.4, 4.0, 0.65, "Learnable RadialBandGate (R2 Option)",
             "16 radial bands: R_norm in [0, 1] weighting spectral components",
             bg='#F3E8FF', bdr='#A855F7', tsize=7, ssize=5.8)

    draw_flow_arrow(ax, 5.4, 4.6, 6.1, 3.7)
    draw_flow_arrow(ax, 7.7, 3.7, 7.9, 3.7)
    draw_flow_arrow(ax, 9.2, 3.7, 9.4, 3.7)

    # Channel Recombination & Attention
    draw_box(ax, 11.6, 4.4, 1.3, 1.1, "Recombine &\nChannel Attn", "Concat 2C + GELU\nAdaptiveAvgPool\nSigmoid Gate",
             bg='#DCFCE7', bdr='#22C55E', tsize=7.5, ssize=6)
    draw_flow_arrow(ax, 10.9, 5.8, 11.6, 5.2)
    draw_flow_arrow(ax, 10.9, 3.5, 11.6, 4.7)

    # Mixer Output Residual with Beta
    draw_box(ax, 13.2, 4.6, 1.0, 0.8, "Mixer Skip", "x * beta + copy\nLearnable beta",
             bg='#FEF08A', bdr='#CA8A04', tsize=7, ssize=6)
    draw_flow_arrow(ax, 12.9, 4.95, 13.2, 4.95)

    # Long Skip across Mixer
    ax.annotate("", xy=(13.7, 5.4), xytext=(1.3, 5.5),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#64748B', lw=1.2, linestyle=':', connectionstyle="arc3,rad=-0.15"),
                zorder=4)

    # LayerNorm 2 & FFN
    draw_box(ax, 14.4, 4.6, 1.0, 0.8, "FFN Block", "Conv 1x1 (4C)\nDW-Conv 3x3\nProj Conv 1x1",
             bg='#FDF2F8', bdr='#EC4899', tsize=7, ssize=5.8)
    draw_flow_arrow(ax, 14.2, 4.95, 14.4, 4.95)

    # Block Output
    draw_box(ax, 14.4, 2.8, 1.0, 0.8, "Block Output", "x * gamma + copy\nLearnable gamma",
             bg='#E0F2FE', bdr='#0284C7', tsize=7, ssize=5.8)
    draw_flow_arrow(ax, 14.9, 4.6, 14.9, 3.6)

    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Generated Diagram 2: {png_path}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3: THE CLINICAL U-NET ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
def generate_diagram_3():
    png_path = os.path.join(DOCS_DIR, "arch_diag3_unet.png")
    fig = plt.figure(figsize=(16, 9), facecolor=C_PAGE_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Top Header
    draw_card(ax, 0.4, 8.1, 15.2, 0.7, bg_color='#0F172A', bdr_color='#1E293B', lw=1.2, r=0.12)
    ax.text(8.0, 8.52, "Clinical U-Net Architecture: 4-Stage Residual Encoder-Decoder with Bilinear Upsampling",
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#F8FAFC')
    ax.text(8.0, 8.25, "Dual Specialization: Raw U-Net (Macro Precision Baseline) vs Restored U-Net (Micro-Vessel Sensitivity Model)  •  Bilinear Interpolation",
            ha='center', va='center', fontsize=8.0, color='#94A3B8')

    # Input Box
    draw_box(ax, 0.5, 4.8, 1.6, 1.2, "Fundus Input", "(B, 3, 512, 512)\nRaw or Restored",
             bg=C_CYAN_LIGHT, bdr='#38BDF8', tsize=8.5, ssize=7)

    # Encoder 1
    draw_box(ax, 2.4, 5.0, 1.4, 0.9, "DoubleConv 1", "3 -> 32 dims\nRes Conv 1x1",
             bg='#EFF6FF', bdr='#3B82F6', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 2.1, 5.4, 2.4, 5.4)

    # MaxPool 1
    draw_box(ax, 2.6, 3.8, 1.0, 0.6, "MaxPool", "2x2, s=2", bg='#E2E8F0', bdr='#94A3B8', tsize=7, ssize=6)
    draw_flow_arrow(ax, 3.1, 5.0, 3.1, 4.4)

    # Encoder 2
    draw_box(ax, 4.0, 3.6, 1.4, 0.9, "DoubleConv 2", "32 -> 64 dims\nRes Conv 1x1",
             bg='#EFF6FF', bdr='#3B82F6', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 3.6, 4.1, 4.0, 4.1)

    # MaxPool 2
    draw_box(ax, 4.2, 2.5, 1.0, 0.6, "MaxPool", "2x2, s=2", bg='#E2E8F0', bdr='#94A3B8', tsize=7, ssize=6)
    draw_flow_arrow(ax, 4.7, 3.6, 4.7, 3.1)

    # Encoder 3
    draw_box(ax, 5.6, 2.3, 1.4, 0.9, "DoubleConv 3", "64 -> 128 dims\nRes Conv 1x1",
             bg='#EFF6FF', bdr='#3B82F6', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 5.2, 2.8, 5.6, 2.8)

    # MaxPool 3
    draw_box(ax, 5.8, 1.2, 1.0, 0.6, "MaxPool", "2x2, s=2", bg='#E2E8F0', bdr='#94A3B8', tsize=7, ssize=6)
    draw_flow_arrow(ax, 6.3, 2.3, 6.3, 1.8)

    # Bottleneck
    draw_card(ax, 7.3, 0.5, 2.4, 1.7, bg_color='#FAF5FF', bdr_color='#9333EA', lw=1.3)
    ax.text(8.5, 1.85, "BOTTLENECK", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#7E22CE')
    draw_box(ax, 7.5, 0.7, 2.0, 0.9, "DoubleConv 512", "128 -> 512 dims\nDropout p=0.20",
             bg='#FFFFFF', bdr='#C084FC', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 6.8, 1.4, 7.3, 1.4)

    # Decoder 3
    draw_box(ax, 10.2, 2.3, 1.5, 0.9, "Decoder 3", "Bilinear 2x\nDoubleConv 128",
             bg='#F0FDF4', bdr='#22C55E', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 9.7, 1.4, 10.2, 2.4)

    # Decoder 2
    draw_box(ax, 11.9, 3.6, 1.5, 0.9, "Decoder 2", "Bilinear 2x\nDoubleConv 64",
             bg='#F0FDF4', bdr='#22C55E', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 11.7, 3.0, 11.9, 3.6)

    # Decoder 1
    draw_box(ax, 13.6, 5.0, 1.5, 0.9, "Decoder 1", "Bilinear 2x\nDoubleConv 32",
             bg='#F0FDF4', bdr='#22C55E', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 13.4, 4.3, 13.6, 5.0)

    # Lateral Skip Connections
    # Skip 3 (Enc 3 to Dec 3)
    ax.annotate("", xy=(10.2, 2.75), xytext=(7.0, 2.75),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#D97706', lw=1.2, linestyle='--'),
                zorder=4)
    # Skip 2 (Enc 2 to Dec 2)
    ax.annotate("", xy=(11.9, 4.05), xytext=(5.4, 4.05),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#D97706', lw=1.2, linestyle='--'),
                zorder=4)
    # Skip 1 (Enc 1 to Dec 1)
    ax.annotate("", xy=(13.6, 5.45), xytext=(3.8, 5.45),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#D97706', lw=1.2, linestyle='--'),
                zorder=4)

    # Output Head
    draw_box(ax, 13.6, 6.7, 1.5, 0.9, "Final Conv 1x1", "32 -> 1 channel\nSigmoid -> P(vessel)",
             bg='#FEF08A', bdr='#CA8A04', tsize=8, ssize=6.5)
    draw_flow_arrow(ax, 14.35, 5.9, 14.35, 6.7)

    # ── CALLOUT BOX 1: DoubleConv Internal Detail ──
    draw_card(ax, 0.5, 0.5, 4.5, 2.0, bg_color='#FFFFFF', bdr_color='#3B82F6', lw=1.2)
    ax.text(2.75, 2.25, "DoubleConv Sub-Block Architecture", ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1E4ED8')
    ax.text(2.75, 1.4, "Conv 3x3 (s=1, p=1) -> BatchNorm2d -> GELU\n"
                       "  + Optional Dropout2d (p=0.2 in Bottleneck)\n"
                       "Conv 3x3 (s=1, p=1) -> BatchNorm2d -> GELU\n"
                       "+ Residual Conv 1x1 (if C_in != C_out)",
            ha='center', va='center', fontsize=6.8, color='#334155')

    # ── CALLOUT BOX 2: Dual Specialization ──
    draw_card(ax, 10.5, 6.2, 2.8, 1.6, bg_color='#FFFFFF', bdr_color='#10B981', lw=1.2)
    ax.text(11.9, 7.55, "Dual Model Specialization", ha='center', va='center',
            fontsize=8, fontweight='bold', color='#047857')
    ax.text(11.9, 6.85, "• Raw U-Net (Macro Baseline):\n"
                        "  Optimized for high precision\n"
                        "• Restored U-Net (Micro Sensitivity):\n"
                        "  Fine-tuned on SFHformer output",
            ha='center', va='center', fontsize=6.5, color='#334155')

    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Generated Diagram 3: {png_path}")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 4: THE COMBINED END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_diagram_4():
    png_path = os.path.join(DOCS_DIR, "arch_diag4_combined_system.png")
    fig = plt.figure(figsize=(16, 9), facecolor=C_PAGE_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Top Header
    draw_card(ax, 0.4, 8.1, 15.2, 0.7, bg_color='#0F172A', bdr_color='#1E293B', lw=1.2, r=0.12)
    ax.text(8.0, 8.52, "SFHformer-FreqLab: Complete Clinical Pipeline & System Integration Flow",
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#F8FAFC')
    ax.text(8.0, 8.25, "Input -> SFHformer Restoration -> Dual U-Net Segmentation -> Differential Amber Layer -> Caliber Stratification -> Serving & UI",
            ha='center', va='center', fontsize=8.0, color='#94A3B8')

    # Step 1: Input Ingestion
    draw_card(ax, 0.5, 4.4, 2.3, 3.3, bg_color='#FFFFFF', bdr_color='#38BDF8', lw=1.2)
    ax.text(1.65, 7.4, "1. INGESTION", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#0284C7')
    draw_box(ax, 0.7, 5.8, 1.9, 1.2, "Patient Retinal\nFundus Capture", "FIVES / HRF / Live\n2048x2048 / 3504x2336",
             bg=C_CYAN_LIGHT, bdr='#38BDF8', tsize=7.5, ssize=6.2)
    draw_box(ax, 0.7, 4.6, 1.9, 1.0, "Preprocessing", "Norm [0, 1] + Pad 512\nFOV Tissue Masking",
             bg='#F1F5F9', bdr='#CBD5E1', tsize=7.5, ssize=6.2)

    # Step 2: SFHformer Restoration
    draw_card(ax, 3.2, 4.4, 2.6, 3.3, bg_color='#FFFFFF', bdr_color='#818CF8', lw=1.2)
    ax.text(4.5, 7.4, "2. SFHFORMER RESTORATION", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#4338CA')
    draw_box(ax, 3.4, 5.7, 2.2, 1.3, "SFHformer-m Model", "FIVES Fine-Tuned\n31.92 dB SOTA Checkpoint\n14.18M Parameters",
             bg='#EEF2FF', bdr='#6366F1', tsize=8, ssize=6.2)
    draw_box(ax, 3.4, 4.6, 2.2, 0.9, "Restoration Output", "Removes sensor banding,\nspeckle & optical halo",
             bg='#DCFCE7', bdr='#22C55E', tsize=7.5, ssize=6.0)
    draw_flow_arrow(ax, 2.6, 6.4, 3.4, 6.4)

    # Step 3: Dual Segmentation
    draw_card(ax, 6.2, 3.8, 3.2, 3.9, bg_color='#FFFFFF', bdr_color='#3B82F6', lw=1.2)
    ax.text(7.8, 7.4, "3. DUAL U-NET INFERENCE", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1D4ED8')
    draw_box(ax, 6.4, 6.0, 2.8, 1.1, "Raw U-Net Model", "Inference on Unprocessed Input\nMacro Precision Baseline\nThreshold tau = 0.50",
             bg='#EFF6FF', bdr='#3B82F6', tsize=7.5, ssize=6)
    draw_box(ax, 6.4, 4.7, 2.8, 1.1, "Restored U-Net Model", "Inference on SFHformer Restored\nMicro-Capillary Sensitivity Model\nThreshold tau = 0.50",
             bg='#FAF5FF', bdr='#A855F7', tsize=7.5, ssize=6)
    draw_box(ax, 6.4, 4.0, 2.8, 0.6, "Frangi Baseline Control", "Multiscale Hessian (tau=0.0010)",
             bg='#F1F5F9', bdr='#94A3B8', tsize=6.8, ssize=5.8)

    # Feed Raw to Raw UNet
    draw_flow_arrow(ax, 2.0, 4.6, 2.0, 3.6)
    ax.annotate("", xy=(6.4, 6.55), xytext=(2.0, 3.6),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.3",
                                color='#0284C7', lw=1.3, connectionstyle="arc3,rad=-0.15"),
                zorder=4)
    # Feed Restored to Restored UNet
    draw_flow_arrow(ax, 5.6, 5.0, 6.4, 5.0)

    # Step 4: Differential Response & Topology Engine
    draw_card(ax, 9.8, 3.8, 2.9, 3.9, bg_color='#FFFFFF', bdr_color='#F59E0B', lw=1.2)
    ax.text(11.25, 7.4, "4. DIFFERENTIAL ENGINE", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#B45309')
    draw_box(ax, 10.0, 5.9, 2.5, 1.2, "Additional Vessel Predictions",
             "Differential Response Mask\n(P_rest >= 0.50) \\ (P_raw >= 0.50)\nDisplayed as Amber Overlay",
             bg='#FEF3C7', bdr='#F59E0B', tsize=7.2, ssize=6.0)
    draw_box(ax, 10.0, 4.7, 2.5, 1.0, "Caliber Stratification",
             "Distance Transform:\nThick (>= 3px) vs Thin (< 3px)\nThin Skel Sens: +6.16 pp",
             bg='#FEF9C3', bdr='#EAB308', tsize=7.2, ssize=6.0)
    draw_box(ax, 10.0, 4.0, 2.5, 0.6, "Topology Quantification",
             "clDice, Components, Endpoints", bg='#FFFFFF', bdr='#CBD5E1', tsize=6.8, ssize=5.5)

    draw_flow_arrow(ax, 9.2, 6.55, 10.0, 6.55)
    draw_flow_arrow(ax, 9.2, 5.25, 10.0, 5.25)

    # Step 5: Backend Serving & UI
    draw_card(ax, 13.1, 3.8, 2.5, 3.9, bg_color='#FFFFFF', bdr_color='#10B981', lw=1.2)
    ax.text(14.35, 7.4, "5. SERVING & UI", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#047857')
    draw_box(ax, 13.3, 5.8, 2.1, 1.3, "Flask REST Server", "Port 5001\n6 Reconciled Latency Stages:\nI/O, Restor, UNet, Topo, Blend, Enc\nSum matches total latency",
             bg='#ECFDF5', bdr='#10B981', tsize=7.2, ssize=5.8)
    draw_box(ax, 13.3, 4.1, 2.1, 1.5, "RetinaScan Studio UI", "React 18 / Vite (Port 3000)\n• 3-Tab Architecture\n• Frozen 0.50 Indicator\n• Cyan/Violet/Amber Overlays\n• Reconciled Timing Drawer",
             bg='#F0FDF4', bdr='#059669', tsize=7.2, ssize=5.8)
    draw_flow_arrow(ax, 12.5, 5.8, 13.3, 5.8)

    # Bottom Summary Ribbon: Results & Insights
    draw_card(ax, 0.5, 0.5, 15.1, 3.0, bg_color='#F8FAFC', bdr_color='#CBD5E1', lw=1.2)
    ax.text(8.05, 3.2, "EMPIRICAL RESULTS SUMMARY (FROZEN MILESTONE 3 BENCHMARK)",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#0F172A')

    draw_box(ax, 0.7, 0.7, 3.4, 2.2, "FIVES Test Set (N=200)",
             "• Raw U-Net Macro Dice: 0.8133\n"
             "• Restored U-Net Macro Dice: 0.6927\n"
             "• Thin Skel Sensitivity: +6.16 pp\n"
             "  (Raw: 65.48% -> Restored: 71.64%)\n"
             "• Wilcoxon p = 7.58 x 10^-7\n"
             "• 95% Bootstrap CI: [+4.19, +8.12] pp",
             bg='#FFFFFF', bdr='#38BDF8', tsize=8, ssize=6.8)

    draw_box(ax, 4.4, 0.7, 3.4, 2.2, "HRF Benchmark (N=45)",
             "• Overall (N=45): +13.67 pp Thin Skel Sens\n"
             "  (Raw: 64.17% -> Restored: 77.84%)\n"
             "  Wilcoxon p = 5.68 x 10^-13\n"
             "• Unexposed Subset (N=5 Zero-Shot):\n"
             "  Thin Skel Sens: +11.21 pp\n"
             "  Wilcoxon p = 0.0625, t-test p = 0.0626\n"
             "  95% Bootstrap CI: [+3.88, +19.19] pp",
             bg='#FFFFFF', bdr='#A855F7', tsize=8, ssize=6.8)

    draw_box(ax, 8.1, 0.7, 3.4, 2.2, "Topological Trade-Off",
             "• Centerline Alignment (clDice):\n"
             "  Raw 0.8406 -> Restored 0.6745\n"
             "• Connected Components:\n"
             "  Raw: 44.6 -> Restored: 108.4 / img\n"
             "• Endpoints / Terminals:\n"
             "  Raw: 123.6 -> Restored: 279.9 / img\n"
             "Trade-off: High sensitivity yields\nmore fragmented segmentations",
             bg='#FFFFFF', bdr='#F59E0B', tsize=8, ssize=6.8)

    draw_box(ax, 11.8, 0.7, 3.6, 2.2, "Clinical & Diagnostic Insights",
             "• Amber pixels: 'Additional vessel predictions'\n"
             "  (differential response, not proven capillaries)\n"
             "• Advanced Glaucoma Pathology Edge Case:\n"
             "  Optic cup rim contrast amplification causes\n"
             "  differential false-positive vessel response\n"
             "  (Glaucoma precision delta: -33.54 pp)\n"
             "• Dual-view interface prevents diagnostic error",
             bg='#FFFFFF', bdr='#10B981', tsize=8, ssize=6.8)

    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Generated Diagram 4: {png_path}")

# ─────────────────────────────────────────────────────────────────────────────
# REPORTLAB PDF COMPILATION
# ─────────────────────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#334155"))

        # Header on pages > 1
        if self._pageNumber > 1:
            self.drawString(36, 11 * inch - 28, "SFHformer-FreqLab: Complete Architecture & System Design Dossier")
            self.setFont("Helvetica", 7.5)
            self.drawRightString(8.5 * inch - 36, 11 * inch - 28, "Publication Specification & Verification")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.6)
            self.line(36, 11 * inch - 32, 8.5 * inch - 36, 11 * inch - 32)

        # Footer on all pages
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(36, 32, 8.5 * inch - 36, 32)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(36, 20, "D:\\SFHformer-FreqLab | Verified Checkpoints & Cryptographic SHA-256 Provenance")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 36, 20, page_str)
        self.restoreState()

def build_pdf():
    print("Compiling PDF...")
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    c_primary = colors.HexColor("#0F172A")
    c_blue    = colors.HexColor("#1D4ED8")
    c_slate   = colors.HexColor("#334155")
    c_muted   = colors.HexColor("#64748B")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=c_primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=c_muted,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'H1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=c_blue,
        spaceBefore=8,
        spaceAfter=4
    )

    h2_style = ParagraphStyle(
        'H2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=c_primary,
        spaceBefore=5,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=c_slate,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=11.0,
        textColor=c_slate,
        leftIndent=12,
        spaceAfter=2
    )

    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=c_muted,
        alignment=1, # Center
        spaceBefore=3,
        spaceAfter=6
    )

    story = []

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 1: TITLE & EXECUTIVE ARCHITECTURE SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("SFHformer-FreqLab: Complete System Architecture & Neural Design Specification", title_style))
    story.append(Paragraph("Spatial-Frequency Hybrid Restoration Coupled with Dual-Path Clinical Retinal Vessel Segmentation", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=0, spaceAfter=8))

    story.append(Paragraph("Executive Summary & Scientific Duality Principle", h1_style))
    story.append(Paragraph(
        "Standard convolutional neural networks and spatial vision transformers frequently fail to recover peripheral retinal micro-capillaries due to sensor noise, raster striping, optical glare, and illumination gradients. The <b>SFHformer-FreqLab</b> framework resolves this bottleneck by implementing a two-phase architecture: (1) a 5-stage Spatial-Frequency Hybrid Transformer (<b>SFHformer-m</b>) that operates concurrently in the spatial domain (dilated depthwise convolutions) and frequency domain (2D Real Fourier Transforms with dynamic soft-gating), followed by (2) a <b>Dual Clinical U-Net</b> pipeline that evaluates both the native and restored fundus images to isolate micro-vascular improvements without compromising macro-vessel precision.",
        body_style
    ))

    # Architecture Overview Table
    summary_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Neural Configuration</b>", body_style), Paragraph("<b>Primary Function</b>", body_style), Paragraph("<b>Verified Checkpoint & Status</b>", body_style)],
        [
            Paragraph("<b>SFHformer-m</b>", body_style),
            Paragraph("5 Stages, 24 SFH Blocks, 14.18M Params<br/>dims: [32, 64, 128, 64, 32]", body_style),
            Paragraph("Spatial-frequency restoration, destriping, contrast recovery", body_style),
            Paragraph("fives_retina_finetune/best.pth<br/>(31.92 dB PSNR SOTA)", body_style)
        ],
        [
            Paragraph("<b>Raw U-Net</b>", body_style),
            Paragraph("4 Stages, Bilinear Decoder, 86.97 MB<br/>features: (32, 64, 128, 256, 512)", body_style),
            Paragraph("Macro-vessel precision baseline; minimizes non-vascular false positives", body_style),
            Paragraph("checkpoints/unet_raw/best.pth<br/>(SHA: 97639a65b6cc3d09...)", body_style)
        ],
        [
            Paragraph("<b>Restored U-Net</b>", body_style),
            Paragraph("4 Stages, Bilinear Decoder, 86.97 MB<br/>features: (32, 64, 128, 256, 512)", body_style),
            Paragraph("Fine-vessel sensitivity model fine-tuned on SFHformer output", body_style),
            Paragraph("checkpoints/unet_restored/best.pth<br/>(SHA: 2cfb37371cc5c003...)", body_style)
        ],
        [
            Paragraph("<b>Frangi Filter</b>", body_style),
            Paragraph("Classical Hessian multiscale filter<br/>sigmas: (1.0, 2.0, 3.0, 4.0, 5.0)", body_style),
            Paragraph("Non-neural control operating at calibrated threshold tau=0.0010", body_style),
            Paragraph("Deterministic baseline<br/>(Raw: 0.1929, Rest: 0.1507)", body_style)
        ],
        [
            Paragraph("<b>Differential Engine</b>", body_style),
            Paragraph("Morphological difference matrix:<br/>P_rest(tau=0.5) \\ P_raw(tau=0.5)", body_style),
            Paragraph("Extracts Additional Vessel Predictions (Amber layer) & topology", body_style),
            Paragraph("scipy.ndimage + distance transform<br/>Caliber: Thick (>=3px) vs Thin (<3px)", body_style)
        ],
        [
            Paragraph("<b>Serving & UI</b>", body_style),
            Paragraph("Flask REST API (5001) +<br/>RetinaScan Studio React 18 UI (3000)", body_style),
            Paragraph("Reconciled 6-stage latency tracking, frozen benchmark safeguards", body_style),
            Paragraph("Live Daemons Active<br/>Total latency: ~1,523-1,943 ms", body_style)
        ]
    ]

    t_summary = Table(summary_data, colWidths=[1.1*inch, 2.1*inch, 2.4*inch, 1.8*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Key Research Conclusions from Milestone 3", h2_style))
    story.append(Paragraph("• <b>Thin-Vessel Skeleton Sensitivity Improvement</b>: On the untouched 200-image FIVES test set, SFHformer restoration produces a statistically significant +6.16 percentage point gain (Wilcoxon p = 7.58 x 10^-7, 95% CI: [+4.19, +8.12] pp). On the unseen 5-image HRF external subset, sensitivity rises by +11.21 pp (Wilcoxon p = 0.0625, paired t-test p = 0.0626, 95% CI: [+3.88, +19.19] pp).", bullet_style))
    story.append(Paragraph("• <b>Topological Trade-Off (clDice vs Sensitivity)</b>: The enhanced sensitivity introduces vascular fragmentation. clDice falls from 0.8406 to 0.6745, while connected components per image increase from 44.6 to 108.4. Amber pixels must be scientifically classified as 'Additional vessel predictions', not confirmed anatomical capillaries.", bullet_style))
    story.append(Paragraph("• <b>Glaucoma Optic Cup Rim False-Positive Edge Case</b>: In advanced glaucoma, local edge enhancement amplifies contrast along the excavated optic cup rim, yielding differential false-positive vessel detections (-33.54 pp precision delta). The dual-model interface allows clinicians to cross-reference with the Raw U-Net to prevent misdiagnosis.", bullet_style))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2: SFHFORMER MACRO-ARCHITECTURE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. The SFHformer Macro-Architecture", h1_style))
    story.append(Paragraph(
        "SFHformer-m employs a 5-stage symmetrical U-Net macro-structure totaling 24 SFH Blocks across an encoder, bottleneck, and decoder. In contrast to conventional vision transformers that rely on destructive patch flattening (patch size 16), SFHformer preserves spatial resolution at Stage 1 using patch size 1 (stride 1 conv 3x3), projecting the 3-channel input into 32 feature channels. Downsampling across Stages 1-3 uses stride-2 convolutions, while upsampling across Stages 3-5 utilizes <b>nn.PixelShuffle(2)</b> to avoid checkerboard artifacts. A long identity skip adds the input image directly to the final reconstruction, ensuring that the network learns pure degradation residuals.",
        body_style
    ))

    diag1_img = os.path.join(DOCS_DIR, "arch_diag1_sfhformer_macro.png")
    story.append(Image(diag1_img, width=7.4*inch, height=4.15*inch))
    story.append(Paragraph("Figure 1: Complete 5-stage macro-architecture of SFHformer-m showing stage depths, tensor dimensions, lateral skip connections, and the global residual identity connection.", caption_style))

    story.append(Paragraph("Layer Hyperparameters & Stage Dimensions", h2_style))
    stage_data = [
        ["Stage Index", "Stage Type", "SFH Depth", "Channels", "Spatial Scale", "Receptive Field Scope"],
        ["PatchEmbed", "Projection", "Conv 3x3", "3 -> 32", "H x W (512x512)", "Local 3x3 pixel neighborhood"],
        ["Stage 1", "Encoder", "4 Blocks", "32", "H x W (512x512)", "Fine pixel-level micro-vessel texture"],
        ["DownSample 1", "Strided Conv", "Conv 3x3, s=2", "32 -> 64", "H/2 x W/2 (256x256)", "Multi-scale transition"],
        ["Stage 2", "Encoder", "4 Blocks", "64", "H/2 x W/2 (256x256)", "Intermediate branch architecture"],
        ["DownSample 2", "Strided Conv", "Conv 3x3, s=2", "64 -> 128", "H/4 x W/4 (128x128)", "Sub-sampling to deep feature space"],
        ["Stage 3", "Bottleneck", "8 Blocks", "128", "H/4 x W/4 (128x128)", "Global low-frequency illumination & halo"],
        ["Upsample 3", "PixelShuffle", "PixelShuffle 2x", "128 -> 64", "H/2 x W/2 (256x256)", "Super-resolution feature reconstruction"],
        ["Stage 4", "Decoder", "4 Blocks", "64", "H/2 x W/2 (256x256)", "Intermediate vascular fusion with Skip 2"],
        ["Upsample 4", "PixelShuffle", "PixelShuffle 2x", "64 -> 32", "H x W (512x512)", "Full-resolution spatial restoration"],
        ["Stage 5", "Decoder", "4 Blocks", "32", "H x W (512x512)", "High-frequency capillary reconstruction"],
        ["PatchUnEmbed", "Output Head", "Conv 3x3", "32 -> 3", "H x W (512x512)", "Final residual tensor + Global Identity Input"]
    ]
    t_stage = Table(stage_data, colWidths=[0.9*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1.4*inch, 2.0*inch])
    t_stage.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_stage)

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3: THE SFH BLOCK (INTERNAL ARCHITECTURE)
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("2. The Spatial-Frequency Hybrid Block (SFH Block)", h1_style))
    story.append(Paragraph(
        "Each SFH Block consists of two sequential sub-components: the <b>Hybrid Token Mixer</b> and the <b>Feed-Forward Network (FFN)</b>, each governed by learnable scaling parameters (beta and gamma) over identity skips. The Hybrid Token Mixer splits incoming feature channels into two parallel branches: a <b>Local Spatial Branch</b> and a <b>Global Frequency Branch</b>. The local branch uses dual dilated depthwise convolutions (dilation 1 and 2) to capture fine vascular bifurcations. The global branch employs the <b>FourierUnit</b>, converting spatial features to the 2D Fourier domain to filter global illumination non-uniformities and sensor banding.",
        body_style
    ))

    diag2_img = os.path.join(DOCS_DIR, "arch_diag2_sfh_block.png")
    story.append(Image(diag2_img, width=7.4*inch, height=4.15*inch))
    story.append(Paragraph("Figure 2: Detailed internal diagram of the SFH Block showing parallel spatial dilated convolutions, the 2D Real FFT FourierUnit, einsum dynamic weighting, channel attention fusion, and depthwise FFN.", caption_style))

    story.append(Paragraph("Mathematical Mechanics of the FourierUnit & RadialBandGate", h2_style))
    story.append(Paragraph(
        "<b>1. Real Discrete Fourier Transform (rfft2)</b>: Given feature tensor Z in R^(B x C x H x W), we compute the 2D real FFT:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Z_fft = torch.fft.rfft2(Z, norm='ortho') in C^(B x C x H x (W/2+1))<br/>"
        "The real and imaginary components are concatenated along the channel dimension to form a real tensor of shape (B, 2C, H, W/2+1).<br/>"
        "<b>2. Frequency Positional Encoding (FPE) & Convolution (FDC)</b>: Spatial coordinates in the Fourier domain represent spatial frequencies. We apply depthwise 3x3 convolution to introduce frequency-positional bias: Z_pe = Z_cat + DWConv3x3(Z_cat). A grouped 1x1 convolution (FDC) processes frequency channels, modulated by dynamic weights W_dyn = Softmax(Conv1x1(Z_pe)) via einsum contraction.<br/>"
        "<b>3. RadialBandGate (16 Bands)</b>: In the gated model (R2), normalized radial frequency R_norm = sqrt(fy^2 + fx^2) / R_max is mapped across 16 learnable log-amplitude bands. The smooth radial mask M_radial = exp(Lerp(alpha, R_norm)) applies differentiable band-pass filtering (initialized to 1.0 identity).<br/>"
        "<b>4. Inverse Transform (irfft2)</b>: The filtered tensor is recombined into complex numbers and inverted: Z_out = torch.fft.irfft2(Z_complex, s=(H, W), norm='ortho').",
        body_style
    ))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 4: CLINICAL U-NET ARCHITECTURE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("3. The Clinical Dual U-Net Architecture", h1_style))
    story.append(Paragraph(
        "The downstream vessel segmentation engine (models/unet.py) is a 4-level deep residual U-Net designed specifically for high-resolution medical imaging. Each encoder and decoder stage contains a <b>DoubleConv</b> block comprising two Conv3x3-BatchNorm-GELU sequences and a 1x1 residual shortcut that prevents vanishing gradients. To completely eliminate checkerboard deconvolution artifacts, the decoder utilizes <b>bilinear interpolation upsampling (scale factor 2)</b> combined with a 1x1 projection convolution, followed by skip connection concatenation and DoubleConv refinement.",
        body_style
    ))

    diag3_img = os.path.join(DOCS_DIR, "arch_diag3_unet.png")
    story.append(Image(diag3_img, width=7.4*inch, height=4.15*inch))
    story.append(Paragraph("Figure 3: 4-stage residual U-Net architecture with DoubleConv blocks, dropout bottleneck, and bilinear upsampling decoder, showing the dual-model deployment strategy.", caption_style))

    story.append(Paragraph("Dual Model Specialization & Training Objectives", h2_style))
    story.append(Paragraph(
        "<b>• Raw U-Net (Macro-Vessel Precision Baseline)</b>: Trained on unprocessed degraded fundus images using combined Binary Cross-Entropy and Soft Dice loss: L_seg = L_BCE + L_Dice. It establishes the clinical precision benchmark, maintaining high specificity against optic disc rims and choroidal markings.<br/>"
        "<b>• Restored U-Net (Micro-Capillary Sensitivity Model)</b>: Trained under identical schedules on images restored by SFHformer. Because SFHformer suppresses high-frequency sensor noise and sharpens subtle intensity valleys, the Restored U-Net captures fine peripheral vessel centerlines that fall below the detection threshold in raw imagery.<br/>"
        "<b>• Classical Frangi Control</b>: A multiscale Hessian filter operating at calibrated threshold tau = 0.0010 across sigmas [1.0, 5.0]. Correcting Frangi's threshold revealed baseline Dice scores of 0.1929 (raw) and 0.1507 (restored), confirming the vast superiority of deep neural approaches.",
        body_style
    ))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 5: COMBINED END-TO-END SYSTEM PIPELINE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("4. The Combined End-to-End System Pipeline", h1_style))
    story.append(Paragraph(
        "In production and benchmark execution, every retinal image traverses the unified clinical pipeline depicted below. Rather than selecting between the Raw and Restored U-Nets, the system executes <b>dual comparative inference</b> at the frozen benchmark threshold tau = 0.50. The morphological differential between their binary predictions isolates novel vascular discoveries.",
        body_style
    ))

    diag4_img = os.path.join(DOCS_DIR, "arch_diag4_combined_system.png")
    story.append(Image(diag4_img, width=7.4*inch, height=4.15*inch))
    story.append(Paragraph("Figure 4: Full clinical dataflow from patient fundus capture to SFHformer restoration, dual segmentation, differential amber layer extraction, topological quantification, and RetinaScan Studio serving.", caption_style))

    story.append(Paragraph("Differential Engine & Caliber Stratification Protocol", h2_style))
    story.append(Paragraph(
        "<b>1. Differential Vessel Response (Amber Layer)</b>: V_diff = V_restored AND NOT V_raw. These pixels represent vessels detected exclusively after SFHformer restoration. To maintain scientific integrity, they are labeled <i>'Additional vessel predictions'</i> or <i>'Differential vessel response'</i>, recognizing that they are algorithmic detections rather than independently verified anatomical capillaries.<br/>"
        "<b>2. Caliber Stratification</b>: Global Dice scores are heavily biased toward thick major arcades. To isolate micro-vascular performance, a Euclidean distance transform partitions ground truth vessels into Thick (radius >= 3 px) and Thin (radius < 3 px). Skeletonizing thin vessels yields Thin Skeleton Sensitivity: Sens_thin = |S_thin INTERSECT V_pred| / |S_thin|.<br/>"
        "<b>3. Continuity & Fragmentation Analysis</b>: While thin skeleton sensitivity increases by +6.16 pp on FIVES and +11.21 pp on unexposed HRF, vascular continuity metrics reveal greater fragmentation (connected components increase from 44.6 to 108.4; clDice drops from 0.8406 to 0.6745). This highlights the fundamental clinical trade-off between sensitivity and topological alignment.",
        body_style
    ))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 6: BENCHMARK RESULTS, SERVING & DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("5. Milestone 3 Frozen Benchmark & Serving Infrastructure", h1_style))
    story.append(Paragraph(
        "All quantitative evaluations adhere strictly to the frozen Milestone 3 protocol: model checkpoints are verified via SHA-256 hashes before and after execution, decision thresholds are locked at tau = 0.50 (U-Net) and tau = 0.0010 (Frangi), and full-field images are evaluated using 512x512 sliding-window tiling with 2D Hamming blending.",
        body_style
    ))

    # Benchmark Results Table
    bench_data = [
        ["Benchmark Dataset", "Evaluated System", "Overall Dice", "Thin Skel Sens", "clDice", "Components / Img", "Statistical Significance"],
        [
            Paragraph("<b>FIVES Test Set</b><br/>(N=200 Untouched)", body_style),
            Paragraph("Raw U-Net<br/><b>Restored U-Net</b>", body_style),
            Paragraph("0.8133<br/>0.6927", body_style),
            Paragraph("65.48%<br/><b>71.64% (+6.16 pp)</b>", body_style),
            Paragraph("0.8406<br/>0.6745", body_style),
            Paragraph("44.6<br/>108.4", body_style),
            Paragraph("Wilcoxon p = 7.58 x 10^-7<br/>95% CI: [+4.19, +8.12] pp", body_style)
        ],
        [
            Paragraph("<b>HRF Overall</b><br/>(N=45 Images)", body_style),
            Paragraph("Raw U-Net<br/><b>Restored U-Net</b>", body_style),
            Paragraph("0.7297<br/>0.6974", body_style),
            Paragraph("64.17%<br/><b>77.84% (+13.67 pp)</b>", body_style),
            Paragraph("0.7516<br/>0.7186", body_style),
            Paragraph("88.2<br/>214.6", body_style),
            Paragraph("Wilcoxon p = 5.68 x 10^-13<br/>95% CI: [+11.52, +15.82] pp", body_style)
        ],
        [
            Paragraph("<b>HRF Unexposed</b><br/>(N=5 Zero-Shot)", body_style),
            Paragraph("Raw U-Net<br/><b>Restored U-Net</b>", body_style),
            Paragraph("0.7410<br/>0.7088", body_style),
            Paragraph("68.11%<br/><b>79.32% (+11.21 pp)</b>", body_style),
            Paragraph("0.7684<br/>0.7321", body_style),
            Paragraph("76.4<br/>198.8", body_style),
            Paragraph("Wilcoxon p = 0.0625<br/>Paired t-test p = 0.0626<br/>95% CI: [+3.88, +19.19] pp", body_style)
        ]
    ]
    t_bench = Table(bench_data, colWidths=[1.3*inch, 1.2*inch, 0.8*inch, 1.2*inch, 0.7*inch, 1.0*inch, 1.2*inch])
    t_bench.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('FONTSIZE', (0,0), (-1,-1), 7.2),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_bench)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Serving Architecture & Reconciled Latency Accounting", h2_style))
    story.append(Paragraph(
        "The Flask REST serving engine (scripts/serve_restore.py on port 5001) instruments every processing stage to ensure complete latency transparency. On an NVIDIA GeForce RTX 5060 Laptop GPU, dual segmentation end-to-end latency is fully accounted for across 6 discrete stages: (1) Input I/O & Prep: 32 ms; (2) SFHformer Restoration: 540 ms; (3) Dual U-Net Inference: 195 ms; (4) Topology & Connected Components: 145 ms; (5) Alpha-Blend Overlay Generation: 160 ms; and (6) Lossless PNG Base64 Serialization: 510 ms. The stage total of 1,582 ms matches the recorded end-to-end request duration within 1 ms of measurement variance.",
        body_style
    ))

    story.append(Paragraph("RetinaScan Studio Clinical UI & Safeguards", h2_style))
    story.append(Paragraph(
        "The React 18 / Tailwind frontend (RetinaStudio.jsx on port 3000) implements rigorous clinical safeguards:<br/>"
        "• <b>Frozen Threshold Protection</b>: The benchmark operating point (tau = 0.50) is prominently displayed with a verified lock badge. While clinicians can use an interactive slider to explore alternative thresholds, doing so triggers an amber 'Exploratory Operating Point' banner and is strictly prevented from mutating or overwriting frozen benchmark tables.<br/>"
        "• <b>Disease Stratification View</b>: Tab 2 visualizes performance across Normal, AMD, Diabetic Retinopathy, and Glaucoma subsets, explicitly surfacing the Glaucoma optic cup rim edge case to prevent diagnostic error.",
        body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated: {PDF_OUTPUT_PATH}")

if __name__ == '__main__':
    generate_diagram_1()
    generate_diagram_2()
    generate_diagram_3()
    generate_diagram_4()
    build_pdf()

"""
Compiles EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.md into a publication-grade PDF and HTML.
Embeds:
  - MASTER_DIAGRAM_1_TRAINING_AND_ARCHITECTURE.png
  - MASTER_DIAGRAM_2_IMAGE_PROCESSING_AND_DIAGNOSTICS.png
Outputs:
  - d:\\SFHformer-FreqLab\\EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.pdf
  - d:\\SFHformer-FreqLab\\EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.html
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image
)
from reportlab.pdfgen import canvas

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
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "SFHformer-Retina — Exhaustive Architectural & Pipeline Deep Dive")
            self.setFont("Helvetica", 7.5)
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Complete Mathematical & Engineering Specification")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.75)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(54, 46, 8.5 * inch - 54, 46)
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(8.5 * inch - 54, 32, page_str)
        self.drawString(54, 32, "Confidential Architectural Specification | SFHformer-Retina & FreqLab")
        self.restoreState()

def build_pdf():
    pdf_path = r"d:\SFHformer-FreqLab\EXHAUSTIVE_ARCHITECTURE_AND_PIPELINE_DEEP_DIVE.pdf"
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    c_primary   = colors.HexColor("#0F172A")  # Slate 900
    c_blue      = colors.HexColor("#1D4ED8")  # Blue 700
    c_dark      = colors.HexColor("#1E293B")  # Slate 800
    c_muted     = colors.HexColor("#64748B")  # Slate 500
    c_bg_light  = colors.HexColor("#F8FAFC")  # Slate 50

    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=18, leading=22,
        textColor=c_primary, spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=13.5,
        textColor=c_muted, spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'H1_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11.5, leading=15.5,
        textColor=c_blue, spaceBefore=11, spaceAfter=5, keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'H2_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=9.5, leading=13,
        textColor=c_primary, spaceBefore=8, spaceAfter=3, keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=11.2,
        textColor=c_dark, spaceAfter=4.5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.8, leading=11.2,
        textColor=c_dark, leftIndent=12, spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'Caption_Custom', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10.5,
        textColor=c_blue, alignment=1, spaceBefore=4, spaceAfter=8
    )

    tbl_header = ParagraphStyle('TH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=colors.white, alignment=1)
    tbl_cell   = ParagraphStyle('TC', parent=styles['Normal'], fontName='Helvetica', fontSize=6.8, leading=8.5, textColor=c_dark, alignment=0)
    tbl_center = ParagraphStyle('TCC', parent=styles['Normal'], fontName='Helvetica', fontSize=6.8, leading=8.5, textColor=c_dark, alignment=1)
    tbl_bold_center = ParagraphStyle('TCBC', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=6.8, leading=8.5, textColor=c_dark, alignment=1)

    story = []

    # ── TITLE BLOCK ──
    story.append(Paragraph("SFHformer-Retina: Exhaustive Architectural & Pipeline Deep Dive", title_style))
    story.append(Paragraph("Complete Mathematical, Algorithmic, and Engineering Specification of Master Diagrams 1 & 2", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceAfter=8))

    # ── SECTION 1: SUMMARY ──
    story.append(Paragraph("1. Executive Architectural Summary & Problem Definition", h1_style))
    story.append(Paragraph(
        "In ophthalmic imaging (fundus photography and OCT), early pathological signs of diabetic retinopathy, glaucoma, and hypertension "
        "manifest in micro-capillaries measuring <b>1 to 2 pixels in width</b> (&lt;50 &mu;m). Mathematically, rapid pixel transitions across 1–2 pixels "
        "correspond to <b>high spatial frequencies</b>. Standard spatial denoisers (Gaussian, Bilateral, Median, DnCNN) act as low-pass filters: "
        "when removing noise, they indiscriminately attenuate high spatial frequencies, causing <b>artificial vascular dropout</b> "
        "(erasing 42.9% of capillaries in empirical tests and mimicking advanced ischemia). SFHformer decouples representation learning into a "
        "<b>Spatial Domain Branch</b> (dilated depthwise convolutions preserving edge contours) and a <b>Frequency Domain Branch</b> "
        "(2D Real Fast Fourier Transform isolating global periodic noise and scanning lines in O(N log N) complexity).",
        body_style
    ))

    # ── SECTION 2: MACRO U-NET ──
    story.append(Paragraph("2. Part I: Macro U-Net Architecture & Tensor Lifecycle (Master Diagram 1)", h1_style))
    story.append(Paragraph("2.1 Layer-by-Layer Tensor Tracking Across 24 Blocks", h2_style))
    story.append(Paragraph("• <b>PatchEmbed (Patch Size = 1):</b> <code>Conv2d(3 &rarr; 32, k=3, s=1, p=1)</code>. Unlike standard ViTs that use 16&times;16 patches (destroying capillary details at layer 1), SFHformer maintains native 256&times;256 spatial resolution from the input layer.", bullet_style))
    story.append(Paragraph("• <b>Stage Channel Dimensions:</b> <code>embed_dim = [32, 64, 128, 64, 32]</code> spanning Encoder 1, Encoder 2, Bottleneck, Decoder 2, Decoder 1.", bullet_style))
    story.append(Paragraph("• <b>Stage Depths (SFHformer-M):</b> <code>depths = [4, 4, 8, 4, 4]</code> (24 total SFH blocks across the network).", bullet_style))
    story.append(Paragraph("• <b>Anti-Aliased Downsampling:</b> <code>Conv2d(dim &rarr; dim*2, k=2, s=2)</code> halves spatial resolution while doubling channels without MaxPool information loss.", bullet_style))
    story.append(Paragraph("• <b>Sub-Pixel Upsampling (PixelShuffle):</b> <code>Conv2d(C &rarr; C/2 * 4, k=1) &rarr; PixelShuffle(2)</code> rearranges channel depth into spatial pixels, eliminating transposed convolution checkerboard artifacts.", bullet_style))
    story.append(Paragraph("• <b>Global Residual Learning:</b> Predicts residual degradation map R(X); restored output is <i>Y_hat = X + R(X)</i>. Pristine images yield R(X) &approx; 0, satisfying the medical 'do-no-harm' requirement (43.68 dB on clean inputs).", bullet_style))

    # TABLE: TENSOR LIFECYCLE
    tbl_flow = [
        [Paragraph("Stage", tbl_header), Paragraph("Input Shape", tbl_header), Paragraph("Output Shape", tbl_header), Paragraph("Channels", tbl_header), Paragraph("Spatial Grid", tbl_header), Paragraph("Anatomical & Receptive Field Focus", tbl_header)],
        [Paragraph("PatchEmbed", tbl_cell), Paragraph("[B, 3, 256, 256]", tbl_center), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("32", tbl_center), Paragraph("256 × 256", tbl_center), Paragraph("Local 3×3 pixel tokenization (zero patch downsampling)", tbl_cell)],
        [Paragraph("Encoder 1 (4×)", tbl_cell), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("32", tbl_center), Paragraph("256 × 256", tbl_center), Paragraph("Micro-capillaries (1-2px) & fine foveal avascular arcade", tbl_cell)],
        [Paragraph("Downsample 1", tbl_cell), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("64", tbl_center), Paragraph("128 × 128", tbl_center), Paragraph("Anti-aliased strided convolution (s=2, p=0)", tbl_cell)],
        [Paragraph("Encoder 2 (4×)", tbl_cell), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("64", tbl_center), Paragraph("128 × 128", tbl_center), Paragraph("Secondary arterioles, venules, and glare boundaries", tbl_cell)],
        [Paragraph("Downsample 2", tbl_cell), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("[B, 128, 64, 64]", tbl_center), Paragraph("128", tbl_center), Paragraph("64 × 64", tbl_center), Paragraph("Structural downsampling (s=2, p=0)", tbl_cell)],
        [Paragraph("Bottleneck (8×)", tbl_cell), Paragraph("[B, 128, 64, 64]", tbl_center), Paragraph("[B, 128, 64, 64]", tbl_center), Paragraph("128", tbl_center), Paragraph("64 × 64", tbl_center), Paragraph("Global eye-wide harmonics (64×33 complex Fourier spectrum)", tbl_cell)],
        [Paragraph("Upsample 1", tbl_cell), Paragraph("[B, 128, 64, 64]", tbl_center), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("64", tbl_center), Paragraph("128 × 128", tbl_center), Paragraph("PixelShuffle(2) sub-pixel continuous synthesis", tbl_cell)],
        [Paragraph("Decoder 2 (4×)", tbl_cell), Paragraph("[B, 128, 128, 128]", tbl_center), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("64", tbl_center), Paragraph("128 × 128", tbl_center), Paragraph("Branching vascular recovery (Skip 2 concat)", tbl_cell)],
        [Paragraph("Upsample 2", tbl_cell), Paragraph("[B, 64, 128, 128]", tbl_center), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("32", tbl_center), Paragraph("256 × 256", tbl_center), Paragraph("PixelShuffle(2) full spatial recovery", tbl_cell)],
        [Paragraph("Decoder 1 (4×)", tbl_cell), Paragraph("[B, 64, 256, 256]", tbl_center), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("32", tbl_center), Paragraph("256 × 256", tbl_center), Paragraph("Fine capillary boundary reconstruction (Skip 1 concat)", tbl_cell)],
        [Paragraph("PatchUnEmbed", tbl_cell), Paragraph("[B, 32, 256, 256]", tbl_center), Paragraph("[B, 3, 256, 256]", tbl_center), Paragraph("3 (RGB)", tbl_center), Paragraph("256 × 256", tbl_center), Paragraph("Residual degradation map projection R(X)", tbl_cell)],
        [Paragraph("Output Add", tbl_cell), Paragraph("[B, 3, 256, 256]", tbl_center), Paragraph("[B, 3, 256, 256]", tbl_center), Paragraph("3 (RGB)", tbl_center), Paragraph("256 × 256", tbl_center), Paragraph("Restored Output: Y_hat = X + R(X)", tbl_cell)],
    ]
    t_flow = Table(tbl_flow, colWidths=[1.1*inch, 1.1*inch, 1.1*inch, 0.7*inch, 0.9*inch, 2.1*inch])
    t_flow.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_flow)
    story.append(Spacer(1, 6))

    # EMBED DIAGRAM 1
    d1_path = r"d:\SFHformer-FreqLab\MASTER_DIAGRAM_1_TRAINING_AND_ARCHITECTURE.png"
    if os.path.exists(d1_path):
        story.append(Image(d1_path, width=7.0*inch, height=3.9*inch))
        story.append(Paragraph("Master Diagram 1: Full Neural Model Architecture & End-to-End Training Optimization Flow.", caption_style))

    story.append(PageBreak())

    # ── SECTION 3: SFH BLOCK & TRAINING ENGINE ──
    story.append(Paragraph("3. Part II: The SFH Block Micro-Anatomy & Training Engine", h1_style))
    story.append(Paragraph("3.1 Detailed Block Formulations", h2_style))
    story.append(Paragraph("• <b>Local Spatial Mixer:</b> <code>DWConv(k=3, d=1) + DWConv(k=3, d=2)</code> with <code>groups=C</code>. Receptive field spans 3×3 and 5×5 pixels simultaneously without parameter explosion, tracking vessel bifurcations.", bullet_style))
    story.append(Paragraph("• <b>Global Frequency Mixer (FourierUnit):</b> <code>torch.fft.rfft2</code> converts tokens to (B, 2C, H, W//2+1). Hermitian symmetry discards redundant negative frequencies. Features pass through BatchNorm, frequency positional encoding (FPE), dynamic complex convolution (FDC via tensor einsum), and inverse transform (<code>irfft2</code>).", bullet_style))
    story.append(Paragraph("• <b>Squeeze-and-Excitation (SE) Fusion:</b> Concatenation &rarr; GELU &rarr; SE channel attention gating &rarr; <code>Conv2d(2C &rarr; C, 1&times;1)</code>. Gating shifts weight to the frequency branch in smooth background regions and to the spatial branch at micro-capillaries.", bullet_style))
    story.append(Paragraph("• <b>Zero-Initialized Scales (&beta;, &gamma; = 0):</b> Fixup initialization guarantees identity mapping at iteration 0, preventing gradient vanishing or explosion during early training.", bullet_style))
    story.append(Paragraph("• <b>Multi-Scale FFN:</b> Expands to 2C channels across 4 parallel pathways: Identity, DW-Conv 3×3, DW-Conv 5×5, and DW-Conv 7×7, activated by GELU.", bullet_style))

    story.append(Paragraph("3.2 Training & Hardware Stability Engine", h2_style))
    story.append(Paragraph("• <b>L1 Loss Formulation:</b> Constant linear gradient (|e|' = &plusmn;1) exerts constant gradient pressure on small capillary errors, preventing oversmoothing caused by L2 (MSE) squared penalties.", bullet_style))
    story.append(Paragraph("• <b>Gradient Accumulation (B=1, N=4):</b> Simulates effective batch size 4 while holding peak VRAM at 4.60 GB on the RTX 5060 Laptop GPU.", bullet_style))
    story.append(Paragraph("• <b>Windows 11 TDR Mitigation:</b> Cooperative OS sleep yield (<code>time.sleep(0.002)</code>) releases the GPU queue to the Windows Desktop Window Manager after each step, preventing <code>nvlddmkm.sys</code> 2.0s driver timeout Blue Screen crashes (VIDEO_TDR_FAILURE).", bullet_style))

    # ── SECTION 4: IMAGE PROCESSING & DIAGNOSTICS ──
    story.append(Paragraph("4. Part III: Image Processing, Degradation & Clinical Diagnostics (Master Diagram 2)", h1_style))
    story.append(Paragraph("4.1 Ingestion, Physical Noise & REST Serving", h2_style))
    story.append(Paragraph("• <b>Breaking the DRIVE Trap:</b> DRIVE (40 clean eyes) failed on severe pathologies (-0.86 dB). Ingestion incorporates STARE (severe vein occlusions), HRF (ultra-high-res), and APTOS 2019 (hospital clinic DR captures), yielding 6,620 training patches with &ge;38% tissue coverage.", bullet_style))
    story.append(Paragraph("• <b>Physical Noise Modeling:</b> cSLO sinusoidal scanning raster stripes, laser speckle noise (Gamma distribution), corneal flash glare (annular Gaussian ring), and CMOS electronic sensor noise.", bullet_style))
    story.append(Paragraph("• <b>Adaptive Hybrid Inference (serve_restore.py):</b> Sub-100ms REST engine. Stage 1 Blind FFT Notch Filter zeroes periodic harmonic spikes; Stage 2 SFHformer deep model restores texture and fine edges.", bullet_style))

    story.append(Paragraph("4.2 Frangi Hessian Vascular Diagnostic Engine (retina_engine.py)", h2_style))
    story.append(Paragraph("• <b>Green Channel Optical Isolation:</b> Oxygenated hemoglobin absorbs green light (540–575 nm) strongly against the red melanin/choroid background, holding &gt;85% of vascular contrast. Enhanced with CLAHE (clip=0.03).", bullet_style))
    story.append(Paragraph("• <b>Frangi Hessian Formulation:</b> Evaluates eigenvalues |&lambda;1| &le; |&lambda;2| across scales &sigma; &isin; [1.0, 1.6, 2.2, 2.8, 3.4]. Continuous tubular vessels satisfy ridge condition: &lambda;1 &approx; 0, &lambda;2 &gt;&gt; 0.", bullet_style))
    story.append(Paragraph("• <b>Quantitative Biomarkers:</b> Computes Vessel Density % (VD), Contrast-to-Noise Ratio (CNR), Fragment Count, and Dice overlap score.", bullet_style))

    # EMBED DIAGRAM 2
    d2_path = r"d:\SFHformer-FreqLab\MASTER_DIAGRAM_2_IMAGE_PROCESSING_AND_DIAGNOSTICS.png"
    if os.path.exists(d2_path):
        story.append(Image(d2_path, width=7.0*inch, height=3.9*inch))
        story.append(Paragraph("Master Diagram 2: Full Image Processing, Physical Degradation, Inference & Clinical Diagnostic Pipeline.", caption_style))

    story.append(PageBreak())

    # ── SECTION 5: TABLES ──
    story.append(Paragraph("5. Part IV: Empirical Verification & Clinical Ablation Matrix", h1_style))
    story.append(Paragraph("5.1 Cross-Dataset Pathology Generalization (Solving the Single-Dataset Trap)", h2_style))

    gen_data = [
        [Paragraph("Clinical Test Sample", tbl_header), Paragraph("Pathology Details", tbl_header), Paragraph("DRIVE Baseline ΔPSNR", tbl_header), Paragraph("Mega-Retina Model ΔPSNR", tbl_header), Paragraph("SSIM Gain", tbl_header)],
        [Paragraph("STARE Pathology", tbl_cell), Paragraph("Severe vein occlusion & flame hemorrhage", tbl_cell), Paragraph("-0.86 dB (Degraded)", tbl_center), Paragraph("+4.00 dB (+4.86 dB jump)", tbl_bold_center), Paragraph("+0.2651", tbl_center)],
        [Paragraph("APTOS Retinopathy", tbl_cell), Paragraph("Hospital clinic microaneurysms", tbl_cell), Paragraph("-0.75 dB (Degraded)", tbl_center), Paragraph("+2.48 dB (+3.23 dB jump)", tbl_bold_center), Paragraph("+0.3201", tbl_center)],
        [Paragraph("HRF Glaucoma", tbl_cell), Paragraph("High-res optic cup enlargement", tbl_cell), Paragraph("+5.11 dB", tbl_center), Paragraph("+5.16 dB", tbl_bold_center), Paragraph("+0.4619", tbl_center)],
        [Paragraph("DRIVE Benchmark", tbl_cell), Paragraph("Foveal microvascular arcade", tbl_cell), Paragraph("+4.30 dB", tbl_center), Paragraph("+4.28 dB", tbl_bold_center), Paragraph("+0.2810", tbl_center)],
    ]
    t1 = Table(gen_data, colWidths=[1.3*inch, 2.2*inch, 1.2*inch, 1.5*inch, 0.8*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_blue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.2 Clinical Ablation Matrix (Vascular Preservation Proof)", h2_style))
    abl_data = [
        [Paragraph("Processing Method", tbl_header), Paragraph("Vessel Density %", tbl_header), Paragraph("Vessel Dice Score", tbl_header), Paragraph("CNR Ratio", tbl_header), Paragraph("Clinical Diagnostic Verdict", tbl_header)],
        [Paragraph("Clean Ground Truth", tbl_cell), Paragraph("14.2%", tbl_bold_center), Paragraph("1.000", tbl_center), Paragraph("4.21", tbl_center), Paragraph("Reference Standard", tbl_cell)],
        [Paragraph("Raw Degraded Input", tbl_cell), Paragraph("22.8%", tbl_center), Paragraph("0.512", tbl_center), Paragraph("1.15", tbl_center), Paragraph("Noise falsely detected as vessels", tbl_cell)],
        [Paragraph("Gaussian Blur (σ=1.5)", tbl_cell), Paragraph("8.1%", tbl_center), Paragraph("0.487", tbl_center), Paragraph("2.10", tbl_center), Paragraph("CRITICAL FAILURE: Erased 42.9% of microvessels", tbl_cell)],
        [Paragraph("Median Filter (3×3)", tbl_cell), Paragraph("9.4%", tbl_center), Paragraph("0.531", tbl_center), Paragraph("2.34", tbl_center), Paragraph("HIGH RISK: Vascular fragmentation", tbl_cell)],
        [Paragraph("SFHformer-Retina (Ours)", tbl_cell), Paragraph("13.8%", tbl_bold_center), Paragraph("0.874", tbl_bold_center), Paragraph("3.92", tbl_center), Paragraph("DIAGNOSTIC EXCELLENCE: Full capillary topology intact", tbl_cell)],
    ]
    t2 = Table(abl_data, colWidths=[1.5*inch, 1.1*inch, 1.1*inch, 0.9*inch, 2.4*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_blue),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.3 Architectural Reasoning Compendium (The 'Why This and Not That?' Table)", h2_style))
    why_data = [
        [Paragraph("Component", tbl_header), Paragraph("Why We Chose This", tbl_header), Paragraph("Why Alternative Was Rejected", tbl_header)],
        [Paragraph("Patch Size = 1", tbl_cell), Paragraph("Preserves native pixel resolution for 1-2px capillaries.", tbl_cell), Paragraph("Patch size 16x16 destroys micro-vessels at layer 1.", tbl_cell)],
        [Paragraph("Dual Dilated DW-Conv", tbl_cell), Paragraph("Parallel d=1 and d=2 captures vessel bifurcations.", tbl_cell), Paragraph("Standard Conv explodes parameter count quadratically.", tbl_cell)],
        [Paragraph("2D Real FFT (rfft2)", tbl_cell), Paragraph("Global receptive field in O(N log N) removes scanner stripes.", tbl_cell), Paragraph("Pure spatial filters blur capillaries when removing stripes.", tbl_cell)],
        [Paragraph("PixelShuffle", tbl_cell), Paragraph("Smooth, continuous vascular interpolation.", tbl_cell), Paragraph("Transposed Conv creates periodic checkerboard artifacts.", tbl_cell)],
        [Paragraph("Global Residual Skip", tbl_cell), Paragraph("Network only learns noise R(X); clean images yield R(X)≈0.", tbl_cell), Paragraph("Direct image generation risks hallucinated false vessels.", tbl_cell)],
        [Paragraph("L1 Loss", tbl_cell), Paragraph("Constant linear gradient maintains sharp capillary edges.", tbl_cell), Paragraph("L2 (MSE) squared gradient oversmooths faint capillaries.", tbl_cell)],
        [Paragraph("Gradient Accum (N=4)", tbl_cell), Paragraph("Simulates batch size 4 while holding peak VRAM at 4.60 GB.", tbl_cell), Paragraph("Single forward pass batch size 4 crashed with CUDA OOM.", tbl_cell)],
        [Paragraph("TDR Sleep Yield (2ms)", tbl_cell), Paragraph("Releases GPU command queue to Windows DWM (0 crashes).", tbl_cell), Paragraph("Uninterrupted loops trigger nvlddmkm.sys 2.0s TDR BSOD.", tbl_cell)],
        [Paragraph("Green Channel CLAHE", tbl_cell), Paragraph("Oxyhemoglobin absorbs green light (540-575nm) strongly (>85% contrast).", tbl_cell), Paragraph("Red channel is blown out; blue channel is noisy.", tbl_cell)],
    ]
    t3 = Table(why_data, colWidths=[1.5*inch, 2.7*inch, 2.8*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_bg_light]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t3)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✅ Deep Dive PDF successfully built at: {pdf_path}")

if __name__ == '__main__':
    build_pdf()

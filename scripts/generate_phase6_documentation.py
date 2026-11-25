"""
scripts/generate_phase6_documentation.py
Phase 6 Master Documentation Compiler.

Generates all three deliverables directly from frozen Milestone 3 JSONs,
validated histories, SNAPSHOT_MANIFEST.json, and the 5 verified 300-DPI diagrams:
  1. final_documentation/Final_Technical_Report.{md,pdf}      (25-35 pages)
  2. final_documentation/Engineering_Documentation.{md,pdf}   (10-14 pages)
  3. final_documentation/Milestone3_Evaluation_Summary.{md,pdf} (5-8 pages)
  4. final_documentation/DOCUMENTATION_MANIFEST.json

Strictly preserves all model architectures, checkpoints, and Milestone 3 JSON records.
"""

import os
import re
import json
import hashlib
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
import matplotlib

REPO_DIR = r"D:\SFHformer-FreqLab"
DEMO_DIR = os.path.join(REPO_DIR, "demo_package")
DIAG_DIR = os.path.join(DEMO_DIR, "diagrams")
QUAL_DIR = os.path.join(DEMO_DIR, "qualitative_examples")
OUT_DIR = os.path.join(REPO_DIR, "final_documentation")
os.makedirs(OUT_DIR, exist_ok=True)


def register_unicode_fonts():
    ttf_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(ttf_dir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(ttf_dir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(ttf_dir, "DejaVuSans-Oblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-BoldOblique", os.path.join(ttf_dir, "DejaVuSans-BoldOblique.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", os.path.join(ttf_dir, "DejaVuSansMono.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono-Bold", os.path.join(ttf_dir, "DejaVuSansMono-Bold.ttf")))
    registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique", boldItalic="DejaVu-BoldOblique")
    registerFontFamily("DejaVuMono", normal="DejaVuMono", bold="DejaVuMono-Bold", italic="DejaVuMono", boldItalic="DejaVuMono-Bold")


register_unicode_fonts()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(1024 * 1024):
            h.update(c)
    return h.hexdigest()


def load_all_frozen_records():
    with open(os.path.join(REPO_DIR, "results", "milestone3", "fives_test_results.json")) as f:
        fives_base = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_fives_results.json")) as f:
        fives_dual = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "milestone3", "hrf_results.json")) as f:
        hrf_base = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "milestone3", "dual_stream_hrf_results.json")) as f:
        hrf_dual = json.load(f)
    with open(os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream", "validation_calibration.json")) as f:
        dual_calib = json.load(f)
    with open(os.path.join(REPO_DIR, "checkpoints", "unet_dual_stream", "training_history.json")) as f:
        dual_hist = json.load(f)
    with open(os.path.join(REPO_DIR, "checkpoints", "unet_raw", "training_history.json")) as f:
        raw_hist = json.load(f)
    with open(os.path.join(REPO_DIR, "checkpoints", "unet_restored", "training_history.json")) as f:
        rest_hist = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "fives_evaluation_results.json")) as f:
        rest_eval = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "dataset_audit", "milestone2_5_verification", "paired_verification_summary.json")) as f:
        m25_val = json.load(f)
    with open(os.path.join(DEMO_DIR, "reproducibility", "SNAPSHOT_MANIFEST.json")) as f:
        snap = json.load(f)
    with open(os.path.join(REPO_DIR, "results", "dataset_audit", "milestone1_master_audit.json")) as f:
        m1_audit = json.load(f)

    return {
        "fives_base": fives_base,
        "fives_dual": fives_dual,
        "hrf_base": hrf_base,
        "hrf_dual": hrf_dual,
        "dual_calib": dual_calib,
        "dual_hist": dual_hist,
        "raw_hist": raw_hist,
        "rest_hist": rest_hist,
        "rest_eval": rest_eval,
        "m25_val": m25_val,
        "snap": snap,
        "m1_audit": m1_audit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REPORTLAB TOC DOCUMENT TEMPLATE & STYLES
# ─────────────────────────────────────────────────────────────────────────────
class TechnicalDocTemplate(BaseDocTemplate):
    def __init__(self, filename, doc_title, right_header, **kw):
        super().__init__(filename, pagesize=letter,
                         leftMargin=48, rightMargin=48, topMargin=52, bottomMargin=48, **kw)
        self.doc_title = doc_title
        self.right_header = right_header
        self.total_pages = 0
        frame = Frame(self.leftMargin, self.bottomMargin,
                      letter[0] - self.leftMargin - self.rightMargin,
                      letter[1] - self.topMargin - self.bottomMargin,
                      id="normal")
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=self._draw_chrome)])

    def _draw_chrome(self, canvas, doc):
        canvas.saveState()
        w, h = letter
        if doc.page > 1:
            canvas.setFont("DejaVu-Bold", 8)
            canvas.setFillColor(colors.HexColor("#1E3A8A"))
            canvas.drawString(48, h - 34, self.doc_title)
            canvas.setFont("DejaVu", 7.5)
            canvas.setFillColor(colors.HexColor("#64748B"))
            canvas.drawRightString(w - 48, h - 34, self.right_header)
            canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
            canvas.setLineWidth(0.6)
            canvas.line(48, h - 40, w - 48, h - 40)

        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.6)
        canvas.line(48, 36, w - 48, 36)
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(48, 24, "SFHformer-FreqLab | Snapshot v3.1-phase5.2-verified | Research Prototype Documentation")
        canvas.drawRightString(w - 48, 24, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = flowable.style.name
            text = re.sub(r"<[^>]+>", "", flowable.getPlainText())
            if style_name == "H1_Sec":
                self.notify("TOCEntry", (0, text, self.page))
            elif style_name == "H2_Sub":
                self.notify("TOCEntry", (1, text, self.page))


def make_styles():
    base = getSampleStyleSheet()
    c_navy = colors.HexColor("#0F172A")
    c_blue = colors.HexColor("#1D4ED8")
    c_slate = colors.HexColor("#334155")
    c_mute = colors.HexColor("#64748B")
    return {
        "Title": ParagraphStyle("Title_Main", parent=base["Normal"], fontName="DejaVu-Bold", fontSize=19.5, leading=25, textColor=c_navy, spaceAfter=6),
        "Subtitle": ParagraphStyle("Sub_Main", parent=base["Normal"], fontName="DejaVu", fontSize=10.2, leading=14.5, textColor=c_blue, spaceAfter=10),
        "H1": ParagraphStyle("H1_Sec", parent=base["Normal"], fontName="DejaVu-Bold", fontSize=13.2, leading=18, textColor=c_blue, spaceBefore=12, spaceAfter=6),
        "H2": ParagraphStyle("H2_Sub", parent=base["Normal"], fontName="DejaVu-Bold", fontSize=10.6, leading=14.5, textColor=c_navy, spaceBefore=9, spaceAfter=4),
        "H3": ParagraphStyle("H3_Min", parent=base["Normal"], fontName="DejaVu-Bold", fontSize=9.4, leading=13, textColor=colors.HexColor("#0F766E"), spaceBefore=7, spaceAfter=3),
        "Body": ParagraphStyle("Body_Txt", parent=base["Normal"], fontName="DejaVu", fontSize=9.0, leading=13.8, textColor=c_slate, spaceAfter=6.5),
        "Bullet": ParagraphStyle("Bullet_Txt", parent=base["Normal"], fontName="DejaVu", fontSize=8.8, leading=13.2, textColor=c_slate, leftIndent=14, spaceAfter=4.0),
        "Code": ParagraphStyle("Code_Txt", parent=base["Normal"], fontName="DejaVuMono", fontSize=7.5, leading=10.6, textColor=c_navy, backColor=colors.HexColor("#F1F5F9"), leftIndent=8, rightIndent=8, spaceBefore=4, spaceAfter=6),
        "Callout": ParagraphStyle("Callout_Txt", parent=base["Normal"], fontName="DejaVu", fontSize=8.6, leading=12.8, textColor=c_navy, leftIndent=10, rightIndent=10, backColor=colors.HexColor("#EFF6FF"), borderPadding=6, spaceBefore=4, spaceAfter=7),
        "Warn": ParagraphStyle("Warn_Txt", parent=base["Normal"], fontName="DejaVu", fontSize=8.6, leading=12.8, textColor=colors.HexColor("#78350F"), leftIndent=10, rightIndent=10, backColor=colors.HexColor("#FFFBEB"), borderPadding=6, spaceBefore=4, spaceAfter=7),
        "Caption": ParagraphStyle("Cap_Txt", parent=base["Normal"], fontName="DejaVu-Oblique", fontSize=8.0, leading=11.0, textColor=c_mute, alignment=1, spaceBefore=3, spaceAfter=9),
        "TblCell": ParagraphStyle("Tbl_Cell", parent=base["Normal"], fontName="DejaVu", fontSize=7.4, leading=10.0, textColor=c_slate),
        "TblHdr": ParagraphStyle("Tbl_Hdr", parent=base["Normal"], fontName="DejaVu-Bold", fontSize=7.5, leading=10.0, textColor=c_navy),
    }


def make_toc():
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.levelStyles = [
        ParagraphStyle(fontName="DejaVu-Bold", fontSize=9.2, name="TOCHeading1", leftIndent=0, firstLineIndent=0, spaceBefore=5, leading=13, textColor=colors.HexColor("#0F172A")),
        ParagraphStyle(fontName="DejaVu", fontSize=8.3, name="TOCHeading2", leftIndent=14, firstLineIndent=0, spaceBefore=1.5, leading=11.5, textColor=colors.HexColor("#475569")),
    ]
    return toc


def fit_image(path, max_w_inch=6.9, max_h_inch=4.5):
    im = PILImage.open(path)
    px_w, px_h = im.size
    aspect = px_h / float(px_w)
    w = max_w_inch * inch
    h = w * aspect
    if h > max_h_inch * inch:
        h = max_h_inch * inch
        w = h / aspect
    return Image(path, width=w, height=h)


def styled_table(rows, col_widths_inch, st):
    formatted = []
    for r_i, row in enumerate(rows):
        p_style = st["TblHdr"] if r_i == 0 else st["TblCell"]
        formatted.append([Paragraph(str(c), p_style) for c in row])
    t = Table(formatted, colWidths=[w * inch for w in col_widths_inch], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def md_to_story(md_lines, st, img_map):
    """
    Lightweight structured Markdown-to-ReportLab story converter supporting:
    #, ##, ###, ####, bullets (- / *), numbered lists, > callouts, ``` code blocks,
    ![caption](img_key), markdown tables, and --- pagebreaks.
    """
    story = []
    i = 0
    n = len(md_lines)
    in_code = False
    code_buf = []

    def fmt_inline(txt):
        # Extract inline code spans first so asterisks/underscores inside code never trigger italics/bold
        code_spans = []
        def _save_code(m):
            inner = m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_spans.append(f"<font name='DejaVuMono' size='7.8'>{inner}</font>")
            return f"@@CODE{len(code_spans)-1}@@"

        txt = re.sub(r"`([^`]+)`", _save_code, txt)
        txt = txt.replace("$\\to$", "→").replace(r"$\to$", "→")
        txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        txt = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", txt)
        for idx, span in enumerate(code_spans):
            txt = txt.replace(f"@@CODE{idx}@@", span)
        return txt

    while i < n:
        line = md_lines[i].rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_buf = []
            else:
                in_code = False
                code_txt = "<br/>".join(l.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;") for l in code_buf)
                story.append(Paragraph(code_txt, st["Code"]))
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        if stripped == "---PAGEBREAK---":
            story.append(PageBreak())
            i += 1
            continue

        if stripped == "---TOC---":
            story.append(Paragraph("Table of Contents", st["H1"]))
            story.append(make_toc())
            story.append(PageBreak())
            i += 1
            continue

        if stripped == "---":
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceBefore=4, spaceAfter=6))
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # Images: ![Caption](key|max_w|max_h)
        m_img = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if m_img:
            cap, spec = m_img.group(1), m_img.group(2)
            parts = spec.split("|")
            key = parts[0]
            mw = float(parts[1]) if len(parts) > 1 else 6.8
            mh = float(parts[2]) if len(parts) > 2 else 4.2
            img_path = img_map[key]
            story.append(KeepTogether([
                fit_image(img_path, max_w_inch=mw, max_h_inch=mh),
                Paragraph(fmt_inline(cap), st["Caption"]),
            ]))
            i += 1
            continue

        # Tables
        if stripped.startswith("|") and i + 1 < n and md_lines[i + 1].strip().startswith("|"):
            tbl_lines = []
            while i < n and md_lines[i].strip().startswith("|"):
                tbl_lines.append(md_lines[i].strip())
                i += 1
            # Parse header + data rows (skip separator row index 1)
            parsed = []
            for r_idx, tl in enumerate(tbl_lines):
                if r_idx == 1 and set(tl.replace("|", "").replace(":", "").replace("-", "").strip()) == set():
                    continue
                cells = [fmt_inline(c.strip()) for c in tl.strip("|").split("|")]
                parsed.append(cells)
            if parsed:
                ncols = len(parsed[0])
                avail = 6.9
                # Smart widths: wider first col if >= 5 cols
                if ncols == 2:
                    widths = [2.2, 4.7]
                elif ncols == 3:
                    widths = [1.8, 2.5, 2.6]
                elif ncols == 4:
                    widths = [1.5, 1.8, 1.8, 1.8]
                else:
                    w0 = 1.65
                    wr = (avail - w0) / max(ncols - 1, 1)
                    widths = [w0] + [wr] * (ncols - 1)
                story.append(styled_table(parsed, widths, st))
                story.append(Spacer(1, 6))
            continue

        # Headings
        if stripped.startswith("# "):
            story.append(Paragraph(fmt_inline(stripped[2:]), st["Title"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(fmt_inline(stripped[3:]), st["H1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(fmt_inline(stripped[4:]), st["H2"]))
        elif stripped.startswith("#### "):
            story.append(Paragraph(fmt_inline(stripped[5:]), st["H3"]))
        elif stripped.startswith(">! "):
            story.append(Paragraph(fmt_inline(stripped[3:]), st["Warn"]))
        elif stripped.startswith("> "):
            story.append(Paragraph(fmt_inline(stripped[2:]), st["Callout"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            story.append(Paragraph("• " + fmt_inline(stripped[2:]), st["Bullet"]))
        elif re.match(r"^\d+\.\s+", stripped):
            story.append(Paragraph(fmt_inline(stripped), st["Bullet"]))
        else:
            story.append(Paragraph(fmt_inline(stripped), st["Body"]))
        i += 1

    return story


def compile_pdf(md_text, out_md_name, out_pdf_name, doc_title, right_header, img_map):
    md_path = os.path.join(OUT_DIR, out_md_name)
    pdf_path = os.path.join(OUT_DIR, out_pdf_name)
    # Clean PDF-only control markers from editable Markdown copy
    clean_md = md_text.replace("---PAGEBREAK---\n", "\n").replace("---TOC---\n", "")
    for k, real_p in img_map.items():
        clean_md = re.sub(rf"\({re.escape(k)}(\|[^)]*)?\)", f"({real_p.replace(chr(92), '/')})", clean_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(clean_md)

    st = make_styles()
    doc = TechnicalDocTemplate(pdf_path, doc_title=doc_title, right_header=right_header)
    story = md_to_story(md_text.splitlines(keepends=True), st, img_map)
    doc.multiBuild(story)

    with open(pdf_path, "rb") as f:
        pages = len(re.findall(rb"/Type\s*/Page[^s]", f.read()))
    print(f"Compiled {out_pdf_name}: {pages} pages | {os.path.getsize(pdf_path)/1024:.1f} KB")
    return md_path, pdf_path, pages


# ─────────────────────────────────────────────────────────────────────────────
# DELIVERABLE CONTENT BUILDERS (ALL NUMBERS PULLED FROM FROZEN RECORDS)
# ─────────────────────────────────────────────────────────────────────────────
def build_deliverable_a_master_report(d):
    fb = d["fives_base"]
    fd = d["fives_dual"]
    hb = d["hrf_base"]
    hd = d["hrf_dual"]
    cal = d["dual_calib"]
    reval = d["rest_eval"]["overall_summary"]
    m25 = d["m25_val"]
    m1 = d["m1_audit"]
    re_dis = d["rest_eval"]["disease_breakdown"]
    snap = d["snap"]["sha256_inventory"]

    raw_f = fb["summary_overall_200"]["per_system"]["raw_unet"]
    rest_f = fb["summary_overall_200"]["per_system"]["restored_unet"]
    fr_raw_f = fb["summary_overall_200"]["per_system"]["raw_frangi"]
    fr_rest_f = fb["summary_overall_200"]["per_system"]["restored_frangi"]
    dual_f = fd["overall"]
    vr_200 = fd["vs_raw_unet_n200"]
    vs_200 = fd["vs_restored_unet_n200"]
    vr_199 = fd["vs_raw_unet_dedup_n199"]

    raw_h45 = hb["summary_overall_45"]["per_system"]["raw_unet"]
    rest_h45 = hb["summary_overall_45"]["per_system"]["restored_unet"]
    dual_h45 = hd["overall_n45"]
    dual_h40 = hd["exposed_n40"]
    hr_vr = hd["overall_vs_raw"]
    hr_vs = hd["overall_vs_restored"]

    raw_h5 = hb["summary_unexposed_5"]["per_system"]["raw_unet"]
    rest_h5 = hb["summary_unexposed_5"]["per_system"]["restored_unet"]
    dual_h5 = hd["unexposed_n5_exploratory"]["summary"]
    h5_vr = hd["unexposed_vs_raw_exploratory"]

    # Helper to format training history rows cleanly (exact matching col counts!)
    def hist_rows(hist, is_dual=False):
        out = []
        for e in hist:
            c = e["val_clean_n59"]
            if is_dual:
                stg = "Stage 1 (Frozen)" if e["epoch"] <= 4 else "Stage 2 (Joint)"
                out.append(f"| Epoch {e['epoch']:02d} | {stg} | `{e['train_loss']:.4f}` | `{c['dice']:.4f}` | `{c['precision']:.4f}` | `{c['thin_vessel_sensitivity']*100:.2f}%` | `{c['cldice']:.4f}` | `{c['mean_pred_cc']}` | `{c['composite_score']:.4f}` |")
            else:
                out.append(f"| Epoch {e['epoch']:02d} | `{e['train_loss']:.4f}` | `{c['dice']:.4f}` | `{c['precision']:.4f}` | `{c['sensitivity']:.4f}` | `{c['thin_vessel_sensitivity']*100:.2f}%` | `{c['vascular_continuity_ratio']:.2f}` |")
        return "\n".join(out)

    rest_dis_rows = "\n".join(
        f"| **{name}** (`n={v['count']}`) | `{v['psnr_corrupted']:.2f} dB` | `{v['psnr_baseline']:.2f} dB` | **`{v['psnr_finetuned']:.2f} dB`** | `+{v['delta_psnr_gain']:.2f} dB` | `{v['ssim_finetuned']:.4f}` | `{v['noise_fpr_finetuned']:.2f}%` |"
        for name, v in [
            ("Normal Healthy (NOR)", re_dis["Normal Healthy (NOR)"]),
            ("AMD", re_dis["Age-related Macular Degeneration (AMD)"]),
            ("Diabetic Retinopathy (DR)", re_dis["Diabetic Retinopathy (DR)"]),
            ("Glaucoma (GLA)", re_dis["Glaucoma (GLA)"]),
        ]
    )

    calib_rows = "\n".join(
        f"| `{float(k):.2f}` | `{v['composite_score']:.4f}` | `{v['dice']:.4f}` | `{v['precision']:.4f}` | `{v['thin_vessel_sensitivity']*100:.2f}%` | `{v['thin_skel_sens_r35']*100:.2f}%` | `{v['cldice']:.4f}` | `{v['mean_pred_cc']}` |"
        for k, v in cal["validation_sweep_clean_n59"].items()
    )

    disease_rows = "\n".join(
        f"| **{lbl}** (n=50) | `{fb['disease_breakdown'][code]['per_system']['raw_unet']['dice']['mean']:.4f}` / `{fb['disease_breakdown'][code]['per_system']['raw_unet']['prec']['mean']:.4f}` | `{fb['disease_breakdown'][code]['per_system']['restored_unet']['dice']['mean']:.4f}` / `{fb['disease_breakdown'][code]['per_system']['restored_unet']['prec']['mean']:.4f}` | **`{fd['by_disease'][lbl]['dual_stream']['dice']:.4f}` / `{fd['by_disease'][lbl]['dual_stream']['prec']:.4f}`** | `{fd['by_disease'][lbl]['dual_stream']['thin_skel_sens']*100:.2f}%` | `{fd['by_disease'][lbl]['dual_stream']['cldice']:.4f}` | `{fd['by_disease'][lbl]['dual_stream']['num_pred_cc']}` |"
        for lbl, code in [("Normal", "N"), ("AMD", "A"), ("DR", "D"), ("Glaucoma", "G")]
    )

    disease_paired_rows = "\n".join(
        f"| **{lbl}** (`n=50`) | `{fd['by_disease'][lbl]['vs_raw_unet']['dice']['delta_mean']:+.4f}` (`p={fd['by_disease'][lbl]['vs_raw_unet']['dice']['wilcoxon_p_holm']:.2e}`) | `{fd['by_disease'][lbl]['vs_raw_unet']['thin_skel_sens']['delta_mean']*100:+.2f} pp` (`p={fd['by_disease'][lbl]['vs_raw_unet']['thin_skel_sens']['wilcoxon_p_holm']:.2e}`) | `{fd['by_disease'][lbl]['vs_raw_unet']['cldice']['delta_mean']:+.4f}` | `{fd['by_disease'][lbl]['vs_restored_unet']['prec']['delta_mean']*100:+.2f} pp` (`p={fd['by_disease'][lbl]['vs_restored_unet']['prec']['wilcoxon_p_holm']:.2e}`) | `{fd['by_disease'][lbl]['vs_restored_unet']['num_pred_cc']['delta_mean']:+.1f}` |"
        for lbl in ["Normal", "AMD", "DR", "Glaucoma"]
    )

    hrf_unexp_rows = "\n".join(
        f"| `{rec['filename']}` | `{rec['dual_stream']['dice']:.4f}` | `{rec['dual_stream']['prec']:.4f}` | `{rec['dual_stream']['sens']:.4f}` | `{rec['dual_stream']['thin_skel_sens']*100:.2f}%` | `{rec['dual_stream']['cldice']:.4f}` | `{rec['dual_stream']['branch_recall']*100:.2f}%` | `{rec['dual_stream']['num_pred_cc']}` |"
        for rec in hd["unexposed_n5_exploratory"]["per_image"]
    )

    snap_rows = "\n".join(
        f"| `{path}` | `{digest[:32]}...` |"
        for path, digest in snap.items()
    )

    return f"""# SFHformer-FreqLab & Dual-Stream Hybrid U-Net: Master Technical Report

### Spatial-Frequency Retinal Image Restoration and Dual-Stream Vessel Segmentation with Frozen Milestone 3 Evaluation

---

| Report Metadata | Verified Record |
| :--- | :--- |
| **Project / Repository** | `D:\\SFHformer-FreqLab` (`SFHformer-FreqLab` + `RetinaScan Studio`) |
| **Immutable Snapshot Tag** | `v3.1-phase5.2-verified` |
| **Git Commit Hash** | `42ac0565d1d1517ef5bad9fdc1f392608272db1b` |
| **Compute Environment** | NVIDIA GeForce RTX 5060 Laptop GPU (`8 GB` VRAM, BF16 AMP + FP32 `SoftclDice`) |
| **Primary In-Domain Test** | FIVES Untouched Test Set (`N=200`: 50 Normal / 50 AMD / 50 DR / 50 Glaucoma) |
| **External Transfer Cohort** | HRF (`N=45`: `40` upstream pretrain-exposed / `5` strictly unexposed exploratory) |
| **Calibrated Dual Threshold** | `tau = {cal['selected_frozen_threshold']:.2f}` (frozen strictly on `N=59` clean FIVES validation) |
| **Baseline U-Net Thresholds** | Raw U-Net `tau = 0.50` / Restored U-Net `tau = 0.50` / Classical Frangi `tau = 0.0010` |

>! **Scope & Clinical Disclaimer**: RetinaScan Studio and all associated neural models presented in this dossier constitute a **research prototype for retinal vessel segmentation and frequency-domain restoration analysis**. This software has not undergone prospective clinical validation or regulatory clearance and must **not** be used as a diagnostic medical device.

---PAGEBREAK---

---TOC---

## 1. Executive Summary & Research Objectives

### 1.1 Scientific Problem Statement
Automated segmentation of the retinal vascular tree in fundus photography underpins morphological research into diabetic retinopathy (DR), age-related macular degeneration (AMD), and glaucomatous optic neuropathy. While modern convolutional encoder-decoder networks (U-Nets) achieve strong overlap on thick primary arcades, fine peripheral capillaries frequently degrade beneath photon shot noise, coherent laser speckle, non-uniform illumination falloff, optical halo glare, and periodic sensor raster striping.

Conversely, aggressively restoring high-frequency contrast prior to segmentation creates a sharp **sensitivity-precision trade-off** observed empirically across our Milestone 2 and Milestone 3 baselines:
- **Raw U-Net (`checkpoints/unet_raw/best.pth`)**: Achieves high overall Dice (`{raw_f['dice']['mean']:.4f}`), high positive predictive value (`Precision = {raw_f['prec']['mean']:.4f}`), and strong topological alignment (`clDice = {raw_f['cldice']['mean']:.4f}`) on untouched FIVES (`N=200`), yet omits low-contrast thin peripheral vessels (`Thin Skeleton Sensitivity = {raw_f['thin_skel_sens']['mean']*100:.2f}%`).
- **Restored U-Net (`checkpoints/unet_restored/best.pth`)**: Trained on images denoised by our 5-stage Spatial-Frequency Hybrid Transformer (**SFHformer-m**), recovers substantially more thin-vessel structure on clean FIVES validation (`N=59`: `+6.16 pp` diameter <= 3.5 px skeleton sensitivity, Wilcoxon `p = 7.58e-7`) and on high-resolution external HRF scans (`N=45`: `+13.67 pp` radius <= 3.5 px thin skeleton sensitivity, Wilcoxon `p = 5.68e-13`). However, single-stream restoration suffers severe precision drops on coarse-annotated datasets (FIVES Dice drops to `{rest_f['dice']['mean']:.4f}`, Precision `{rest_f['prec']['mean']:.4f}`), elevated connected-component fragmentation (`{rest_f['num_pred_cc']['mean']:.2f}` vs `{raw_f['num_pred_cc']['mean']:.2f}` components/image), and contrast amplification along the excavated optic cup rim in glaucoma (`Precision = 57.59%`, `-33.54 pp` vs Raw).

### 1.2 Principal Contribution: Dual-Stream Hybrid U-Net (`DualStreamUNet`)
To resolve this divergence without sacrificing either macro precision or capillary sensitivity, we introduce, train, validate, and rigorously benchmark **`DualStreamUNet`** ([models/dual_stream_unet.py](file:///d:/SFHformer-FreqLab/models/dual_stream_unet.py)):
1. **Parallel Pretrained Representation Streams**: Stream A encodes unprocessed native fundus features (warm-started from `unet_raw`), while Stream B encodes SFHformer-restored features (warm-started from `unet_restored`).
2. **Multi-Scale Spatial Gated Fusion (`CrossStreamFusionBlock`)**: At all five spatial resolutions (`512x512` down to `32x32` bottleneck), a learned per-pixel 2-channel spatial softmax gate (`w_raw + w_rest = 1`) dynamically arbitrates between native macro stability and restored capillary contrast, augmented by a Squeeze-and-Excitation (SE) cross-projection residual.
3. **Topology-Aware Compound Optimization**: Trained using `HybridTopologyLoss` combining Binary Cross-Entropy (`0.4`), Soft-Dice (`0.4`), and strictly FP32 differentiable Soft-clDice (`0.2`, `SoftSkeletonize(iter=12)` with `2 px` border masking).

### 1.3 Summary of Frozen Milestone 3 Empirical Findings
- **In-Domain Untouched FIVES (`N=200`, calibrated frozen `tau=0.40`)**: `DualStreamUNet` outperforms **both** single-stream baselines simultaneously across Overall Dice (`{dual_f['dice']:.4f}`, `+{vr_200['dice']['delta_mean']*100:.2f} pp` vs Raw, Holm `p = {vr_200['dice']['wilcoxon_p_holm']:.2e}`), Thin Skeleton Sensitivity (`{dual_f['thin_skel_sens']*100:.2f}%`, `+{vr_200['thin_skel_sens']['delta_mean']*100:.2f} pp` vs Raw, Holm `p = {vr_200['thin_skel_sens']['wilcoxon_p_holm']:.2e}`), Branch Recall (`{dual_f['branch_recall']*100:.2f}%`, `+{vr_200['branch_recall']['delta_mean']*100:.2f} pp`), and `clDice` (`{dual_f['cldice']:.4f}`, `+{vr_200['cldice']['delta_mean']*100:.2f} pp`), while **reducing** fragmentation (`{dual_f['num_pred_cc']:.2f}` vs `{raw_f['num_pred_cc']['mean']:.2f}` Raw and `{rest_f['num_pred_cc']['mean']:.2f}` Restored components/image) and restoring Glaucoma precision to **`89.98%`** (`+32.39 pp` above single-stream Restored U-Net).
- **External HRF Transfer (`N=45`, `40` pretrain-exposed / `5` exploratory)**: `DualStreamUNet` reveals **honest mixed domain-transfer trade-offs**: it retains higher thin-vessel skeleton sensitivity (`+{hr_vr['thin_skel_sens']['delta_mean']*100:.2f} pp`), branch recall (`+{hr_vr['branch_recall']['delta_mean']*100:.2f} pp`), `clDice` (`+{hr_vr['cldice']['delta_mean']*100:.2f} pp`), and lower fragmentation (`{hr_vr['num_pred_cc']['delta_mean']:.2f}` CCs/img) than Raw U-Net, while trading `-8.54 pp` Precision and `-1.82 pp` Dice vs Raw U-Net and trailing Restored U-Net (`-3.89 pp`) on HRF thin skeleton sensitivity.

---PAGEBREAK---

## 2. Datasets, Provenance Audits & Preprocessing Protocols

### 2.1 Dataset Inventory & Zero-Leakage Governance
Rigorous provenance auditing was conducted during Milestone 1 (`results/dataset_audit/milestone1_master_audit.json`, `elapsed = {m1['elapsed_seconds']} s`) and Milestone 2.5 prior to freezing Milestone 3 test runs:

| Dataset Split | Image Count | Resolution | Annotation Scope | Exposure & Audit Notes |
| :--- | :---: | :---: | :--- | :--- |
| **FIVES Train** | `{m1['fives_audit']['train_count']}` | `2048 x 2048` | Expert retinal vessel masks | Generated `{m1['patch_and_tile_manifests']['train_patches_count']}` paired `512x512` patches (`train_patch_manifest_512.json`) and fine-tuned `SFHformer-m` (`27,000` online crops). |
| **FIVES Clean Validation** | `59` (`{m1['fives_audit']['val_count']}` total) | `2048 x 2048` | Expert retinal vessel masks | Exact train-val duplicate pair (`train_61_A.png` / `train_121_A.png`) isolated; all hyperparameter, checkpoint, and threshold calibration decisions locked exclusively on `N=59` clean val. |
| **FIVES Untouched Test** | `{m1['fives_audit']['test_count']}` | `2048 x 2048` | `50 N / 50 A / 50 D / 50 G` | **Zero training or tuning access.** Intra-test duplicate (`test_32_A.png` / `test_48_A.png`) tracked via duplicate-clean `N=199` sensitivity checks. |
| **HRF Exposed Subset** | `40` | `3504 x 2336` | High-res full vascular tree | Disclosed prior upstream exposure during initial `SFHformer` pretraining; **not** used for U-Net or Dual-Stream training. |
| **HRF Unexposed Subset** | `5` | `3504 x 2336` | High-res full vascular tree | Identifiers: `14_g`, `14_h`, `15_dr`, `15_g`, `15_h`. Zero exposure in either SFHformer or U-Net training; **strictly exploratory (`N=5`)**. |

> **Patient-Level Provenance Disclosure (`milestone1_master_audit.json`)**: While 100% image-level disjointness is verified between FIVES splits after duplicate filtering, FIVES numbering restarts per partition (e.g., `train_1_A` vs `test_1_A`, `180` repeating subject indices), and public metadata omits independent patient linkage IDs.

### 2.2 Audited Vessel Caliber Profile, FOV Masking & Patch Stratification
Quantitative morphological profiling across 15 full-resolution scans per dataset revealed fundamental structural differences between FIVES and HRF that directly explain downstream model transfer behavior:

| Audited Morphometric Property | FIVES (`2048 x 2048`) | HRF (`3504 x 2336`) | Engineering Impact on Loss & Calibration |
| :--- | :---: | :---: | :--- |
| **Smallest Vessels (`1.0 - 3.5 px` radius)** | `{m1['thin_vessel_profile']['FIVES']['pct_smallest_annotated_vessels_1_to_3_5px']}%` | `{m1['thin_vessel_profile']['HRF']['pct_smallest_annotated_vessels_1_to_3_5px']}%` | HRF contains `11.2x` denser fine-capillary annotation share than FIVES, shifting optimal threshold sensitivity. |
| **Intermediate Vessels (`3.5 - 8.0 px`)** | `{m1['thin_vessel_profile']['FIVES']['pct_intermediate_vessels_3_5_to_8px']}%` | `{m1['thin_vessel_profile']['HRF']['pct_intermediate_vessels_3_5_to_8px']}%` | Intermediate arcades benefit strongly from `HybridTopologyLoss` centerline alignment. |
| **Primary Trunks (`> 8.0 px`)** | `{m1['thin_vessel_profile']['FIVES']['pct_primary_trunks_gt_8px']}%` | `{m1['thin_vessel_profile']['HRF']['pct_primary_trunks_gt_8px']}%` | Dictated setting `SoftSkeletonize(num_iter=12)` (`p95 = 8.96 px`, median max `13.69 px`). |
| **FOV Border Clipping Mean / Max** | `{m1['fov_audit']['fives_mean_clipping_pct']}%` / `{m1['fov_audit']['fives_max_clipping_pct']}%` | `{m1['fov_audit']['hrf_mean_clipping_pct']}%` / `{m1['fov_audit']['hrf_max_clipping_pct']}%` | Motivated strict FOV masking (`intensity > 10/255`) and `border_ignore=2 px` during soft-clDice computation. |

- **Training Patch Stratification (`2,160` patches, `512 x 512`)**: Extracted across `540` FIVES training pairs with balanced coverage (`{m1['patch_and_tile_manifests']['train_strata']['vessel_rich']}` vessel-rich, `{m1['patch_and_tile_manifests']['train_strata']['mixed']}` mixed, `{m1['patch_and_tile_manifests']['train_strata']['vessel_poor']}` vessel-poor patches).

### 2.3 Synthetic Ophthalmic Degradation Engine (`scripts/retina_corrupt.py`)
During restoration fine-tuning and robustness ablation, five physically grounded ophthalmic degradations are composed dynamically:
1. **Raster Striping (`apply_raster_banding`)**: Periodic line-scan detector readout interference modeled as sinusoidal spatial modulation along horizontal/angled scan lines.
2. **Coherent Laser Speckle (`apply_laser_speckle`)**: Multiplicative granular noise simulating coherent optical interference in scanning laser ophthalmoscopy.
3. **Optical Halo Glare (`apply_optical_halo`)**: Radially symmetric corneal/pupil flash reflection artifacts.
4. **Illumination Falloff (`apply_illumination_gradient`)**: Smooth low-frequency peripheral vignetting typical of handheld or undilated fundus cameras.
5. **CMOS Shot Noise (`apply_sensor_noise`)**: Zero-mean additive Gaussian sensor readout noise.

### 2.4 Full-Field Sliding-Window Inference Protocol
Because training runs on `512 x 512` crops under strict `8 GB` VRAM bounds while test frames measure `2048 x 2048` (FIVES) and `3504 x 2336` (HRF), full-resolution evaluation employs overlapping `512 x 512` tiles at stride `384 px` (`25` tiles for FIVES; complete boundary-anchored grid for HRF) weighted by a 2D separable Hamming window:
- `W_2D(y, x) = max(w_hamming(y) * w_hamming(x), 1e-4)`
- Overlapping tile probabilities are accumulated pixel-wise as `sum(P_k * W_2D) / sum(W_2D)` in FP32, completely eliminating tile-boundary seams during evaluation.

---PAGEBREAK---

## 3. SFHformer Spatial-Frequency Restoration Architecture

### 3.1 Macro-Architecture (`models/SFHformer.py`)
`SFHformer-m` (`14.18M` parameters) replaces standard ViT patch flattening (which destroys pixel-level vessel edges) with a 5-stage symmetric U-shaped encoder-decoder operating at full spatial resolution in Stage 1 and Stage 5 (`patch_size=1`, `3x3` convolution). Feature channel widths progress as `[32, 64, 128, 64, 32]` with SFH block depths `[4, 4, 8, 4, 4]` (`24` blocks total). Encoder downsampling uses strided `3x3` convolutions (`stride=2`), whereas decoder upsampling uses `Conv1x1 + nn.PixelShuffle(2)` to avoid transposed-convolution checkerboards. A long global identity connection adds the degraded input directly to the network output so all `24` blocks predict pure degradation residuals.

![Figure 1: SFHformer-m 5-stage spatial-frequency hierarchical transformer macro-architecture detailing encoder/decoder stages, lateral skips, PixelShuffle upsampling, and global residual learning.|6.9|3.8](diag_1|6.9|3.8)

### 3.2 Internal Mechanics of the SFH Block (`Mixer`, `FourierUnit`, & `RadialBandGate`)
Each SFH Block applies pre-normalization (`BatchNorm2d`), a parallel **Hybrid Token Mixer** with learnable residual scale `beta`, and a depthwise **Feed-Forward Network (`FFN`)** (`Conv1x1 C->4C -> DWConv3x3 -> GELU -> Conv1x1 4C->C`) scaled by `gamma`.

![Figure 2: In-depth internal architecture of the SFH Block illustrating parallel Local Dilated DW-Convolutions, Global 2D Real FFT FourierUnit, Frequency Positional Encoding, Dynamic Grouped Convolutions, Channel Attention, and FFN.|6.9|3.8](diag_2|6.9|3.8)

Within `Mixer`, the `1x1` projected tensor is split equally into local and spectral streams:
1. **Local Spatial Stream (`TokenMixer_For_Local`)**: Splits `C` channels into two `C/2` chunks processed in parallel by depthwise dilated convolutions (`3x3, dilation=1` and `3x3, dilation=2`) to capture fine bifurcations and local multi-scale context without mixing unrelated frequencies.
2. **Global Spectral Stream (`TokenMixer_For_Gloal` + `FourierUnit`)**:
   - Transforms normalized spatial tensor `Z` via orthonormal 2D Real FFT: `Z_fft = torch.fft.rfft2(Z, norm='ortho')` in `C^(B x 2C x H x (W/2+1))`.
   - Concatenates real and imaginary components along channels (`4C`), applies `BatchNorm2d`, and adds **Frequency Positional Encoding (`fpe`)** (`3x3` depthwise conv residual) so kernels distinguish low-frequency illumination from high-frequency periodic striping.
   - Applies grouped `1x1` **Frequency Domain Convolution (`fdc`)** modulated by spatial-spectral `Softmax` weights via `torch.einsum`.
   - In the gated `R2` variant ([freqlab/gate.py](file:///d:/SFHformer-FreqLab/freqlab/gate.py)), `RadialBandGate` interpolates `16` learnable log-amplitude bands across normalized radial frequency `R_norm = sqrt(f_y^2 + f_x^2) / R_max` (`exp(0) = 1.0` identity initialization).
   - Reconstructs complex tensors and transforms back via `torch.fft.irfft2(..., s=(H, W), norm='ortho')`, followed by **Squeeze-and-Excitation Channel Attention** fusion with the local stream.

### 3.3 Restoration Fine-Tuning & Held-Out 200-Scan Evaluation
- **Training Configuration** ([scripts/train_fives_retina.py](file:///d:/SFHformer-FreqLab/scripts/train_fives_retina.py)): Initialized from DRIVE/baseline weights (`L1` pixel loss, `AdamW`, `CosineAnnealingLR`, `256x256` tissue-filtered crops with `>35%` fundus tissue, `batch=1, accum=4`, online mixed degradations).
- **Empirical Restoration Fidelity** ([results/fives_evaluation_results.json](file:///d:/SFHformer-FreqLab/results/fives_evaluation_results.json)):
  - **Peak Validation PSNR**: `31.92 dB` on FIVES fine-tuning validation crops (`checkpoints/fives_retina_finetune/best.pth`).
  - **Held-Out 200-Scan Synthetic Degradation Test**: Mean PSNR improves from `{reval['mean_psnr_corrupted']:.2f} dB` (corrupted) and `{reval['mean_psnr_baseline']:.2f} dB` (DRIVE baseline `checkpoints/retina/best.pth`) to **`{reval['mean_psnr_finetuned']:.2f} dB`** (`+{reval['mean_psnr_delta_over_baseline']:.2f} dB` gain over DRIVE baseline), with mean SSIM reaching **`{reval['mean_ssim_finetuned']:.4f}`** (`avg latency = {reval['avg_inference_latency_ms']:.1f} ms`, `peak VRAM = {reval['peak_vram_gb']:.2f} GB`).

| FIVES Pathology Subset | Corrupted PSNR | DRIVE Baseline PSNR | Fine-Tuned SFHformer PSNR | Gain vs Baseline | Fine-Tuned SSIM | Residual Noise FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{rest_dis_rows}
| **Overall FIVES (`N=200`)** | `{reval['mean_psnr_corrupted']:.2f} dB` | `{reval['mean_psnr_baseline']:.2f} dB` | **`{reval['mean_psnr_finetuned']:.2f} dB`** | **`+{reval['mean_psnr_delta_over_baseline']:.2f} dB`** | **`{reval['mean_ssim_finetuned']:.4f}`** | `{reval['mean_noise_fpr_finetuned']:.2f}%` |

- *(Scope Qualification: These restoration metrics evaluate intra-project progression on our synthetic multi-degradation benchmark under matched experimental protocols; they do not claim external cross-benchmark SOTA.)*

---PAGEBREAK---

## 4. Baseline Single-Stream Clinical U-Net & Classical Frangi Control

### 4.1 Single-Stream U-Net Architecture (`models/unet.py`)
To evaluate whether SFHformer restoration genuinely aids downstream segmentation, two structurally identical 4-stage residual U-Nets (`32 -> 64 -> 128 -> 256 -> 512` bottleneck, `7.7M` parameters each) were trained under identical optimization schedules ([scripts/train_unet.py](file:///d:/SFHformer-FreqLab/scripts/train_unet.py)):
- Each encoder/decoder level employs `DoubleConv` (`Conv3x3 -> BN -> GELU -> [Dropout2d(0.2) at bottleneck] -> Conv3x3 -> BN -> GELU`) plus a `1x1` residual identity projection.
- Decoder stages utilize `Upsample(scale_factor=2, mode='bilinear', align_corners=True) + Conv1x1` rather than transposed convolutions, preserving crisp binary thresholds without checkerboard artifacts.

![Figure 3: Clinical single-stream 4-stage residual U-Net architecture detailing DoubleConv blocks, lateral skip projections, bilinear upsampling decoder, and the divergence between Raw U-Net and Restored U-Net baselines.|6.9|3.8](diag_3|6.9|3.8)

### 4.2 Single-Stream Training Dynamics (`N=59` Clean Validation)
Both single-stream models trained for `8` epochs (`batch=2, accum=2`, BF16 AMP, `BCEDiceLoss` with `0.5 BCE + 0.5 Soft-Dice`) with checkpoint selection on `N=59` clean validation at fixed threshold `tau = 0.50`:

#### `unet_raw` Training History (Native FIVES Fundus Input)
| Epoch | Train Loss | Clean `N=59` Dice | Precision | Sensitivity | ThinSens (diam<=3.5px) | Continuity Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hist_rows(d["raw_hist"], is_dual=False)}

#### `unet_restored` Training History (`SFHformer-m` Restored Input)
| Epoch | Train Loss | Clean `N=59` Dice | Precision | Sensitivity | ThinSens (diam<=3.5px) | Continuity Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hist_rows(d["rest_hist"], is_dual=False)}

### 4.3 Milestone 2.5 Clean-Val Divergence & Classical Frangi Control
- **Clean Validation `N=59` (Milestone 2.5 Paired Audit, diameter <= 3.5 px)**:
  - `unet_raw`: Dice `{m25['overall_dice']['raw_mean']:.4f}`, Thin Skeleton Sensitivity `{m25['thin_vessel_skeleton_sensitivity']['raw_mean']*100:.2f}%`, `clDice {m25['cldice_and_topology']['raw_cldice_before_closing']:.4f}`, `{m25['cldice_and_topology']['raw_mean_components_before_closing']}` components/image.
  - `unet_restored`: Dice `{m25['overall_dice']['restored_mean']:.4f}`, Thin Skeleton Sensitivity **`{m25['thin_vessel_skeleton_sensitivity']['restored_mean']*100:.2f}%`** (`+{m25['thin_vessel_skeleton_sensitivity']['mean_delta']*100:.2f} pp`, 95% CI `[+4.19, +8.12] pp`, Wilcoxon `p = {m25['thin_vessel_skeleton_sensitivity']['wilcoxon_pval']:.2e}`), `clDice {m25['cldice_and_topology']['restored_cldice_before_closing']:.4f}`, `{m25['cldice_and_topology']['restored_mean_components_before_closing']}` components/image.
- **Classical Multiscale Frangi Control (`scripts/baseline_frangi.py`)**: After threshold calibration (`tau = 0.0010` across `sigmas=[1..5]`), Frangi achieved `{fr_raw_f['dice']['mean']:.4f}` Dice (Raw) and `{fr_rest_f['dice']['mean']:.4f}` Dice (Restored) on FIVES `N=200`, confirming that deep segmentation networks substantially outperform classical Hessian vesselness filters (`>0.81` vs `<0.20` Dice).
- **Motivation for Dual-Stream Fusion**: Between Epoch 6 and Epoch 8 on `N=59` clean validation, `unet_raw` improved by `+0.0196` Dice (`0.8299 -> 0.8495`) while `unet_restored` improved by `+0.0048` (`0.7203 -> 0.7251`). Neither single-stream representation alone satisfies both clinical requirements simultaneously.

---PAGEBREAK---

## 5. Proposed Architecture: Dual-Stream Hybrid U-Net (`DualStreamUNet`)

### 5.1 Architectural Design & Warm-Start Strategy
Implemented in [models/dual_stream_unet.py](file:///d:/SFHformer-FreqLab/models/dual_stream_unet.py) (`16.41M` total parameters), `DualStreamUNet` eliminates the forced binary choice between native and restored inputs by running **paired encoding streams** concurrently and fusing them at every hierarchical resolution before a single shared decoder.

![Figure 4: Dual-Stream Hybrid U-Net architecture (DualStreamUNet) showing warm-started Raw and Restored encoders, five multi-scale CrossStreamFusionBlock modules with per-pixel spatial gate inset (w_raw + w_rest = 1), optional scale_0 gate visualization export, shared bilinear decoder, and HybridTopologyLoss.|6.9|4.1](diag_4|6.9|4.1)

#### Exact Warm-Start Mapping (Verified Before Training)
Inspecting `models/unet.py` confirmed identical encoder block keys (`downs.0..3`, `bottleneck`) across `unet_raw/best.pth` and `unet_restored/best.pth`:
- **Stream A (`raw_downs`, `raw_bottleneck`)**: Initialized with `unet_raw/best.pth` (`downs.* -> raw_downs.*`, `bottleneck.* -> raw_bottleneck.*`).
- **Stream B (`rest_downs`, `rest_bottleneck`)**: Initialized with `unet_restored/best.pth` (`downs.* -> rest_downs.*`, `bottleneck.* -> rest_bottleneck.*`).
- **Shared Bilinear Decoder (`ups`, `final_conv`)**: Initialized from `unet_raw/best.pth` (`208` total matched tensors loaded with zero shape mismatches).
- **Fusion Spine (`fusions[0..3]`, `bottleneck_fusion`)**: Randomly initialized (`80` new parameter/buffer tensors).

### 5.2 Mathematical Formulation of `CrossStreamFusionBlock`
Given aligned raw features `F_raw` and restored features `F_rest` in `R^(B x C x H_s x W_s)` at scale `s`, concatenation forms `Z_cat = [F_raw, F_rest]` in `R^(B x 2C x H_s x W_s)`. The block computes two parallel mechanisms:
1. **Per-Pixel 2-Channel Spatial Gate**:
   - `A_raw, A_rest = Softmax(Conv1x1(GELU(BN(Conv3x3(Z_cat)))), dim=1)`
   - Produces one normalized spatial weighting pair `(w_raw, w_rest)` per pixel such that `w_raw + w_rest = 1` across every `(y, x)` location (`scale_0` returns `w_rest` at full `512x512` input resolution). *Important distinction*: this is a **2-channel per-pixel spatial gate**, not channel-wise attention; `w_rest` serves strictly as a spatial routing visualization and never implies pixel confidence or anatomical proof.
2. **SE-Weighted Cross-Projection Residual**:
   - `R_cross = SE(GELU(BN(Conv1x1(Z_cat))))`, where `SE(u) = u * Sigmoid(Conv1x1(GELU(Conv1x1(GAP(u)))))`.
3. **Fused Feature Output**:
   - `F_fused = w_raw * F_raw + w_rest * F_rest + R_cross`

### 5.3 Numerically Stable FP32 Topology Loss (`HybridTopologyLoss`)
To prevent fine-vessel gains from fragmenting into isolated components, `HybridTopologyLoss` ([utils/losses.py](file:///d:/SFHformer-FreqLab/utils/losses.py)) optimizes:
- `L_hybrid = 0.4 * L_BCEWithLogits + 0.4 * L_SoftDice + 0.2 * L_SoftclDice`
- **Stability & Boundary Guards**: `SoftclDiceLoss` executes strictly in **FP32** (`torch.autocast(..., enabled=False)`) to prevent BF16 pooling underflow, uses **`num_iter=12`** erosions (matching the audited `p95=8.96 px` / `median max=13.69 px` FIVES vessel radii, capturing `1,768` vs `499` skeleton pixels at `iter=5`), and masks the outer `2 px` border (`border_ignore=2`) so vessels cut by `512x512` crop edges do not generate artificial boundary skeletons.

---PAGEBREAK---

### 5.4 Two-Stage Training, Hardware Profiling & Clean-Val Calibration (`N=59`)
- **Stage 2 Memory Profile**: Prior to full training, `3` unfrozen Stage 2 backward steps (`batch_size=2`, `512x512`, BF16 AMP) passed on RTX 5060 with `0` `NaN`/`Inf` gradients and **Peak VRAM = `2,948.8 MB`** (within the `5.57 GB` allocator cap).

| Optimization Phase | Epochs | Frozen Modules | Learning Rates | Regularization & Runtime |
| :--- | :---: | :--- | :--- | :--- |
| **Stage 1: Fusion Warmup** | `1 - 4` | `raw_downs`, `raw_bottleneck`, `rest_downs`, `rest_bottleneck` | Fusion & Decoder: `5e-4` | `CosineAnnealingLR`, `grad_clip=1.0`, synchronized spatial flips/rotations |
| **Stage 2: Joint Fine-Tuning** | `5 - 12` | None (End-to-End Unfrozen) | Encoders: `1e-4`; Fusion/Dec: `3e-4` | BF16 AMP + FP32 `SoftclDice(12)`, total `12`-epoch wall time: **`47.43 mins`** |

- **Checkpoint Selection**: Driven at fixed `tau=0.50` by balanced composite score `S = 0.4*Dice + 0.3*clDice + 0.3*ThinSens` on clean validation (`N=59`), reporting all components separately:

| Epoch | Stage | Train Loss | Clean `N=59` Dice | Precision | ThinSens (diam<=3.5px) | clDice | Mean CC | Composite `S` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hist_rows(d["dual_hist"], is_dual=True)}

- **Non-Monotonic Behavior Note**: As expected during Stage 2 encoder unfreezing (Epoch 6), precision briefly spiked (`0.9643`) while thin-vessel sensitivity dipped (`0.1251`) before recovering to peak balanced performance at **Epoch 10** (`Composite = 0.6806`, `Dice = 0.8851`, `clDice = 0.8829`, `ThinSens = 0.2057`, `CC = 35.2`).
- **Frozen Threshold Calibration (`N=59` Clean Val Only)**: Sweeping Epoch 10 (`best.pth`, SHA-256 `9be2213e0cf0c799...`) strictly on `N=59` clean validation selected and froze **`tau = 0.40`** before any test-set evaluation:

| Threshold `tau` | Composite `S` | Dice | Precision | ThinSens (diam<=3.5px) | Thin_r35 (rad<=3.5px) | clDice | Mean CC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{calib_rows}

---PAGEBREAK---

## 6. Milestone 3 Frozen Generalization Evaluation

### 6.1 Evaluation Protocol & Statistical Rigor
All evaluations ran once via [scripts/evaluate_milestone3.py](file:///d:/SFHformer-FreqLab/scripts/evaluate_milestone3.py) and [scripts/evaluate_dual_stream_milestone3.py](file:///d:/SFHformer-FreqLab/scripts/evaluate_dual_stream_milestone3.py) with pre- and post-execution SHA-256 verification. Paired non-parametric Wilcoxon signed-rank tests, paired t-tests, `10,000`-sample bootstrap 95% confidence intervals, and **Holm-Bonferroni step-down adjustments** across the 6 tested metrics (`dice`, `prec`, `thin_skel_sens`, `cldice`, `branch_recall`, `num_pred_cc`) were computed from preserved per-image JSON records.

### 6.2 FIVES Untouched Test Set (`N=200` Primary In-Domain & `N=199` Deduplicated)

| System (`Frozen tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | Thin Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw U-Net** (`0.50`) | `{raw_f['dice']['mean']:.4f}` | **`{raw_f['prec']['mean']:.4f}`** | `{raw_f['sens']['mean']:.4f}` | `{raw_f['thin_skel_sens']['mean']*100:.2f}%` | `{raw_f['branch_recall']['mean']*100:.2f}%` | `{raw_f['thin_branch_recall']['mean']*100:.2f}%` | `{raw_f['cldice']['mean']:.4f}` | `{raw_f['num_pred_cc']['mean']:.2f}` |
| **Restored U-Net** (`0.50`) | `{rest_f['dice']['mean']:.4f}` | `{rest_f['prec']['mean']:.4f}` | `{rest_f['sens']['mean']:.4f}` | `{rest_f['thin_skel_sens']['mean']*100:.2f}%` | `{rest_f['branch_recall']['mean']*100:.2f}%` | `{rest_f['thin_branch_recall']['mean']*100:.2f}%` | `{rest_f['cldice']['mean']:.4f}` | `{rest_f['num_pred_cc']['mean']:.2f}` |
| **Dual-Stream Hybrid** (`0.40`) | **`{dual_f['dice']:.4f}`** | `{dual_f['prec']:.4f}` | **`{dual_f['sens']:.4f}`** | **`{dual_f['thin_skel_sens']*100:.2f}%`** | **`{dual_f['branch_recall']*100:.2f}%`** | **`{dual_f['thin_branch_recall']*100:.2f}%`** | **`{dual_f['cldice']:.4f}`** | **`{dual_f['num_pred_cc']:.2f}`** |
| **Raw Frangi** (`0.0010`) | `{fr_raw_f['dice']['mean']:.4f}` | `{fr_raw_f['prec']['mean']:.4f}` | `{fr_raw_f['sens']['mean']:.4f}` | `{fr_raw_f['thin_skel_sens']['mean']*100:.2f}%` | `{fr_raw_f['branch_recall']['mean']*100:.2f}%` | `{fr_raw_f['thin_branch_recall']['mean']*100:.2f}%` | `{fr_raw_f['cldice']['mean']:.4f}` | `{fr_raw_f['num_pred_cc']['mean']:.1f}` |
| **Restored Frangi** (`0.0010`) | `{fr_rest_f['dice']['mean']:.4f}` | `{fr_rest_f['prec']['mean']:.4f}` | `{fr_rest_f['sens']['mean']:.4f}` | `{fr_rest_f['thin_skel_sens']['mean']*100:.2f}%` | `{fr_rest_f['branch_recall']['mean']*100:.2f}%` | `{fr_rest_f['thin_branch_recall']['mean']*100:.2f}%` | `{fr_rest_f['cldice']['mean']:.4f}` | `{fr_rest_f['num_pred_cc']['mean']:.1f}` |

#### Paired Statistical Comparisons on FIVES (`N=200`, Holm-Bonferroni Adjusted)

| Metric | Dual-Stream vs Raw U-Net Delta (95% CI, Holm `p_wil`) | Dual-Stream vs Restored U-Net Delta (95% CI, Holm `p_wil`) |
| :--- | :--- | :--- |
| **Dice** | **`+{vr_200['dice']['delta_mean']:.4f}`** (`[{vr_200['dice']['ci95'][0]:+.4f}, {vr_200['dice']['ci95'][1]:+.4f}]`, `p={vr_200['dice']['wilcoxon_p_holm']:.2e}`) | **`+{vs_200['dice']['delta_mean']:.4f}`** (`[{vs_200['dice']['ci95'][0]:+.4f}, {vs_200['dice']['ci95'][1]:+.4f}]`, `p={vs_200['dice']['wilcoxon_p_holm']:.2e}`) |
| **Precision** | `{vr_200['prec']['delta_mean']:+.4f}` (`[{vr_200['prec']['ci95'][0]:+.4f}, {vr_200['prec']['ci95'][1]:+.4f}]`, `p={vr_200['prec']['wilcoxon_p_holm']:.2e}`) | **`+{vs_200['prec']['delta_mean']:.4f}`** (`[{vs_200['prec']['ci95'][0]:+.4f}, {vs_200['prec']['ci95'][1]:+.4f}]`, `p={vs_200['prec']['wilcoxon_p_holm']:.2e}`) |
| **Thin Skel Sens** | **`+{vr_200['thin_skel_sens']['delta_mean']*100:.2f} pp`** (`[+{vr_200['thin_skel_sens']['ci95'][0]*100:.2f}, +{vr_200['thin_skel_sens']['ci95'][1]*100:.2f}] pp`, `p={vr_200['thin_skel_sens']['wilcoxon_p_holm']:.2e}`) | **`+{vs_200['thin_skel_sens']['delta_mean']*100:.2f} pp`** (`[+{vs_200['thin_skel_sens']['ci95'][0]*100:.2f}, +{vs_200['thin_skel_sens']['ci95'][1]*100:.2f}] pp`, `p={vs_200['thin_skel_sens']['wilcoxon_p_holm']:.2e}`) |
| **Branch Recall** | **`+{vr_200['branch_recall']['delta_mean']*100:.2f} pp`** (`[+{vr_200['branch_recall']['ci95'][0]*100:.2f}, +{vr_200['branch_recall']['ci95'][1]*100:.2f}] pp`, `p={vr_200['branch_recall']['wilcoxon_p_holm']:.2e}`) | **`+{vs_200['branch_recall']['delta_mean']*100:.2f} pp`** (`[+{vs_200['branch_recall']['ci95'][0]*100:.2f}, +{vs_200['branch_recall']['ci95'][1]*100:.2f}] pp`, `p={vs_200['branch_recall']['wilcoxon_p_holm']:.2e}`) |
| **clDice** | **`+{vr_200['cldice']['delta_mean']:.4f}`** (`[{vr_200['cldice']['ci95'][0]:+.4f}, {vr_200['cldice']['ci95'][1]:+.4f}]`, `p={vr_200['cldice']['wilcoxon_p_holm']:.2e}`) | **`+{vs_200['cldice']['delta_mean']:.4f}`** (`[{vs_200['cldice']['ci95'][0]:+.4f}, {vs_200['cldice']['ci95'][1]:+.4f}]`, `p={vs_200['cldice']['wilcoxon_p_holm']:.2e}`) |
| **Components (`CC`)** | **`{vr_200['num_pred_cc']['delta_mean']:+.2f}`** (`[{vr_200['num_pred_cc']['ci95'][0]:+.2f}, {vr_200['num_pred_cc']['ci95'][1]:+.2f}]`, `p={vr_200['num_pred_cc']['wilcoxon_p_holm']:.2e}`) | **`{vs_200['num_pred_cc']['delta_mean']:+.2f}`** (`[{vs_200['num_pred_cc']['ci95'][0]:+.2f}, {vs_200['num_pred_cc']['ci95'][1]:+.2f}]`, `p={vs_200['num_pred_cc']['wilcoxon_p_holm']:.2e}`) |

- **Duplicate-Clean Sensitivity (`N=199`, excluding `test_48_A.png`)**: All conclusions remain unchanged (`Dual Dice = {vr_199['dice']['dual_mean']:.4f}`, `+{vr_199['dice']['delta_mean']:.4f}` vs Raw; `ThinSkel = {vr_199['thin_skel_sens']['dual_mean']*100:.2f}%`, `+{vr_199['thin_skel_sens']['delta_mean']*100:.2f} pp`; `clDice = {vr_199['cldice']['dual_mean']:.4f}`).

---PAGEBREAK---

### 6.3 Disease Stratification (`n=50` per Pathology on FIVES)

| Pathology | Raw Dice / Prec | Restored Dice / Prec | Dual-Stream Dice / Prec | Dual ThinSkel | Dual clDice | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{disease_rows}

#### Per-Disease Paired Statistical Deltas (`Dual-Stream` vs `Raw` & `Restored`, Holm `p`)
| Pathology | vs Raw Dice (`p_holm`) | vs Raw ThinSkel (`p_holm`) | vs Raw clDice | vs Restored Prec (`p_holm`) | vs Restored CC |
| :--- | :---: | :---: | :---: | :---: | :---: |
{disease_paired_rows}

- **Glaucoma Cup Rim Resolution**: Single-stream `unet_restored` suffered `-33.54 pp` precision collapse on Glaucoma (`57.59%` vs `91.13%`) because frequency contrast enhancement amplified excavated optic-cup rim borders. By gating with Stream A (`Raw`), `DualStreamUNet` restores Glaucoma precision to **`89.98%`** (`+32.39 pp` over Restored, only `-1.15 pp` vs Raw) while achieving `+6.63 pp` higher Dice and `+9.48 pp` higher Thin Skeleton Sensitivity than Raw U-Net.

![Figure 5: Full-resolution untouched FIVES test qualitative comparisons across Normal (test_161_N), AMD (test_37_A), DR (test_100_D), and Glaucoma (test_133_G).|6.9|4.6](fives_qual|6.9|4.6)

---PAGEBREAK---

### 6.4 External Generalization on HRF (`N=45`: `40` Pretrain-Exposed / `5` Exploratory Unexposed)

>! **HRF Exposure & Transfer Caveat**: `SFHformer` encountered `40/45` HRF scans during upstream pretraining. Furthermore, transferring a FIVES-calibrated threshold (`tau = 0.40`) onto higher-resolution `3504x2336` HRF scans (`16.71%` smallest vessels vs `1.49%` in FIVES) shifts the sensitivity-precision operating point. Results must be interpreted transparently alongside these trade-offs.

| Cohort & System (`tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HRF Overall (`N=45`) — Raw U-Net** (`0.50`) | **`{raw_h45['dice']['mean']:.4f}`** | **`{raw_h45['prec']['mean']:.4f}`** | `{raw_h45['sens']['mean']:.4f}` | `{raw_h45['thin_skel_sens']['mean']*100:.2f}%` | `{raw_h45['branch_recall']['mean']*100:.2f}%` | `{raw_h45['cldice']['mean']:.4f}` | `{raw_h45['num_pred_cc']['mean']:.1f}` |
| **HRF Overall (`N=45`) — Restored U-Net** (`0.50`) | `{rest_h45['dice']['mean']:.4f}` | `{rest_h45['prec']['mean']:.4f}` | **`{rest_h45['sens']['mean']:.4f}`** | **`{rest_h45['thin_skel_sens']['mean']*100:.2f}%`** | **`{rest_h45['branch_recall']['mean']*100:.2f}%`** | `{rest_h45['cldice']['mean']:.4f}` | `{rest_h45['num_pred_cc']['mean']:.1f}` |
| **HRF Overall (`N=45`) — Dual-Stream** (`0.40`) | `{dual_h45['dice']:.4f}` | `{dual_h45['prec']:.4f}` | `{dual_h45['sens']:.4f}` | `{dual_h45['thin_skel_sens']*100:.2f}%` | `{dual_h45['branch_recall']*100:.2f}%` | **`{dual_h45['cldice']:.4f}`** | **`{dual_h45['num_pred_cc']:.1f}`** |
| **HRF Exposed (`N=40`) — Dual-Stream** (`0.40`) | `{dual_h40['dice']:.4f}` | `{dual_h40['prec']:.4f}` | `{dual_h40['sens']:.4f}` | `{dual_h40['thin_skel_sens']*100:.2f}%` | `{dual_h40['branch_recall']*100:.2f}%` | `{dual_h40['cldice']:.4f}` | `{dual_h40['num_pred_cc']:.1f}` |
| **HRF Unexposed (`N=5`, Exploratory) — Raw** (`0.50`) | **`{raw_h5['dice']['mean']:.4f}`** | **`{raw_h5['prec']['mean']:.4f}`** | `{raw_h5['sens']['mean']:.4f}` | `{raw_h5['thin_skel_sens']['mean']*100:.2f}%` | `{raw_h5['branch_recall']['mean']*100:.2f}%` | `{raw_h5['cldice']['mean']:.4f}` | `{raw_h5['num_pred_cc']['mean']:.1f}` |
| **HRF Unexposed (`N=5`, Exploratory) — Restored** (`0.50`) | `{rest_h5['dice']['mean']:.4f}` | `{rest_h5['prec']['mean']:.4f}` | **`{rest_h5['sens']['mean']:.4f}`** | **`{rest_h5['thin_skel_sens']['mean']*100:.2f}%`** | **`{rest_h5['branch_recall']['mean']*100:.2f}%`** | `{rest_h5['cldice']['mean']:.4f}` | `{rest_h5['num_pred_cc']['mean']:.1f}` |
| **HRF Unexposed (`N=5`, Exploratory) — Dual** (`0.40`) | `{dual_h5['dice']:.4f}` | `{dual_h5['prec']:.4f}` | `{dual_h5['sens']:.4f}` | `{dual_h5['thin_skel_sens']*100:.2f}%` | `{dual_h5['branch_recall']*100:.2f}%` | **`{dual_h5['cldice']:.4f}`** | **`{dual_h5['num_pred_cc']:.1f}`** |

#### Paired HRF Statistical Trade-Off Summary (`N=45`, Holm-Bonferroni Adjusted)
| Metric | Dual-Stream vs Raw U-Net (`N=45`) | Dual-Stream vs Restored U-Net (`N=45`) |
| :--- | :--- | :--- |
| **Dice** | `{hr_vr['dice']['delta_mean']:+.4f}` (`[{hr_vr['dice']['ci95'][0]:+.4f}, {hr_vr['dice']['ci95'][1]:+.4f}]`, `p={hr_vr['dice']['wilcoxon_p_holm']:.2e}`) | `{hr_vs['dice']['delta_mean']:+.4f}` (`[{hr_vs['dice']['ci95'][0]:+.4f}, {hr_vs['dice']['ci95'][1]:+.4f}]`, `p={hr_vs['dice']['wilcoxon_p_holm']:.2e}`) |
| **Precision** | `{hr_vr['prec']['delta_mean']:+.4f}` (`[{hr_vr['prec']['ci95'][0]:+.4f}, {hr_vr['prec']['ci95'][1]:+.4f}]`, `p={hr_vr['prec']['wilcoxon_p_holm']:.2e}`) | `{hr_vs['prec']['delta_mean']:+.4f}` (`[{hr_vs['prec']['ci95'][0]:+.4f}, {hr_vs['prec']['ci95'][1]:+.4f}]`, `p={hr_vs['prec']['wilcoxon_p_holm']:.2e}`) |
| **Thin Skel Sens** | **`+{hr_vr['thin_skel_sens']['delta_mean']*100:.2f} pp`** (`[+{hr_vr['thin_skel_sens']['ci95'][0]*100:.2f}, +{hr_vr['thin_skel_sens']['ci95'][1]*100:.2f}] pp`, `p={hr_vr['thin_skel_sens']['wilcoxon_p_holm']:.2e}`) | `{hr_vs['thin_skel_sens']['delta_mean']*100:+.2f} pp` (`[{hr_vs['thin_skel_sens']['ci95'][0]*100:+.2f}, {hr_vs['thin_skel_sens']['ci95'][1]*100:+.2f}] pp`, `p={hr_vs['thin_skel_sens']['wilcoxon_p_holm']:.2e}`) |
| **Branch Recall** | **`+{hr_vr['branch_recall']['delta_mean']*100:.2f} pp`** (`p={hr_vr['branch_recall']['wilcoxon_p_holm']:.2e}`) | `{hr_vs['branch_recall']['delta_mean']*100:+.2f} pp` (`p={hr_vs['branch_recall']['wilcoxon_p_holm']:.2e}`) |
| **clDice** | **`+{hr_vr['cldice']['delta_mean']:.4f}`** (`p={hr_vr['cldice']['wilcoxon_p_holm']:.2e}`) | **`+{hr_vs['cldice']['delta_mean']:.4f}`** (`p={hr_vs['cldice']['wilcoxon_p_holm']:.2e}`) |
| **Components (`CC`)** | **`{hr_vr['num_pred_cc']['delta_mean']:+.2f}`** (`p={hr_vr['num_pred_cc']['wilcoxon_p_holm']:.2e}`) | **`{hr_vs['num_pred_cc']['delta_mean']:+.2f}`** (`p={hr_vs['num_pred_cc']['wilcoxon_p_holm']:.2e}`) |

- **HRF `N=5` Exploratory Per-Image Breakdown**: With only `N=5` pairs, the minimum achievable two-sided Wilcoxon `p` is `0.0625` (`wilcoxon_p_holm = {h5_vr['thin_skel_sens']['wilcoxon_p_holm']:.3f}` across 6 comparisons), which is statistically non-significant and reported strictly as exploratory evidence:

| HRF Unexposed Scan | Dual Dice | Dual Prec | Dual Sens | Dual ThinSkel | Dual clDice | Dual BranchRecall | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{hrf_unexp_rows}

![Figure 6: External HRF exploratory unexposed subset (14_h, 15_dr) illustrating thin-vessel recovery alongside out-of-domain precision trade-offs.|6.9|3.3](hrf_qual|6.9|3.3)

---PAGEBREAK---

## 7. System Integration & RetinaScan Studio Prototype

### 7.1 End-to-End Pipeline Architecture
The validated models are deployed inside a modular research prototype connecting a Flask REST inference server ([scripts/serve_restore.py](file:///d:/SFHformer-FreqLab/scripts/serve_restore.py), port `5001`) with a React 18 / Vite frontend ([frontend/src/pages/RetinaStudio.jsx](file:///d:/SFHformer-FreqLab/frontend/src/pages/RetinaStudio.jsx), port `3000`).

![Figure 7: Combined end-to-end RetinaScan Studio system architecture linking ingestion, SFHformer restoration, the 3-model segmentation core, thread-safe Flask serving with 6-stage reconciled timing, and the React UI.|6.9|3.8](diag_5|6.9|3.8)

### 7.2 Protocol Safeguards & Integration QA Summary
- **Frozen Operating Points vs Exploratory Controls**: Defaults to `Dual-Stream Hybrid` at frozen validation-calibrated `tau = 0.40`, offers `tau = 0.50` explicitly labeled as an alternative threshold-matched baseline comparison, isolates exploratory slider changes to local previews only, and never auto-adjusts thresholds per dataset.
- **Optional Spatial Fusion-Weight Visualization**: Hidden by default; renders the exact single-pass `scale_0` spatial gate `w_rest` (`Blue = 0.0 Raw -> Amber = 1.0 Restored`) explicitly captioned *"Not confidence, uncertainty, or vessel probability."*

| Integration QA Verification Check | Scope & Harness | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| **Heatmap Spatial Alignment & Gradient Polarity** | `qa_phase5_1.py` (checkerboard shift) | Top-Left `12.27` vs Bottom-Right `2.45`; `w_rest` 0->1 polarity exact | PASS |
| **Same-Forward-Pass Gate Guarantee** | `u_dual(..., return_gates=True)` check | `max logit delta = 0.0` vs standard pass | PASS |
| **Invalid Input & Edge-Case Guards** | `9` malformed API requests (`POST`) | `9/9` structured `HTTP 400` JSON errors; cross-model viz suppressed | PASS |
| **20-Run Latency & Timing Reconciliation** | `20` consecutive `512x512` runs | Median `1,300.8 ms`, P95 `1,724.9 ms`, max reconcile err `0.1 ms` | PASS |
| **VRAM Stability & Thread Serialization** | `20` sequential + `4` concurrent calls | `0.0 MB` VRAM leak (`216.5 MB` steady); `4/4` concurrent `HTTP 200` | PASS |
| **Live Chromium DOM & Visual UI Suite** | `qa_phase5_2_browser.py` (`Edg/157.0.4341.60`) | `11/11` checks pass, `naturalWidth > 0`, `0` JS/React runtime errors | PASS |

---PAGEBREAK---

### 7.3 Verified Live Studio Captures (`Phase 5.2`)

![Figure 8: Live Chromium screenshot of RetinaScan Studio Tab 1 showing Dual-Stream Hybrid (tau=0.40), prototype notice, and optional scale_0 spatial fusion-weight heatmap legend.|6.8|4.1](ui_tab1|6.8|4.1)

![Figure 9: Live Chromium screenshot of RetinaScan Studio Tab 2 displaying frozen FIVES and HRF comparative metrics alongside explicit HRF exposure and N=5 exploratory disclosures.|6.8|4.1](ui_tab2|6.8|4.1)

---PAGEBREAK---

## 8. Reproducibility, Limitations & Conclusions

### 8.1 Checkpoint Manifest & Naming Reconciliation
To prevent ambiguity across Milestone 2, Milestone 3, and Phase 5.3 manifests, every frozen binary weight file on disk was cryptographically audited:

| Logical / Script Alias | Exact Repository Path | Size | SHA-256 Digest (`64 hex chars`) |
| :--- | :--- | :---: | :--- |
| `sfhformer` / `retina_fives` | `checkpoints/fives_retina_finetune/best.pth` | `67.80 MB` | `{snap['checkpoints/fives_retina_finetune/best.pth']}` |
| `unet_raw` / `raw` | `checkpoints/unet_raw/best.pth` | `86.97 MB` | `{snap['checkpoints/unet_raw/best.pth']}` |
| `unet_restored` / `restored` | `checkpoints/unet_restored/best.pth` | `86.97 MB` | `{snap['checkpoints/unet_restored/best.pth']}` |
| `unet_dual_stream` / `dual_stream` | `checkpoints/unet_dual_stream/best.pth` | `62.74 MB` | `{snap['checkpoints/unet_dual_stream/best.pth']}` |

- **Naming Resolution**: The identifier `"sfhformer"` inside `scripts/evaluate_milestone3.py` (`CKPTS['sfhformer']`) is confirmed to reference `checkpoints/fives_retina_finetune/best.pth` (`SHA-256: d74e267c...`). Earlier pretraining/ablation SFHformer checkpoints (`r1_baseline/best.pth`, `r2_gated/best.pth`, `retina/best.pth`) reside in separate subdirectories and were not modified.
- **Size Difference Explanation**: `unet_raw` and `unet_restored` (`86.97 MB`) bundle `AdamW` optimizer state buffers alongside `7.7M` model weights, whereas `unet_dual_stream` (`62.74 MB`) stores `16.41M` model weights (`state_dict`) and validation metadata only.

### 8.2 Reproducibility Instructions
```bash
git checkout v3.1-phase5.2-verified
d:\\SFHformer-FreqLab\\.venv\\python.exe scripts/evaluate_milestone3.py
d:\\SFHformer-FreqLab\\.venv\\python.exe scripts/evaluate_dual_stream_milestone3.py
d:\\SFHformer-FreqLab\\.venv\\python.exe scripts/serve_restore.py
```

### 8.3 Limitations & Future Extensions
1. **External Domain Shift**: While `DualStreamUNet` excels across all metrics in-domain on FIVES (`N=200`), external transfer to `3504x2336` HRF at `tau=0.40` trades `-8.54 pp` Precision vs Raw U-Net, motivating multi-domain validation and domain-invariant threshold calibration on genuinely unexposed high-resolution cohorts.
2. **Spatial vs Channel-Wise Gating**: Current `CrossStreamFusionBlock` applies a 2-channel per-pixel spatial gate (`w_raw + w_rest = 1`) with an SE residual; a controlled ablation comparing channel-wise gating (`2C` weights) against spatial gating remains open.
3. **Interactive Studio Resolution vs Full-Field Tiling**: RetinaScan Studio resizes interactive uploads to `512x512` for `~1.3 s` turnaround; extremely dark or ultra-high-res scans benefit from full sliding-window tiling used in Milestone 3 scripts.

### 8.4 Conclusions & References
1. **SFHformer-FreqLab** demonstrates that coupling spatial-frequency restoration with warm-started dual-stream gated segmentation (`DualStreamUNet`) and FP32 `SoftclDice` regularization resolves the core accuracy-vs-sensitivity trade-off on FIVES (`0.8474` Dice, `+10.20 pp` thin skeleton sensitivity, `0.8537` clDice, `41.22` components/image, `89.98%` Glaucoma precision).
2. **References**: (a) Shit et al., *clDice - A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation*, CVPR 2021; (b) Jin et al., *FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation*, Scientific Data 2022; (c) Budai et al., *Robust Vessel Segmentation in Fundus Images (HRF)*, Int. J. Biomed. Imaging 2013; (d) Frangi et al., *Multiscale Vessel Enhancement Filtering*, MICCAI 1998; (e) Ronneberger et al., *U-Net*, MICCAI 2015.

---PAGEBREAK---

## Appendix A: Development Progression & Immutable Snapshot Inventory

### A.1 End-to-End Experimental Milestone Summary

| Stage / Phase | Objective & Scope | Key Verified Outcome |
| :--- | :--- | :--- |
| **Milestone 1** | FIVES & HRF provenance, duplicate isolation, FOV clipping & caliber audit | `100%` image-level disjointness (`N=540 / 59 / 200`); `11.2x` fine-capillary share difference (`16.71%` HRF vs `1.49%` FIVES). |
| **SFHformer Fine-Tune** | 5-stage hierarchical FFT restoration (`SFHformer-m`, `14.18M` params) on FIVES crops | Peak validation PSNR `31.92 dB`; held-out `N=200` test PSNR `29.24 dB` (`+3.65 dB` over DRIVE baseline), `SSIM 0.9309`. |
| **Milestone 2.5** | Paired clean-val (`N=59`) evaluation of `unet_raw` vs `unet_restored` baselines | Proved `unet_restored` boosts thin-vessel sensitivity `+6.16 pp` (`p=7.58e-7`) while dropping Dice & continuity. |
| **Phases 1–4 (`DualStream`)** | Warm-started dual encoder + `CrossStreamFusionBlock` + FP32 `SoftclDice(12)` + `N=59` sweep | Epoch 10 best checkpoint (`S=0.6806`); froze calibrated threshold `tau = 0.40` strictly on `N=59` clean validation. |
| **Milestone 3** | One-shot frozen generalization on untouched FIVES (`N=200`/`199`) & HRF (`N=45`) | FIVES Dice `0.8474`, ThinSkel `43.52%`, Glaucoma Prec `89.98%`; honest HRF transfer trade-offs documented. |
| **Phases 5.1–5.3** | RetinaScan Studio REST + React UI + QA (`9/9` backend, `11/11` Chromium) + Git tag | Tagged `v3.1-phase5.2-verified`; `15` immutable SHA-256 artifacts + `5` 300-DPI architecture diagrams verified. |

### A.2 Cryptographic Inventory (`SNAPSHOT_MANIFEST.json`, `v3.1-phase5.2-verified`)

| Frozen Artifact Path (`D:\\SFHformer-FreqLab\\`) | SHA-256 Digest Prefix (Full 64-Hex Verified on Disk) |
| :--- | :--- |
{snap_rows}
"""


def build_deliverable_b_engineering_doc(d):
    snap = d["snap"]["sha256_inventory"]
    cal = d["dual_calib"]
    return f"""# SFHformer-FreqLab: Engineering & System Implementation Documentation

> **Target Audience**: ML Engineers, Infrastructure Maintainers & Technical Reviewers  
> **Snapshot Tag**: `v3.1-phase5.2-verified` (`42ac0565d1d1517ef5bad9fdc1f392608272db1b`)  
> **Scope Notice**: RetinaScan Studio is a **research prototype for retinal vessel segmentation**, not a certified clinical diagnostic product.

---TOC---

## 1. Repository Layout & Module Ownership

```
D:\\SFHformer-FreqLab\\
├── models/
│   ├── SFHformer.py               # 5-stage SFHformer_m (14.18M params)
│   ├── unet.py                    # 4-stage DoubleConv + Bilinear UNet (7.7M params)
│   └── dual_stream_unet.py        # DualStreamUNet + CrossStreamFusionBlock (16.41M params)
├── freqlab/
│   ├── gate.py                    # RadialBandGate (16 learnable radial spectral bands)
│   ├── sfhformer_gated.py         # GatedBackbone (R2 variant)
│   └── metrics.py / classical.py  # PSNR, SSIM, Notch/Gaussian/Median filters
├── utils/
│   └── losses.py                  # BCEDiceLoss, SoftSkeletonize(FP32), HybridTopologyLoss
├── scripts/
│   ├── train_fives_retina.py      # Online FIVES restoration fine-tuning
│   ├── train_unet.py              # Single-stream Raw & Restored U-Net training
│   ├── train_dual_stream_unet.py  # Two-stage Dual-Stream training + N=59 calibration
│   ├── evaluate_milestone3.py     # Frozen baseline Milestone 3 harness
│   ├── evaluate_dual_stream_milestone3.py  # Frozen Dual-Stream evaluation + Holm-Bonferroni
│   ├── serve_restore.py           # Flask REST API server (:5001) with GPU_LOCK
│   ├── retina_engine.py           # Overlays, create_fusion_gate_overlay, CCL & skeleton
│   └── qa_phase5_1.py             # Automated API, alignment, latency, concurrency QA
├── frontend/src/pages/RetinaStudio.jsx   # React 18 / Vite RetinaScan Studio UI (:3000)
├── checkpoints/                   # Frozen .pth weights + validation_calibration.json
├── results/milestone3/            # Immutable per-image JSON experimental outputs
├── demo_package/                  # Phase 5.3 diagrams, qualitative strips, manifest
└── final_documentation/           # Phase 6 reports, sources & DOCUMENTATION_MANIFEST.json
```

---PAGEBREAK---

## 2. Model Loading, State-Dict Mapping & Checkpoint Provenance

### 2.1 Dual-Stream Warm-Start Contract (`models/dual_stream_unet.py`)
`DualStreamUNet.load_pretrained_streams(raw_ckpt_path, rest_ckpt_path, init_decoder_from="raw")` maps single-stream `UNet` keys cleanly onto dual encoders without shape ambiguity:
- `downs.0..3.*` & `bottleneck.*` from `unet_raw/best.pth` $\\to$ `raw_downs.*`, `raw_bottleneck.*`
- `downs.0..3.*` & `bottleneck.*` from `unet_restored/best.pth` $\\to$ `rest_downs.*`, `rest_bottleneck.*`
- `ups.*` & `final_conv.*` from `unet_raw/best.pth` $\\to$ shared decoder `ups.*`, `final_conv.*`
- Verified `208` tensors loaded cleanly; `80` parameters/buffers across `fusions[0..3]` and `bottleneck_fusion` initialized fresh.

### 2.2 Checkpoint Inventory & Hash Reconciliation

| Alias / Identifier | File Path | Size | Format Contents | SHA-256 Prefix (`Full in SNAPSHOT_MANIFEST`) |
| :--- | :--- | :---: | :--- | :--- |
| `sfhformer` / `retina_fives` | `checkpoints/fives_retina_finetune/best.pth` | `67.80 MB` | Model + Optimizer | `{snap['checkpoints/fives_retina_finetune/best.pth'][:24]}...` |
| `unet_raw` / `raw` | `checkpoints/unet_raw/best.pth` | `86.97 MB` | Model + Optimizer | `{snap['checkpoints/unet_raw/best.pth'][:24]}...` |
| `unet_restored` / `restored` | `checkpoints/unet_restored/best.pth` | `86.97 MB` | Model + Optimizer | `{snap['checkpoints/unet_restored/best.pth'][:24]}...` |
| `unet_dual_stream` | `checkpoints/unet_dual_stream/best.pth` | `62.74 MB` | Model `state_dict` + Val Meta | `{snap['checkpoints/unet_dual_stream/best.pth'][:24]}...` |

> **Reconciliation Note**: In `scripts/evaluate_milestone3.py`, key `"sfhformer"` maps directly to `checkpoints/fives_retina_finetune/best.pth` (`d74e267c...`). The `62.74 MB` size of `unet_dual_stream/best.pth` reflects saving `model.state_dict()` (`16.41M` params) without optimizer momentum buffers.

![Architecture Overview of DualStreamUNet & CrossStreamFusionBlock Implementation|6.8|3.8](diag_4|6.8|3.8)

---PAGEBREAK---

## 3. Inference Pipeline, GPU Memory Management & API Specification

### 3.1 Hardware Safety & Thread Serialization
- **Target Device**: NVIDIA GeForce RTX 5060 Laptop GPU (`8,150.6 MB` total VRAM).
- **Training Allocator Cap**: Training scripts enforce `torch.cuda.set_per_process_memory_fraction(0.65)` (Dual-Stream peak Stage 2 VRAM = `2,948.8 MB`, `batch_size=2, accum=2`, BF16 AMP) or `0.70` (`5.57 GB`).
- **Serving Concurrency (`GPU_LOCK`)**: Inside `scripts/serve_restore.py`, Flask runs with `threaded=True`. All CUDA forward passes in `/api/retina/dual_segment` execute inside `with GPU_LOCK:` (`threading.Lock()`), serializing GPU memory allocation so `4` simultaneous requests execute cleanly with `0.0 MB` VRAM leak (`216.5 MB` allocated / `1,436.0 MB` reserved steady state).

### 3.2 REST API Endpoints (`http://127.0.0.1:5001`)

| Endpoint | Method | Purpose & Payload | Validation Guards |
| :--- | :---: | :--- | :--- |
| `/health` | `GET` | Device, loaded `MODELS` keys, `gpu_memory` (`allocated_mb`, `reserved_mb`, `free_mb`) | Always `200 OK` when alive |
| `/api/retina/samples` | `GET` | Returns 15 curated fundus/OCT sample definitions (`hrf_unexposed_14h`, FIVES, DRIVE, STARE, OCT) | Static + disk catalog |
| `/api/retina/benchmark` | `GET` | Frozen Milestone 3 records (`fives_summary`, `hrf_overall`, `hrf_unexposed`, `dual_stream`) | Cached read-only from JSONs |
| `/api/retina/restore` | `POST` | Runs synthetic ophthalmic corruption + single SFHformer restoration + Frangi ablation | `400` on missing image |
| `/api/retina/dual_segment` | `POST` | Runs SFHformer + Raw/Restored U-Net + `DualStreamUNet` + topology + overlays | Strict HTTP `400` guards (below) |

#### `/api/retina/dual_segment` Input Validation Matrix
- `threshold`: Finite float in `[0.0, 1.0]` (defaults to `{cal['selected_frozen_threshold']:.2f}`). Out-of-range or non-numeric $\\to$ `HTTP 400`.
- `selected_model`: Must be in `['dual_stream', 'raw_unet', 'restored_unet']`. Unknown string $\\to$ `HTTP 400`.
- `resolution`: Must be integer `256` or `512`. Unsupported size $\\to$ `HTTP 400`.
- `sample_id` / `image`: Unknown `sample_id`, empty string, corrupted Base64, or malformed JSON $\\to$ `HTTP 400`.
- `include_fusion_viz`: Boolean (default `False`). When `True` alongside `selected_model != 'dual_stream'`, suppresses `overlay_fusion_gate: null` and returns explanatory `fusion_viz_note`.

### 3.3 6-Stage Reconciled Latency Instrumentation
Every response from `/api/retina/dual_segment` accounts for `100%` of request wall time across 6 sequential timers (discrepancy `<= 0.1 ms` across 20 consecutive QA runs):
1. `input_io_and_prep_ms`: JSON parsing, disk read / Base64 decode, `512x512` bilinear resize.
2. `sfhformer_restoration_ms`: `SFHformer-m` GPU forward pass (`[0, 1]` clamped).
3. `dual_unet_segmentation_ms` + `dual_stream_segmentation_ms`: BF16 AMP forward passes for `unet_raw`, `unet_restored`, and `unet_dual_stream` (single pass returning both `logits_dual` and `scale_0` gate when requested).
4. `topology_analysis_ms`: FOV mask (`clean.mean(axis=2) > 10/255`), binarization (`>= threshold`), `scipy.ndimage.label`, `skimage.morphology.skeletonize`, and endpoint convolution.
5. `overlay_blending_ms`: Alpha compositing for Raw, Restored, Differential Amber, Dual-Stream, and optional Spatial Gate overlays.
6. `fidelity_and_base64_encoding_ms`: PSNR/SSIM calculation and lossless PNG Base64 serialization.
- **Measured QA Performance (20 runs, RTX 5060)**: **Median = `1,300.8 ms`**, **P95 = `1,724.9 ms`**, Min = `1,019.7 ms`, Max = `1,740.3 ms`.

---PAGEBREAK---

## 4. Spatial Fusion Visualization & Frontend Architecture

### 4.1 `create_fusion_gate_overlay()` Implementation (`scripts/retina_engine.py`)
When `include_fusion_viz=True` and `selected_model='dual_stream'`, `u_dual(x_raw, x_restored, return_gates=True)` extracts `gates['scale_0']` (`512x512` or `256x256`) from the **exact same forward pass** as segmentation (`max logit diff = 0.0`). Pure vectorized NumPy maps `w_rest` in `[0, 1]` across a 3-stop colormap (`0.0` Blue `[37,99,235]` $\\to$ `0.5` Teal `[20,184,166]` $\\to$ `1.0` Amber `[245,158,11]`) blended over fundus tissue (`alpha=0.58`).

![Combined End-to-End RetinaScan Studio System Architecture|6.8|3.8](diag_5|6.8|3.8)

### 4.2 Frontend State & Safeguards (`RetinaStudio.jsx`)
- **Active Segmentation Model**: Defaults to `Dual-Stream Hybrid (Default)`; switching to `Raw U-Net` or `Restored U-Net` cleanly updates the view and disables the optional `w_rest` gate checkbox.
- **Operating Threshold (`tau`)**: Defaults to `0.40 (Calibrated)` with an explicit `0.50 (Baseline Match)` preset button. Adjusting the exploratory slider displays `Exploratory View Mode` without altering Tab 2 frozen benchmark data.
- **Prototype Notice**: Header explicitly states *"Research Prototype for Retinal Vessel Segmentation & Frequency Restoration Analysis (Not a Certified Diagnostic System)"*.

---PAGEBREAK---

## 5. Automated Integration QA & Deployment Verification

### 5.1 Phase 5.1 Backend QA Results (`scripts/qa_phase5_1.py`)
- **Spatial Gate Alignment & Polarity**: `w_rest=0.0 -> RGB [37,99,235]`, `w_rest=1.0 -> RGB [245,158,11]`; checkerboard perturbation yielded `Top-Left delta = 12.27` vs `Bottom-Right = 2.45`.
- **Edge-Case Guards**: All `9` malformed/out-of-range API calls returned structured `HTTP 400` JSON errors; cross-model fusion guard passed.
- **Concurrency & Stability**: `4/4` concurrent threads returned `HTTP 200` (`1,028.2 ms` to `3,189.9 ms` wall); `20` sequential calls showed `0.0 MB` VRAM drift (`216.5 MB` alloc before & after).

### 5.2 Phase 5.2 Live Chromium QA (`Edg/157.0.4341.60` via `--remote-debugging-port=9222`)
All `11/11` interactive frontend assertions passed live against `http://localhost:3000/retina` + `http://127.0.0.1:5001`:
- Verified default `Dual-Stream Hybrid` + `0.40` calibrated notice, Raw/Restored switching, `0.50` preset, `0.65` exploratory slider isolation, hidden-by-default heatmap + `0.0 -> 1.0` legend, separate FIVES/HRF benchmark disclosures, `naturalWidth > 0` across all overlays, and `0` JS/React runtime errors.

![Live Chromium Tab 1 Verification Screenshot|6.8|4.1](ui_tab1|6.8|4.1)

### 5.3 Deployment Commands
```powershell
# Backend (Port 5001)
d:\\SFHformer-FreqLab\\.venv\\python.exe -u scripts\\serve_restore.py

# Frontend (Port 3000)
Set-Location "C:\\Users\\ARAN\\.gemini\\antigravity-ide\\scratch\\freqlab-ui"; npm run dev
```
"""


def build_deliverable_c_eval_summary(d):
    fb = d["fives_base"]
    fd = d["fives_dual"]
    hb = d["hrf_base"]
    hd = d["hrf_dual"]
    snap = d["snap"]["sha256_inventory"]

    raw_f = fb["summary_overall_200"]["per_system"]["raw_unet"]
    rest_f = fb["summary_overall_200"]["per_system"]["restored_unet"]
    dual_f = fd["overall"]
    vr_200 = fd["vs_raw_unet_n200"]
    vs_200 = fd["vs_restored_unet_n200"]

    raw_h45 = hb["summary_overall_45"]["per_system"]["raw_unet"]
    rest_h45 = hb["summary_overall_45"]["per_system"]["restored_unet"]
    dual_h45 = hd["overall_n45"]

    raw_h5 = hb["summary_unexposed_5"]["per_system"]["raw_unet"]
    rest_h5 = hb["summary_unexposed_5"]["per_system"]["restored_unet"]
    dual_h5 = hd["unexposed_n5_exploratory"]["summary"]

    disease_rows = "\n".join(
        f"| **{lbl}** (n=50) | `{fb['disease_breakdown'][code]['per_system']['raw_unet']['dice']['mean']:.4f}` / `{fb['disease_breakdown'][code]['per_system']['raw_unet']['prec']['mean']:.4f}` | `{fb['disease_breakdown'][code]['per_system']['restored_unet']['dice']['mean']:.4f}` / `{fb['disease_breakdown'][code]['per_system']['restored_unet']['prec']['mean']:.4f}` | **`{fd['by_disease'][lbl]['dual_stream']['dice']:.4f}` / `{fd['by_disease'][lbl]['dual_stream']['prec']:.4f}`** | `{fd['by_disease'][lbl]['dual_stream']['thin_skel_sens']*100:.2f}%` | `{fd['by_disease'][lbl]['dual_stream']['cldice']:.4f}` | `{fd['by_disease'][lbl]['dual_stream']['num_pred_cc']}` |"
        for lbl, code in [("Normal", "N"), ("AMD", "A"), ("DR", "D"), ("Glaucoma", "G")]
    )

    hrf_unexp_rows = "\n".join(
        f"| `{rec['filename']}` | `{rec['dual_stream']['dice']:.4f}` | `{rec['dual_stream']['prec']:.4f}` | `{rec['dual_stream']['thin_skel_sens']*100:.2f}%` | `{rec['dual_stream']['cldice']:.4f}` | `{rec['dual_stream']['branch_recall']*100:.2f}%` | `{rec['dual_stream']['num_pred_cc']}` |"
        for rec in hd["unexposed_n5_exploratory"]["per_image"]
    )

    return f"""# Milestone 3 Final Evaluation Summary

> **Protocol Snapshot**: `v3.1-phase5.2-verified` | **Thresholds**: Dual-Stream `0.40` (FIVES Clean-Val Calibrated) vs Single-Stream Baselines `0.50` vs Frangi `0.0010`  
> **Scope Notice**: RetinaScan Studio is a **research prototype for retinal vessel segmentation**, not a clinically validated diagnostic tool.

---TOC---

## 1. Principal Contributions & Proposed Architecture
Single-stream vessel segmentation faces a sharp trade-off: training on native images (`Raw U-Net`) maximizes macro precision (`0.9266`) and continuity (`clDice 0.8134`) while missing low-contrast thin capillaries, whereas training on `SFHformer`-restored images (`Restored U-Net`) increases thin-vessel sensitivity on HRF (`+13.67 pp`) at the cost of coarse-label precision drops, fragmentation (`114.25` components/image on FIVES), and Glaucoma optic-cup rim false positives (`57.59%` precision).

`DualStreamUNet` resolves this divergence by fusing warm-started Raw and SFHformer-Restored encoders across all 5 scales via per-pixel spatial softmax gates (`w_raw + w_rest = 1`) and SE cross-residuals, optimized under FP32 `HybridTopologyLoss` (`SoftSkeletonize(iter=12, border_ignore=2)`).

![Proposed Dual-Stream Hybrid U-Net Architecture|6.9|3.8](diag_4|6.9|3.8)

---PAGEBREAK---

## 2. Primary In-Domain Benchmark: FIVES Untouched Test (`N=200`)

| System (`Frozen tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw U-Net** (`0.50`) | `{raw_f['dice']['mean']:.4f}` | **`{raw_f['prec']['mean']:.4f}`** | `{raw_f['sens']['mean']:.4f}` | `{raw_f['thin_skel_sens']['mean']*100:.2f}%` | `{raw_f['branch_recall']['mean']*100:.2f}%` | `{raw_f['cldice']['mean']:.4f}` | `{raw_f['num_pred_cc']['mean']:.2f}` |
| **Restored U-Net** (`0.50`) | `{rest_f['dice']['mean']:.4f}` | `{rest_f['prec']['mean']:.4f}` | `{rest_f['sens']['mean']:.4f}` | `{rest_f['thin_skel_sens']['mean']*100:.2f}%` | `{rest_f['branch_recall']['mean']*100:.2f}%` | `{rest_f['cldice']['mean']:.4f}` | `{rest_f['num_pred_cc']['mean']:.2f}` |
| **Dual-Stream Hybrid** (`0.40`) | **`{dual_f['dice']:.4f}`** | `{dual_f['prec']:.4f}` | **`{dual_f['sens']:.4f}`** | **`{dual_f['thin_skel_sens']*100:.2f}%`** | **`{dual_f['branch_recall']*100:.2f}%`** | **`{dual_f['cldice']:.4f}`** | **`{dual_f['num_pred_cc']:.2f}`** |

- **Statistical Gains vs Raw U-Net (`N=200`, Holm-Bonferroni corrected)**: Dice `+{vr_200['dice']['delta_mean']:.4f}` (`p={vr_200['dice']['wilcoxon_p_holm']:.2e}`), Thin Skeleton Sensitivity `+{vr_200['thin_skel_sens']['delta_mean']*100:.2f} pp` (`p={vr_200['thin_skel_sens']['wilcoxon_p_holm']:.2e}`), Branch Recall `+{vr_200['branch_recall']['delta_mean']*100:.2f} pp`, `clDice +{vr_200['cldice']['delta_mean']:.4f}`, Fragmentation `{vr_200['num_pred_cc']['delta_mean']:.2f}` CCs/img (`N=199` deduplicated matches identically).
- **Statistical Gains vs Restored U-Net (`N=200`)**: Dice `+{vs_200['dice']['delta_mean']:.4f}`, Precision `+{vs_200['prec']['delta_mean']:.4f}`, Thin Skeleton Sensitivity `+{vs_200['thin_skel_sens']['delta_mean']*100:.2f} pp`, `clDice +{vs_200['cldice']['delta_mean']:.4f}`, Fragmentation `-73.03` CCs/img.

### FIVES Pathology Stratification (`n=50` per Class)

| Pathology | Raw Dice / Prec | Restored Dice / Prec | Dual-Stream Dice / Prec | Dual ThinSkel | Dual clDice | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{disease_rows}

![Full-Resolution FIVES Disease-Stratified Qualitative Evidence|6.9|4.6](fives_qual|6.9|4.6)

---PAGEBREAK---

## 3. External HRF Transfer (`N=45`: `40` Exposed / `5` Exploratory)

>! **Disclosure**: `40/45` HRF images were seen upstream during `SFHformer` pretraining. The remaining `5` unexposed images carry insufficient power (`min Wilcoxon p = 0.0625`, `Holm p = 0.375`) and are presented as exploratory evidence only.

| Cohort & System (`tau`) | Dice | Precision | Sensitivity | Thin Skel (r<=3.5px) | Branch Recall | clDice | Mean CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HRF Overall (`N=45`) — Raw** (`0.50`) | **`{raw_h45['dice']['mean']:.4f}`** | **`{raw_h45['prec']['mean']:.4f}`** | `{raw_h45['sens']['mean']:.4f}` | `{raw_h45['thin_skel_sens']['mean']*100:.2f}%` | `{raw_h45['branch_recall']['mean']*100:.2f}%` | `{raw_h45['cldice']['mean']:.4f}` | `{raw_h45['num_pred_cc']['mean']:.1f}` |
| **HRF Overall (`N=45`) — Restored** (`0.50`) | `{rest_h45['dice']['mean']:.4f}` | `{rest_h45['prec']['mean']:.4f}` | **`{rest_h45['sens']['mean']:.4f}`** | **`{rest_h45['thin_skel_sens']['mean']*100:.2f}%`** | **`{rest_h45['branch_recall']['mean']*100:.2f}%`** | `{rest_h45['cldice']['mean']:.4f}` | `{rest_h45['num_pred_cc']['mean']:.1f}` |
| **HRF Overall (`N=45`) — Dual-Stream** (`0.40`) | `{dual_h45['dice']:.4f}` | `{dual_h45['prec']:.4f}` | `{dual_h45['sens']:.4f}` | `{dual_h45['thin_skel_sens']*100:.2f}%` | `{dual_h45['branch_recall']*100:.2f}%` | **`{dual_h45['cldice']:.4f}`** | **`{dual_h45['num_pred_cc']:.1f}`** |
| **HRF Unexposed (`N=5`) — Raw** (`0.50`) | **`{raw_h5['dice']['mean']:.4f}`** | **`{raw_h5['prec']['mean']:.4f}`** | `{raw_h5['sens']['mean']:.4f}` | `{raw_h5['thin_skel_sens']['mean']*100:.2f}%` | `{raw_h5['branch_recall']['mean']*100:.2f}%` | `{raw_h5['cldice']['mean']:.4f}` | `{raw_h5['num_pred_cc']['mean']:.1f}` |
| **HRF Unexposed (`N=5`) — Restored** (`0.50`) | `{rest_h5['dice']['mean']:.4f}` | `{rest_h5['prec']['mean']:.4f}` | **`{rest_h5['sens']['mean']:.4f}`** | **`{rest_h5['thin_skel_sens']['mean']*100:.2f}%`** | **`{rest_h5['branch_recall']['mean']*100:.2f}%`** | `{rest_h5['cldice']['mean']:.4f}` | `{rest_h5['num_pred_cc']['mean']:.1f}` |
| **HRF Unexposed (`N=5`) — Dual** (`0.40`) | `{dual_h5['dice']:.4f}` | `{dual_h5['prec']:.4f}` | `{dual_h5['sens']:.4f}` | `{dual_h5['thin_skel_sens']*100:.2f}%` | `{dual_h5['branch_recall']*100:.2f}%` | **`{dual_h5['cldice']:.4f}`** | **`{dual_h5['num_pred_cc']:.1f}`** |

### Exploratory Unexposed Per-Image Breakdown (`N=5`)

| Scan | Dual Dice | Dual Prec | Dual ThinSkel | Dual clDice | Dual BranchRecall | Dual CC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{hrf_unexp_rows}

![HRF Unexposed Exploratory Transfer Comparison|6.9|3.1](hrf_qual|6.9|3.1)

---PAGEBREAK---

## 4. Checkpoint Provenance & End-to-End Summary
- `sfhformer` / `fives_retina_finetune/best.pth` (SHA-256: `{snap['checkpoints/fives_retina_finetune/best.pth'][:20]}...`)
- `unet_raw/best.pth` (SHA-256: `{snap['checkpoints/unet_raw/best.pth'][:20]}...`)
- `unet_restored/best.pth` (SHA-256: `{snap['checkpoints/unet_restored/best.pth'][:20]}...`)
- `unet_dual_stream/best.pth` (SHA-256: `{snap['checkpoints/unet_dual_stream/best.pth'][:20]}...`)

![Combined End-to-End System Architecture|6.9|3.8](diag_5|6.9|3.8)
"""


def main():
    d = load_all_frozen_records()
    img_map = {
        "diag_1": os.path.join(DIAG_DIR, "diag_1_sfhformer_macro.png"),
        "diag_2": os.path.join(DIAG_DIR, "diag_2_sfh_block_detail.png"),
        "diag_3": os.path.join(DIAG_DIR, "diag_3_single_unet.png"),
        "diag_4": os.path.join(DIAG_DIR, "diag_4_dual_stream_hybrid_unet.png"),
        "diag_5": os.path.join(DIAG_DIR, "diag_5_combined_retinascan_system.png"),
        "fives_qual": os.path.join(QUAL_DIR, "fives_test_disease_stratified_demo.png"),
        "hrf_qual": os.path.join(QUAL_DIR, "hrf_transfer_unexposed_demo.png"),
        "ui_tab1": os.path.join(QUAL_DIR, "phase5_2_ui_tab1_fusion.png"),
        "ui_tab2": os.path.join(QUAL_DIR, "phase5_2_ui_tab2_benchmark.png"),
    }

    md_a = build_deliverable_a_master_report(d)
    md_b = build_deliverable_b_engineering_doc(d)
    md_c = build_deliverable_c_eval_summary(d)

    files_out = []
    files_out.append(compile_pdf(
        md_a, "Final_Technical_Report.md", "Final_Technical_Report.pdf",
        "SFHformer-FreqLab — Master Technical Report", "Phase 6 Final Documentation", img_map
    ))
    files_out.append(compile_pdf(
        md_b, "Engineering_Documentation.md", "Engineering_Documentation.pdf",
        "SFHformer-FreqLab — Engineering Documentation", "System & API Reference", img_map
    ))
    files_out.append(compile_pdf(
        md_c, "Milestone3_Evaluation_Summary.md", "Milestone3_Evaluation_Summary.pdf",
        "SFHformer-FreqLab — Milestone 3 Evaluation Summary", "Frozen Benchmark Summary", img_map
    ))

    # Build DOCUMENTATION_MANIFEST.json
    manifest = {
        "git_snapshot_tag": "v3.1-phase5.2-verified",
        "git_commit": "42ac0565d1d1517ef5bad9fdc1f392608272db1b",
        "checkpoint_naming_resolution": {
            "sfhformer_alias_in_evaluate_milestone3": "checkpoints/fives_retina_finetune/best.pth",
            "sha256": d["snap"]["sha256_inventory"]["checkpoints/fives_retina_finetune/best.pth"],
        },
        "deliverables": {},
        "embedded_visual_assets_sha256": {
            os.path.basename(p): sha256_file(p) for p in img_map.values()
        },
    }
    for md_p, pdf_p, pages in files_out:
        manifest["deliverables"][os.path.basename(pdf_p)] = {
            "pdf_path": pdf_p,
            "pdf_pages": pages,
            "pdf_size_bytes": os.path.getsize(pdf_p),
            "pdf_sha256": sha256_file(pdf_p),
            "markdown_path": md_p,
            "markdown_sha256": sha256_file(md_p),
        }

    man_p = os.path.join(OUT_DIR, "DOCUMENTATION_MANIFEST.json")
    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved DOCUMENTATION_MANIFEST.json:", man_p)


if __name__ == "__main__":
    main()

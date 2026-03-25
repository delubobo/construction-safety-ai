"""
ReportLab PDF generator for PPE safety inspection reports.
Produces: header, annotated photo, violation table, corrective action details,
and a signature block.
"""

import io
from datetime import datetime
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)

_PAGE_W, _PAGE_H = letter
_MARGIN = 0.75 * inch

_SEVERITY_COLORS = {
    "High":   colors.HexColor("#FF3B30"),
    "Medium": colors.HexColor("#FF9500"),
    "Low":    colors.HexColor("#FFCC00"),
}


def _pil_to_rl_image(pil_img: Image.Image, max_width: float, max_height: float) -> RLImage:
    """Convert a PIL image to a ReportLab Image flowable, scaled to fit."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    orig_w, orig_h = pil_img.size
    scale = min(max_width / orig_w, max_height / orig_h)
    return RLImage(buf, width=orig_w * scale, height=orig_h * scale)


def build_report(
    annotated_image: Image.Image,
    detections: list[dict],
    site_name: str = "Construction Site",
    project_name: str = "AWS Data Center Expansion",
    inspector_name: str = "Safety Officer",
    image_filename: str = "uploaded_image",
) -> bytes:
    """
    Generate a PDF safety report and return it as bytes.

    Args:
        annotated_image: PIL Image with bounding boxes already drawn.
        detections:      Full detection list from PPEDetector.detect().
        site_name:       Editable site name shown in the report header.
        project_name:    Project name shown in the report header.
        inspector_name:  Name for the signature block.
        image_filename:  Original filename, shown in the report.

    Returns:
        PDF content as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["Normal"]

    bold_style = ParagraphStyle(
        "bold_normal", parent=normal, fontName="Helvetica-Bold", fontSize=10
    )
    small = ParagraphStyle(
        "small", parent=normal, fontSize=8, textColor=colors.grey
    )

    violations = [d for d in detections if d.get("is_violation")]
    now = datetime.now()

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("PPE Safety Inspection Report", h1))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1C1C1E")))
    story.append(Spacer(1, 0.15 * inch))

    meta_data = [
        ["Project:", project_name,   "Date:", now.strftime("%B %d, %Y")],
        ["Site:",    site_name,       "Time:", now.strftime("%H:%M")],
        ["Image:",   image_filename,  "Inspector:", inspector_name],
    ]
    meta_table = Table(meta_data, colWidths=[1 * inch, 2.5 * inch, 1 * inch, 2.5 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN",   (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── Summary Banner ───────────────────────────────────────────────────────
    vcount = len(violations)
    summary_color = (
        colors.HexColor("#FF3B30") if vcount >= 3 else
        colors.HexColor("#FF9500") if vcount >= 1 else
        colors.HexColor("#34C759")
    )
    summary_text = (
        f"{vcount} PPE Violation{'s' if vcount != 1 else ''} Detected"
        if vcount > 0
        else "No PPE Violations Detected"
    )
    summary_table = Table([[Paragraph(f"<b>{summary_text}</b>", bold_style)]], colWidths=[7 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), summary_color),
        ("TEXTCOLOR",    (0, 0), (-1, -1), colors.white),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.2 * inch))

    # ── Annotated Photo ──────────────────────────────────────────────────────
    story.append(Paragraph("Annotated Site Photo", h2))
    story.append(Spacer(1, 0.1 * inch))

    max_img_w = _PAGE_W - 2 * _MARGIN
    max_img_h = 3.5 * inch
    rl_img = _pil_to_rl_image(annotated_image, max_img_w, max_img_h)
    story.append(rl_img)
    story.append(Spacer(1, 0.25 * inch))

    # ── Violation Table ──────────────────────────────────────────────────────
    story.append(Paragraph("Violation Summary", h2))
    story.append(Spacer(1, 0.1 * inch))

    if violations:
        header_row = ["#", "Violation", "Severity", "Confidence", "Corrective Action"]
        table_data = [header_row]
        for i, d in enumerate(violations, 1):
            action = d.get("corrective_action", "")
            # Truncate to ~80 chars for table readability; full text in detail section
            short_action = action[:90] + "…" if len(action) > 90 else action
            table_data.append([
                str(i),
                d.get("label", d.get("class", "Unknown")),
                d.get("severity", "—"),
                f"{d['confidence']:.0%}",
                Paragraph(short_action, ParagraphStyle("small_wrap", parent=normal, fontSize=8)),
            ])

        col_widths = [0.3 * inch, 1.5 * inch, 0.7 * inch, 0.75 * inch, 3.75 * inch]
        viol_table = Table(table_data, colWidths=col_widths, repeatRows=1)

        table_style = [
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#1C1C1E")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 1), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("ALIGN",         (0, 0), (3, -1),  "CENTER"),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D1D6")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F7")]),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        # Colour severity column cells
        for i, d in enumerate(violations, 1):
            sev = d.get("severity", "Medium")
            sev_color = _SEVERITY_COLORS.get(sev, colors.orange)
            table_style.append(("TEXTCOLOR", (2, i), (2, i), sev_color))
            table_style.append(("FONTNAME",  (2, i), (2, i), "Helvetica-Bold"))

        viol_table.setStyle(TableStyle(table_style))
        story.append(viol_table)
    else:
        story.append(Paragraph("No violations detected in this image.", normal))

    story.append(Spacer(1, 0.3 * inch))

    # ── Full Corrective Actions ──────────────────────────────────────────────
    if violations:
        story.append(Paragraph("Required Corrective Actions", h2))
        story.append(Spacer(1, 0.1 * inch))
        for i, d in enumerate(violations, 1):
            story.append(Paragraph(
                f"<b>{i}. {d.get('label', d.get('class'))} [{d.get('severity')}]</b>",
                bold_style,
            ))
            story.append(Paragraph(d.get("corrective_action", ""), normal))
            story.append(Spacer(1, 0.1 * inch))

    # ── Signature Block ──────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#D1D1D6")))
    story.append(Spacer(1, 0.15 * inch))

    sig_data = [
        [
            Paragraph("<b>Safety Officer Signature:</b>", bold_style),
            Paragraph("_______________________________", normal),
            Paragraph(f"<b>Date:</b> {now.strftime('%B %d, %Y')}", bold_style),
        ]
    ]
    sig_table = Table(sig_data, colWidths=[2 * inch, 3 * inch, 2 * inch])
    sig_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(sig_table)
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(
        f"Inspector Name: {inspector_name}",
        ParagraphStyle("inspector", parent=normal, fontSize=9, textColor=colors.grey),
    ))

    doc.build(story)
    return buf.getvalue()

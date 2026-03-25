"""
PIL-based bounding box and label overlay for PPE detection results.
No OpenCV GUI dependency — works headlessly on HF Spaces.
"""

from PIL import Image, ImageDraw, ImageFont

# Colour palette: severity → (box_color, text_bg)
_SEVERITY_COLORS = {
    "High":   ("#FF3B30", "#FF3B30"),   # red
    "Medium": ("#FF9500", "#FF9500"),   # orange
    "Low":    ("#FFCC00", "#FFCC00"),   # yellow
}
_SAFE_COLOR = "#34C759"   # green for compliant detections
_TEXT_COLOR = "#FFFFFF"
_BOX_WIDTH = 3
_FONT_SIZE = 16


def _get_font(size: int = _FONT_SIZE) -> ImageFont.ImageFont:
    """Load a font, falling back to PIL default if truetype not available."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


def annotate_image(image: Image.Image, detections: list[dict]) -> Image.Image:
    """
    Draw bounding boxes and labels on a PIL Image.

    Args:
        image:      Original PIL Image (RGB).
        detections: List of detection dicts from PPEDetector.detect().

    Returns:
        New PIL Image with annotations drawn (original is not modified).
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = _get_font(_FONT_SIZE)
    font_small = _get_font(max(12, _FONT_SIZE - 2))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        severity = det.get("severity", "Medium")
        is_viol = det.get("is_violation", True)

        box_color = _SEVERITY_COLORS.get(severity, ("#FF9500", "#FF9500"))[0] if is_viol else _SAFE_COLOR

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=_BOX_WIDTH)

        # Build label text
        label = det.get("label", det.get("class", "Unknown"))
        conf_pct = int(det["confidence"] * 100)
        text = f"{label} {conf_pct}%"

        # Measure text size
        try:
            bbox_text = draw.textbbox((0, 0), text, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
        except AttributeError:
            # Older PIL fallback
            tw, th = draw.textsize(text, font=font)

        padding = 4
        # Position label above box; clamp to image top
        label_y0 = max(0, y1 - th - padding * 2)
        label_y1 = label_y0 + th + padding * 2
        label_x1 = min(x1 + tw + padding * 2, annotated.width)

        # Draw filled label background
        draw.rectangle([x1, label_y0, label_x1, label_y1], fill=box_color)
        draw.text((x1 + padding, label_y0 + padding), text, fill=_TEXT_COLOR, font=font)

    return annotated


def pil_to_bytes(image: Image.Image, format: str = "JPEG", quality: int = 90) -> bytes:
    """Serialize PIL image to bytes for Streamlit st.image() or file download."""
    import io
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    return buf.getvalue()

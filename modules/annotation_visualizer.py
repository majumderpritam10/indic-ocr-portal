"""Draw REAL annotation boxes on a document image.

Boxes come straight from the 'bbox' field of annotations_master.jsonl (pixel coordinates
[x1, y1, x2, y2] on the recorded page size, normally 1200x1600). If the displayed image has a
different size than the recorded size, the boxes are scaled proportionally and the caller is told.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

FIELD_COLORS = {
    "name": (220, 38, 38),
    "date": (37, 99, 235),
    "address": (5, 150, 105),
    "signature": (147, 51, 234),
    "total_amount": (234, 120, 8),
    "recipient": (219, 39, 119),
    "district": (13, 148, 136),
    "phone_number": (120, 83, 48),
}
_FALLBACK = [(71, 85, 105), (190, 24, 93), (21, 128, 61), (126, 34, 206)]


def color_for(field: str):
    if field in FIELD_COLORS:
        return FIELD_COLORS[field]
    return _FALLBACK[sum(map(ord, field or "x")) % len(_FALLBACK)]


def valid_bbox(b) -> bool:
    try:
        return b is not None and len(b) == 4 and all(v is not None for v in b) and b[2] > b[0] and b[3] > b[1]
    except TypeError:
        return False


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, records: list[dict], recorded_size: tuple[int, int] | None = None):
    """Return (annotated_image, n_drawn, scaled_flag). `records` need 'field_name' and 'bbox'."""
    img = image.convert("RGB").copy()
    W, H = img.size
    sx = sy = 1.0
    scaled = False
    if recorded_size and recorded_size[0] and recorded_size[1] and (W, H) != tuple(recorded_size):
        sx, sy = W / recorded_size[0], H / recorded_size[1]
        scaled = True
    draw = ImageDraw.Draw(img)
    lw = max(2, round(W / 400))
    font = _font(max(12, round(W / 55)))
    n = 0
    for rec in records:
        b = rec.get("bbox")
        if not valid_bbox(b):
            continue
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        col = color_for(rec.get("field_name", ""))
        draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        label = str(rec.get("field_name", ""))
        tw = draw.textlength(label, font=font)
        th = font.size if hasattr(font, "size") else 12
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 4], fill=col)
        draw.text((x1 + 4, ty + 1), label, fill=(255, 255, 255), font=font)
        n += 1
    return img, n, scaled

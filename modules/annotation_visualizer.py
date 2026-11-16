"""Draw REAL annotation boxes on a document image.

Boxes come straight from the 'bbox' field of annotations_master.jsonl (pixel coordinates
[x1, y1, x2, y2] on the recorded page size, normally 1200x1600). If the displayed image has a
different size than the recorded size, the boxes are scaled proportionally and the caller is told.
"""
from __future__ import annotations

import cv2
import numpy as np
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


# ───────────────── measured "tight" text boxes (computed by the website, NOT part of the dataset) ─────────────────
def text_mask(image: Image.Image) -> np.ndarray:
    """Boolean mask of dark (text) pixels: Otsu threshold, then tiny specks (noise) are removed."""
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 8
    return keep[labels]


def measure_text_extent(mask: np.ndarray, bbox, sx: float = 1.0, sy: float = 1.0, min_pixels: int = 20):
    """Smallest rectangle around the dark pixels that lie inside the annotated box.
    Returns {'box': [x1,y1,x2,y2], 'w_frac': .., 'h_frac': ..} in displayed-image pixels, or None."""
    if not valid_bbox(bbox):
        return None
    H, W = mask.shape
    x1, y1 = max(0, int(round(bbox[0] * sx))), max(0, int(round(bbox[1] * sy)))
    x2, y2 = min(W, int(round(bbox[2] * sx))), min(H, int(round(bbox[3] * sy)))
    if x2 <= x1 or y2 <= y1:
        return None
    ys, xs = np.where(mask[y1:y2, x1:x2])
    if ys.size < min_pixels:
        return None
    tx1, tx2, ty1, ty2 = x1 + xs.min(), x1 + xs.max() + 1, y1 + ys.min(), y1 + ys.max() + 1
    return {"box": [int(tx1), int(ty1), int(tx2), int(ty2)],
            "w_frac": (tx2 - tx1) / (x2 - x1), "h_frac": (ty2 - ty1) / (y2 - y1)}


def measure_all(image: Image.Image, records: list[dict], recorded_size=None) -> dict:
    """{record index: measurement} for every record that has a valid box."""
    W, H = image.size
    sx = sy = 1.0
    if recorded_size and recorded_size[0] and recorded_size[1] and (W, H) != tuple(recorded_size):
        sx, sy = W / recorded_size[0], H / recorded_size[1]
    mask = text_mask(image)
    out = {}
    for i, rec in enumerate(records):
        m = measure_text_extent(mask, rec.get("bbox"), sx, sy)
        if m:
            out[i] = m
    return out


def _dashed_rect(draw, box, width, dash=10, color=(0, 0, 0)):
    x1, y1, x2, y2 = box
    for x in range(int(x1), int(x2), dash * 2):
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
    for y in range(int(y1), int(y2), dash * 2):
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)


def draw_tight_boxes(image: Image.Image, measured: dict, only_indexes=None) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    lw = max(2, round(img.size[0] / 500))
    for i, m in measured.items():
        if only_indexes is not None and i not in only_indexes:
            continue
        _dashed_rect(draw, m["box"], lw, dash=max(6, img.size[0] // 120))
    return img

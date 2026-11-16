"""Small image helpers shared by every page."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Protect the server from "decompression bomb" images.
Image.MAX_IMAGE_PIXELS = 80_000_000


def human_size(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def read_image(path):
    """Read an image from disk (unicode-path safe). Returns ndarray or None."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img).convert("RGB")
    return Image.fromarray(cv2.cvtColor(to_bgr(img), cv2.COLOR_BGR2RGB))


def pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def limit_size(img: np.ndarray, max_side: int = 3000):
    """Down-scale very large images so processing stays fast. Returns (img, note|None)."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, None
    s = max_side / m
    out = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out, f"Large image ({w}×{h}) was scaled down to {out.shape[1]}×{out.shape[0]} for processing."


def _b64_png(img: np.ndarray, max_side: int = 1000) -> str:
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(encode_png(img)).decode()


def before_after_slider_html(before: np.ndarray, after: np.ndarray, height: int = 640,
                             label_before: str = "Before", label_after: str = "After") -> str:
    """Self-contained before/after slider (plain HTML + JS, no extra libraries)."""
    before, after = to_bgr(before), to_bgr(after)
    ah, aw = after.shape[:2]
    if before.shape[:2] != (ah, aw):
        before = cv2.resize(before, (aw, ah), interpolation=cv2.INTER_AREA)
    b64_b, b64_a = _b64_png(before), _b64_png(after)
    return f"""
<div style="font-family:Segoe UI,Arial,sans-serif">
  <div style="position:relative;height:{height}px;background:#f1f4fa;border:1px solid #cfd8ea;border-radius:10px;overflow:hidden">
    <img src="data:image/png;base64,{b64_a}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain">
    <img id="top" src="data:image/png;base64,{b64_b}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:contain;clip-path:inset(0 50% 0 0)">
    <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:#5b3fd6"></div>
    <span style="position:absolute;top:8px;left:10px;background:#12244a;color:#fff;padding:2px 8px;border-radius:6px;font-size:12px">{label_before}</span>
    <span style="position:absolute;top:8px;right:10px;background:#5b3fd6;color:#fff;padding:2px 8px;border-radius:6px;font-size:12px">{label_after}</span>
  </div>
  <input id="rng" type="range" min="0" max="100" value="50" style="width:100%;margin-top:8px;accent-color:#5b3fd6">
</div>
<script>
 const r=document.getElementById('rng'),t=document.getElementById('top'),b=document.getElementById('bar');
 r.addEventListener('input',()=>{{t.style.clipPath='inset(0 '+(100-r.value)+'% 0 0)';b.style.left=r.value+'%';}});
</script>"""


def load_upload(filename: str, data: bytes, pdf_page: int = 0) -> dict:
    """Decode an already-validated upload fully in memory (nothing is written to disk).

    Returns dict(image=BGR ndarray, kind, pages, frames, note, width, height).
    Raises ValueError with a human-readable message on failure.
    """
    ext = Path(filename).suffix.lower()
    notes = []
    pages = 1
    frames = 1
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ValueError("PDF support needs the 'pymupdf' package (pip install pymupdf).") from e
        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as e:
            raise ValueError(f"Could not open this PDF: {e}") from e
        if doc.needs_pass:
            raise ValueError("This PDF is password-protected; please upload an unlocked copy.")
        pages = doc.page_count
        if pages == 0:
            raise ValueError("This PDF has no pages.")
        idx = min(max(pdf_page, 0), pages - 1)
        page = doc.load_page(idx)
        zoom = min(200 / 72, 3500 / max(page.rect.width, page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pil = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        notes.append(f"Rendered PDF page {idx + 1} of {pages}.")
        kind = "PDF"
    else:
        try:
            pil = Image.open(io.BytesIO(data))
            frames = getattr(pil, "n_frames", 1)
            pil.load()
        except Image.DecompressionBombError as e:
            raise ValueError("Image has too many pixels and was rejected for safety.") from e
        except Exception as e:
            raise ValueError(f"The image could not be decoded ({e}). The file may be corrupt.") from e
        pil = ImageOps.exif_transpose(pil)
        if pil.mode in ("RGBA", "LA") or (pil.mode == "P" and "transparency" in pil.info):
            rgba = pil.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            pil = bg
        elif pil.mode.startswith("I"):
            arr = np.array(pil).astype(np.float64)
            hi = arr.max() or 1.0
            pil = Image.fromarray((arr / hi * 255).astype(np.uint8)).convert("RGB")
        else:
            pil = pil.convert("RGB")
        kind = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".tif": "TIFF", ".tiff": "TIFF"}[ext]
        if frames > 1:
            notes.append(f"Multi-page image: only the first of {frames} frames is processed.")
    img = pil_to_bgr(pil)
    img, note = limit_size(img)
    if note:
        notes.append(note)
    return {"image": img, "kind": kind, "pages": pages, "frames": frames,
            "note": " ".join(notes), "width": img.shape[1], "height": img.shape[0]}

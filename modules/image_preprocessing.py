"""Image preprocessing pipeline — same operations and parameters as scripts/preprocess_images.py
(Phase 7 of the data-engineering project):

    grayscale  ->  CLAHE (clipLimit=2.0, 8x8 tiles)  ->  bilateral filter (d=5, sigma=50/50)
               ->  Otsu threshold  ->  deskew (only if |angle| > 0.5 degrees)

The functions are re-implemented here (instead of imported) because the original script runs
a whole batch job and writes files the moment it is imported. This module never writes anything
and never modifies the input image.

`run_pipeline` returns a record of every stage together with a `ran` flag, so the website only
reports a step as done when the step really executed.
"""
from __future__ import annotations

import cv2
import numpy as np

from utils.image_utils import encode_png

MAX_DESKEW_DEG = 10.5  # refuse to "correct" larger angles: they are almost always mis-estimates


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, grid=(8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid).apply(gray)


def apply_bilateral(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)


def apply_otsu(gray: np.ndarray) -> np.ndarray:
    _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out


def _profile_score(mask: np.ndarray, angle: float) -> float:
    """How 'line-like' the text is after rotating by `angle`: variance of the row sums. It is largest when text
    lines (and the Bengali/Hindi headline bar) are exactly horizontal."""
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return float(np.var(rot.sum(axis=1)))


def _count_text_lines(mask: np.ndarray) -> int:
    """Number of separate horizontal text bands (rows of text) in a boolean text mask."""
    occupied = mask.sum(axis=1) >= 2
    bands, run, gap = 0, 0, 0
    min_gap = max(2, int(0.005 * mask.shape[0]))
    for v in occupied:
        if v:
            run += 1
            gap = 0
        else:
            gap += 1
            if gap == min_gap and run >= 3:
                bands += 1
                run = 0
    if run >= 3:
        bands += 1
    return bands


def _estimate_rect(mask: np.ndarray) -> float:
    """Tilt of the minimum-area rectangle around all text pixels. Exact for page-like blocks with several lines."""
    ys, xs = np.where(mask)
    if len(xs) > 200_000:
        idx = np.random.default_rng(0).choice(len(xs), 200_000, replace=False)
        xs, ys = xs[idx], ys[idx]
    angle = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))[-1]
    if angle > 45:       # OpenCV >= 4.5 reports (0, 90]; fold into (-45, 45]
        angle -= 90
    elif angle <= -45:
        angle += 90
    return float(round(angle, 2))


def _estimate_profile(mask: np.ndarray) -> float:
    """Projection-profile search over -10..+10 degrees; precise for a single word or line."""
    h, w = mask.shape
    m = mask.astype(np.float32)
    s = 500.0 / max(h, w)
    if s < 1.0:
        m = cv2.resize(m, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), 1.3)   # fair comparison between re-sampled angles and angle 0
    base = _profile_score(m, 0.0)
    coarse = np.arange(-10.0, 10.01, 0.5)
    best = float(coarse[int(np.argmax([_profile_score(m, float(a)) for a in coarse]))])
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    fscores = [_profile_score(m, float(a)) for a in fine]
    best = float(fine[int(np.argmax(fscores))])
    return float(round(best, 2)) if max(fscores) >= base * 1.03 else 0.0


def estimate_skew(binary: np.ndarray) -> float:
    """Skew angle in degrees: the rotation (same sign convention as `deskew`) that makes the text horizontal.
    Pages with 4 or more text lines use the rectangle method (exact for forms); a single word or a few lines use the
    projection-profile method (the rectangle method gives wrong angles for a lone word)."""
    mask = binary < 128
    if int(mask.sum()) < 100:
        return 0.0
    if _count_text_lines(mask) >= 4:
        return _estimate_rect(mask)
    return _estimate_profile(mask)


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.2:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    border = 255 if img.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def quality_metrics(img: np.ndarray) -> dict:
    """Metrics computed from the pixels of the image you pass in (never hard-coded)."""
    gray = to_gray(img)
    return {
        "contrast": round(float(np.std(gray)), 2),                       # std-dev of intensities
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),  # Laplacian variance
        "brightness": round(float(np.mean(gray)), 2),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "size_bytes_png": len(encode_png(img)),
    }


def run_pipeline(img: np.ndarray, *, grayscale=True, clahe=True, bilateral=True,
                 otsu=True, deskew_on=True) -> dict:
    """Run the selected steps in the fixed order above.

    Returns {'final': ndarray, 'stages': [{'name','image','ran','note'}], 'skew_angle': float|None,
             'steps_run': [names]}
    """
    stages = []
    cur = to_gray(img) if (grayscale or clahe or bilateral or otsu or deskew_on) else img.copy()
    stages.append({"name": "Original", "image": img, "ran": True, "note": "Input image (unchanged)"})

    def step(name, enabled, fn):
        nonlocal cur
        if not enabled:
            stages.append({"name": name, "image": cur, "ran": False, "note": "Skipped (switched off)"})
            return
        cur = fn(cur)
        stages.append({"name": name, "image": cur, "ran": True, "note": "Done"})

    step("Grayscale", grayscale, lambda x: x)  # conversion already applied above
    step("CLAHE", clahe, apply_clahe)
    step("Bilateral filter", bilateral, apply_bilateral)
    step("Otsu threshold", otsu, apply_otsu)

    skew = None
    too_little = False
    if deskew_on:
        too_little = int(np.count_nonzero((cur if otsu else apply_otsu(cur)) < 128)) < 200
    if deskew_on and too_little:
        stages.append({"name": "Deskew", "image": cur, "ran": False,
                       "note": "Skipped: almost no text was found, so the tilt cannot be measured"})
    elif deskew_on:
        if not otsu:
            # skew estimation needs a binary-like picture; use a temporary Otsu copy for measuring only
            skew = estimate_skew(apply_otsu(cur))
        else:
            skew = estimate_skew(cur)
        if abs(skew) > MAX_DESKEW_DEG:
            stages.append({"name": "Deskew", "image": cur, "ran": False,
                           "note": f"Estimated angle {skew}° is outside ±{MAX_DESKEW_DEG}°; not applied"})
        elif abs(skew) <= 0.5:
            stages.append({"name": "Deskew", "image": cur, "ran": True,
                           "note": f"Measured {skew}° — within tolerance, no rotation needed"})
        else:
            cur = deskew(cur, skew)
            stages.append({"name": "Deskew", "image": cur, "ran": True, "note": f"Rotated by {skew}°"})
    else:
        stages.append({"name": "Deskew", "image": cur, "ran": False, "note": "Skipped (switched off)"})

    return {"final": cur, "stages": stages, "skew_angle": skew,
            "steps_run": [s["name"] for s in stages if s["ran"] and s["name"] != "Original"]}

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

MAX_DESKEW_DEG = 15.0  # refuse to "correct" larger angles: they are almost always mis-estimates


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


def estimate_skew(binary: np.ndarray) -> float:
    """Skew angle in degrees (positive = text is rotated counter-clockwise), via the
    minimum-area rectangle around the dark (text) pixels — the same idea as the project script."""
    ys, xs = np.where(binary < 128)
    if len(xs) < 100:
        return 0.0
    if len(xs) > 200_000:  # subsample: the rectangle only depends on the outline
        idx = np.random.default_rng(0).choice(len(xs), 200_000, replace=False)
        xs, ys = xs[idx], ys[idx]
    pts = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(pts)[-1]
    # OpenCV >= 4.5 reports (0, 90]; fold into (-45, 45]
    if angle > 45:
        angle -= 90
    elif angle <= -45:
        angle += 90
    return float(round(angle, 2))


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
        ys, xs = np.where((cur if otsu else apply_otsu(cur)) < 128)
        if ys.size == 0:
            too_little = True
        else:
            h_txt, w_txt = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            # a single line of text (very wide, very short block) has too little geometry to measure tilt
            too_little = (w_txt / h_txt) > 5 or h_txt < 0.10 * cur.shape[0] and w_txt < 0.5 * cur.shape[1]
    if deskew_on and too_little:
        stages.append({"name": "Deskew", "image": cur, "ran": False,
                       "note": "Skipped: only a single line / very small block of text was found, which is too little to measure tilt reliably"})
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

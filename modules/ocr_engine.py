"""OCR integration layer.

IMPORTANT — what is (and is not) connected:
  * The data-engineering repository you shared contains NO trained OCR model.
  * Therefore this module offers two real back-ends and never invents text:
      1. "team"      -> your teammates' model, plugged in through modules/team_ocr_plugin.py
                        (see modules/team_ocr_plugin_template.py for the 3-function contract).
      2. "tesseract" -> the open-source Tesseract engine (needs the 'tesseract-ocr' program plus
                        the ben / hin / eng language packs). It is a generic BASELINE, not the project's own model,
                        and the website labels it that way.
  * If neither back-end works, an OcrUnavailable error explains exactly what is missing.
"""
from __future__ import annotations

import importlib
import time

import cv2

TESS_LANG = {"bn": "ben", "hi": "hin", "en": "eng"}
PACK_NAMES = {"ben": "tesseract-ocr-ben", "hin": "tesseract-ocr-hin", "eng": "tesseract-ocr-eng"}


class OcrUnavailable(RuntimeError):
    """Raised when no usable OCR back-end exists. The message is safe to show to the user."""


# ───────────────────────── team model plug-in ─────────────────────────
def _load_team_plugin():
    try:
        return importlib.import_module("modules.team_ocr_plugin"), None
    except ModuleNotFoundError as e:
        if e.name == "modules.team_ocr_plugin":
            return None, None  # simply not installed
        return None, f"team_ocr_plugin failed to import: missing package '{e.name}'"
    except Exception as e:  # noqa: BLE001
        return None, f"team_ocr_plugin failed to import: {e}"


def team_status() -> dict:
    mod, err = _load_team_plugin()
    if mod is None:
        return {"available": False,
                "detail": err or "No team OCR model installed (modules/team_ocr_plugin.py not found)."}
    if not hasattr(mod, "run_ocr"):
        return {"available": False, "detail": "modules/team_ocr_plugin.py has no run_ocr() function."}
    return {"available": True, "detail": getattr(mod, "MODEL_NAME", "Team OCR model")}


# ───────────────────────── Tesseract ─────────────────────────
def tesseract_status() -> dict:
    try:
        import pytesseract
    except ImportError:
        return {"available": False, "langs": [], "version": None,
                "detail": "Python package 'pytesseract' is not installed (pip install pytesseract)."}
    try:
        version = str(pytesseract.get_tesseract_version())
        langs = sorted(pytesseract.get_languages(config=""))
        return {"available": True, "langs": langs, "version": version, "detail": f"Tesseract {version}"}
    except Exception:  # noqa: BLE001
        return {"available": False, "langs": [], "version": None,
                "detail": "The Tesseract program is not installed on this machine "
                          "(Windows: install from github.com/UB-Mannheim/tesseract/wiki; "
                          "Streamlit Cloud: add packages.txt — already included in this project)."}


def backend_summary() -> list[dict]:
    t = team_status()
    s = tesseract_status()
    rows = [
        {"id": "team", "label": "Team OCR model (plug-in)", "available": t["available"], "detail": t["detail"]},
        {"id": "tesseract", "label": "Tesseract OCR (baseline)", "available": s["available"],
         "detail": s["detail"] + (f" — language packs: {', '.join(s['langs'])}" if s["available"] else "")},
    ]
    return rows


def _pad(image, px=None):
    """White border around the page. Tesseract often finds nothing when text touches the image edge
    (typical for tightly cropped single-line images)."""
    h, w = image.shape[:2]
    px = px or max(30, min(h, w) // 10)
    value = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(image, px, px, px, px, cv2.BORDER_CONSTANT, value=value)


def _tesseract_once(image, lang, psm, timeout):
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, lang=lang, config=f"--psm {psm}",
                                     output_type=Output.DICT, timeout=timeout)
    lines, confs, order = {}, [], []
    for i, word in enumerate(data["text"]):
        w = (word or "").strip()
        if not w:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(w)
        try:
            c = float(data["conf"][i])
            if c >= 0:
                confs.append(c)
        except (TypeError, ValueError):
            pass
    return "\n".join(" ".join(lines[k]) for k in order), confs


def _tesseract_run(image, lang_codes, psm, timeout):
    st = tesseract_status()
    if not st["available"]:
        raise OcrUnavailable(st["detail"])
    wanted = [TESS_LANG[c] for c in lang_codes if c in TESS_LANG]
    missing = [w for w in wanted if w not in st["langs"]]
    if missing:
        raise OcrUnavailable(
            "Tesseract language data missing: " + ", ".join(missing) + ". "
            "Install it (Ubuntu/Streamlit Cloud: " + ", ".join(PACK_NAMES[m] for m in missing) +
            " in packages.txt; Windows: re-run the Tesseract installer and tick the language) "
            "— the OCR was NOT run, so no text is shown.")
    lang = "+".join(wanted)
    h0, w0 = image.shape[:2]
    if max(h0, w0) < 600:   # tiny crops: enlarge so the letters are big enough for Tesseract
        f = min(4.0, 1200 / max(h0, w0))
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    padded = _pad(image)
    tried, text, confs = [], "", []
    for p in [psm] + [x for x in (6, 7, 11) if x != psm]:   # only goes on if the previous mode found nothing
        text, confs = _tesseract_once(padded, lang, p, timeout)
        tried.append(p)
        if text.strip():
            break
    return {
        "text": text,
        "confidence": round(sum(confs) / len(confs), 1) if confs else None,
        "word_count": len(confs),
        "backend": "Tesseract OCR (baseline engine, not the project model)",
        "languages": lang,
        "psm_used": tried[-1],
        "psm_tried": tried,
    }


def run_ocr(image, lang_codes, backend: str = "auto", psm: int = 3, timeout: int = 120) -> dict:
    """Run real OCR. `image` is a numpy array (gray or BGR). Returns a dict with 'text',
    'confidence' (mean word confidence 0-100 or None), 'backend', 'languages', 'seconds'."""
    t0 = time.time()
    use = backend
    if backend == "auto":
        use = "team" if team_status()["available"] else "tesseract"
    if use == "team":
        mod, err = _load_team_plugin()
        if mod is None or not hasattr(mod, "run_ocr"):
            raise OcrUnavailable(err or "Team OCR plug-in is not installed.")
        out = mod.run_ocr(image, list(lang_codes))
        if not isinstance(out, dict) or "text" not in out:
            raise OcrUnavailable("team_ocr_plugin.run_ocr() must return a dict containing 'text'.")
        res = {"text": out["text"], "confidence": out.get("confidence"), "word_count": out.get("word_count"),
               "backend": getattr(mod, "MODEL_NAME", "Team OCR model"), "languages": "+".join(lang_codes)}
    elif use == "tesseract":
        res = _tesseract_run(image, lang_codes, psm, timeout)
    else:
        raise OcrUnavailable(f"Unknown OCR back-end '{backend}'.")
    res["seconds"] = round(time.time() - t0, 2)
    return res


def rank(res: dict):
    """Which read is better? Reads the engine itself is unsure about (confidence < 30%, or fewer than 2 letters) are
    treated as unreliable garbage and always lose to a reliable read, however many letters they contain.
    Among reliable reads, engine confidence is weighted by how many letters were found."""
    letters = sum(ch.isalpha() for ch in res.get("text", ""))
    conf = res["confidence"] if res.get("confidence") is not None else 50.0
    reliable = conf >= 30 and letters >= 2
    return (1 if reliable else 0, conf * letters / (letters + 4.0) if reliable else conf)


def run_ocr_best(variants, lang_codes, backend: str = "auto", psm: int = 3, timeout: int = 120,
                 accept_conf: float = 75.0) -> dict:
    """Try the image variants one after another and keep the best read.
    `variants` = [(name, image), ...]. Stops early when a read is already good (confidence >= accept_conf with
    at least 3 letters), so clean scans cost one pass. A team model is run on the first variant only.
    The result gets 'variant' (the winner) and 'attempts' (what was tried) so the page can show it honestly."""
    use_team = backend == "team" or (backend == "auto" and team_status()["available"])
    if use_team:
        variants = variants[:1]
    best, attempts = None, []
    for name, img in variants:
        res = run_ocr(img, lang_codes, backend, psm, timeout)
        score = rank(res)
        letters = sum(ch.isalpha() for ch in res.get("text", ""))
        attempts.append({"variant": name, "confidence": res.get("confidence"), "letters": letters})
        if best is None or score > best[0]:
            best = (score, res, name)
        if letters >= 3 and (res.get("confidence") or 0) >= accept_conf:
            break
    res = best[1]
    res["variant"], res["attempts"] = best[2], attempts
    return res

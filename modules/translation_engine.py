"""Translation integration layer.

Order of back-ends in "auto" mode (first one that is configured and works wins):
  1. team   -> your teammates' translator (modules/team_translation_plugin.py, translate(text, source, target) -> str)
  2. azure  -> Microsoft Azure Translator (official API, free tier ~2 million characters/month).
               Needs two secrets: AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION
               (Streamlit Cloud: App settings -> Secrets; on your PC: .streamlit/secrets.toml). Never put keys in code.
  3. mymemory -> free MyMemory service via 'deep-translator'. If MYMEMORY_EMAIL is set (any real email address, no sign-up)
               its daily free limit rises from ~5,000 to ~50,000 characters, so it is tried BEFORE Google.
  4. google -> free Google Translate web endpoint via 'deep-translator' (unofficial, easily rate-limited)
If every back-end fails, TranslationUnavailable explains why; a fake translation is never produced.
"""
from __future__ import annotations

import importlib
import os
import time

LANGS = {"en": "English", "hi": "Hindi", "bn": "Bengali"}
_CHUNK = 4000
_MM_CHUNK = 450
_MM_NAMES = {"en": "english", "hi": "hindi", "bn": "bengali"}
_AZURE_URL = "https://api.cognitive.microsofttranslator.com/translate"


class TranslationUnavailable(RuntimeError):
    """Message is safe to show to the user (it never contains secrets)."""


# ───────────────────────── configuration ─────────────────────────
def _secret(name: str, default: str = "") -> str:
    val = os.environ.get(name, "")
    if val:
        return val.strip()
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:  # noqa: BLE001  (no secrets file is perfectly normal)
        return default


def azure_config():
    key, region = _secret("AZURE_TRANSLATOR_KEY"), _secret("AZURE_TRANSLATOR_REGION")
    endpoint = _secret("AZURE_TRANSLATOR_ENDPOINT", _AZURE_URL) or _AZURE_URL
    if key:
        return {"key": key, "region": region, "endpoint": endpoint.rstrip("/")}
    return None


def _team_plugin():
    try:
        mod = importlib.import_module("modules.team_translation_plugin")
        return mod if hasattr(mod, "translate") else None
    except Exception:  # noqa: BLE001
        return None


def backend_summary() -> list[dict]:
    try:
        import deep_translator  # noqa: F401
        free = {"available": True, "detail": "deep-translator installed (needs internet when used; unofficial, may be rate-limited)"}
    except ImportError:
        free = {"available": False, "detail": "Package 'deep-translator' is not installed (pip install deep-translator)."}
    az = azure_config()
    return [
        {"id": "team", "label": "Team translation model/API (plug-in)", "available": _team_plugin() is not None,
         "detail": "modules/team_translation_plugin.py found" if _team_plugin() else
                   "No team translator installed (modules/team_translation_plugin.py not found)."},
        {"id": "azure", "label": "Microsoft Azure Translator (official API)", "available": az is not None,
         "detail": "Key configured (not yet verified until a translation is run)" if az else
                   "Not configured: add AZURE_TRANSLATOR_KEY and AZURE_TRANSLATOR_REGION to Secrets."},
        {"id": "mymemory", "label": "Free service: MyMemory (+ Google fallback) via deep-translator", **free,
         "detail": free["detail"] + ("; MYMEMORY_EMAIL set (higher daily limit)" if _secret("MYMEMORY_EMAIL")
                                      else "; tip: set MYMEMORY_EMAIL in Secrets for a 10x higher daily limit")},
    ]


# ───────────────────────── helpers ─────────────────────────
def _chunks(text: str, limit: int = _CHUNK):
    cur, size = [], 0
    for line in text.split("\n"):
        if size + len(line) + 1 > limit and cur:
            yield "\n".join(cur)
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        yield "\n".join(cur)


def _azure(text: str, source, target: str, cfg: dict) -> str:
    """Official Azure Translator REST call. Each non-empty line is sent as its own element,
    so the line structure of the OCR text is preserved exactly."""
    import requests

    headers = {"Ocp-Apim-Subscription-Key": cfg["key"], "Content-Type": "application/json; charset=UTF-8"}
    if cfg["region"]:
        headers["Ocp-Apim-Subscription-Region"] = cfg["region"]
    params = {"api-version": "3.0", "to": target}
    if source in LANGS:
        params["from"] = source
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip()]
    out = list(lines)
    batch, size = [], 0

    def flush(batch_idx):
        body = [{"Text": lines[i]} for i in batch_idx]
        r = requests.post(cfg["endpoint"], params=params, headers=headers, json=body, timeout=25)
        if r.status_code != 200:
            hint = {401: "key or region is wrong", 403: "free quota used up or access denied", 429: "too many requests"}.get(r.status_code, "service error")
            raise RuntimeError(f"Azure returned HTTP {r.status_code} ({hint})")
        data = r.json()
        for i, item in zip(batch_idx, data):
            out[i] = item["translations"][0]["text"]

    for i in idx:
        if batch and (size + len(lines[i]) > 20000 or len(batch) >= 200):
            flush(batch)
            batch, size = [], 0
        batch.append(i)
        size += len(lines[i])
    if batch:
        flush(batch)
    return "\n".join(out)


def _google(text: str, source, target: str) -> str:
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source=source if source in LANGS else "auto", target=target)
    parts = []
    for chunk in _chunks(text):
        last = None
        for attempt in range(3):
            try:
                parts.append(tr.translate(chunk) or "")
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        else:
            raise last
    return "\n".join(parts)


def _mymemory(text: str, source, target: str) -> str:
    from deep_translator import MyMemoryTranslator
    if source not in _MM_NAMES:
        raise ValueError("MyMemory needs the document language to be known (not 'auto').")
    email = _secret("MYMEMORY_EMAIL")
    try:
        tr = (MyMemoryTranslator(source=_MM_NAMES[source], target=_MM_NAMES[target], email=email) if email
              else MyMemoryTranslator(source=_MM_NAMES[source], target=_MM_NAMES[target]))
    except TypeError:  # older deep-translator without the email option
        tr = MyMemoryTranslator(source=_MM_NAMES[source], target=_MM_NAMES[target])
    return "\n".join(tr.translate(c) or "" for c in _chunks(text, _MM_CHUNK))


# ───────────────────────── public API ─────────────────────────
def translate(text: str, source: str | None, target: str, backend: str = "auto") -> dict:
    """Translate `text`. `source` may be None/'auto'. Returns {'text','backend','seconds','note'}."""
    if target not in LANGS:
        raise TranslationUnavailable(f"Unsupported target language '{target}'.")
    if not text or not text.strip():
        raise TranslationUnavailable("There is no extracted text to translate.")
    if source == target:
        return {"text": text, "backend": "none", "seconds": 0.0,
                "note": "Source and target language are the same, so no translation was needed."}
    t0 = time.time()
    attempts = []   # human-readable reasons earlier back-ends were skipped/failed
    if backend != "auto":
        order = [backend]
    elif _secret("MYMEMORY_EMAIL"):
        order = ["team", "azure", "mymemory", "google"]
    else:
        order = ["team", "azure", "google", "mymemory"]

    for use in order:
        try:
            if use == "team":
                mod = _team_plugin()
                if mod is None:
                    continue
                out, name = mod.translate(text, source or "auto", target), getattr(mod, "MODEL_NAME", "Team translation model")
            elif use == "azure":
                cfg = azure_config()
                if cfg is None:
                    continue
                out, name = _azure(text, source, target, cfg), "Microsoft Azure Translator (official API)"
            elif use == "google":
                out, name = _google(text, source, target), "Google Translate (via deep-translator, unofficial free endpoint)"
            elif use == "mymemory":
                out, name = _mymemory(text, source, target), "MyMemory (via deep-translator, free fallback)"
            else:
                raise TranslationUnavailable(f"Unknown translation back-end '{use}'.")
        except ImportError:
            attempts.append(f"{use}: package not installed")
            continue
        except Exception as e:  # noqa: BLE001
            attempts.append(f"{use}: {type(e).__name__}" + (f" — {e}" if use == "azure" else ""))
            continue
        note = ("Earlier back-ends failed, so a fallback was used (" + "; ".join(attempts) + ")") if attempts else ""
        return {"text": out, "backend": name, "seconds": round(time.time() - t0, 2), "note": note}

    why = ("; ".join(attempts)) if attempts else "no translation back-end is configured"
    raise TranslationUnavailable(
        f"Translation failed ({why}). Free services are often rate-limited — wait a minute and retry, or configure the "
        "official Azure Translator key (see README, section 'Reliable translation'). No translation was produced.")

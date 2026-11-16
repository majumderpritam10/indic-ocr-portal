"""Script / language detection.

How it works (and what it does NOT do):
  * The only real detector available in this project is Unicode-block analysis, the same
    rule the data-engineering audit used (CHK_11: Bengali U+0980-09FF, Devanagari U+0900-097F,
    Basic Latin). It needs TEXT, so on an uploaded image we first run a quick OCR pass over
    all three languages and then measure which script the recognised letters belong to.
  * The number reported is the SHARE OF RECOGNISED LETTERS in that script. It is a plain
    character ratio — NOT a probability from a machine-learning model — and the website labels it that way.
"""
from __future__ import annotations

from modules.paths import LANG_NAMES

_BN = (0x0980, 0x09FF)
_HI = (0x0900, 0x097F)


def script_counts(text: str) -> dict:
    bn = hi = en = other = 0
    for ch in text:
        o = ord(ch)
        if _BN[0] <= o <= _BN[1] and ch.isalpha():
            bn += 1
        elif _HI[0] <= o <= _HI[1] and ch.isalpha():
            hi += 1
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            en += 1
        elif ch.isalpha():
            other += 1
    return {"bn": bn, "hi": hi, "en": en, "other": other}


def analyse_text(text: str) -> dict:
    """Return {'script': 'bn'|'hi'|'en'|None, 'shares': {...}, 'present': [...], 'letters': n}."""
    c = script_counts(text)
    total = sum(c.values())
    if total == 0:
        return {"script": None, "shares": {}, "present": [], "letters": 0}
    shares = {k: c[k] / total for k in ("bn", "hi", "en", "other")}
    main = max(("bn", "hi", "en"), key=lambda k: c[k])
    if c[main] == 0:
        main = None
    present = [k for k in ("bn", "hi", "en") if shares[k] >= 0.10]
    return {"script": main, "shares": shares, "present": present, "letters": total}


def language_label(code: str | None) -> str:
    return LANG_NAMES.get(code, "Unknown") if code else "Unknown"


def ocr_languages_for(script: str | None, present: list[str] | None = None) -> list[str]:
    """Which language packs to use for the real OCR pass. English is always added because
    Indian documents almost always contain Latin words/numbers/currency."""
    langs = []
    if script and script != "en":
        langs.append(script)
    for p in (present or []):
        if p not in langs and p != "en":
            langs.append(p)
    langs.append("en")
    return langs

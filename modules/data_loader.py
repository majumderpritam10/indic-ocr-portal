"""Loads the REAL project files (manifest, annotations, catalogs, reports).

Nothing here invents data: if a file is missing, an empty DataFrame / None is returned
and the page shows a clear 'file not found' message.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.paths import (AUG_DIR, CORRECTED_DIR, DATA_DIR, FINAL_IMAGES_DIR, PREP_IMAGES_DIR,
                           RAW_IMAGES_DIR, REPORTS_DIR, norm_rel, resolve)


def _csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame:
    """data/corrected/manifest_master.csv — the usable corpus (originals + augmented)."""
    df = _csv(CORRECTED_DIR / "manifest_master.csv")
    if df.empty:
        return df
    df["image_rel"] = df["image_path"].map(norm_rel)
    df["has_image"] = df["image_rel"].map(lambda p: resolve(p).exists())
    df["doc_type"] = df["doc_type"].fillna("unknown")
    df["kind"] = df["is_augmented"].map(lambda v: "augmented" if bool(v) else "original")
    return df


@st.cache_data(show_spinner=False)
def load_annotations() -> pd.DataFrame:
    """data/corrected/annotations_master.jsonl — one row per annotated field."""
    path = CORRECTED_DIR / "annotations_master.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_augmented_catalog() -> pd.DataFrame:
    return _csv(AUG_DIR / "augmented_catalog.csv")


@st.cache_data(show_spinner=False)
def load_quarantine_manifest() -> pd.DataFrame:
    return _csv(CORRECTED_DIR / "quarantine" / "quarantined_manifest.csv")


@st.cache_data(show_spinner=False)
def load_quarantine_annotations() -> pd.DataFrame:
    path = CORRECTED_DIR / "quarantine" / "quarantined_annotations.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_preprocessed_catalog() -> pd.DataFrame:
    return _csv(DATA_DIR / "preprocessed" / "preprocessed_catalog.csv")


@st.cache_data(show_spinner=False)
def load_disposition_log() -> pd.DataFrame:
    return _csv(CORRECTED_DIR / "disposition_log.csv")


@st.cache_data(show_spinner=False)
def load_report_csv(name: str) -> pd.DataFrame:
    return _csv(REPORTS_DIR / name)


@st.cache_data(show_spinner=False)
def load_report_json(name: str):
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def report_image(name: str):
    p = REPORTS_DIR / name
    return p if p.exists() else None


def raw_image_path(doc_id: str):
    p = RAW_IMAGES_DIR / f"{doc_id}.png"
    return p if p.exists() else None


def prep_image_path(doc_id: str):
    p = PREP_IMAGES_DIR / f"{doc_id}.png"
    return p if p.exists() else None


def final_image_path(doc_id: str):
    p = FINAL_IMAGES_DIR / f"{doc_id}.png"
    return p if p.exists() else None


def augmented_image_path(parent_id: str, recipe: str):
    p = AUG_DIR / "images" / f"{parent_id}_aug_{recipe}.png"
    return p if p.exists() else None


@st.cache_data(show_spinner=False)
def sha256_of(path_str: str) -> str:
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

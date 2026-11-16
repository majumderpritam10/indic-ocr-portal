"""Screen 2 — Before / After image preprocessing.

Tab 1 (main demo): a REAL noisy page from the dataset (the project's own augmented copy with noise, blur and
        rotation) is cleaned by the preprocessing pipeline right now, in front of you.
Tab 2: the project's stored raw image next to its stored preprocessed image. In this synthetic dataset the raw
        page is blank (the text was rendered later), which the page says openly.
Every metric is calculated from the image pixels; nothing is hard-coded.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules import data_loader as dl
from modules import ui
from modules.image_preprocessing import quality_metrics, run_pipeline
from modules.paths import AUG_DIR, LANG_NAMES
from utils.image_utils import before_after_slider_html, encode_png, human_size, read_image, to_pil


def _metric_table(before, after, before_bytes, after_bytes):
    mb, ma = quality_metrics(before), quality_metrics(after)
    rows = [
        ("Contrast (std-dev of pixel intensity)", mb["contrast"], ma["contrast"]),
        ("Sharpness (Laplacian variance)", mb["sharpness"], ma["sharpness"]),
        ("Image dimensions", f"{mb['width']} × {mb['height']}", f"{ma['width']} × {ma['height']}"),
        ("File size", human_size(before_bytes), human_size(after_bytes)),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Metric", "Before", "After"]).astype(str),
                 hide_index=True, use_container_width=True)
    return mb, ma


@st.cache_data(show_spinner=False)
def _run_cached(path_str: str, opts: tuple):
    """Run the real pipeline on one dataset image (cached so toggling widgets stays fast)."""
    src = read_image(path_str)
    if src is None:
        return None, None
    return src, run_pipeline(src, **dict(opts))


# ───────────────────────── tab 1: live demo on noisy pages ─────────────────────────
def _live_tab():
    st.markdown("Pick a document. **Before** = the dataset's own noisy copy (noise, blur, small rotation added by the "
                "project's augmentation phase). **After** = that same image cleaned by the pipeline, run live right now.")
    man = dl.load_manifest()
    files = {}
    for recipe in ("moderate", "mild"):
        for p in sorted((AUG_DIR / "images").glob(f"*_aug_{recipe}.png")):
            parent = p.name.split("_aug_")[0]
            files.setdefault(parent, {})[recipe] = p
    if not files:
        st.warning("No augmented images are included in this deployment, so the live demo has nothing to process.")
        return
    lang_of = dict(zip(man["document_id"], man["script"])) if not man.empty else {}
    parents = sorted(files)
    c1, c2, c3 = st.columns([1, 2, 1.4])
    lang = c1.selectbox("Language", ["All"] + list(LANG_NAMES),
                        format_func=lambda c: "All" if c == "All" else f"{LANG_NAMES[c]} ({c})", key="ba_lang")
    if lang != "All":
        parents = [p for p in parents
                   if lang_of.get(f"{p}_aug_moderate", lang_of.get(f"{p}_aug_mild")) == lang]
    if not parents:
        st.info("No documents for this language.")
        return
    doc = c2.selectbox("Document", parents, key="ba_doc",
                       format_func=lambda d: f"{d}  ·  noise levels: {', '.join(sorted(files[d]))}")
    level = c3.radio("Noise level", sorted(files[doc], key=lambda r: r != "moderate"), horizontal=True, key="ba_level",
                     help="'moderate' is clearly noisy; 'mild' is only slightly degraded.")
    path = files[doc][level]

    cols = st.columns(5)
    opts = {
        "grayscale": cols[0].checkbox("Grayscale", True, key="ba_g"),
        "clahe": cols[1].checkbox("CLAHE", True, key="ba_c"),
        "bilateral": cols[2].checkbox("Bilateral", True, key="ba_b"),
        "otsu": cols[3].checkbox("Otsu", True, key="ba_o"),
        "deskew_on": cols[4].checkbox("Deskew", True, key="ba_d"),
    }
    with st.spinner("Running the preprocessing pipeline…"):
        src, res = _run_cached(str(path), tuple(sorted(opts.items())))
    if src is None:
        st.error("The image file could not be read.")
        return

    ran = ", ".join(res["steps_run"]) or "none"
    st.success(f"Pipeline finished just now on `{path.name}`. Steps that actually ran: {ran}.")
    tab_side, tab_slider, tab_stages = st.tabs(["Side by side", "Slider", "Each step"])
    with tab_side:
        a, b = st.columns(2)
        a.markdown("**Before** (noisy dataset image)")
        a.image(to_pil(src), use_container_width=True)
        b.markdown("**After** (pipeline output)")
        b.image(to_pil(res["final"]), use_container_width=True)
    with tab_slider:
        components.html(before_after_slider_html(src, res["final"], label_before="Before", label_after="After"), height=720)
    with tab_stages:
        tabs = st.tabs([s["name"] for s in res["stages"]])
        for t, s in zip(tabs, res["stages"]):
            with t:
                st.image(to_pil(s["image"]), use_container_width=True)
                (st.caption if s["ran"] else st.warning)(f"{s['name']}: {s['note']}")

    st.subheader("Image quality metrics (measured now from the pixels)")
    _metric_table(src, res["final"], path.stat().st_size, len(encode_png(res["final"])))
    st.caption("'After' file size = the cleaned image encoded as PNG in memory. Dataset files are never modified.")


# ───────────────────────── tab 2: stored raw vs stored preprocessed ─────────────────────────
def _stored_tab():
    cat = dl.load_preprocessed_catalog()
    man = dl.load_manifest()
    if cat.empty:
        ui.missing_file_msg("data/preprocessed/preprocessed_catalog.csv")
        return
    cat = cat[cat["document_id"].map(lambda d: dl.raw_image_path(d) is not None and dl.prep_image_path(d) is not None)]
    if cat.empty:
        st.warning("No document has both a raw image and a preprocessed image in this deployment.")
        return
    st.markdown("The two images below are the files stored in the project: `data/raw/...` and `data/preprocessed/...`.")
    c1, c2 = st.columns([1, 2])
    lang = c1.selectbox("Language", ["All"] + list(LANG_NAMES),
                        format_func=lambda c: "All" if c == "All" else f"{LANG_NAMES[c]} ({c})", key="st_lang")
    sub = (cat if lang == "All" else cat[cat["script"] == lang]).sort_values("document_id")
    split_map = dict(zip(man["document_id"], man["split"])) if not man.empty else {}
    doc = c2.selectbox("Document ID", sub["document_id"].tolist(), key="st_doc",
                       format_func=lambda d: f"{d}  ·  split: {split_map.get(d, 'not in usable manifest')}")
    raw_p, prep_p = dl.raw_image_path(doc), dl.prep_image_path(doc)
    raw, prep = read_image(raw_p), read_image(prep_p)
    if raw is None or prep is None:
        st.error("One of the two image files could not be read (it may be corrupt).")
        return
    a, b = st.columns(2)
    a.markdown("**Original (raw, as stored)**")
    a.image(to_pil(raw), use_container_width=True)
    b.markdown("**Preprocessed (as stored)**")
    b.image(to_pil(prep), use_container_width=True)
    mb, _ = _metric_table(raw, prep, raw_p.stat().st_size, prep_p.stat().st_size)
    if mb["contrast"] == 0:
        st.markdown(
            '<div class="note-box"><b>Why is the original empty?</b> In this synthetic dataset the raw page is a blank white '
            'canvas (contrast 0.00). Phase 7 of the project <i>rendered the annotated text onto it</i> and then applied '
            'grayscale → CLAHE → bilateral → Otsu → deskew. So “before” has no text and “after” does. The project\'s own '
            'report says the same (raw contrast 0.00 → 22.82). For a meaningful cleaning demo use the first tab.</div>',
            unsafe_allow_html=True)
    row = cat[cat["document_id"] == doc].iloc[0]
    st.caption(f"Project catalog values for comparison: preprocessed contrast {row.get('prep_contrast_std', '—')}, "
               f"sharpness {row.get('prep_blur_laplacian_var', '—')}, skew at preprocessing time {row.get('skew_angle_deg', '—')}°. "
               "The table above is re-measured now from the files.")


def render():
    ui.page_header("🖼️", "Before / After Image Preprocessing",
                   "See the preprocessing pipeline clean a noisy document, with every number measured from the real image.")
    t1, t2 = st.tabs(["Noisy page → cleaned (live demo)", "Stored raw vs stored preprocessed"])
    with t1:
        _live_tab()
    with t2:
        _stored_tab()

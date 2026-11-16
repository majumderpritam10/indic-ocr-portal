"""Screen 2 — Before / After image preprocessing (uses real dataset images; metrics computed live)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from modules import data_loader as dl
from modules import ui
from modules.image_preprocessing import quality_metrics, run_pipeline
from modules.paths import LANG_NAMES
from utils.image_utils import before_after_slider_html, encode_png, human_size, read_image, to_bgr, to_pil


def _metric_table(before, after, before_bytes, after_bytes):
    mb, ma = quality_metrics(before), quality_metrics(after)
    rows = [
        ("Contrast (std-dev of pixel intensity)", mb["contrast"], ma["contrast"]),
        ("Sharpness (Laplacian variance)", mb["sharpness"], ma["sharpness"]),
        ("Image dimensions", f"{mb['width']} × {mb['height']}", f"{ma['width']} × {ma['height']}"),
        ("File size", human_size(before_bytes), human_size(after_bytes)),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Before", "After"]).astype(str)
    st.dataframe(df, hide_index=True, use_container_width=True)
    return mb, ma


def render():
    ui.page_header("🖼️", "Before / After Image Preprocessing",
                   "Compare the original dataset image with its preprocessed version; every number is calculated from the actual image files.")
    cat = dl.load_preprocessed_catalog()
    man = dl.load_manifest()
    if cat.empty:
        ui.missing_file_msg("data/preprocessed/preprocessed_catalog.csv")
        return

    cat = cat[cat["document_id"].map(lambda d: dl.raw_image_path(d) is not None and dl.prep_image_path(d) is not None)]
    if cat.empty:
        st.warning("No document has both a raw image and a preprocessed image in this deployment.")
        return

    c1, c2 = st.columns([1, 2])
    lang = c1.selectbox("Language", ["All"] + list(LANG_NAMES),
                        format_func=lambda c: "All" if c == "All" else f"{LANG_NAMES[c]} ({c})")
    sub = cat if lang == "All" else cat[cat["script"] == lang]
    sub = sub.sort_values("document_id")
    split_map = dict(zip(man["document_id"], man["split"])) if not man.empty else {}
    doc = c2.selectbox("Document ID", sub["document_id"].tolist(),
                       format_func=lambda d: f"{d}  ·  split: {split_map.get(d, 'not in usable manifest')}")

    raw_p, prep_p = dl.raw_image_path(doc), dl.prep_image_path(doc)
    raw, prep = read_image(raw_p), read_image(prep_p)
    if raw is None or prep is None:
        st.error("One of the two image files could not be read (it may be corrupt).")
        return

    tab_side, tab_slider = st.tabs(["Side by side", "Slider"])
    with tab_side:
        a, b = st.columns(2)
        a.markdown("**Original (raw)**")
        a.image(to_pil(raw), use_container_width=True)
        b.markdown("**Preprocessed**")
        b.image(to_pil(prep), use_container_width=True)
    with tab_slider:
        components.html(before_after_slider_html(raw, prep, label_before="Original", label_after="Preprocessed"),
                        height=720)

    st.subheader("Image quality metrics")
    mb, ma = _metric_table(raw, prep, raw_p.stat().st_size, prep_p.stat().st_size)
    if mb["contrast"] == 0:
        st.markdown(
            '<div class="note-box"><b>Why is the original so empty?</b> In this synthetic dataset the raw image is a blank '
            'white page (contrast 0.00). Phase 7 of your pipeline <i>renders the annotated text onto the page</i> and then '
            'applies grayscale → CLAHE → bilateral → Otsu → deskew. So “before” has no text and “after” does. '
            'This is exactly what the project\'s own report says (raw mean contrast 0.00 → 22.82), '
            'so these values are real, not placeholders. To see the cleaning effect on a <i>noisy</i> page, use the live demo below.</div>',
            unsafe_allow_html=True)
    row = cat[cat["document_id"] == doc].iloc[0]
    st.caption(f"Cross-check with the project catalog: preprocessed contrast {row.get('prep_contrast_std', '—')}, "
               f"sharpness {row.get('prep_blur_laplacian_var', '—')}, skew measured at preprocessing time "
               f"{row.get('skew_angle_deg', '—')}°. The values above are re-measured now from the image files.")

    st.divider()
    st.subheader("Run the preprocessing pipeline live")
    sources = {"Raw original (blank canvas)": raw_p, "Preprocessed dataset image": prep_p}
    for recipe in ("mild", "moderate"):
        ap = dl.augmented_image_path(doc, recipe)
        if ap is not None:
            sources[f"Augmented ({recipe}: noise / blur / rotation)"] = ap
    default = [k for k in sources if k.startswith("Augmented")]
    choice = st.selectbox("Image to process", list(sources), index=list(sources).index(default[-1]) if default else 0)

    cols = st.columns(5)
    opts = {
        "grayscale": cols[0].checkbox("Grayscale", True),
        "clahe": cols[1].checkbox("CLAHE", True),
        "bilateral": cols[2].checkbox("Bilateral", True),
        "otsu": cols[3].checkbox("Otsu", True),
        "deskew_on": cols[4].checkbox("Deskew", True),
    }
    if st.button("Run pipeline", type="primary"):
        src_path = sources[choice]
        src = read_image(src_path)
        if src is None:
            st.error("Could not read the selected image.")
            return
        with st.spinner("Processing…"):
            res = run_pipeline(src, **opts)
        st.session_state["ba_result"] = {"doc": doc, "choice": choice, "src": src, "res": res,
                                        "src_bytes": src_path.stat().st_size}

    out = st.session_state.get("ba_result")
    if out and out["doc"] == doc and out["choice"] == choice:
        res = out["res"]
        ran = ", ".join(res["steps_run"]) or "none"
        st.success(f"Pipeline finished. Steps that actually ran: {ran}.")
        tabs = st.tabs([s["name"] for s in res["stages"]])
        for t, s in zip(tabs, res["stages"]):
            with t:
                st.image(to_pil(s["image"]), use_container_width=True)
                (st.caption if s["ran"] else st.warning)(f"{s['name']}: {s['note']}")
        final = res["final"]
        _metric_table(out["src"], final, out["src_bytes"], len(encode_png(final)))
        st.caption("'After' file size is the size of the processed image encoded as PNG in memory. "
                   "Your original dataset files are never modified or overwritten.")
        components.html(before_after_slider_html(out["src"], final, label_before="Before", label_after="After"),
                        height=720)
    elif out:
        st.info("Selection changed — click **Run pipeline** to process this image.")

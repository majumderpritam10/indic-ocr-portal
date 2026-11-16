"""Screen 1 — Document Inspector: browse real dataset documents with their real annotation boxes."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from modules import data_loader as dl
from modules import ui
from modules.annotation_visualizer import color_for, draw_boxes, draw_tight_boxes, measure_all, valid_bbox
from modules.paths import LANG_NAMES, resolve


def _fmt(v):
    """Show a coordinate list exactly as stored, or a dash if the dataset has none."""
    if isinstance(v, (list, tuple)) and len(v) > 0 and all(x is not None for x in v):
        return "[" + ", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in v) + "]"
    return "—"


def render():
    ui.page_header("🔎", "Document Inspector",
                   "Pick a document from the dataset and see its image with the annotation boxes drawn from the real annotation file.")
    man, ann = dl.load_manifest(), dl.load_annotations()
    if man.empty:
        ui.missing_file_msg("data/corrected/manifest_master.csv")
        return
    if ann.empty:
        ui.missing_file_msg("data/corrected/annotations_master.jsonl")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        lang = c1.selectbox("Language", ["All"] + list(LANG_NAMES), format_func=lambda c: "All" if c == "All" else f"{LANG_NAMES[c]} ({c})")
        types = ["All"] + sorted(man["doc_type"].dropna().unique())
        dtype = c2.selectbox("Document type", types)
        split = c3.selectbox("Dataset split", ["All", "train", "validation", "test"])
        kind = c4.selectbox("Document kind", ["Original documents", "Augmented images", "Both"])
        only_img = st.checkbox("Only list documents whose image file is included in this deployment", value=True)

    view = man.copy()
    if lang != "All":
        view = view[view["script"] == lang]
    if dtype != "All":
        view = view[view["doc_type"] == dtype]
    if split != "All":
        view = view[view["split"] == split]
    if kind == "Original documents":
        view = view[~view["is_augmented"]]
    elif kind == "Augmented images":
        view = view[view["is_augmented"]]
    if only_img:
        view = view[view["has_image"]]
    view = view.sort_values("document_id")

    if view.empty:
        st.info("No documents match these filters. Try widening them.")
        return

    labels = {r.document_id: f"{r.document_id}  ·  {LANG_NAMES.get(r.script, r.script)}  ·  {r.doc_type}  ·  {r.split}"
              for r in view.itertuples()}
    st.caption(f"{len(view)} documents match. Click the box below and type part of an ID (e.g. INDIC_0042) to search.")
    doc_id = st.selectbox("Document ID", list(labels), format_func=lambda d: labels[d])
    row = view[view["document_id"] == doc_id].iloc[0]

    left, right = st.columns([1, 1.15], gap="large")
    measured = {}
    records = []
    if not ann.empty:
        records = ann[ann["document_id"] == doc_id].to_dict("records")
        for r in records:  # clean NaN bboxes into None
            if not isinstance(r.get("bbox"), (list, tuple)):
                r["bbox"] = None

    with left:
        st.subheader("Document image")
        img_path = resolve(row["image_rel"])
        if not img_path.exists():
            st.warning(f"The image file `{row['image_rel']}` is listed in the manifest but is not included in this deployment.")
        else:
            pil = Image.open(img_path).convert("RGB")
            fields = sorted({r["field_name"] for r in records})
            show = st.multiselect("Fields to draw", fields, default=fields)
            draw_on = st.toggle("Show bounding boxes", value=True)
            tight_on = st.toggle("Also show measured tight text boxes (dashed black)", value=False,
                                 help="Measured by this website from the image pixels. They are NOT part of the dataset; "
                                      "they show how much of each annotated box the text really fills.")
            if records:
                measured = measure_all(pil, records, (int(row["image_width"]), int(row["image_height"])))
            if draw_on and records:
                shown = [r for r in records if r["field_name"] in show]
                out, n, scaled = draw_boxes(pil, shown, (int(row["image_width"]), int(row["image_height"])))
                if tight_on and measured:
                    idx = {i for i, r in enumerate(records) if r["field_name"] in show}
                    out = draw_tight_boxes(out, measured, idx)
                st.image(out, use_container_width=True)
                if tight_on:
                    st.caption("Solid coloured boxes = dataset annotations. Dashed black boxes = tight text extent measured "
                               "from the pixels by this website (approximate; may include the small grey field label).")
                st.caption(f"{n} box(es) drawn from the annotation file · image {pil.size[0]}×{pil.size[1]} px"
                           + (" · boxes were scaled to match the displayed image size" if scaled else
                              " · box coordinates map 1:1 to the image pixels"))
                chips = " ".join(
                    f'<span style="background:rgb{color_for(f)};color:#fff;padding:2px 9px;border-radius:6px;font-size:.8rem">{f}</span>'
                    for f in show)
                st.markdown(chips, unsafe_allow_html=True)
                if str(doc_id).startswith("INDIC"):
                    st.caption("ℹ️ These boxes are read from this document's own annotation record. In the synthetic INDIC_* "
                               "pages the generator placed the 8 fields on the same fixed rows (left edge x = 80), so the layout "
                               "looks alike from page to page; the box widths, heights and text differ. Open INDIC_0001 and "
                               "INDIC_0103 to compare, or an EXT_* document, which has a different layout.")
            else:
                st.image(pil, use_container_width=True)
                if draw_on and not records:
                    st.info("No annotations exist for this document, so there is nothing to draw.")

    with right:
        st.subheader("Document details")
        meta = {
            "Document ID": row["document_id"], "Language": f"{LANG_NAMES.get(row['script'], row['script'])} ({row['script']})",
            "Document type": row["doc_type"], "Split": row["split"], "Kind": row["kind"],
            "Source dataset": row.get("source_dataset", "—"), "Origin": row.get("source_origin", "—"),
            "Image size": f"{row['image_width']}×{row['image_height']}", "Image file": row["image_rel"],
            "Annotations (manifest)": row.get("num_annotations", "—"),
        }
        if row["is_augmented"]:
            meta["Parent document"] = row.get("parent_document_id", "—")
            meta["Augmentation recipe"] = row.get("augmentation_recipe", "—")
        st.table(pd.DataFrame({"Value": [str(v) for v in meta.values()]}, index=list(meta)))

    st.subheader("Annotations")
    if not records:
        st.warning("No annotation records were found for this document in annotations_master.jsonl.")
        return
    table = pd.DataFrame([{
        "Field": r.get("field_name"),
        "Raw text": r.get("text_raw"),
        "Normalized text": r.get("text"),
        "Absolute bbox [x1,y1,x2,y2]": _fmt(r.get("bbox")),
        "LayoutLM bbox (0-1000)": _fmt(r.get("bbox_layoutlm")),
        "YOLO bbox [cx,cy,w,h]": _fmt(r.get("bbox_yolo")),
        "Normalized bbox (0-1)": _fmt(r.get("bbox_norm")),
        "Text fills box (width × height)": (f"{measured[i]['w_frac']:.0%} × {measured[i]['h_frac']:.0%}" if i in measured else "—"),
        "Changed by normalization": (r.get("text_raw") is not None and r.get("text_raw") != r.get("text")),
    } for i, r in enumerate(records)])
    if measured:
        import statistics
        st.caption(f"Measured from the image: on this document the text fills a median of "
                   f"{statistics.median(m['w_frac'] for m in measured.values()):.0%} of each annotated box's width and "
                   f"{statistics.median(m['h_frac'] for m in measured.values()):.0%} of its height. "
                   "The boxes are valid and inside the page but looser than the text.")
    st.dataframe(table, use_container_width=True, hide_index=True)
    missing = [r["field_name"] for r in records if not valid_bbox(r.get("bbox"))]
    if missing:
        st.warning("These fields have no valid bounding box in the dataset, so no box is drawn: " + ", ".join(missing))

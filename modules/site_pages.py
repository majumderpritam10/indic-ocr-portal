"""Home, Dataset Explorer, Data Quality, Dataset Splitting, Final Audit and About pages.
All numbers are read from the real project files; missing files produce a clear message."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import data_loader as dl
from modules import ocr_engine, translation_engine, ui
from modules.dataset_analytics import STATUS_ICON, status_table
from modules.paths import LANG_NAMES


def _show_block(obj, title: str | None = None):
    """Generic, honest renderer for report JSON: lists of dicts -> table, flat dicts -> key/value table."""
    if title:
        st.markdown(f"##### {title}")
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        st.dataframe(pd.DataFrame(obj).astype(str), hide_index=True, use_container_width=True)
    elif isinstance(obj, dict) and all(not isinstance(v, (dict, list)) for v in obj.values()):
        st.dataframe(pd.DataFrame({"Item": list(obj), "Value": [str(v) for v in obj.values()]}),
                     hide_index=True, use_container_width=True)
    elif obj in (None, [], {}):
        st.caption("Nothing recorded.")
    else:
        st.json(obj, expanded=False)


# ───────────────────────── Home ─────────────────────────
def home():
    ui.page_header("🏠", "Verifiable Indic Document OCR",
                   "University of Engineering and Management, Kolkata · Department of Computer Science and Engineering")
    man = dl.load_manifest()
    st.write("Explore a Bengali · Hindi · English document dataset, inspect its annotations, watch the image preprocessing "
             "pipeline work, and try OCR and translation on your own documents.")
    if not man.empty:
        orig = man[~man["is_augmented"]]
        c = st.columns(4)
        c[0].metric("Usable images", f"{len(man):,}")
        c[1].metric("Original documents", f"{len(orig):,}")
        c[2].metric("Languages", "3 (bn · hi · en)")
        ann = dl.load_annotations()
        c[3].metric("Annotation records", f"{len(ann):,}" if not ann.empty else "n/a")
    else:
        ui.missing_file_msg("data/corrected/manifest_master.csv")

    st.subheader("Where to start")
    c = st.columns(4)
    cards = [("🔎 Document Inspector", "See a document with its real annotation boxes.", "pages/inspector"),
             ("🖼️ Before / After", "Compare raw and preprocessed images, with live metrics.", "pages/before_after"),
             ("📤 Live OCR Sandbox", "Upload a document, extract and translate text.", "pages/sandbox"),
             ("📊 Analytics & Downloads", "Statistics, audit results and verified downloads.", "pages/analytics")]
    for col, (t, d, _) in zip(c, cards):
        with col.container(border=True):
            st.markdown(f"**{t}**")
            st.caption(d)

    st.subheader("What is connected right now")
    rows = []
    for b in ocr_engine.backend_summary():
        rows.append({"Component": b["label"], "Working": "✅ yes" if b["available"] else "❌ no", "Detail": b["detail"]})
    for b in translation_engine.backend_summary():
        rows.append({"Component": b["label"], "Working": "✅ yes" if b["available"] else "❌ no", "Detail": b["detail"]})
    try:
        import fitz  # noqa: F401
        rows.append({"Component": "PDF reading (PyMuPDF)", "Working": "✅ yes", "Detail": "PDF uploads enabled"})
    except ImportError:
        rows.append({"Component": "PDF reading (PyMuPDF)", "Working": "❌ no", "Detail": "pip install pymupdf to enable PDF uploads"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("This table is checked live. The repository itself contains no trained OCR or translation model; "
               "teammates' models plug in through modules/team_ocr_plugin.py and modules/team_translation_plugin.py.")


# ───────────────────────── Dataset Explorer ─────────────────────────
def explorer():
    ui.page_header("🗂️", "Dataset Explorer", "Browse the manifest and search the annotation text.")
    man, ann = dl.load_manifest(), dl.load_annotations()
    if man.empty:
        ui.missing_file_msg("data/corrected/manifest_master.csv")
        return
    tab1, tab2 = st.tabs(["Documents", "Search annotation text"])
    with tab1:
        c = st.columns(4)
        lang = c[0].selectbox("Language", ["All"] + list(LANG_NAMES), key="ex_lang")
        split = c[1].selectbox("Split", ["All", "train", "validation", "test"], key="ex_split")
        kind = c[2].selectbox("Kind", ["All", "original", "augmented"], key="ex_kind")
        q = c[3].text_input("Document ID contains", key="ex_q")
        v = man
        if lang != "All":
            v = v[v["script"] == lang]
        if split != "All":
            v = v[v["split"] == split]
        if kind != "All":
            v = v[v["kind"] == kind]
        if q:
            v = v[v["document_id"].str.contains(q, case=False, na=False)]
        cols = ["document_id", "split", "kind", "script", "doc_type", "num_annotations", "image_width", "image_height", "has_image"]
        st.dataframe(v[cols], hide_index=True, use_container_width=True)
        st.caption(f"{len(v):,} of {len(man):,} documents. 'has_image' = the image file is included in this deployment.")
        st.download_button("⬇ Download this filtered table (CSV)", v[cols].to_csv(index=False).encode("utf-8"),
                           "filtered_manifest.csv", "text/csv")
    with tab2:
        if ann.empty:
            ui.missing_file_msg("data/corrected/annotations_master.jsonl")
            return
        term = st.text_input("Text to find (Bengali, Hindi or English)")
        if term:
            hits = ann[ann["text"].fillna("").str.contains(term, case=False, regex=False)]
            st.write(f"{len(hits):,} annotation records contain “{term}”.")
            st.dataframe(hits[["document_id", "field_name", "text_raw", "text", "script", "split"]].head(300),
                         hide_index=True, use_container_width=True)


# ───────────────────────── Data Quality ─────────────────────────
def data_quality():
    ui.page_header("🧪", "Data Quality", "Results of the data-engineering quality phases, shown as recorded in the project reports.")
    tabs = st.tabs(["Image quality", "Missing values", "Duplicates", "Bounding boxes", "Annotation cleaning"])
    with tabs[0]:
        iq = dl.load_report_json("image_quality.json")
        if not iq:
            ui.missing_file_msg("reports/image_quality.json")
        else:
            c = st.columns(3)
            c[0].metric("Images analysed", iq.get("total_images"))
            c[1].metric("Usable", iq.get("usable_images"))
            c[2].metric("Corrupt (quarantined)", iq.get("corrupt_images"))
            d = iq.get("defect_counts", {})
            if d:
                df = pd.DataFrame({"defect": list(d), "images": list(d.values())})
                st.plotly_chart(px.bar(df, x="defect", y="images", title="Defects found in the RAW images (Phase 5)",
                                       color_discrete_sequence=["#dc2626"], text="images"), use_container_width=True)
            st.info("Phase 5 flagged all raw images as blank placeholders because the synthetic raw pages contain no text; "
                    "Phase 7 rendered the annotated text and preprocessed them. See Before / After Preprocessing.")
            _show_block(iq.get("thresholds_used"), "Thresholds used")
        p = dl.report_image("image_quality_charts.png")
        if p:
            st.image(str(p), caption="Project report chart: image quality", use_container_width=True)
    with tabs[1]:
        mv = dl.load_report_json("missing_values.json")
        if not mv:
            ui.missing_file_msg("reports/missing_values.json")
        else:
            _show_block(mv.get("image_level", {}).get("summary"), "Image-level missing values")
            _show_block(mv.get("annotation_level", {}).get("summary"), "Annotation-level missing values")
            _show_block(mv.get("handling_rules_applied"), "Handling rules applied")
    with tabs[2]:
        dj = dl.load_report_json("duplicates.json")
        if not dj:
            ui.missing_file_msg("reports/duplicates.json")
        else:
            img = dj.get("image_duplicates", {})
            _show_block({k: v for k, v in img.items() if not isinstance(v, (dict, list))}, "Image duplicates (raw data, Phase 4)")
            _show_block(dj.get("annotation_duplicates", {}) and {k: v for k, v in dj["annotation_duplicates"].items()
                                                                  if not isinstance(v, (dict, list))}, "Annotation duplicates")
        res = dl.load_report_json("duplicate_resolution_report.json")
        if res:
            st.caption(f"Corrected-dataset duplicate resolution: {res.get('total_records')} records handled "
                       "(see Quarantine tab in Dataset Analytics).")
    with tabs[3]:
        bj = dl.load_report_json("bbox_validation.json")
        if not bj:
            ui.missing_file_msg("reports/bbox_validation.json")
        else:
            _show_block({k: v for k, v in bj.items() if not isinstance(v, (dict, list))}, "Bounding-box validation (Phase 6b, before correction)")
            _show_block(bj.get("page_dimensions_reference"), "Page dimensions reference")
        p = dl.report_image("bbox_validation_charts.png")
        if p:
            st.image(str(p), use_container_width=True)
    with tabs[4]:
        cj = dl.load_report_json("clean_annotations_summary.json")
        if not cj:
            ui.missing_file_msg("reports/clean_annotations_summary.json")
        else:
            _show_block({k: v for k, v in cj.items() if not isinstance(v, (dict, list))}, "Annotation cleaning summary")
            _show_block(cj.get("field_distribution"), "Field distribution")


# ───────────────────────── Dataset Splitting ─────────────────────────
def splitting():
    ui.page_header("✂️", "Dataset Splitting", "How documents were divided into train, validation and test sets.")
    man = dl.load_manifest()
    summ = dl.load_report_csv("corrected_split_summary.csv")
    if not summ.empty:
        st.markdown("##### Split summary (from reports/corrected_split_summary.csv)")
        st.dataframe(summ, hide_index=True, use_container_width=True)
    else:
        ui.missing_file_msg("reports/corrected_split_summary.csv")
    if not man.empty:
        df = man.groupby(["split", "script", "kind"]).size().reset_index(name="images")
        df["language"] = df["script"].map(lambda c: LANG_NAMES.get(c, c))
        st.plotly_chart(px.bar(df, x="split", y="images", color="language", facet_col="kind",
                               title="Images per split, by language (computed from the manifest)"), use_container_width=True)
    audit = dl.load_report_csv("corrected_final_audit_report.csv")
    if not audit.empty:
        st.markdown("##### Leakage checks from the final audit")
        keep = audit[audit["check_id"].isin(["CHK_03", "CHK_04", "CHK_05", "CHK_06", "CHK_07", "CHK_08"])]
        st.dataframe(status_table(keep[["check_id", "check_name", "actual", "status"]]), hide_index=True, use_container_width=True)
    p = dl.report_image("split_distribution_charts.png")
    if p:
        st.image(str(p), caption="Project report chart: split distribution", use_container_width=True)


# ───────────────────────── Final Audit ─────────────────────────
def final_audit():
    ui.page_header("✅", "Final Audit", "The project's own final audit scorecard — warnings are shown, not hidden.")
    summ = dl.load_report_json("corrected_final_audit_summary.json")
    audit = dl.load_report_csv("corrected_final_audit_report.csv")
    if not summ and audit.empty:
        ui.missing_file_msg("reports/corrected_final_audit_summary.json")
        return
    if summ:
        s = summ.get("scorecard_summary", {})
        status = str(summ.get("final_dataset_status", "n/a"))
        st.markdown(f"### Status: {ui.pill(status, 'ok' if status.upper() == 'READY' else 'warn')}", unsafe_allow_html=True)
        c = st.columns(3)
        c[0].metric("PASS", s.get("PASS", 0))
        c[1].metric("WARNING", s.get("WARNING", 0))
        c[2].metric("FAIL", s.get("FAIL", 0))
        st.caption(f"Audit run: {summ.get('run_timestamp', 'n/a')}")
    if not audit.empty:
        st.dataframe(status_table(audit), hide_index=True, use_container_width=True)
    if summ and summ.get("documented_findings"):
        _show_block(summ["documented_findings"], "Documented findings")
    hv = dl.load_report_json("handover_verification_report.json")
    if hv:
        st.markdown("##### Handover verification")
        st.write(f"Verdict: **{hv.get('handover_verdict', 'n/a')}**")
        for k, v in hv.get("verification_points", {}).items():
            with st.expander(k.replace("_", " ")):
                st.json(v, expanded=False)
    p = dl.report_image("corrected_final_audit_charts.png")
    if p:
        st.image(str(p), caption="Project report chart: corrected final audit", use_container_width=True)


# ───────────────────────── About ─────────────────────────
def about():
    ui.page_header("ℹ️", "About this project", "Verifiable Indic Document OCR — University of Engineering and Management, Kolkata (CSE)")
    st.markdown("""
**Goal.** Build and verify a Bengali · Hindi · English document dataset for OCR model development, and demonstrate it
in an interactive portal.

**Data-engineering phases (see the Data Quality and Final Audit pages):** inspection → missing values → duplicate
detection → image quality → annotation cleaning → bounding-box validation → preprocessing (grayscale, CLAHE, bilateral
filter, Otsu, deskew) → augmentation → integration → train/validation/test split → final audit and correction.

**What this website does for real**
- Reads the real manifest, annotations and reports in `data/` and `reports/`.
- Draws the stored bounding boxes; measures image quality from the actual pixels.
- Runs the preprocessing pipeline on uploaded documents, in memory.
- Runs OCR and translation through the back-ends listed on the Home page; if a back-end is missing it says so.

**Honest limits**
- The dataset is largely *synthetic*: text was rendered onto blank pages from the annotation file.
  Results on it do not predict accuracy on real scanned documents.
- The repository has no trained OCR or translation model. The baseline OCR (Tesseract) and the free translation
  service are generic stand-ins until the team's models are plugged in.
- Script detection counts Unicode letters in a first OCR pass. It is not a probabilistic classifier, so no
  "confidence %" is displayed.
""")

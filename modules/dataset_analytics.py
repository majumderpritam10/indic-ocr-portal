"""Screen 4 — Dataset analytics & download hub. All figures are computed from the real project files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import data_loader as dl
from modules import ui
from modules.paths import APP_ROOT, LANG_NAMES
from utils.image_utils import human_size

COLORS = {"original": "#2f5fd0", "augmented": "#5b3fd6", "train": "#2f5fd0", "validation": "#f59e0b", "test": "#10b981"}
STATUS_ICON = {"PASS": "✅ PASS", "WARNING": "⚠️ WARNING", "FAIL": "❌ FAIL"}


@st.cache_data(show_spinner=False)
def _read_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


def status_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "status" in df.columns:
        df["status"] = df["status"].map(lambda s: STATUS_ICON.get(str(s).upper(), str(s)))
    return df


def _lang_name(code):
    return LANG_NAMES.get(code, str(code))


def render():
    ui.page_header("📊", "Dataset Analytics & Download Hub",
                   "Live statistics, audit results and verifiable downloads — computed from the real dataset files.")
    man, ann = dl.load_manifest(), dl.load_annotations()
    if man.empty:
        ui.missing_file_msg("data/corrected/manifest_master.csv")
        return
    aug_cat, quar = dl.load_augmented_catalog(), dl.load_quarantine_manifest()
    qann = dl.load_quarantine_annotations()
    orig, aug = man[~man["is_augmented"]], man[man["is_augmented"]]

    t_over, t_charts, t_man, t_audit, t_quar, t_dl = st.tabs(
        ["Overview", "Charts", "Manifest", "Audit & quality", "Quarantine", "Downloads"])

    # ───────────── overview ─────────────
    with t_over:
        c = st.columns(5)
        c[0].metric("Usable images (total documents)", f"{len(man):,}")
        c[1].metric("Original documents", f"{len(orig):,}")
        c[2].metric("Augmented images", f"{len(aug):,}")
        c[3].metric("Total annotations", f"{len(ann):,}" if not ann.empty else "n/a")
        c[4].metric("Quarantined documents", f"{len(quar):,}" if not quar.empty else "n/a")

        st.markdown("**Documents by language** (original documents · all usable images incl. augmented)")
        c = st.columns(3)
        for col, code in zip(c, ["bn", "hi", "en"]):
            col.metric(f"{_lang_name(code)} ({code})", f"{(orig['script'] == code).sum():,}",
                       f"all images: {(man['script'] == code).sum():,}", delta_color="off")
        st.markdown("**Documents by split** (original documents · all usable images)")
        c = st.columns(3)
        for col, sp in zip(c, ["train", "validation", "test"]):
            col.metric(sp.capitalize(), f"{(orig['split'] == sp).sum():,}",
                       f"all images: {(man['split'] == sp).sum():,}", delta_color="off")
        notes = [f"Augmented catalog (data/augmented/augmented_catalog.csv) lists {len(aug_cat):,} augmented images; "
                 f"{len(aug):,} of them are in the usable manifest."] if not aug_cat.empty else []
        if not ann.empty:
            notes.append(f"Annotation records: {int(ann['is_augmented'].sum()):,} belong to augmented images and "
                         f"{int((~ann['is_augmented']).sum()):,} to original documents.")
        for n in notes:
            st.caption(n)
        st.caption("Augmented images are used for training only (audit check CHK_04).")

    # ───────────── charts ─────────────
    with t_charts:
        a, b = st.columns(2)
        lang_df = man.groupby(["script", "kind"]).size().reset_index(name="count")
        lang_df["language"] = lang_df["script"].map(_lang_name)
        fig = px.bar(lang_df, x="language", y="count", color="kind", barmode="group", title="Language distribution",
                     color_discrete_map=COLORS, text="count")
        a.plotly_chart(fig, use_container_width=True)

        type_df = orig.groupby("doc_type").size().reset_index(name="count")
        fig = px.bar(type_df, x="doc_type", y="count", title="Document type distribution (original documents)",
                     color_discrete_sequence=["#2f5fd0"], text="count")
        b.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        sp_df = man.groupby(["split", "kind"]).size().reset_index(name="count")
        sp_df["split"] = pd.Categorical(sp_df["split"], ["train", "validation", "test"], ordered=True)
        fig = px.bar(sp_df.sort_values("split"), x="split", y="count", color="kind", title="Train / validation / test distribution",
                     color_discrete_map=COLORS, text="count")
        a.plotly_chart(fig, use_container_width=True)
        kd = man.groupby("kind").size().reset_index(name="count")
        fig = px.pie(kd, names="kind", values="count", hole=0.5, title="Original vs augmented images",
                     color="kind", color_discrete_map=COLORS)
        b.plotly_chart(fig, use_container_width=True)
        if not ann.empty:
            fd = ann.groupby("field_name").size().reset_index(name="count").sort_values("count", ascending=False)
            st.plotly_chart(px.bar(fd, x="field_name", y="count", title="Annotations per field name",
                                   color_discrete_sequence=["#5b3fd6"], text="count"), use_container_width=True)

    # ───────────── manifest preview ─────────────
    with t_man:
        st.caption(f"data/corrected/manifest_master.csv — {len(man):,} rows. Showing the first 200 after filtering.")
        f1, f2, f3 = st.columns(3)
        fl = f1.selectbox("Language", ["All"] + list(LANG_NAMES), key="m_lang")
        fs = f2.selectbox("Split", ["All", "train", "validation", "test"], key="m_split")
        fk = f3.selectbox("Kind", ["All", "original", "augmented"], key="m_kind")
        v = man
        if fl != "All":
            v = v[v["script"] == fl]
        if fs != "All":
            v = v[v["split"] == fs]
        if fk != "All":
            v = v[v["kind"] == fk]
        cols = ["document_id", "parent_document_id", "split", "kind", "augmentation_recipe", "script", "doc_type",
                "source_origin", "num_annotations", "image_width", "image_height", "image_rel", "sha256"]
        st.dataframe(v[[c for c in cols if c in v.columns]].head(200), use_container_width=True, hide_index=True)
        st.caption(f"{len(v):,} rows match.")

    # ───────────── audit + quality ─────────────
    with t_audit:
        summ = dl.load_report_json("corrected_final_audit_summary.json")
        audit = dl.load_report_csv("corrected_final_audit_report.csv")
        if summ:
            s = summ.get("scorecard_summary", {})
            st.markdown(f"**Final dataset status (from the project audit): `{summ.get('final_dataset_status', 'n/a')}`** · "
                        f"PASS {s.get('PASS', 0)} · WARNING {s.get('WARNING', 0)} · FAIL {s.get('FAIL', 0)}  \n"
                        f"Audit run: {summ.get('run_timestamp', 'n/a')}")
        if audit.empty:
            ui.missing_file_msg("reports/corrected_final_audit_report.csv")
        else:
            st.markdown("##### Audit check results")
            st.dataframe(status_table(audit), use_container_width=True, hide_index=True)
            warn = audit[audit["status"].astype(str).str.upper() == "WARNING"]
            if len(warn):
                st.warning("Warnings preserved from the original audit: " + "; ".join(warn["check_name"]))
        st.markdown("##### Data quality information")
        if summ:
            st.dataframe(pd.DataFrame({"Metric": list(summ["corpus_metrics"]), "Value": list(summ["corpus_metrics"].values())})
                         .astype(str), hide_index=True, use_container_width=True)
        if not ann.empty:
            q = {
                "Annotation records checked now": len(ann),
                "Records with is_bbox_valid = True": int(ann["is_bbox_valid"].sum()),
                "Records flagged needs_manual_annotation": int(ann["needs_manual_annotation"].sum()),
                "Records with empty text": int((ann["text"].fillna("").str.strip() == "").sum()),
                "Distinct documents with annotations": int(ann["document_id"].nunique()),
            }
            st.dataframe(pd.DataFrame({"Check (computed live from annotations_master.jsonl)": list(q), "Value": [str(x) for x in q.values()]}),
                         hide_index=True, use_container_width=True)

    # ───────────── quarantine ─────────────
    with t_quar:
        if quar.empty:
            st.info("No quarantine records are available in this deployment.")
        else:
            st.caption(f"data/corrected/quarantine/quarantined_manifest.csv — {len(quar)} documents removed from the usable set"
                       + (f"; {len(qann)} quarantined annotation records." if not qann.empty else "."))
            rc = quar.groupby("quarantine_reason").size().reset_index(name="documents")
            st.dataframe(rc, hide_index=True, use_container_width=True)
            show = [c for c in ["document_id", "script", "doc_type", "source_dataset", "quarantine_reason", "disposition", "image_path"] if c in quar.columns]
            st.dataframe(quar[show], hide_index=True, use_container_width=True)

    # ───────────── downloads ─────────────
    with t_dl:
        render_downloads()


_FILES = [
    ("annotations_master.jsonl", "data/corrected/annotations_master.jsonl", "All usable annotation records (originals + augmented)", "application/x-ndjson"),
    ("manifest_master.csv", "data/corrected/manifest_master.csv", "Manifest of all usable images with split, language, hashes", "text/csv"),
    ("train/annotations.jsonl", "data/corrected/train/annotations.jsonl", "Training-split annotations", "application/x-ndjson"),
    ("validation/annotations.jsonl", "data/corrected/validation/annotations.jsonl", "Validation-split annotations", "application/x-ndjson"),
    ("test/annotations.jsonl", "data/corrected/test/annotations.jsonl", "Test-split annotations", "application/x-ndjson"),
    ("annotations_augmented.jsonl", "data/augmented/annotations_augmented.jsonl", "Annotations of augmented images", "application/x-ndjson"),
    ("quarantined_manifest.csv", "data/corrected/quarantine/quarantined_manifest.csv", "Quarantined documents and reasons", "text/csv"),
    ("corrected_final_audit_report.csv", "reports/corrected_final_audit_report.csv", "Final audit scorecard (12 checks)", "text/csv"),
    ("corrected_final_audit_summary.json", "reports/corrected_final_audit_summary.json", "Final audit summary incl. documented findings", "application/json"),
    ("handover_verification_report.json", "reports/handover_verification_report.json", "Handover verification report", "application/json"),
    ("corrected_split_summary.csv", "reports/corrected_split_summary.csv", "Train/val/test split summary", "text/csv"),
]


def render_downloads():
    st.caption("Only files that really exist in this deployment get a download button. "
               "The SHA-256 lets anyone verify a downloaded file against the original.")
    for label, rel, desc, mime in _FILES:
        p = APP_ROOT / rel
        c1, c2 = st.columns([3, 1])
        if p.exists():
            c1.markdown(f"**{label}** — {desc}  \n<small>{human_size(p.stat().st_size)} · SHA-256 `{dl.sha256_of(str(p))[:24]}…`</small>",
                        unsafe_allow_html=True)
            c2.download_button("⬇ Download", _read_bytes(str(p)), p.name, mime, key=f"dl_{rel}", use_container_width=True)
        else:
            c1.markdown(f"**{label}** — {desc}")
            c2.caption("Not available")

    st.markdown("**Master dataset archive (images.zip)**")
    cands = [APP_ROOT / "images.zip", APP_ROOT / "data" / "images.zip", APP_ROOT / "data" / "final" / "images.zip"]
    found = next((p for p in cands if p.exists()), None)
    if found:
        st.download_button("⬇ Download images.zip", _read_bytes(str(found)), "images.zip", "application/zip", key="dl_zip")
    else:
        st.info("Unavailable: no images.zip (or other master image archive) exists in this project, so no link is offered. "
                "Individual images are included under data/final/images.")

    st.markdown("**Extracted OCR text (from the Live OCR Sandbox, this session)**")
    res = st.session_state.get("sb_result")
    if res and res.get("ocr") and res["ocr"]["text"].strip():
        st.download_button("⬇ Download latest extracted text", res["ocr"]["text"].encode("utf-8-sig"),
                           f"{Path(res['name']).stem}_ocr.txt", "text/plain", key="dl_ocr_hub")
    else:
        st.caption("No OCR text yet — run a document in the Live OCR Sandbox first.")

"""Verifiable Indic Document OCR Portal — run with:  python -m streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="Indic Document OCR Portal", page_icon="📄", layout="wide",
                   initial_sidebar_state="expanded")

from pathlib import Path  # noqa: E402

from modules import document_inspector, live_sandbox, page_before_after, site_pages, ui  # noqa: E402
from modules import dataset_analytics  # noqa: E402

ui.inject_css()
_logo = Path(__file__).parent / "assets" / "logo.svg"
if _logo.exists():
    st.logo(str(_logo), size="large")

pages = {
    "": [
        st.Page(site_pages.home, title="Home", icon=":material/home:", default=True),
        st.Page(document_inspector.render, title="Document Inspector", icon=":material/image_search:", url_path="inspector"),
        st.Page(page_before_after.render, title="Before / After Preprocessing", icon=":material/compare:", url_path="before-after"),
        st.Page(live_sandbox.render, title="Live OCR Sandbox", icon=":material/upload_file:", url_path="live-ocr"),
        st.Page(dataset_analytics.render, title="Dataset Analytics", icon=":material/bar_chart:", url_path="analytics"),
    ],
    "Dataset & project": [
        st.Page(site_pages.explorer, title="Dataset Explorer", icon=":material/table_view:", url_path="explorer"),
        st.Page(site_pages.data_quality, title="Data Quality", icon=":material/fact_check:", url_path="data-quality"),
        st.Page(site_pages.splitting, title="Dataset Splitting", icon=":material/call_split:", url_path="splitting"),
        st.Page(site_pages.final_audit, title="Final Audit", icon=":material/verified:", url_path="final-audit"),
        st.Page(site_pages.about, title="About Project", icon=":material/info:", url_path="about"),
    ],
}
st.navigation(pages).run()

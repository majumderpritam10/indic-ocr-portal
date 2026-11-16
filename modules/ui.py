"""Shared look & feel (matches the mock-up: dark navy sidebar, light blue page, white cards, purple accent)."""
from __future__ import annotations

import streamlit as st

NAVY = "#12244a"
BLUE = "#2f5fd0"
PURPLE = "#5b3fd6"

_CSS = f"""
<style>
.stApp {{ background:#eef3fb; }}
[data-testid="stSidebar"] {{ background:{NAVY}; }}
[data-testid="stSidebar"] * {{ color:#e8eefc; }}
[data-testid="stSidebarNavLink"] {{ border-radius:8px; margin:2px 0; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background:{BLUE}; }}
[data-testid="stSidebarNavLink"]:hover {{ background:rgba(255,255,255,.10); }}
[data-testid="stNavSectionHeader"] {{ opacity:.6; }}
.block-container {{ padding-top:2.2rem; max-width:1250px; }}
h1, h2, h3 {{ letter-spacing:-0.01em; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#ffffff; border-radius:12px; }}
div[data-testid="stMetric"] {{ background:#fff; border:1px solid #dfe6f3; padding:12px 14px; border-radius:10px; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background:{PURPLE}; border-color:{PURPLE}; color:#fff; }}
div[data-testid="stMetricValue"] {{ font-size:1.35rem; }}
.page-head {{ display:flex; gap:14px; align-items:center; margin-bottom:.4rem; }}
.page-head .ico {{ font-size:2rem; background:#fff; border-radius:12px; padding:6px 12px; border:1px solid #dfe6f3; }}
.page-head h1 {{ margin:0; font-size:1.9rem; color:{NAVY}; }}
.page-head p {{ margin:0; color:#475569; }}
.stepper {{ display:flex; align-items:flex-start; justify-content:space-between; background:#fff;
           border-radius:12px; padding:14px 22px; margin:.6rem 0 1rem 0; border:1px solid #dfe6f3; }}
.step {{ flex:1; text-align:center; position:relative; font-size:.9rem; font-weight:600; color:#64748b; }}
.step .dot {{ width:30px; height:30px; border-radius:50%; margin:0 auto 4px auto; line-height:30px;
             background:#cbd5e1; color:#fff; font-weight:700; }}
.step.done .dot {{ background:{BLUE}; }}
.step.done {{ color:{BLUE}; }}
.step.active .dot {{ background:{PURPLE}; box-shadow:0 0 0 4px rgba(91,63,214,.18); }}
.step.active {{ color:{PURPLE}; }}
.step:not(:last-child)::after {{ content:""; position:absolute; top:15px; left:calc(50% + 22px); width:calc(100% - 44px);
                                height:3px; background:#cbd5e1; }}
.step.done:not(:last-child)::after {{ background:{BLUE}; }}
.note-box {{ background:#fff8e6; border:1px solid #f3d98b; border-radius:8px; padding:10px 14px; font-size:.92rem; }}
.ok-box {{ background:#e9f9ef; border:1px solid #9bdcb3; border-radius:8px; padding:10px 14px; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600; }}
.pill.ok {{ background:#dcfce7; color:#166534; }} .pill.warn {{ background:#fef3c7; color:#92400e; }}
.pill.bad {{ background:#fee2e2; color:#991b1b; }}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str):
    st.markdown(
        f'<div class="page-head"><div class="ico">{icon}</div><div><h1>{title}</h1><p>{subtitle}</p></div></div>',
        unsafe_allow_html=True)


def stepper(active: int, done: set[int]):
    names = ["Upload", "Preprocess", "Detect Language", "OCR Extract", "Translate"]
    html = '<div class="stepper">'
    for i, n in enumerate(names, 1):
        cls = "done" if i in done else ("active" if i == active else "")
        mark = "✓" if i in done else str(i)
        html += f'<div class="step {cls}"><div class="dot">{mark}</div>{n}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def pill(text: str, kind: str = "ok") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def missing_file_msg(what: str):
    st.warning(f"**{what}** was not found in this deployment, so nothing is shown here. "
               "Nothing has been made up to fill the gap.")

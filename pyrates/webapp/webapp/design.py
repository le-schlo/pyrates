from __future__ import annotations

import streamlit as st


GLOBAL_STYLE = """
<style>
/***** Force light theme appearance *****/
:root { color-scheme: light; }
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background-color: #ffffff !important;
  color: #111111 !important;
}
/* Sidebar and header bar */
section[data-testid="stSidebar"] { background-color: #f7f7f9 !important; color: #111111 !important; }
header, [data-testid="stHeader"] { background: #ffffff !important; color: #111111 !important; }
/* Ensure left menu text and icons are readable */
section[data-testid="stSidebar"] * { color: #111111 !important; }
/* Inputs and widgets */
input, textarea, select, .stTextInput, .stTextArea, .stSelectbox, .stFileUploader {
  color: #111111 !important;
  background-color: #ffffff !important;
}
/* Some Streamlit class hooks for input backgrounds */
[data-baseweb="input"]>div, [data-baseweb="textarea"]>textarea, [data-baseweb="select"]>div {
  background-color: #ffffff !important; color: #111111 !important;
}
/* Buttons */
.stButton>button, .stDownloadButton>button { background-color: #2b6cb0 !important; color: #ffffff !important; border: 0 !important; }
.stDownloadButton>button:hover, .stButton>button:hover { background-color: #2c5282 !important; }
/* Dataframe tweaks */
.block-container { padding-top: 2rem; }
/* Ensure SVG overlays look good on light bg */
svg { background: transparent !important; }

/***** Tabs banner helpers *****/
.tab-banner { padding: 6px 10px; border-radius: 8px; color: #fff; margin: 0 0 0.5rem 0; font-weight: 600; }
.tab-single { background: linear-gradient(90deg, #6a5acd, #7b68ee); }
.tab-batch { background: linear-gradient(90deg, #2e8b57, #3cb371); }
</style>
"""


def apply_global_styles():
    """Inject global CSS once per page to keep style centralized."""
    st.markdown(GLOBAL_STYLE, unsafe_allow_html=True)


def tab_banner(label: str, variant: str = "single"):
    cls = "tab-single" if variant == "single" else "tab-batch"
    st.markdown(f'<div class="tab-banner {cls}">{label}</div>', unsafe_allow_html=True)

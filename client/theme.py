"""Minimal styling: font imports, subtle typography, and pure dark black canvas."""
import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #000000;
}

h1, h2, h3, h4 {
    font-weight: 600;
    letter-spacing: -0.015em;
    color: #F8FAFC;
}

[data-testid="stMetricValue"], [data-testid="stDataFrame"], code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    /* Scale the dashboard with the screen instead of capping at 1220px:
       the Visual Inspection Studio should use the full width on large monitors. */
    max-width: min(94vw, 1760px);
}
"""

def apply() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

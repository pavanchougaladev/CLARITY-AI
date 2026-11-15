"""Minimal styling: fonts and spacing only. Colours live in .streamlit/config.toml."""
import streamlit as st

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:wght@600&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }
[data-testid="stDataFrame"] { font-family: 'JetBrains Mono', monospace; }
.block-container { padding-top: 2rem; max-width: 1200px; }
"""

def apply() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

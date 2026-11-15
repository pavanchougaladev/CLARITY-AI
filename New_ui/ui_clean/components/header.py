"""Sidebar: name, numbered navigation, status."""
import streamlit as st
from config import NAV

def sidebar() -> str:
    with st.sidebar:
        st.markdown("### CLARITY\\*")
        st.caption("Lunar co-registration and 3D mapping  \nISRO SAC capstone project")
        page = st.radio("Navigation", NAV, label_visibility="collapsed")
        st.divider()
        st.caption(":green[●] Pipeline ready  \nMAGSAC++ v2.4  \nMPS GPU acceleration")
        st.caption("Lead: Mahim S Muradimath")
    return page

def sample_badge(result: dict | None) -> None:
    if result is None or result.get("is_sample"):
        st.badge("Sample data", icon=":material/science:", color="orange")

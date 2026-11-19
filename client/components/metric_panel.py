from __future__ import annotations
"""Quality-gate and KPI cards with clean typography and status badges."""
from typing import Optional, List, Tuple
import streamlit as st

def gate_card(title: str, value: str, caption: str, passed: Optional[bool] = None) -> None:
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.caption(title.upper())
        if passed is not None:
            c2.badge("Passed" if passed else "Breached", color="green" if passed else "red")
        st.markdown(f"### {value}")
        st.caption(caption)

def kpis(items: List[Tuple[str, str, str]]) -> None:
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        col.metric(label, value, help_, delta_color="off", border=True)

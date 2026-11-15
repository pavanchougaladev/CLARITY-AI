"""Quality-gate and KPI cards."""
import streamlit as st

def gate_card(title: str, value: str, caption: str, passed: bool | None = None) -> None:
    with st.container(border=True):
        st.caption(title)
        st.markdown(f"#### {value}")
        if passed is not None:
            st.badge("Passed" if passed else "Failed", color="green" if passed else "red")
        st.caption(caption)

def kpis(items: list[tuple[str, str, str]]) -> None:
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        col.metric(label, value, help_, delta_color="off", border=True)

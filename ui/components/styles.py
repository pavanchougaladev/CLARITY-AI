"""
UI Theme and Styles Component
=============================
ISRO LunarFusion Co-Registration Suite — UI Styling
Injects cosmic space aesthetics, high-refresh rate CSS tokens, and live telemetry HUD.
"""

import os
import streamlit as st
from typing import Optional

THEME_CSS_PATH = os.path.join(os.path.dirname(__file__), "..", "static", "theme.css")


def apply_custom_theme() -> None:
    """Inject ISRO cosmic space theme, custom fonts, and telemetry HUD styling into Streamlit."""
    try:
        if os.path.exists(THEME_CSS_PATH):
            with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
                raw_html = f.read()
            st.html(raw_html)
        else:
            # Fallback basic styling
            st.html("""
            <style>
                body, .stApp {
                    background-color: #05060F !important;
                    color: #E8ECF5 !important;
                }
            </style>
            """)
    except Exception as exc:
        st.warning(f"Failed to load visual theme styles: {exc}")


def render_metric_card(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    badge: Optional[str] = None,
    badge_color: str = "#34D399",
) -> str:
    """Generate HTML markup for a styled glassmorphic telemetry card."""
    badge_html = f"""
    <span style="background:rgba(255,255,255,0.05); border:1px solid {badge_color}; border-radius:6px; font-size:0.68rem; padding:2px 7px; color:{badge_color}; font-weight:600; font-family:'JetBrains Mono',monospace;">
        {badge}
    </span>
    """ if badge else ""

    sub_html = f"""
    <div style="font-size:0.72rem; color:#CBD5E1; margin-top:4px; line-height:1.4;">
        {subtext}
    </div>
    """ if subtext else ""

    return f"""
    <div style="background:rgba(16,26,54,0.65); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.10); border-radius:12px; padding:14px 16px; box-shadow:0 4px 20px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.12); transition:border-color 0.18s ease, transform 0.18s ease;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:0.75rem; color:#94A3B8; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">{label}</span>
            {badge_html}
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.4rem; font-weight:700; color:#E8ECF5; letter-spacing:-0.02em;">
            {value}
        </div>
        {sub_html}
    </div>
    """

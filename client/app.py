"""Run with: streamlit run client/app.py"""
import sys
from pathlib import Path

_UI_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _UI_DIR.parent
_SERVER_DIR = _PROJECT_ROOT / "server"
for p in (str(_UI_DIR), str(_PROJECT_ROOT), str(_SERVER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st

st.set_page_config(page_title="CLARITY*", page_icon=":material/dark_mode:", layout="wide")

import theme
from components.header import sidebar
from views import view_dashboard, view_registration, view_dem_3d, view_crater_hazard, view_evaluation
from config import NAV

theme.apply()
ROUTES = [view_dashboard, view_registration, view_dem_3d, view_crater_hazard, view_evaluation]
page = sidebar()
ROUTES[NAV.index(page)].render()

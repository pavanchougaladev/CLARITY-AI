"""Run with: streamlit run ui_clean/app.py"""
import streamlit as st

st.set_page_config(page_title="CLARITY*", page_icon=":material/dark_mode:", layout="wide")

import theme
from components.header import sidebar
from views import view_dashboard, view_registration, view_dem_3d, view_crater_hazard, view_evaluation

theme.apply()
ROUTES = [view_dashboard, view_registration, view_dem_3d, view_crater_hazard, view_evaluation]
page = sidebar()
from config import NAV
ROUTES[NAV.index(page)].render()

# CLARITY* clean UI (Streamlit)

1. Copy `ui_clean/` and `.streamlit/config.toml` into your project root (back up your old `.streamlit/` first).
2. `pip install -r requirements.txt`
3. `streamlit run ui_clean/app.py`

The UI runs on sample data out of the box (orange "Sample data" badge).
To use real results, edit `ui_clean/services.py`: set `USE_REAL_PIPELINE = True` and fill in
`_real_registration()` to return the same keys as the demo result (points, inliers, rmse,
confidence, entropy, matches, craters, gates, grade). Swap `make_dem()` for your DEM array.
Placeholders: the second preset in `config.py`, and benchmark numbers in `config.BENCH`.

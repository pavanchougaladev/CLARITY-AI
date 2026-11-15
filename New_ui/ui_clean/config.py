"""Constants shared by all views."""
NAV = ["01  Overview", "02  Co-registration studio", "03  3D topography & DEM",
       "04  Crater & hazard AI", "05  Evaluation & benchmarks"]
PRESETS = {
    "Chandrayaan-2 TMC-2 vs OHRC (high contrast)": {
        "seed": 3,
        "src": "Chandrayaan-2 TMC-2 nadir, 5.0 m/px, solar incidence 42.1°",
        "ref": "Chandrayaan-2 OHRC, 0.25 m/px, solar incidence 38.4°"},
    "Second pair (edit in config.py)": {
        "seed": 7,
        "src": "[SENSOR], [RES] m/px, solar incidence [DEG]°",
        "ref": "[SENSOR], [RES] m/px, solar incidence [DEG]°"},
}
GSD_REF = 0.25      # m/px of the OHRC reference
DEM_SPACING = 5.0   # metres between DEM grid cells
BENCH = [  # method, inliers, rmse, reprojection, seconds, grade
    ("LoFTR + RoMa (ours)", 1420, 0.38, 0.42, 2.8, "A"),
    ("LightGlue + DISK", 610, 0.74, 0.82, 1.9, "B"),
    ("SIFT", 86, 1.82, 2.14, 0.9, "C"),
    ("ORB", 32, 3.41, 4.10, 0.4, "D"),
]

# 🌔 ISRO Lunar Image Registration Pipeline (SIH26166)

A production-grade, deep-learning powered software suite built for the Indian Space Research Organisation (ISRO). This pipeline performs fully automated, sub-pixel accurate co-registration of heterogeneous lunar imagery from Chandrayaan-1 and Chandrayaan-2 (OHRC, TMC-2).

---

## 🎯 Features

This software solves critical challenges for ISRO by providing:

1. **Multi-Modal Robustness:** Uses advanced Deep Learning (LoFTR & LightGlue) that looks at global context rather than just local corners, allowing it to match craters even when shadows are cast in completely opposite directions.
2. **Sub-Pixel Accuracy & Pyramid Matching:** Employs Phase Correlation and coarse-to-fine Gaussian Pyramids for extreme accuracy and speed.
3. **Photometric Correction (Hapke):** Normalizes illumination differences using Hapke BRDF or Lunar-Lambert models before matching.
4. **NASA Ames Stereo Pipeline (ASP) Integration:** Automatically exports tie-points in ASP-compatible formats.
5. **GeoTIFF Support:** Reads and preserves geospatial coordinates natively using Rasterio.
6. **Pushbroom Jitter Correction:** Uses Thin-Plate Spline (TPS) transformation to correct non-linear distortions.
7. **Batch Processing:** Run thousands of pairs overnight with pass/fail thresholds and comprehensive JSON reports.
8. **Interactive UI:** Streamlit dashboard featuring warped overlays (Checkerboard, Blend, Difference), Plotly match inspectors, and historical tracking.
9. **FastAPI Backend & Docker:** Ready to be deployed as a microservice on ISRO servers.

---

## 🛠️ Tools & Technologies Used

- **Frontend UI:** `Streamlit`, `Plotly`
- **Backend API:** `FastAPI`, `Uvicorn`
- **Core Vision Processing:** `OpenCV`, `rasterio`
- **Deep Learning Framework:** `PyTorch`, `Kornia` (LoFTR, LightGlue)
- **Mathematical Analysis:** `NumPy`, `Pandas`, `SciPy`

---

## 📂 Codebase Architecture 

- `ui/app.py`: Streamlit frontend with interactive Match Editor, Warped Overlays, and Batch Processing Dashboard.
- `api/main.py`: FastAPI backend for running the pipeline as a REST service.
- `src/pipeline.py`: Core orchestrator.
- `src/batch.py`: Batch processing engine.
- `src/history.py`: Registration history tracking database.
- `src/preprocess/`: Hapke photometric correction, CLAHE, and Gaussian pyramids.
- `src/features/`: LoFTR / LightGlue deep matchers.
- `src/matching/`: MAGSAC++ outlier rejection and Thin-Plate Spline estimation.
- `src/evaluate/`: Confidence-weighted RMSE and match density analytics.
- `src/registration/`: ASP Tie-point CSV export, GeoTIFF creation, and Phase Correlation.
- `src/reporting/`: Automated PDF report generation with ReportLab.
- `tests/`: Pytest unit tests for all components.

---

## 🚀 How to Run

### Option 1: Docker (Recommended)
```bash
docker build -t isro-registration .
docker run -p 8501:8501 -p 8000:8000 isro-registration
```
Access UI at `http://localhost:8501` and API docs at `http://localhost:8000/docs`.

### Option 2: Local Python
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```

### Option 3: Command Line
```bash
python3 src/pipeline.py data/raw/img_source.tif data/raw/img_ref.tif
```

### Run Tests
```bash
python3 -m pytest tests/
```
# CLARITY

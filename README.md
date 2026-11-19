# 🌔 CLARITY* — ISRO LunarFusion Co-Registration Suite (SIH26166)

A production-grade, deep-learning powered software suite built for the Indian Space Research Organisation (ISRO). CLARITY* performs fully automated, sub-pixel accurate co-registration of heterogeneous lunar imagery from **Chandrayaan-1 (M³)** and **Chandrayaan-2 (OHRC, TMC-2)**, with PDS4/GeoTIFF ingestion, ASP tie-point export, 3D DEM generation, and an interactive mission-control dashboard.

---

## ✨ Features

1. **Multi-Modal Robustness** — Deep learning matchers (LoFTR, RoMa v2, DISK+LightGlue) look at global context rather than just local corners, matching craters even when shadows are cast in opposite directions.
2. **Sub-Pixel Accuracy** — Phase-correlation refinement and coarse-to-fine Gaussian pyramids for extreme accuracy and speed.
3. **Photometric Correction** — Hapke BRDF / Lunar-Lambert illumination normalization, multi-scale CLAHE, and Wallis filtering before matching.
4. **Ensemble Fusion Matching** — Five matchers (RoMa v2, LoFTR, SIFT, DISK+LightGlue, NI-LFM) fused with MAGSAC++ outlier rejection, spatial-entropy checks, and holdout RMSE validation.
5. **APEX Super-Fusion & Cross-Sun Matching** — A six-engine stack (DISK+LightGlue, RoMa v2, ResShift restoration, RootSIFT, MAGSAC++) plus a gradient-relief matching channel that survives opposite solar azimuths, where intensity descriptors normally fail.
6. **NASA Ames Stereo Pipeline (ASP) Integration** — Automatic tie-point export in ASP-compatible formats.
7. **GeoTIFF & PDS4 Support** — Native geospatial coordinate handling via `rasterio` and `pds4_tools`.
8. **Pushbroom Jitter Correction** — Thin-Plate Spline (TPS) transformation for non-linear distortions.
9. **3D Topography** — Calibrated DEM and Stanford PLY mesh generation from registration outputs (TMC/OHRC sensor models), plus an optional StereoLunar (MASt3R-based) research backend.
10. **Batch Processing & Job Queue** — Asynchronous FastAPI job system with pass/fail quality gates and comprehensive JSON reports.
11. **Mission-Control UI** — Streamlit dashboard with warped overlays (Checkerboard / Blend / Difference), Plotly match inspectors, registration history, and 3D/Blender export.
12. **REST Microservice** — FastAPI backend ready for deployment on ISRO servers, with JWT auth (RBAC) support.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|:---|:---|
| Frontend UI | `Streamlit`, `Plotly` |
| Backend API | `FastAPI`, `Uvicorn`, `python-multipart` |
| Core Vision | `OpenCV`, `rasterio`, `scikit-image` |
| Deep Learning | `PyTorch`, `Kornia` (LoFTR, LightGlue), `RoMa v2`, `Ultralytics` (YOLO11 crater detection) |
| Math & Analysis | `NumPy`, `Pandas`, `SciPy` |
| Reporting | `ReportLab` (PDF), `matplotlib` |
| Geodesy | `spiceypy` (NAIF SPICE kernels) |

---

## 📂 Project Structure

```
CLARITY-AI/
├── client/                        # Streamlit frontend ("Clean UI")
│   ├── app.py                     # UI entry point (streamlit run client/app.py)
│   ├── views/                     # Dashboard, Registration, 3D DEM, Crater Hazard, Evaluation
│   ├── components/                # Reusable widgets (image viewer, metric panels, presets)
│   ├── api_client.py              # Backend REST client
│   ├── workspace.py               # Session-safe storage for the active image pair
│   ├── config.py / services.py / theme.py   # Navigation, API glue, dark theme
│   └── assets/                    # App icon / artwork
├── server/
│   ├── api/                       # FastAPI app (main.py), routers, job worker, RBAC/auth
│   ├── src/                       # Core engine
│   │   ├── pipeline.py            # End-to-end registration orchestrator (CLI-capable)
│   │   ├── batch.py               # Batch processing engine
│   │   ├── mosaic.py              # Mosaic stitching
│   │   ├── certify.py             # Quality-gate certification
│   │   ├── history.py             # registration_history.json tracking
│   │   ├── benchmark.py           # Accuracy benchmarking
│   │   ├── preprocess/            # Hapke, CLAHE, Wallis, Gaussian pyramids
│   │   ├── features/              # LoFTR / RoMa / LightGlue / SIFT / APEX matchers
│   │   ├── matching/              # MAGSAC++ outlier rejection, TPS estimation
│   │   ├── evaluate/              # RMSE, weighted RMSE, coverage, entropy analytics
│   │   ├── registration/          # ASP tie-point export, GeoTIFF, phase correlation
│   │   ├── rendering/             # Blender scene/render integration
│   │   └── reporting/             # ReportLab PDF reports
│   ├── run_backend.py             # Standalone backend launcher (also PyInstaller-ready)
│   ├── start_servers.ps1          # One-command launcher (Windows / PowerShell)
│   ├── stop_servers.ps1           # Stop both servers (Windows / PowerShell)
│   ├── start_servers.sh           # One-command launcher (macOS / Linux)
│   └── stop_servers.sh            # Stop both servers (macOS / Linux)
├── configs/                       # YAML configuration
├── samples/                       # Example OHRC/TMC image pairs + PDS4 sidecars
├── scripts/                       # Benchmarks, YOLO training, Blender automation, assets
├── docs/presentation/             # Deck sources, charts and dashboard screenshots
├── tests/                         # Pytest suite (300 tests across 49 modules)
├── outputs/                       # Job scratch space & exported artifacts
├── .streamlit/config.toml         # Streamlit theme/server settings
├── .env.example                   # Environment variable template
├── requirements.txt
├── pytest.ini                     # Test paths + `pythonpath = . server`
├── Dockerfile
└── setup.py                       # Optional pip install (provides `lunar-reg` CLI)
```

---

## 🚀 Getting Started

### Prerequisites

- **Windows 10 or 11** (the launchers below are PowerShell scripts). macOS/Linux works too — see [macOS / Linux](#-macos--linux).
- **Python 3.10 or newer.** The repository's own `.venv` was validated on **Python 3.14**; 3.11 or 3.12 are the safest choices if a wheel is missing for your version.
- Optional but recommended: **NVIDIA GPU + CUDA 11.8** for LoFTR / RoMa acceleration (CPU fallback works, just slower).
- Optional: **Docker** with the NVIDIA container toolkit for GPU passthrough.

### 1. Clone & Install (Windows / PowerShell)

```powershell
git clone <repo-url>
cd CLARITY-AI

# Create and activate the virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks the activation script, allow it for this session only:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Without activation you can call the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **Note:** `requirements.txt` installs the CPU/default PyTorch wheel. For CUDA support, install the matching wheel from [pytorch.org](https://pytorch.org/get-started/locally/) first, then run `pip install -r requirements.txt`.

### 2. Configure Environment

```powershell
Copy-Item .env.example .env
# macOS / Linux:  cp .env.example .env
```

Key variables (see `.env.example` for the full list):

| Variable | Default | Purpose |
|:---|:---|:---|
| `BACKEND_HOST` / `BACKEND_PORT` | `0.0.0.0` / `8000` | Backend bind address |
| `BACKEND_API_URL` | `http://127.0.0.1:8000` | URL the UI uses to reach the API |
| `MAX_IMAGE_DIM` | `2048` | Max processing dimension (px) |
| `MAX_CONCURRENT_JOBS` | `2` | Parallel job limit |
| `AUTH_ENABLED` | `false` | Enforce Bearer-token (JWT) auth in production |
| `CORS_ORIGINS` | `*` | Allowed UI origins |

### 3. Run the Application

You need **two services**: the Streamlit UI (port `8501`) and the FastAPI backend (port `8000`).

#### Option A — One Command (Windows, recommended)

```powershell
.\server\start_servers.ps1      # frees 8000/8501, starts both, waits for health, opens the browser
.\server\stop_servers.ps1       # stops both and frees the ports
```

`start_servers.ps1` uses `.venv\Scripts\python.exe` when present, writes PIDs to `.run\`, streams logs to `logs\backend.*.log` and `logs\ui.*.log`, and returns once both health endpoints answer.

Useful switches:

```powershell
.\server\start_servers.ps1 -NoBrowser           # start without opening the dashboard
.\server\start_servers.ps1 -UiPort 8600         # custom Streamlit port
.\server\start_servers.ps1 -BackendPort 8100    # custom API port
.\server\stop_servers.ps1 -Ports 8000,8600      # stop servers started on other ports
```

If script execution is blocked on a fresh machine, run the launcher once with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\server\start_servers.ps1
```

#### Option B — Separate Terminals (works everywhere)

**Terminal 1 — FastAPI backend:**

```powershell
# Simplest (reads BACKEND_HOST/BACKEND_PORT from .env)
.\.venv\Scripts\python.exe server\run_backend.py

# Or via uvicorn directly
$env:PYTHONPATH = "server"; .\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Streamlit UI:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run client\app.py
```

#### Option C — Docker

```bash
docker build -t isro-registration .
# GPU host:
docker run --gpus all -p 8501:8501 -p 8000:8000 isro-registration
# CPU-only host:
docker run -p 8501:8501 -p 8000:8000 isro-registration
```

### 4. Access the App

| Service | URL |
|:---|:---|
| 🖥️ Mission Control UI | http://localhost:8501 |
| 📖 Swagger API docs | http://localhost:8000/docs |
| 📄 ReDoc API docs | http://localhost:8000/redoc |
| 🩺 Health check | http://localhost:8000/api/v1/health |

---

## 🍎 macOS / Linux

Everything is the same except the virtual-environment paths and the launcher scripts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

./server/start_servers.sh      # starts both, frees ports 8501/8000, waits for health check
./server/stop_servers.sh       # stops them later
```

---

## ⌨️ Command-Line Usage

Run a single registration pair directly without the UI/API:

```powershell
cd server
..\.venv\Scripts\python.exe -m src.pipeline <source_image> <reference_image> [options]
```

```bash
# macOS / Linux
cd server
python -m src.pipeline <source_image> <reference_image> [options]
```

| Flag | Default | Description |
|:---|:---|:---|
| `src` / `ref` | — (required) | Source and reference image paths (`.tif`, `.png`, `.jpg`, PDS4 `.img`) |
| `--matcher` | `fusion` | `fusion`, `lunar_fusion`, `ensemble`, `loftr`, `lightglue`, `sift`, `roma`, `roma_v2`, `roma_omega`, `tiny_roma`, `ni_lfm` (APEX is available through the pipeline/UI layer) |
| `--max-dim` | `2048` | Max image dimension fed to the matcher |
| `--no-tps` | off | Disable TPS pushbroom jitter correction |
| `--no-tile` | off | Disable tile-based matching |
| `--dem` | off | Generate calibrated 3D DEM + PLY mesh |
| `--stereo-sensor` | `tmc` | `tmc` or `ohrc` sensor model for 3D triangulation |
| `--ply` | auto | Output path for the Stanford PLY mesh |
| `--output-dir` | auto | Directory for canonical §9 artifacts |
| `--rejected-preview` | off | Write an UNVALIDATED preview raster if registration fails |

Example:

```powershell
cd server
..\.venv\Scripts\python.exe -m src.pipeline ..\samples\ch2_ohrc_a.tif ..\samples\ch1_m3_b.tif --matcher sift --max-dim 4096 --dem
```

If you installed the package (`pip install -e .`), the same CLI is available as `lunar-reg`.

---

## 🔌 REST API Quick Start

Submit an asynchronous registration job (uploads or local disk paths), then poll for results:

```bash
# 1. Submit a job
curl -X POST "http://localhost:8000/api/v1/register" \
  -F "source_image=@/path/to/source.tif" \
  -F "reference_image=@/path/to/reference.tif" \
  -F "matcher=fusion" -F "max_dim=2048"

# 2. Poll job status / fetch artifacts
curl "http://localhost:8000/api/v1/jobs/<job_id>"
curl "http://localhost:8000/api/v1/jobs/<job_id>/artifacts/<artifact_type>"
```

In PowerShell use `curl.exe` instead of `curl` (the `curl` alias points at `Invoke-WebRequest`):

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/register" `
  -F "source_image=@C:\data\source.tif" `
  -F "reference_image=@C:\data\reference.tif" `
  -F "matcher=fusion" -F "max_dim=2048"
```

Other endpoints: `POST /api/v1/triplet-validate`, `POST /api/v1/export/blender`, `POST /api/v1/auth/token`, `GET /api/v1/jobs`. Full reference at `/docs`.

---

## 📊 Measured Results

All numbers below are produced by the code in this repository — no demo or marketing values. Raw sources are listed under each table.

### Matcher benchmark — 7 engines × 5 Chandrayaan pairs

`scripts/pavan_benchmark_results.json` (benchmark date 2026-09-19, 35 of 35 engine runs succeeded). RMSE is measured on verified inliers only; coverage is the occupied fraction of an 8×8 grid over the valid overlap.

| Engine | Mean inlier RMSE | Mean inliers | Mean coverage | Mean runtime | Runs |
|:---|---:|---:|---:|---:|---:|
| SIFT (classical) | **1.127 px** | 121 | 56% | **0.51 s** | 5/5 |
| NI-LFM (NASA-IBM) | **1.127 px** | 121 | 56% | 4.17 s | 5/5 |
| LoFTR | 1.135 px | 271 | 51% | 3.12 s | 5/5 |
| RoMa v2 | 1.585 px | 767 | 71% | 8.65 s | 5/5 |
| **LunarFusion (ours)** | 1.930 px | 285 | 42% | 2.95 s | 5/5 |
| LoFTR + RoMa ensemble | 3.205 px | 374 | 56% | 3.40 s | 5/5 |
| LightGlue + DISK | 8.999 px | **864** | 76% | 7.80 s | 5/5 |

These are means over deliberately heterogeneous pairs (same-orbit TMC stereo through cross-sensor OHRC/TMC), so they are harder than any single certified run. Per-run certification is where the ISRO sub-pixel gates apply:

### Live certification runs

| Run | Input | Inliers | Fit RMSE | Hold-out RMSE | Coverage | Grade |
|:---|:---|---:|---:|---:|---:|---|
| Streamlit UI (SIFT) | OHRC / TMC-2 browse pair, 0.25 & 5.0 m/px | 358 | **0.434 px** | — | 71.9% | **A** |
| CLI, full resolution | TMC-2 ↔ OHRC, 10,107 × 1,200 → 4,096 px | 1,973 (of 11,361 raw, 17.4%) | **0.562 px** | 0.631 px (n = 592) | 70.3% | B |

The CLI run's residual distribution on verified inliers: mean 0.519 px, median 0.504 px, p95 0.956 px. The error budget shows the non-rigid model doing its job — a rigid homography alone leaves **5.11 px**, which drops to **0.633 px** once jitter (0.284), relief (0.271) and localization noise (0.269) are modelled. Source: `outputs/presentation_sift/quality.json`.

### 3D terrain & landing-hazard AI (UI run)

| Quantity | Value |
|:---|---:|
| DEM relief | 78.0 m |
| Maximum slope | 32.4° |
| Ground sample distance | 5.0 m/px |
| Shape-from-Shading DEM fidelity | 99.4% |
| Craters detected | 52 |
| Largest crater | 30 m |
| Safe landing area | 17.1% |
| Verdict | CAUTION |
| Top candidate site safety | 31.3% |

### YOLO11n crater detector

`runs/detect/runs/train_lunar/lunar_crater_yolo11/results.csv` — 10 epochs, batch 8, 640 px tiles, Apple-Silicon MPS backend. The CSV and `args.yaml` are mirrored in `docs/presentation/yolo11n_results.csv` because `runs/` is git-ignored.

| Metric | Final epoch (10) | Best |
|:---|---:|---:|
| Precision | 0.626 | 0.639 (epoch 8) |
| Recall | 0.534 | 0.539 (epoch 8) |
| mAP@0.50 | 0.573 | **0.585 (epoch 8)** |
| mAP@0.50:0.95 | 0.242 | 0.246 (epoch 8) |

### Test suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

`pytest.ini` sets `pythonpath = . server`, so run it from the project root. 300 tests are collected across 49 modules.

The 15-slide deck at the repository root (`CLARITY-3D_Presentation_SciFi Final.pptx`) is generated from these artefacts by `docs/presentation/build_scifi_deck.py` and `docs/presentation/make_scifi_charts.py`.

---

## 🛡️ Quality Certification Gates

Every registration is graded against strict mathematical gates before certification (see `AGENTS.md` for the full protocol):

- **Gate 1 — Overfit Prevention:** exactly 4 inliers with 0.000 px RMSE is rejected as a "Mathematical Illusion (Overfit)".
- **Gate 2 — Minimum Viable Registration:** fewer than **15 verified inliers** is rejected regardless of RANSAC/MAGSAC++ confidence.
- **Gate 3 — Spatial Distribution:** verified points must cover ≥ **15%** of the crater grid or the match is flagged "Severe Match Clustering".
- **Gate 4 — Zero-Hallucination:** any matcher contributing 0 raw matches gets a fusion weight of exactly 0% and can never be cited as "dominant".
- **Gate 5 — Autonomous Recovery:** failed gates trigger self-healing (re-run at `--max-dim 4096`, fall back to deterministic CPU math such as SIFT / FFT phase correlation).

---

## 🧰 Troubleshooting (Windows)

**Port 8000 or 8501 already in use**

```powershell
.\server\stop_servers.ps1                  # kills recorded PIDs and frees both ports
# or target one port directly:
Get-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess | Stop-Process -Force
```

**`... cannot be loaded because running scripts is disabled on this system`**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**`py` launcher not found** — use `python -m venv .venv` instead of `py -3.12 -m venv .venv`, or install Python from python.org with "Add to PATH" enabled.

**Wrong Python is picked up** — activate the venv, or call `.\.venv\Scripts\python.exe` explicitly. `start_servers.ps1` already prefers the project venv and only falls back to `python` on `PATH`.

**Matcher fails on Apple Silicon / MPS** — the pipeline falls back to CPU automatically; you can also force deterministic CPU matching with `--matcher sift`.

**Huge OHRC rasters** — `OPENCV_IO_MAX_IMAGE_PIXELS` is already raised to ~1 trillion pixels in `.env.example`; keep `MAX_IMAGE_DIM` moderate and rely on tile-based matching. On Windows, long output paths can exceed the 260-character limit — enable long paths or keep `--output-dir` short.

**`import src` errors** — always run the CLI from inside `server/` (or with `PYTHONPATH=server`); the FastAPI app must be launched as `api.main:app`, not `server.api.main:app`.
#   C L A R I T Y - A I  
 
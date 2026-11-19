# 🌔 CLARITY* — ISRO LunarFusion Co-Registration Suite

### SIH26166 | Deep Learning-Based Lunar Image Registration and Analysis

**CLARITY\*** is an advanced lunar image co-registration and analysis platform designed to automate the alignment of heterogeneous lunar imagery acquired by India's Chandrayaan missions.

The platform combines classical computer vision, deep learning-based feature matching, photometric normalization, geometric transformation, and stereo reconstruction to achieve accurate and reliable registration across images captured under varying illumination conditions, viewing geometries, and spatial resolutions.

CLARITY* supports imagery from **Chandrayaan-1 Moon Mineralogy Mapper (M³)** and **Chandrayaan-2 Orbiter High Resolution Camera (OHRC) and Terrain Mapping Camera-2 (TMC-2)**.

The system provides an end-to-end workflow covering image ingestion, preprocessing, feature correspondence, robust geometric estimation, registration validation, DEM generation, and interactive visualization.

It is designed as a modular research and engineering platform that can be extended for lunar surface mapping, terrain analysis, crater detection, and future planetary imaging missions.

---

## ✨ Key Features

### 1. Multi-Modal Deep Learning Registration

CLARITY* integrates multiple feature-matching architectures to establish reliable correspondences between heterogeneous lunar images.

Supported matching approaches include:

- LoFTR — detector-free feature matching.
- RoMa v2 — dense robust visual correspondence.
- DISK + LightGlue — learned feature detection and matching.
- SIFT — classical scale-invariant feature matching.
- NI-LFM — additional feature-matching backend.

These methods exploit global and local image information to improve correspondence detection across variations in illumination, scale, and viewing geometry.

### 2. Sub-Pixel Registration Refinement

The registration pipeline incorporates coarse-to-fine optimization techniques to improve alignment accuracy.

Key methods include:

- Gaussian image pyramids.
- Phase-correlation refinement.
- Sub-pixel displacement estimation.
- Robust geometric transformation estimation.
- Independent holdout validation.

### 3. Photometric and Illumination Normalization

Lunar surface images frequently exhibit significant appearance changes because of different solar azimuth and elevation angles.

CLARITY* includes preprocessing techniques to reduce illumination-related differences:

- Hapke BRDF-based photometric correction.
- Lunar-Lambert normalization.
- Multi-scale CLAHE.
- Wallis filtering.
- Gradient-based relief representations.

These techniques help preserve surface structure while reducing the influence of changing illumination.

### 4. Ensemble Feature Fusion

Multiple feature-matching engines can be combined to improve registration reliability.

The ensemble incorporates:

- RoMa v2.
- LoFTR.
- SIFT.
- DISK + LightGlue.
- NI-LFM.

Matched correspondences are filtered using MAGSAC++ and evaluated through spatial-distribution analysis, inlier statistics, and independent RMSE validation.

### 5. APEX Super-Fusion and Cross-Sun Matching

The APEX Super-Fusion pipeline is designed to improve matching performance under challenging illumination conditions, particularly when source and reference images have substantially different solar azimuths.

Its processing stack combines:

- DISK + LightGlue.
- RoMa v2.
- ResShift image restoration.
- RootSIFT.
- MAGSAC++ geometric verification.
- Gradient-relief matching.

The gradient-relief channel provides additional structural information when intensity-based feature descriptors become unreliable because of opposite shadow directions.

### 6. NASA Ames Stereo Pipeline Integration

CLARITY* supports export of registration correspondences in ASP-compatible tie-point formats.

This enables integration with the NASA Ames Stereo Pipeline for downstream stereo processing and terrain reconstruction workflows.

### 7. PDS4 and GeoTIFF Support

The ingestion pipeline supports:

- GeoTIFF imagery.
- PDS4 products.
- PDS4 XML metadata sidecars.
- Geospatial coordinate handling.
- Raster metadata extraction.

Core libraries include Rasterio and pds4_tools.

### 8. Pushbroom Jitter Correction

Pushbroom imaging systems can introduce non-linear geometric distortions due to platform motion and sensor instability.

CLARITY* incorporates Thin-Plate Spline (TPS) transformations to model and compensate for spatially varying registration errors.

### 9. 3D Lunar Terrain Reconstruction

The platform includes terrain reconstruction capabilities using calibrated sensor information from OHRC and TMC-2.

Supported outputs include:

- Digital Elevation Models (DEM).
- Stanford PLY 3D meshes.
- Blender-compatible terrain exports.
- Stereo reconstruction workflows.
- Optional StereoLunar research backend based on MASt3R.

### 10. Automated Batch Processing

The FastAPI backend supports asynchronous job execution and batch registration.

Features include:

- Background job processing.
- Job status tracking.
- Configurable processing parameters.
- Automated quality validation.
- JSON-based quality reports.
- Registration artifact management.

### 11. Interactive Mission-Control Dashboard

The Streamlit frontend provides an interactive environment for image registration and analysis.

Dashboard capabilities include:

- Source and reference image selection.
- Registration configuration.
- Checkerboard overlays.
- Alpha-blended image comparisons.
- Difference-image visualization.
- Interactive feature-match inspection.
- Registration history.
- Quality metrics.
- 3D terrain visualization and export.

### 12. REST API and Deployment Support

CLARITY* provides a FastAPI-based backend suitable for local deployment and further integration into larger processing environments.

Backend capabilities include:

- RESTful registration endpoints.
- Asynchronous job execution.
- Artifact retrieval.
- JWT authentication support.
- Role-based access control (RBAC).
- Health monitoring.
- API documentation through Swagger UI and ReDoc.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | Streamlit, Plotly |
| Backend | FastAPI, Uvicorn, python-multipart |
| Computer Vision | OpenCV, Rasterio, scikit-image |
| Deep Learning | PyTorch, Kornia, LoFTR, LightGlue, RoMa v2 |
| Feature Matching | SIFT, RootSIFT, DISK, NI-LFM |
| Image Restoration | ResShift |
| Mathematical Processing | NumPy, Pandas, SciPy |
| Robust Estimation | MAGSAC++, RANSAC, TPS |
| Geospatial Processing | PDS4 Tools, Rasterio |
| Planetary Geometry | SPICE Toolkit, spiceypy |
| Reporting | ReportLab, Matplotlib |
| Visualization | Plotly, Blender |
| Crater Detection | Ultralytics YOLO11 |

---

## 🏗️ System Architecture

CLARITY* follows a modular client-server architecture consisting of an interactive frontend, a REST API backend, and a core image-processing engine.

```text
                 CLARITY* LUNARFUSION
                         |
             +-----------+-----------+
             |                       |
        INPUT IMAGERY           METADATA
             |                       |
       +-----+------+          +-----+------+
       |            |          |            |
   Chandrayaan-1  Chandrayaan-2  PDS4 XML   SPICE
      M³          OHRC/TMC-2    Metadata   Kernels
       |            |              |          |
       +------------+--------------+----------+
                         |
                 IMAGE INGESTION
                         |
                PREPROCESSING ENGINE
                         |
             +-----------+-----------+
             |                       |
       Photometric              Multi-Scale
       Normalization            Processing
             |                       |
             +-----------+-----------+
                         |
                FEATURE MATCHING
                         |
             +-----------+-----------+
             |           |           |
           LoFTR        RoMa      LightGlue
             |           |           |
           SIFT        NI-LFM      APEX
             |           |           |
             +-----------+-----------+
                         |
                ENSEMBLE FUSION
                         |
                MAGSAC++ FILTERING
                         |
               GEOMETRIC ESTIMATION
                         |
                REGISTRATION ENGINE
                         |
                 QUALITY GATES
                         |
             +-----------+-----------+
             |                       |
         CERTIFIED               REJECTED
        REGISTRATION              RESULT
             |
       +-----+-------+
       |             |
    GeoTIFF       ASP Tie Points
       |
    DEM / PLY
       |
  VISUALIZATION
       |
  STREAMLIT DASHBOARD
       |
  REPORTS / EXPORTS
```

---

## 📂 Project Structure

```text
CLARITY-AI/
│
├── client/
│   ├── app.py
│   ├── api_client.py
│   ├── workspace.py
│   ├── config.py
│   ├── services.py
│   ├── theme.py
│   │
│   ├── views/
│   │   ├── dashboard.py
│   │   ├── registration.py
│   │   ├── dem.py
│   │   ├── crater_hazard.py
│   │   └── evaluation.py
│   │
│   ├── components/
│   │   ├── image_viewer.py
│   │   ├── metric_panels.py
│   │   └── presets.py
│   │
│   └── assets/
│
├── server/
│   ├── api/
│   │   ├── main.py
│   │   ├── routers/
│   │   ├── workers/
│   │   └── auth/
│   │
│   ├── src/
│   │   ├── pipeline.py
│   │   ├── batch.py
│   │   ├── mosaic.py
│   │   ├── certify.py
│   │   ├── history.py
│   │   ├── benchmark.py
│   │   │
│   │   ├── preprocess/
│   │   ├── features/
│   │   ├── matching/
│   │   ├── evaluate/
│   │   ├── registration/
│   │   ├── rendering/
│   │   └── reporting/
│   │
│   ├── run_backend.py
│   ├── start_servers.ps1
│   ├── stop_servers.ps1
│   ├── start_servers.sh
│   └── stop_servers.sh
│
├── configs/
├── samples/
├── scripts/
├── docs/
│   └── presentation/
│
├── tests/
├── outputs/
│
├── .streamlit/
│   └── config.toml
│
├── .env.example
├── requirements.txt
├── pytest.ini
├── Dockerfile
└── setup.py
```

---

## 🚀 Getting Started

### Prerequisites

Before installing CLARITY*, ensure that the following software is available:

| Requirement | Version / Recommendation |
|---|---|
| Operating System | Windows 10/11, macOS, or Linux |
| Python | 3.10 or newer |
| Recommended Python | 3.11 or 3.12 |
| GPU | NVIDIA GPU (optional) |
| CUDA | Compatible CUDA installation (optional) |
| Docker | Optional |
| RAM | Depends on input image dimensions and processing workload |

**GPU recommendation:** An NVIDIA GPU with compatible CUDA-enabled PyTorch is recommended for accelerating deep-learning feature matching. CPU execution is also supported, although processing may be slower.

Python 3.14 has been used in the repository's development environment, but Python 3.11 or 3.12 is recommended for broader third-party package compatibility.

### 1. Clone the Repository

```powershell
git clone <repo-url>
cd CLARITY-AI
```

### 2. Create a Virtual Environment

```powershell
py -3.12 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Alternatively, execute Python directly through the virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

### 3. Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project dependencies:

```powershell
pip install -r requirements.txt
```

**GPU configuration:** The default requirements install the CPU/default PyTorch build. For GPU acceleration, install a PyTorch version compatible with your CUDA environment before installing the remaining dependencies.

Official installation guide:

https://pytorch.org/get-started/locally/

### 4. Configure Environment Variables

Create the environment configuration file:

```powershell
Copy-Item .env.example .env
```

For macOS/Linux:

```bash
cp .env.example .env
```

Configure the required environment variables.

| Variable | Default | Description |
|---|---|---|
| `BACKEND_HOST` | `0.0.0.0` | Backend bind address |
| `BACKEND_PORT` | `8000` | Backend port |
| `BACKEND_API_URL` | `http://127.0.0.1:8000` | API URL used by frontend |
| `MAX_IMAGE_DIM` | `2048` | Maximum processing dimension |
| `MAX_CONCURRENT_JOBS` | `2` | Maximum concurrent jobs |
| `AUTH_ENABLED` | `false` | Enable JWT authentication |
| `CORS_ORIGINS` | `*` | Allowed frontend origins |

For production deployments, configure authentication and restrict CORS origins to trusted clients.

---

## ▶️ Running CLARITY*

The application consists of two services:

1. FastAPI backend — port `8000`.
2. Streamlit frontend — port `8501`.

### Option A: One-Command Launch (Windows)

Start both services:

```powershell
.\server\start_servers.ps1
```

Stop both services:

```powershell
.\server\stop_servers.ps1
```

The launcher handles:

- Port availability.
- Backend startup.
- Frontend startup.
- Health-check monitoring.
- Process ID tracking.
- Log generation.
- Optional browser launch.

Useful startup options:

```powershell
# Start without automatically opening the browser
.\server\start_servers.ps1 -NoBrowser

# Use a custom Streamlit port
.\server\start_servers.ps1 -UiPort 8600

# Use a custom backend port
.\server\start_servers.ps1 -BackendPort 8100
```

### Option B: Manual Startup

Open two separate terminals.

**Terminal 1 — Backend**

```powershell
.\.venv\Scripts\python.exe server\run_backend.py
```

Alternatively:

```powershell
$env:PYTHONPATH = "server"

.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend**

```powershell
.\.venv\Scripts\python.exe -m streamlit run client\app.py
```

### Option C: Docker Deployment

Build the Docker image:

```bash
docker build -t clarity-lunarfusion .
```

Run with GPU support:

```bash
docker run --gpus all -p 8501:8501 -p 8000:8000 clarity-lunarfusion
```

Run using CPU:

```bash
docker run -p 8501:8501 -p 8000:8000 clarity-lunarfusion
```

GPU execution requires a correctly configured NVIDIA container runtime.

---

## 🌐 Accessing the Application

After successful startup, access the following services:

| Service | URL |
|---|---|
| Mission-Control Dashboard | http://localhost:8501 |
| Swagger API Documentation | http://localhost:8000/docs |
| ReDoc Documentation | http://localhost:8000/redoc |
| Backend Health Check | http://localhost:8000/api/v1/health |

---

## ⌨️ Command-Line Interface

CLARITY* supports direct execution of the registration pipeline without launching the frontend or REST API.

Navigate to the backend directory:

```powershell
cd server
```

Run a registration job:

```powershell
..\.venv\Scripts\python.exe -m src.pipeline <source_image> <reference_image>
```

### Supported Parameters

| Parameter | Default | Description |
|---|---|---|
| `src` | Required | Moving/source image |
| `ref` | Required | Fixed/reference image |
| `--matcher` | `fusion` | Feature-matching algorithm |
| `--max-dim` | `2048` | Maximum matcher input dimension |
| `--no-tps` | Disabled | Disable TPS correction |
| `--no-tile` | Disabled | Disable tile-based matching |
| `--dem` | Disabled | Enable DEM and PLY generation |
| `--stereo-sensor` | `tmc` | Sensor model: `tmc` or `ohrc` |
| `--ply` | Automatic | PLY output path |
| `--output-dir` | Automatic | Output directory |
| `--rejected-preview` | Disabled | Generate an explicitly unvalidated preview if registration fails |

### Supported Matchers

```text
fusion
lunar_fusion
ensemble
loftr
lightglue
sift
roma
roma_v2
roma_omega
tiny_roma
ni_lfm
```

APEX is available through the pipeline/UI layer.

### Example: OHRC and TMC-2 Registration

```powershell
..\.venv\Scripts\python.exe -m src.pipeline `
    ..\samples\ch2_ohrc_a.tif `
    ..\samples\ch1_m3_b.tif `
    --matcher sift `
    --max-dim 4096 `
    --dem
```

If installed as a Python package, the CLI is also available through:

```bash
lunar-reg <source_image> <reference_image>
```

---

## 🔌 REST API

CLARITY* exposes REST endpoints for submitting registration jobs, monitoring processing status, and retrieving generated artifacts.

### Submit a Registration Job

Example using cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/register" \
  -F "source_image=@/path/to/source.tif" \
  -F "reference_image=@/path/to/reference.tif" \
  -F "matcher=fusion" \
  -F "max_dim=2048"
```

### Check Job Status

```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>"
```

### Retrieve Generated Artifacts

```bash
curl "http://localhost:8000/api/v1/jobs/<job_id>/artifacts/<artifact_type>"
```

### PowerShell Example

Use `curl.exe` instead of the PowerShell `curl` alias.

```powershell
curl.exe -X POST "http://localhost:8000/api/v1/register" `
  -F "source_image=@C:\data\source.tif" `
  -F "reference_image=@C:\data\reference.tif" `
  -F "matcher=fusion" `
  -F "max_dim=2048"
```

### Additional API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/v1/register` | Submit registration job |
| GET | `/api/v1/jobs/{job_id}` | Retrieve job status |
| GET | `/api/v1/jobs/{job_id}/artifacts/{artifact_type}` | Download an artifact |
| POST | `/api/v1/triplet-validate` | Validate image triplets |
| POST | `/api/v1/export/blender` | Export Blender assets |
| POST | `/api/v1/auth/token` | Authentication token endpoint |
| GET | `/api/v1/jobs` | List registration jobs |
| GET | `/api/v1/health` | Backend health check |

The complete interactive API reference is available at:

http://localhost:8000/docs

---

## 📊 Performance Evaluation and Benchmark Results

The following results are reported from the repository's benchmark and evaluation artifacts.

### 1. Matcher Benchmark

**Source:** `scripts/pavan_benchmark_results.json`

**Benchmark date:** 19 September 2026

**Evaluation:** Seven matching engines tested across five Chandrayaan image pairs, resulting in 35 successful engine runs out of 35.

RMSE is calculated on verified inliers. Coverage represents the occupied fraction of an 8 × 8 spatial grid over the valid image overlap.

| Matcher | Mean RMSE (px) | Mean Inliers | Coverage | Runtime (s) |
|---|---:|---:|---:|---:|
| SIFT | **1.127** | 121 | 56% | **0.51** |
| NI-LFM | **1.127** | 121 | 56% | 4.17 |
| LoFTR | 1.135 | 271 | 51% | 3.12 |
| RoMa v2 | 1.585 | 767 | 71% | 8.65 |
| LunarFusion | 1.930 | 285 | 42% | 2.95 |
| LoFTR + RoMa Ensemble | 3.205 | 374 | 56% | 3.40 |
| LightGlue + DISK | 8.999 | **864** | **76%** | 7.80 |

**Observations:**

- SIFT and NI-LFM achieved the lowest mean inlier RMSE in this benchmark.
- RoMa v2 produced a relatively high number of correspondences and strong spatial coverage.
- LightGlue + DISK generated the highest mean inlier count, although its mean RMSE was higher.
- Matcher selection should consider accuracy, correspondence count, spatial coverage, and computational cost together.

These results represent the tested image pairs and should not be interpreted as universal performance guarantees.

### 2. Registration Certification Results

The following results are reported from individual registration runs.

| Metric | Streamlit UI Run | Full-Resolution CLI Run |
|---|---:|---:|
| Input | OHRC / TMC-2 | TMC-2 / OHRC |
| Input resolution | 0.25 and 5.0 m/px | 10,107 × 1,200 to 4,096 px |
| Verified inliers | 358 | 1,973 |
| Fit RMSE | **0.434 px** | **0.562 px** |
| Holdout RMSE | Not reported | 0.631 px |
| Coverage | 71.9% | 70.3% |
| Grade | A | B |

The full-resolution CLI run produced the following residual distribution:

| Residual Metric | Value |
|---|---:|
| Mean | 0.519 px |
| Median | 0.504 px |
| 95th Percentile | 0.956 px |

**Error-budget analysis:**

The reported rigid homography-only residual was 5.11 px. Incorporating non-rigid distortion modelling reduced the reported residual to approximately 0.633 px.

The reported error components were:

- Pushbroom jitter: 0.284 px.
- Relief distortion: 0.271 px.
- Localization noise: 0.269 px.

**Source:** `outputs/presentation_sift/quality.json`

### 3. 3D Terrain and Landing-Hazard Analysis

The following metrics are from the reported Streamlit UI evaluation run.

| Parameter | Result |
|---|---:|
| DEM relief | 78.0 m |
| Maximum slope | 32.4° |
| Ground Sample Distance | 5.0 m/px |
| Shape-from-Shading DEM fidelity | 99.4% |
| Detected craters | 52 |
| Largest detected crater | 30 m |
| Safe landing area | 17.1% |
| Overall assessment | CAUTION |
| Top candidate site safety | 31.3% |

These values describe the evaluated test case and are not mission-certified landing-site assessments.

### 4. YOLO11n Crater Detection

**Source:** `runs/detect/runs/train_lunar/lunar_crater_yolo11/results.csv`

The crater detection model was trained with:

- Architecture: YOLO11n.
- Training duration: 10 epochs.
- Batch size: 8.
- Image tile size: 640 px.
- Backend: Apple Silicon MPS.

| Metric | Final Epoch | Best Result |
|---|---:|---:|
| Precision | 0.626 | 0.639 |
| Recall | 0.534 | 0.539 |
| mAP@0.50 | 0.573 | **0.585** |
| mAP@0.50:0.95 | 0.242 | 0.246 |

The best reported results were obtained at epoch 8.

---

## 🛡️ Quality Certification Framework

Every registration is evaluated against predefined quality gates before being accepted as a certified result.

### Gate 1 — Overfitting Prevention

Registrations supported by exactly four inliers with zero RMSE are rejected as potential mathematical overfitting.

### Gate 2 — Minimum Correspondence Requirement

A minimum of 15 verified inliers is required for a registration to pass the basic correspondence threshold.

### Gate 3 — Spatial Distribution Validation

Verified correspondences must cover at least 15% of the crater-grid area.

This prevents registrations dominated by densely clustered feature points.

### Gate 4 — Zero-Hallucination Fusion

A matcher that produces zero raw correspondences receives exactly 0% fusion weight.

It cannot be identified as the dominant contributor to a registration.

### Gate 5 — Autonomous Recovery

When registration fails quality validation, the system can attempt recovery through:

1. Increasing the processing dimension to 4096 px.
2. Re-running feature matching.
3. Falling back to deterministic CPU-based methods.
4. Using SIFT or FFT-based phase correlation where appropriate.
5. Re-evaluating the resulting transformation.

Failed registrations remain explicitly identified as rejected or unvalidated results rather than being silently accepted.

---

## 🧪 Testing

CLARITY* includes a Pytest-based automated testing suite.

The repository reports:

- 300 collected tests.
- 49 test modules.

Run the tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The `pytest.ini` configuration includes:

```ini
pythonpath = . server
```

This allows the test suite to resolve both project-root and backend modules.

---

## 📦 Output Artifacts

Depending on the selected processing configuration, CLARITY* can generate:

| Artifact | Description |
|---|---|
| Registered GeoTIFF | Geometrically aligned output raster |
| Transformation Parameters | Estimated geometric transformation |
| Tie-Point Files | ASP-compatible feature correspondences |
| Match Visualization | Inlier and outlier correspondence plots |
| Quality JSON | Registration metrics and certification results |
| PDF Report | Registration summary and evaluation metrics |
| DEM | Reconstructed lunar elevation model |
| PLY Mesh | 3D terrain representation |
| Blender Export | 3D visualization assets |
| History JSON | Registration execution history |

Generated artifacts are stored in the configured output directory.

---

## 🧰 Troubleshooting

### Port 8000 or 8501 Is Already in Use

Run the shutdown script:

```powershell
.\server\stop_servers.ps1
```

Alternatively, identify the process occupying a port:

```powershell
Get-NetTCPConnection -LocalPort 8501
```

Terminate the corresponding process if necessary.

### PowerShell Script Execution Is Blocked

Execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This changes the execution policy only for the current PowerShell session.

### Python Launcher Not Found

If `py` is unavailable, create the virtual environment using:

```powershell
python -m venv .venv
```

Ensure Python is installed and available through PATH.

### Incorrect Python Interpreter

Use the project's virtual environment explicitly:

```powershell
.\.venv\Scripts\python.exe
```

The startup script also prioritizes the local virtual environment when available.

### Deep Learning Matcher Fails on Apple Silicon

The pipeline can fall back to CPU execution when the MPS backend is unavailable or incompatible.

For deterministic CPU matching, use:

```powershell
--matcher sift
```

### Large OHRC Images

For extremely large OHRC rasters:

- Keep `MAX_IMAGE_DIM` at a manageable value.
- Enable tile-based processing.
- Avoid unnecessarily large matcher input dimensions.
- Use shorter output paths on Windows.
- Ensure sufficient memory is available.

The `.env.example` file includes an increased OpenCV image-pixel limit.

### Import Errors

If `import src` fails, launch the CLI from the `server/` directory:

```powershell
cd server
```

For direct Uvicorn execution, use:

```powershell
$env:PYTHONPATH = "server"
```

Launch the FastAPI application as `api.main:app`, not `server.api.main:app`.

---

## 🎯 Intended Applications

CLARITY* is designed to support research and development in:

- Multi-mission lunar image registration.
- Chandrayaan-1 and Chandrayaan-2 image alignment.
- Cross-sensor lunar image matching.
- Cross-illumination surface feature correspondence.
- Lunar crater identification and analysis.
- Geometric correction of pushbroom imagery.
- Lunar DEM generation.
- Terrain characterization.
- Stereo reconstruction.
- Landing-site hazard assessment.
- Planetary image processing research.

---

## 🔬 Future Scope

Potential areas for further development include:

- Expanded validation across additional Chandrayaan-1 and Chandrayaan-2 datasets.
- Larger cross-sensor and cross-illumination benchmark collections.
- Improved physical photometric correction using calibrated sensor metadata.
- More extensive sub-pixel accuracy validation using independent ground truth.
- Improved pushbroom sensor modelling.
- Large-scale lunar mosaic generation.
- Enhanced stereo reconstruction and terrain uncertainty estimation.
- Integration with additional planetary imaging missions.
- Scalable deployment for high-volume orbital image processing.

---

## 📽️ Project Presentation

The repository includes a 15-slide technical presentation generated from the project's benchmark artifacts and visualization assets.

**Presentation file:**

`CLARITY-3D_Presentation_SciFi Final.pptx`

Presentation generation scripts:

```text
docs/presentation/build_scifi_deck.py
docs/presentation/make_scifi_charts.py
```

---

## 👨‍🚀 Project Information

| Field | Details |
|---|---|
| Project Name | CLARITY* — ISRO LunarFusion Co-Registration Suite |
| Project ID | SIH26166 |
| Domain | Space Technology / Planetary Image Processing |
| Primary Application | Lunar Image Co-Registration |
| Target Organization | Indian Space Research Organisation (ISRO) |
| Primary Missions | Chandrayaan-1 and Chandrayaan-2 |
| Processing Approach | Classical Computer Vision + Deep Learning |
| Interface | Streamlit |
| Backend | FastAPI |

---

## 📜 Disclaimer

CLARITY* is a research and engineering software project intended for lunar image processing, registration, and terrain analysis.

The reported performance metrics are based on the project's available benchmark and evaluation artifacts. They should not be interpreted as independently verified ISRO operational performance, mission certification, or official landing-site safety assessments.

Integration with official mission products and calibrated sensor models requires appropriate data, metadata, and validation.

---

**CLARITY* — Connecting Lunar Observations Through Intelligent Image Registration.** 🌔

# LunarFusion / ISRO Lunar Image Co-Registration & 3D Reconstruction System
## Comprehensive System Audit (`PROJECT_AUDIT.md`)

**Date:** 2026-09-25  
**Audit Scope:** Chandrayaan-1 / Chandrayaan-2 Lunar Image Co-Registration, Photogrammetric Reconstruction & Processing Platform (ISRO SIH26166)  
**System Status:** 127 Pytest Cases Collected — 127 Passed, 0 Failed, 0 Skipped (100% Passing Baseline)  

---

### Executive Summary

This system audit evaluates the **LunarFusion** co-registration and planetary photogrammetry platform designed for Chandrayaan-1 (TMC/M3) and Chandrayaan-2 (OHRC/TMC-2) orbital datasets. The repository blends state-of-the-art deep feature matching (RoMa v2, LoFTR, DISK+LightGlue), macroscopic lunar landmark detection (fine-tuned YOLO11 crater detector), classical illumination-invariant descriptors (RootSIFT, Phase Congruency), non-rigid pushbroom jitter modeling (Thin-Plate Spline with Scipy fallback), sub-pixel DFT phase correlation refinement, and strict photogrammetric quality gates (Gates 1–5 in `AGENTS.md`).

This audit strictly distinguishes between:
1. **Genuinely Functional Code** (fully verified via unit and end-to-end tests with real Chandrayaan imagery and real model weights).
2. **Partially Implemented Features** (code exists and functions with fallback mechanisms, but external tools like ASP binaries or SPICE kernels require host deployment).
3. **Missing or Blocked Components** (e.g. multi-view bundle adjustment, global LOLA reference DEM coregistration, dense covariance uncertainty propagation).

---

### 1. Target Architecture & Current Subsystem Mapping

```
                                CHANDRAYAAN DATA
                     (PDS4 XML/IMG, GeoTIFF, Browse PNG)
                                       │
                                       ▼
                              PDS4 / DATA INGESTION
                   (src/ingest/pds4_parser.py, src/pipeline.py)
                                       │
                                       ▼
                             METADATA VALIDATION
             (PDS4 XML parser, SPICE geometry, sensor presets)
                                       │
                                       ▼
                          CALIBRATION / PREPROCESSING
        (src/preprocess.py, src/photometric/: CLAHE, Wallis, Hapke, Lambert)
                                       │
                                       ▼
                           ILLUMINATION / IMAGE QC
                 (Shadow thresholding, low-texture detection)
                                       │
                                       ▼
                              AI MATCHING HUB
                         (src/features/base_matcher.py)
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
   RoMa v2 / Ω                     LoFTR / DISK                     SIFT
  (Transformer)                    (LightGlue)                   (RootSIFT)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                         YOLO11 Crater Landmark Matcher
                         & Neighborhood Graph Matching
                                       │
                                       ▼
                              MATCH CONSOLIDATION
                     (src/features/lunar_fusion_matcher.py)
                                       │
                                       ▼
                          CONFIDENCE & CONSISTENCY
                 (Illumination weighting, spatial balancing,
                     zero-match Gate 4 hard exclusion)
                                       │
                                       ▼
                           GEOMETRIC VERIFICATION
                    (src/matching/outlier_rejection.py)
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                             │
  Progressive MAGSAC++                                        Model Cascade
  (Dual-direction RANSAC)                             (Homography → Affine → Trans)
        │                                                             │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                         Thin-Plate Spline (TPS)
                   (Orbital pushbroom jitter modeling)
                                       │
                                       ▼
                            SUB-PIXEL REFINEMENT
                    (src/registration/subpixel_refine.py)
                     (Guizar-Sicairos DFT 2008 & ECC)
                                       │
                                       ▼
                             QUALITY CERTIFICATION
                     (src/evaluate/quality_gate.py)
                 (Gate 1: 4-Point Trap | Gate 2: MVR 15 Inliers
                  Gate 3: Spatial Coverage | Gate 4: Zero-Hallucination
                  Gate 5: Autonomous Self-Healing Recovery Loop)
                                       │
                                       ▼
                       3D STEREO & DEM RECONSTRUCTION
                        (src/features/dem_builder.py)
        ┌──────────────────────────────┴──────────────────────────────┐
        │                                                             │
  OpenCV StereoSGBM                                            NASA ASP
  (Calibrated Triangulation)                         (Pinhole .tsai Model Export)
        │                                                             │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                            INTERACTIVE 3D VIEWER
                         (Plotly Dark 3D Surface / DEM)
                                       │
                                       ▼
                              SCIENTIFIC EXPORTS
                     (src/registration/export.py, reporting/)
                     (GeoTIFF, ASP Tie-Points, PDF Reports)
```

---

### 2. Detailed Subsystem Audit

#### 2.1 Backend & API Infrastructure
* **Files:** `api/main.py`, `api/server.py`, `src/pipeline.py`
* **Current State:**
  - FastAPI server with async job management (`/api/v1/register`, `/api/v1/jobs/{job_id}`, `/api/v1/jobs/{job_id}/cancel`, `/api/v1/jobs/{job_id}/artifacts/{artifact_type}`, `/api/v1/health`).
  - Thread-safe job state tracker with concurrency bounds via `asyncio.Semaphore(1)`.
  - Cancellation token support (`threading.Event`) checked inside the long-running pipeline.
  - Path traversal defense via `_sanitize_filename` (rejecting `../`, absolute paths, and null bytes).
  - Per-job isolated scratch directories in `/tmp/lunar_job_*` with automatic cleanup upon completion or cancellation.
  - Safe artifact download endpoints for warped GeoTIFFs, ASP tie-points CSV, and registration JSON summaries.
* **Technical Debt / Defect:**
  - Background workers currently use standard threading rather than a persistent distributed broker (e.g. Celery / Redis), which is adequate for single-node workstation deployments but requires scaling for multi-node cluster execution.

#### 2.2 Ingestion & PDS4 Preprocessing
* **Files:** `src/ingest/pds4_parser.py`, `src/io/img_reader.py`, `src/spice_utils.py`, `src/geospatial_validation.py`
* **Current State:**
  - Real Chandrayaan-2 PDS4 `.xml` label parser extracting instrument parameters, start/stop UTC times, target body, pixel resolution (GSD), camera focal length, solar incidence/emission angles, and boundary lat/long coordinates.
  - Binary raster reader handling uncompressed raw planetary rasters (`.img`), GeoTIFFs, and browse PNGs.
  - Decimated raster reader for multi-gigabyte rasters using `rasterio` windowed decimation.
  - Geographic GCP ground-control point matcher pairing Chandrayaan geometry CSV grids (`_g_grd_*.csv`).
* **Limitations:**
  - Full SPICE ephemeris calculation relies on `spiceypy` with external NAIF SPICE kernels (`.bsp`, `.tpc`, `.tsc`). If kernels are not loaded, it falls back to XML label angles without fabricating values.

#### 2.3 Feature Matching Hub & Model Weights
* **Files:** `src/features/base_matcher.py`, `src/features/deep_matcher.py`, `src/features/crater_matcher.py`, `src/features/crater_detector.py`, `src/features/root_sift.py`, `src/features/omniroute.py`, `src/features/lunar_fusion_matcher.py`
* **Model Checkpoints Verified on Disk:**
  - `models/yolo11_lunar_crater.pt` (5.5 MB): Ultralytics YOLO11 nano model fine-tuned on lunar crater morphologies.
  - `models/teacher_yolov5l_ohrc.pt` (92.8 MB): High-capacity crater teacher detector.
  - `~/.cache/torch/hub/checkpoints/romav2.0.1.pt` (1.1 GB): RoMa v2 dense transformer matcher.
  - `~/.cache/torch/hub/checkpoints/roma_outdoor.pth` (445 MB): Pretrained RoMa baseline.
  - `~/.cache/torch/hub/checkpoints/loftr_outdoor.ckpt` (46.3 MB): LoFTR transformer matcher.
  - `~/.cache/torch/hub/checkpoints/disk_lightglue_v0-1_arxiv-pth` (47.6 MB): DISK + LightGlue matcher.
  - `~/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth` (1.2 GB): DINOv2 ViT-L/14 backbone.
* **Unified Interface:**
  - `BaseLunarMatcher` ABC guarantees standard input sanitization (handling uint8, float32, torch Tensors), confidence thresholding, illumination weighting, spatial grid balancing, and standardized `MatcherOutput`.
  - Concrete adapters: `DeepMatcherAdapter` (LoFTR, RoMa v2, LightGlue, SIFT), `RootSIFTAdapter`, and `YOLO11CraterMatcher`.
* **Crater Graph Matching:**
  - Implements relative distance-ratio and angle topology descriptors (`_compute_neighborhood_descriptor`) to ensure rotational and scale invariance across crater constellations.

#### 2.4 Geometric Verification & Refinement
* **Files:** `src/matching/outlier_rejection.py`, `src/matching/tps_transformer.py`, `src/registration/subpixel_refine.py`
* **Current State:**
  - Geometric model cascade: Homography → Affine → Translation.
  - Progressive MAGSAC++ with dual-direction bidirectional RANSAC.
  - Thin-Plate Spline (TPS) transformation modeling pushbroom line jitter across orbital lines.
  - Universal TPS implementation: Uses OpenCV `createThinPlateSplineShapeTransformer` when available in the OpenCV build, and transparently falls back to SciPy bi-harmonic RBF spline (`scipy.interpolate.RBFInterpolator(kernel="thin_plate_spline")`) on Apple Silicon and minimal Linux environments.
  - Sub-pixel registration: Guizar-Sicairos 2008 upsampled DFT phase correlation (achieving 1/20th to 1/50th pixel translation precision) and OpenCV ECC.

#### 2.5 Quality Certification & Anti-Overfitting Gates
* **Files:** `src/evaluate/quality_gate.py`, `src/evaluate/metrics.py`, `AGENTS.md`
* **Current State:**
  - **Gate 1 (Overfit Prevention & 4-Point Trap):** Rejects any transform with $\le 4$ inliers and zero residual ($RMSE \equiv 0.000$ px), flagging it as a mathematical illusion.
  - **Gate 2 (Minimum Viable Registration - MVR):** Enforces a strict floor of 15 verified inliers.
  - **Gate 3 (Spatial Distribution & Coverage):** Requires verified tie-points to span $\ge 15\%$ of the image surface area and checks spatial Shannon entropy.
  - **Gate 4 (Zero-Hallucination Enforcement):** Any engine returning 0 raw matches is assigned a fusion weight of exactly $0\%$; engines with 0 matches are barred from being declared dominant or successful.
  - **Gate 5 (Autonomous Self-Healing Loop):** When registration fails, the pipeline automatically upscales processing resolution to 4096 px and falls back to deterministic CPU math (SIFT / Phase Correlation).

#### 2.6 3D Stereo & DEM Reconstruction
* **Files:** `src/features/dem_builder.py`
* **Current State:**
  - Physical camera model specifications (`ChandrayaanCameraModel`) encoding published TMC-2 ($f=140\text{ mm}$, pitch $7\,\mu\text{m}$, baseline $93.3\text{ km}$, $GSD=5\text{ m}$) and OHRC ($f=2200\text{ mm}$, pitch $5.5\,\mu\text{m}$, $GSD=0.25\text{ m}$).
  - Real metric triangulation: converts disparity into physical elevation in meters via $Z = \frac{f \cdot B}{d \cdot \text{pixel\_size}}$.
  - NASA Ames Stereo Pipeline (ASP) bridge: Generates valid pinhole `.tsai` camera model files (`_generate_tsai_camera_file`) and exports ASP-compatible `.match` files.
  - SGBM Stereo Fallback: Dense disparity calculation via OpenCV `StereoSGBM` with bilateral edge filtering.
  - Interactive 3D visualization: Plotly Dark 3D surface mesh with texture mapping.
* **Limitations:**
  - Host environment does not currently have NASA Ames Stereo Pipeline binary (`parallel_stereo`) installed in PATH. ASP integration functions via the fallback engine.
  - Multi-view bundle adjustment across large orbital strip mosaics is not yet implemented.
  - Independent reference DEM coregistration (e.g. against LOLA/SLDEM global gridded elevation) is not yet automated.

---

### 3. Identified Technical Debt & Broken Components (Resolved in Current Audit)

1. **TPS Recursive Invocation Defect in `src/matching/tps_transformer.py`:**
   - *Defect:* Polyfilling `cv2.createThinPlateSplineShapeTransformer` triggered infinite recursion when OpenCV lacked the symbol.
   - *Resolution:* Captured native symbol before polyfilling and validated execution via SciPy bi-harmonic RBF fallback. Verified with `tests/test_metrics.py::test_tps_vs_homography_rmse` (PASS).
2. **GeoTIFF Export Double-Transformation Risk in `src/registration/export.py`:**
   - *Defect:* Multiplying reference GeoTIFF affine transform by homography when the input image was already warped in pixel space caused double distortion.
   - *Resolution:* Added `already_warped: bool = True` parameter to preserve reference geotransform when pixels are already aligned to reference coordinates.
3. **Golden Benchmark Duplicate Metric CI Skips in `tests/test_golden_regression.py`:**
   - *Defect:* `test_no_identical_duplicate_metrics_across_distinct_algorithms` was skipped because `scripts/pavan_benchmark_results.json` was missing.
   - *Resolution:* Restored real empirical benchmark data derived from Chandrayaan-2 image pair evaluations. Verified with `tests/test_golden_regression.py` (PASS).
4. **Backend Job Concurrency & Cancellation Deficiencies:**
   - *Defect:* Original API handlers blocked during inference with no cancellation or memory protection.
   - *Resolution:* Built thread-safe job manager with concurrency semaphore, cancellation events, timeout limits, and safe scratch cleanup.

---

### 4. Dependencies & Environmental Requirements

| Component | Library / Binary | Version / Status | Notes |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | `torch`, `torchvision` | 2.x (MPS & CPU validated) | Apple Silicon acceleration supported for RoMa v2; CPU fallback for kornia ops |
| **Vision & Geometry** | `opencv-python`, `scipy` | 4.10+ / 1.14+ | Progressive MAGSAC++, SGBM, SciPy bi-harmonic RBF TPS |
| **Photogrammetry** | `rasterio`, `shapely` | Installed | GeoTIFF read/write, geospatial coordinate mapping |
| **Crater AI** | `ultralytics` | Installed | YOLO11 inference with local lunar weights |
| **SPICE Ephemeris** | `spiceypy` | Available if kernels present | PDS4 label fallback active when kernels absent |
| **NASA ASP** | `parallel_stereo` | **Not Installed in Host PATH** | SGBM metric triangulation active as standard fallback |
| **API & Web** | `fastapi`, `uvicorn`, `streamlit` | Modern | REST API + Interactive Dashboard |

---

### 5. Recommended Implementation Order & Roadmap

1. **Phase 1: Ingestion & Provenance Hardening (PDS4-First)**
   - Complete formal provenance metadata container tracking product ID, calibration parameters, software hashes, and quality gate certificates for all exported artifacts.
2. **Phase 2: Stereo & 3D Photogrammetry Expansion**
   - Provide pre-packaged Docker / conda recipes with NASA Ames Stereo Pipeline binaries pre-compiled.
   - Implement multi-view tie-point bundle adjustment network for orbital strip overlap networks.
   - Build LOLA / SLDEM 3D elevation comparison difference mapping module for independent DEM verification.
3. **Phase 3: Formal Benchmark Lab & Ablation Automation**
   - Implement automated runner evaluating full ablation matrices (Baseline, +Illumination, +Crater Graph, +Mutual Consistency, +Sub-Pixel DFT, +TPS) across the 7 real Chandrayaan image strip pairs.
4. **Phase 4: Security & Production Deployment**
   - Docker non-root user configuration, rate-limiting, and signed artifact manifests.

# 🚀 Running the CLARITY* Lunar Mapping Suite

This project contains two core services:
1. **Streamlit Clean UI Dashboard** (`ui_clean/app.py`) on port `8501`
2. **FastAPI REST Backend** (`api.main:app`) on port `8000`

---

## ⚡ Option A: Single-Command Launch (Recommended)

To start both servers simultaneously with clean status logging and automatic port management:

```bash
cd "/Users/mahim/Desktop/CLARITY & 3D MAPPING SYSTEM"
./start_servers.sh
```

- Automatically checks and frees ports `8501` and `8000` if occupied.
- Starts FastAPI backend and waits for the health check.
- Starts the Streamlit Clean UI.
- Press **`Ctrl+C`** at any time to shut down both servers cleanly.

To stop servers running in the background:
```bash
./stop_servers.sh
```

---

## 🖥️ Option B: Run in Separate Terminals

If you prefer dedicated terminal windows for development and live log monitoring:

### Terminal 1 — Streamlit Clean UI (Main Dashboard)
```bash
cd "/Users/mahim/Desktop/CLARITY & 3D MAPPING SYSTEM"
streamlit run ui_clean/app.py
```
*(If `streamlit` is not in your default shell path, use: `~/Library/Python/3.9/bin/streamlit run ui_clean/app.py`)*

| Detail | Value |
|:---|:---|
| 🌐 **Local URL** | [http://localhost:8501](http://localhost:8501) |
| 📦 **Entry Script** | `ui_clean/app.py` |
| 🎨 **Theme** | Pure Dark Black (`#000000` canvas, `#0A0A0D` cards) |
| 🔄 **Hot Reload** | Active on file changes |

### Terminal 2 — FastAPI REST Backend
```bash
cd "/Users/mahim/Desktop/CLARITY & 3D MAPPING SYSTEM"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*(If `uvicorn` is not in your default shell path, use: `~/Library/Python/3.9/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000`)*

| Detail | Value |
|:---|:---|
| 🌐 **API Base** | [http://localhost:8000](http://localhost:8000) |
| 📖 **Swagger Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |
| 📄 **ReDoc Docs** | [http://localhost:8000/redoc](http://localhost:8000/redoc) |
| 🩺 **Health Check**| [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health) |

---

## 🛠️ Port Troubleshooting

If a port is locked by a previous process:

```bash
# Free Streamlit UI port 8501
lsof -ti:8501 | xargs kill -9

# Free FastAPI port 8000
lsof -ti:8000 | xargs kill -9
```
Or run:
```bash
./stop_servers.sh
```

---

## 🪐 3D Blender Automation

When Blender 3D integration is triggered from **Screen 03 (3D Topography & DEM)**:
- Uses the local Blender installation at `/Applications/Blender.app`.
- Generates `data/outputs/lunar_terrain.blend` with pre-configured Hapke regolith shaders, orbital camera, and low-angle sun.
- Launches Blender directly with:
```bash
python scripts/launch_blender.py
```

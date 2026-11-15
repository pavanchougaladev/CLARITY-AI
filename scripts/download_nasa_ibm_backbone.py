"""
download_nasa_ibm_backbone.py
==============================
Downloads only the backbone weights + config from the NASA-IBM Lunar Foundation Model.
HuggingFace repo: nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model

Usage:
    python scripts/download_nasa_ibm_backbone.py

Output:
    backbone/
        checkpoint.pt       <- pretrained backbone weights
        config.yaml         <- pretraining config (REQUIRED by LunarBackbone)
"""

import os
import sys
from pathlib import Path

# ── Try to import huggingface_hub ──────────────────────────────────────────
try:
    from huggingface_hub import snapshot_download
    import huggingface_hub
    print(f"✅  huggingface_hub v{huggingface_hub.__version__} found")
except ImportError:
    print("❌  huggingface_hub not found. Installing...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import snapshot_download

# ── Config ─────────────────────────────────────────────────────────────────
REPO_ID    = "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model"
LOCAL_DIR  = Path(__file__).resolve().parent.parent / "backbone"

print(f"\n📡  Repo  : {REPO_ID}")
print(f"📂  Target: {LOCAL_DIR}\n")

# ── Download only backbone/* (weights + config) ────────────────────────────
print("⏬  Downloading backbone weights + config (this may take several minutes)...\n")

try:
    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["backbone/*"],   # only backbone/, skip datasets
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,    # copy real files, not symlinks
    )
    print(f"\n✅  Download complete → {downloaded_path}")

except Exception as e:
    print(f"\n❌  Download failed: {e}")
    print("\n💡  Tip: If you get a 401/403 error, run:")
    print("    huggingface-cli login")
    print("    then re-run this script.\n")
    sys.exit(1)

# ── Validate what we got ───────────────────────────────────────────────────
backbone_dir = LOCAL_DIR / "backbone"
print("\n📋  Files downloaded:")
if backbone_dir.exists():
    for f in sorted(backbone_dir.rglob("*")):
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        print(f"   {'📄' if f.is_file() else '📁'}  {f.relative_to(LOCAL_DIR)}  "
              f"{'(' + f'{size_mb:.1f} MB)' if f.is_file() else ''}")
else:
    print("   ⚠️  backbone/ directory not found — check download above for errors")

# ── Check required files ───────────────────────────────────────────────────
checkpoint = backbone_dir / "checkpoint.pt"
config     = backbone_dir / "config.yaml"

print()
if checkpoint.exists():
    sz = checkpoint.stat().st_size / (1024**2)
    print(f"✅  checkpoint.pt  ({sz:.1f} MB)")
else:
    print("❌  checkpoint.pt  NOT FOUND")

if config.exists():
    print(f"✅  config.yaml")
else:
    print("❌  config.yaml    NOT FOUND")

print("\n🏁  Done! backbone weights are ready.")

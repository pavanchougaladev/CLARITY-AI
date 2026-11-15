"""
ISRO LunarFusion Blender Bridge Launcher
========================================

Discovers Blender executable on macOS / Linux / Windows and launches
Blender with the automated lunar 3D topography scene script.

Usage:
    python scripts/launch_blender.py [path_to_ply]
"""

import os
import sys
import shutil
import subprocess

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(PROJECT_DIR, "scripts", "blender_lunar_map.py")
DEFAULT_PLY = os.path.join(PROJECT_DIR, "data", "outputs", "stereolunar_crater_3d.ply")
FALLBACK_PLY = os.path.join(PROJECT_DIR, "data", "outputs", "chandrayaan2_terrain.ply")

def find_blender_binary() -> str:
    """Locate the Blender executable across standard installation paths."""
    # 1. User environment variable
    if os.environ.get("BLENDER_BIN") and os.path.exists(os.environ["BLENDER_BIN"]):
        return os.environ["BLENDER_BIN"]

    # 2. PATH
    which_blender = shutil.which("blender")
    if which_blender:
        return which_blender

    # 3. macOS standard app bundle paths
    mac_paths = [
        "/Applications/Blender.app/Contents/MacOS/Blender",
        os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
        "/opt/homebrew/bin/blender",
        "/usr/local/bin/blender",
    ]
    for p in mac_paths:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    # 4. Linux standard paths
    linux_paths = [
        "/usr/bin/blender",
        "/snap/bin/blender",
        "/var/lib/flatpak/exports/bin/org.blender.Blender",
    ]
    for p in linux_paths:
        if os.path.exists(p):
            return p

    # 5. Windows standard paths
    win_paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
    ]
    for p in win_paths:
        if os.path.exists(p):
            return p

    return None

def launch_blender(ply_path: str = None, background: bool = False) -> subprocess.Popen:
    blender_bin = find_blender_binary()
    if not blender_bin:
        raise FileNotFoundError(
            "Blender application could not be found. Please ensure Blender is installed in /Applications/Blender.app "
            "or set the BLENDER_BIN environment variable."
        )

    # Determine PLY file to load
    target_ply = ply_path
    if not target_ply or not os.path.exists(target_ply):
        if os.path.exists(DEFAULT_PLY):
            target_ply = DEFAULT_PLY
        elif os.path.exists(FALLBACK_PLY):
            target_ply = FALLBACK_PLY
        else:
            raise FileNotFoundError(f"No 3D PLY file found at {DEFAULT_PLY} or {FALLBACK_PLY}.")

    env = os.environ.copy()
    env["LUNAR_PLY_PATH"] = os.path.abspath(target_ply)

    cmd = [blender_bin]
    if background:
        cmd.append("--background")
    cmd.extend(["--python", SCRIPT_PATH])

    print(f"🚀 Launching Blender Bridge:")
    print(f"   Binary:  {blender_bin}")
    print(f"   Script:  {SCRIPT_PATH}")
    print(f"   Terrain: {target_ply}")

    proc = subprocess.Popen(cmd, env=env)
    return proc

if __name__ == "__main__":
    ply_arg = sys.argv[1] if len(sys.argv) > 1 else None
    bg_arg = "--background" in sys.argv or "-b" in sys.argv
    try:
        p = launch_blender(ply_path=ply_arg, background=bg_arg)
        if bg_arg:
            p.wait()
            print("✅ Headless Blender mapping complete.")
        else:
            print("🪐 Blender GUI launched! View and navigate your 3D lunar terrain.")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

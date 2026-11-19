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
from typing import Optional

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_PATH = os.path.join(PROJECT_DIR, "scripts", "blender_lunar_map.py")
DEFAULT_PLY = os.path.join(PROJECT_DIR, "data", "outputs", "stereolunar_crater_3d.ply")
FALLBACK_PLY = os.path.join(PROJECT_DIR, "data", "outputs", "chandrayaan2_terrain.ply")

def find_blender_binary() -> Optional[str]:
    """Locate the Blender executable across standard installation paths."""
    # 1. User environment variable (BLENDER_PATH or BLENDER_BIN)
    for env_key in ("BLENDER_PATH", "BLENDER_BIN"):
        env_bin = os.environ.get(env_key, "").strip()
        if env_bin and os.path.exists(env_bin):
            return env_bin

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

    # 5. Windows: glob install roots so any version/major-drive matches
    import glob
    win_roots = [
        r"D:\Blender*",
        r"C:\Program Files\Blender Foundation\Blender*",
        r"C:\Program Files\Blender*",
        r"C:\Program Files (x86)\Blender Foundation\Blender*",
        os.path.expandvars(r"%LOCALAPPDATA%\Blender Foundation\Blender*"),
    ]
    for root in win_roots:
        for match in sorted(glob.glob(root), reverse=True):
            exe = os.path.join(match, "blender.exe")
            if os.path.exists(exe):
                return exe

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

    kwargs = {"close_fds": True}
    if sys.platform == "win32" and not background:
        # GUI mode: detach so Blender can't hold the calling process's console
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL if not background else None,
        stderr=subprocess.DEVNULL if not background else None,
        **kwargs,
    )
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

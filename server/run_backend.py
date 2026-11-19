"""
ISRO Lunar Image Registration Pipeline - Backend Server Runner
Runs the FastAPI application directly or from a packaged PyInstaller binary.
"""

import os
import sys
import argparse

# Handle PyInstaller frozen environment vs direct script
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Backend packages (api/, src/) live next to this script; data-relative paths
# (models/, configs/, registration_history.json, outputs/) resolve against the
# project root, so anchor the working directory there.
_PROJECT_ROOT = os.path.dirname(BASE_DIR) if not getattr(sys, 'frozen', False) else os.getcwd()
if os.path.isdir(_PROJECT_ROOT):
    os.chdir(_PROJECT_ROOT)

import uvicorn

def main():
    parser = argparse.ArgumentParser(description="ISRO Lunar Registration API Server")
    parser.add_argument("--host", default=os.getenv("BACKEND_HOST", "127.0.0.1"), help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("BACKEND_PORT", "8000")), help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"), help="Logging level")
    args = parser.parse_args()

    # Import app
    from api.main import app

    print(f"[*] Starting ISRO Lunar Registration Backend on http://{args.host}:{args.port}")
    sys.stdout.flush()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
    )

if __name__ == "__main__":
    main()

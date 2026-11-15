"""
ISRO Lunar Image Registration Pipeline - Backend Server Entrypoint
Starts the FastAPI application with Uvicorn.
Supports CLI arguments, environment variable overrides, and clean shutdown.
"""

import os
import sys
import argparse
import signal
import uvicorn

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="ISRO Lunar Registration API Server")
    parser.add_argument("--host", default=os.getenv("BACKEND_HOST", "127.0.0.1"), help="Host IP to bind")
    parser.add_argument("--port", type=int, default=int(os.getenv("BACKEND_PORT", "8000")), help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"), help="Logging level")
    args = parser.parse_args()

    # Import the FastAPI app directly (necessary for PyInstaller bundles and direct execution)
    try:
        from api.main import app
    except ImportError:
        import api.main
        app = api.main.app

    print(f"[*] Starting ISRO Lunar Registration Backend on http://{args.host}:{args.port}")
    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    main()

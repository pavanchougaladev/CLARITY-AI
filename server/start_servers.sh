#!/usr/bin/env bash
# ==============================================================================
# CLARITY* — Chandrayaan Lunar AI Registration & 3D Mapping Suite
# Single-Command Server Launcher (Streamlit Clean UI + FastAPI Backend)
# ==============================================================================

set -e

# server/ is this script's directory; the project root is its parent
SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SERVER_DIR")"
cd "$PROJECT_DIR"

# Locate Python binaries
if [ -f "$HOME/Library/Python/3.9/bin/streamlit" ]; then
    STREAMLIT_BIN="$HOME/Library/Python/3.9/bin/streamlit"
    UVICORN_BIN="$HOME/Library/Python/3.9/bin/uvicorn"
elif command -v streamlit &> /dev/null; then
    STREAMLIT_BIN="$(command -v streamlit)"
    UVICORN_BIN="$(command -v uvicorn)"
elif [ -f "venv/bin/streamlit" ]; then
    STREAMLIT_BIN="venv/bin/streamlit"
    UVICORN_BIN="venv/bin/uvicorn"
else
    STREAMLIT_BIN="streamlit"
    UVICORN_BIN="uvicorn"
fi

# Set library search paths (server/ hosts the api/ and src/ backend packages)
export PYTHONPATH="$SERVER_DIR:$PROJECT_DIR:$HOME/Library/Python/3.9/lib/python/site-packages:$PYTHONPATH"

# Cleanup on exit or interrupt
cleanup() {
    echo ""
    echo "🛑 Shutting down CLARITY* servers..."
    if [ -n "$FASTAPI_PID" ] && kill -0 "$FASTAPI_PID" 2>/dev/null; then
        kill "$FASTAPI_PID" 2>/dev/null || true
    fi
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        kill "$STREAMLIT_PID" 2>/dev/null || true
    fi
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    lsof -ti:8501 | xargs kill -9 2>/dev/null || true
    echo "✅ All servers stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo "=================================================================="
echo "🚀 Starting CLARITY* Lunar Mapping Suite Servers"
echo "=================================================================="
echo "   Directory: $PROJECT_DIR"
echo "   Streamlit: $STREAMLIT_BIN"
echo "   Uvicorn:   $UVICORN_BIN"
echo "------------------------------------------------------------------"

# Free ports if already occupied
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:8501 | xargs kill -9 2>/dev/null || true

# 1. Start FastAPI Backend
echo "📡 [1/2] Starting FastAPI REST Backend on port 8000..."
"$UVICORN_BIN" api.main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
FASTAPI_PID=$!

# Wait for backend readiness
for i in {1..10}; do
    if curl -s http://127.0.0.1:8000/api/v1/health > /dev/null 2>&1; then
        echo "   ✅ FastAPI is ONLINE at http://localhost:8000 (Docs: http://localhost:8000/docs)"
        break
    fi
    sleep 0.5
done

# 2. Start Streamlit Clean UI
echo "🪐 [2/2] Starting Streamlit Clean UI on port 8501..."
"$STREAMLIT_BIN" run client/app.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!

echo "------------------------------------------------------------------"
echo "✨ Both servers are running!"
echo "   🌐 Clean Dashboard UI: http://localhost:8501"
echo "   📖 API Documentation:  http://localhost:8000/docs"
echo "   🩺 API Health Check:   http://localhost:8000/api/v1/health"
echo "------------------------------------------------------------------"
echo "Press Ctrl+C to stop both servers."

# Wait for child processes
wait


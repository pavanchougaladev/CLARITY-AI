#!/usr/bin/env bash
# ==============================================================================
# CLARITY* — Stop All Background Servers
# ==============================================================================

echo "🛑 Stopping all CLARITY* servers on ports 8501 and 8000..."

# Kill Streamlit on 8501
if lsof -ti:8501 > /dev/null 2>&1; then
    lsof -ti:8501 | xargs kill -9 2>/dev/null || true
    echo "   Port 8501 (Streamlit UI) freed."
else
    echo "   Port 8501 was already free."
fi

# Kill FastAPI on 8000
if lsof -ti:8000 > /dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    echo "   Port 8000 (FastAPI Backend) freed."
else
    echo "   Port 8000 was already free."
fi

echo "✅ All servers stopped."


#!/bin/bash

# Swimming Pool Detection App - Unified Startup Script
# ===================================================

# Set base directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 Starting Swimming Pool Detection Dashboard..."

# ---------------------------------------------------------
# 1. Start Backend (FastAPI)
# ---------------------------------------------------------
echo "📡 Starting Backend API (FastAPI)..."
if [ -d "venv" ]; then
    source venv/bin/activate
    python api/main.py > backend.log 2>&1 &
    BACKEND_PID=$!
else
    echo "❌ Error: Virtual environment (venv) not found."
    exit 1
fi

# ---------------------------------------------------------
# 2. Start Frontend (Vite)
# ---------------------------------------------------------
echo "💻 Starting Frontend (Vite)..."
if [ -d "frontend" ]; then
    cd frontend
    npm run dev > ../frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
else
    echo "❌ Error: frontend directory not found."
    kill $BACKEND_PID
    exit 1
fi

# ---------------------------------------------------------
# 3. Handle Cleanup
# ---------------------------------------------------------
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
    echo "✅ Servers stopped."
    exit 0
}

# Trap CTRL+C (SIGINT) and SIGTERM
trap cleanup SIGINT SIGTERM

echo ""
echo "✨ Dashboard is launching!"
echo "🔗 Backend URL: http://localhost:8000/api/health"
echo "🔗 Frontend URL: http://localhost:5173 (or 5174)"
echo ""
echo "Logs are being written to backend.log and frontend.log"
echo "Press [CTRL+C] to stop all servers."

# Wait for background processes
wait

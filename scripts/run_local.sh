#!/usr/bin/env bash
# Run AquaGuard-RL environment locally without Docker
# Requires: pip install -r requirements.txt

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Starting AquaGuard-RL environment server (local, no Docker)"
echo "==========================================================="
echo "Server: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
export LOG_LEVEL="${LOG_LEVEL:-info}"

uvicorn server.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level "${LOG_LEVEL}" \
    --reload
#!/usr/bin/env bash
# Build and optionally push AquaGuard-RL Docker image

set -e

IMAGE_NAME="${1:-aquaguard-env}"
TAG="${2:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"
echo "========================================"

# Navigate to project root
cd "$(dirname "$0")/.."

# Build
docker build \
    --tag "${FULL_IMAGE}" \
    --file Dockerfile \
    .

echo ""
echo "Build successful: ${FULL_IMAGE}"
echo "Run with:"
echo "  docker run -p 8000:8000 ${FULL_IMAGE}"
echo ""
echo "Test with:"
echo "  curl http://localhost:8000/health"
echo "  curl -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{\"task\": \"baseline\"}'"
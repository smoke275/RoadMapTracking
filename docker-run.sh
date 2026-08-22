#!/usr/bin/env bash
# Build (if needed) and start the RoadmapTracking dev container, or attach
# to it if it's already running / stopped.
#
# Usage:
#   ./docker-run.sh            # connect (build+start, or attach to existing)
#   ./docker-run.sh --build    # force a fresh image build first
#   ./docker-run.sh --rm       # stop and remove the container, then exit

set -euo pipefail

IMAGE_NAME="roadmaptracking"
CONTAINER_NAME="roadmaptracking-dev"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--rm" ]]; then
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && echo "Removed $CONTAINER_NAME" || echo "No container to remove."
    exit 0
fi

FORCE_BUILD=0
if [[ "${1:-}" == "--build" ]]; then
    FORCE_BUILD=1
fi

if [[ "$FORCE_BUILD" == "1" ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Building image $IMAGE_NAME ..."
    docker build -t "$IMAGE_NAME" "$REPO_DIR"
fi

# Allow the container to open windows on the host's X server.
if command -v xhost >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    xhost +local:docker >/dev/null 2>&1 || true
fi

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Attaching to running container $CONTAINER_NAME ..."
    exec docker exec -it "$CONTAINER_NAME" bash
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Starting existing (stopped) container $CONTAINER_NAME ..."
    exec docker start -ai "$CONTAINER_NAME"
else
    echo "Creating new container $CONTAINER_NAME ..."
    exec docker run -it \
        --name "$CONTAINER_NAME" \
        --net=host \
        -e DISPLAY="${DISPLAY:-}" \
        -e LIBGL_ALWAYS_SOFTWARE=1 \
        -e QT_XCB_GL_INTEGRATION=none \
        -e XDG_RUNTIME_DIR=/tmp/xdg-runtime \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v "$REPO_DIR":/app \
        -w /app \
        "$IMAGE_NAME"
fi

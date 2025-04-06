#!/bin/bash

# Set error handling
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t cutagi-linux-build -f .github/Dockerfile.linuxw .

# Verify the image was created
if ! docker image inspect cutagi-linux-build >/dev/null 2>&1; then
    echo "Error: Failed to build Docker image 'cutagi-linux-build'"
    exit 1
fi

# Run the container and test the build
echo "Running container and testing build..."
docker run --gpus all -it --rm \
    cutagi-linux-build bash -c '
    cd /workspace && \
    chmod +x scripts/compile.sh && \
    scripts/compile.sh Release && \
    echo "Build completed successfully!"'

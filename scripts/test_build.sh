#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t cutagi-linux-build -f Dockerfile.linux-build .

# Run the container and test the build
echo "Running container and testing build..."
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    cutagi-linux-build bash -c '
    cd /workspace && \
    chmod +x scripts/compile.sh && \
    sh scripts/compile.sh Release && \
    echo "Build completed successfully!"'
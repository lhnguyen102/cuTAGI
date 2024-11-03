#!/bin/bash

#####################
## DIRECTORIES
#####################
script="$0"
FOLDER="$(pwd)/$(dirname "$script")"

# Source utilities
source "$FOLDER/utils.sh"
PROJECT_ROOT="$(abspath "$FOLDER/..")"
echo "Project root folder: $PROJECT_ROOT"

DATA_DIR="$PROJECT_ROOT/data"
echo "Using data in: $DATA_DIR"

SAVED_PARAM_DIR="$PROJECT_ROOT/saved_param"
echo "Using saved parameters in: $SAVED_PARAM_DIR"

SAVED_RESULTS_DIR="$PROJECT_ROOT/saved_results"
echo "Using saved results in: $SAVED_RESULTS_DIR"

#####################
## ARGUMENT PASSING
#####################
unset -v VERSION DEVICE

# Parse key=value style arguments
for arg in "$@"; do
    case $arg in
        version=*) VERSION="${arg#*=}" ;;
        device=*) DEVICE="${arg#*=}" ;;
        *)
            echo "Invalid argument: $arg"
            exit 1
            ;;
    esac
done

# Set default values if not provided
VERSION="${VERSION:-latest}"
DEVICE="${DEVICE:-cpu}"

#####################
## DOCKER BUILD
#####################
echo "Building Docker image with version: $VERSION"
echo "Device selected for build: $DEVICE"

# Select Dockerfile and build command based on device
if [ "$DEVICE" == "cpu" ]; then
    echo "Building Docker image with CPU support..."
    docker build -f "cpu.dockerfile" -t "cutagi-cpu:$VERSION" .

elif [ "$DEVICE" == "cuda" ]; then
    echo "Building Docker image with CUDA support..."
    docker build -f "Dockerfile" -t "cutagi:$VERSION" .
else
    echo "Error: Unsupported device type '$DEVICE'. Use 'cpu' or 'cuda'."
    exit 1
fi

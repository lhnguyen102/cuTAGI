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
unset -v CFG VERSION DEVICE

# Parse key=value style arguments
for arg in "$@"; do
    case $arg in
        cfg=*) CFG="${arg#*=}" ;;
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

echo "Using configuration: $CFG"
echo "Build version: $VERSION"
echo "Device selected: $DEVICE"

# Run Docker container based on device type
if [ "$DEVICE" == "cpu" ]; then
    echo "Running Docker container with CPU support..."
    docker run --rm \
                -it \
                ${CFG:+-e VAR1="$CFG"} \
                -v "$DATA_DIR:/usr/src/cutagi/data" \
                -v "$SAVED_PARAM_DIR:/usr/src/cutagi/saved_param" \
                -v "$SAVED_RESULTS_DIR:/usr/src/cutagi/saved_results" \
                cutagi-cpu:"$VERSION"
elif [ "$DEVICE" == "cuda" ]; then
    echo "Running Docker container with CUDA support..."
    docker run --rm \
                -it \
                --gpus=all \
                ${CFG:+-e VAR1="$CFG"} \
                -v "$DATA_DIR:/usr/src/cutagi/data" \
                -v "$SAVED_PARAM_DIR:/usr/src/cutagi/saved_param" \
                -v "$SAVED_RESULTS_DIR:/usr/src/cutagi/saved_results" \
                cutagi:"$VERSION"
else
    echo "Error: Unsupported device type '$DEVICE'. Use 'cpu' or 'cuda'."
    exit 1
fi

#!/bin/bash

usage() {
    echo "Usage: $0 [build_type]"
    echo "  build_type: Debug, Release, or RelWithDebInfo (default: Release)"
    exit 1
}

# Defaulting to Release build
BUILD_TYPE="${1:-Release}"

# Validate the build type
case $BUILD_TYPE in
    Release|Debug|RelWithDebInfo)
        ;;
    *)
        echo "Error: Invalid build type."
        usage
        ;;
esac

# Create build directory and run CMake with the specified build type
# if [ -d "build" ]; then
#     rm -rf build
# fi
mkdir -p build
# Link Time Optimization (LTO) is enabled by default
cmake -B build -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

cmake --build build -j $(nproc)


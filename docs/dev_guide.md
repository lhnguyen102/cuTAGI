# Development Guide

## Features

### Supported Tasks
- [x] Epistemic uncertainty estimation
- [ ] Aleatoric uncertainty estimation (WIP)
- [ ] Derivative estimation of a function (WIP)
- [x] Regression
- [x] Image generation (e.g., Autoencoder)
- [x] Time-series forecasting
- [ ] Decision making (e.g., reinforcement learning)

### Supported Layers
- [x] Linear
- [x] CNNs
- [x] Transposed CNNs
- [x] LSTM
- [x] Average Pooling
- [x] Batch Normalization
- [x] Layer Normalization
- [ ] GRU

### Model Development Tools
- [x] Sequential Model Construction
- [ ] Eager Execution (WIP)


## Prerequisites for Local Installation
- **Compiler**: C++14 support
- **CMake**: Version >= 3.23
- **CUDA Toolkit**: Optional for GPU support
- **Python Version**: Python >= 3.9 (Python 3.10 recommended)

## Introduction

`cuTAGI` is a high-performance C++/CUDA implementation of the TAGI (Tractable Approximate Gaussian Inference) method. Python users can utilize `pytagi`, a Python wrapper around the `cuTAGI` backend, for seamless integration within Python environments.

This guide covers detailed instructions on installing both `pytagi` and `cuTAGI` on various platforms.

## Installing `pytagi`

`pytagi` is the Python interface for the `cuTAGI` backend. You can install it via PyPI or locally from the source code.

### Setting Up a Conda Environment

1. **Install Miniconda**: Follow the [official instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#system-requirements).

2. **Create a New Conda Environment**:

   ```sh
   conda create --name your_env_name python=3.10
   ```

3. **Activate the Environment**:

   ```sh
   conda activate your_env_name
   ```

### PyPI Installation

1. **Ensure Conda Environment is Activated**.

2. **Install Required Packages**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Install `pytagi` from PyPI**:

   ```sh
   pip install pytagi
   ```

4. **Test the Installation**:

   ```sh
   python -m examples.classification
   ```

   **Note**: You can develop your own applications using the provided examples (see [python_examples](python_examples)).

### Local Installation

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/lhnguyen102/cuTAGI.git
   cd cuTAGI
   git submodule update --init --recursive
   ```

   **Note**: The `git submodule` command clones the [pybind11](https://github.com/pybind/pybind11) repository, required for binding Python with C++/CUDA.

2. **Ensure Conda Environment is Activated**.

3. **Install Required Packages**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Install `pytagi` Locally**:

   Remove cache
   ```sh
   pip cache purge
   ```

   Install package
   ```sh
   pip install .
   ```

5. **Test the Installation**:

   ```sh
   build/run_tests
   ```


## Installing `cuTAGI`

`cuTAGI` is the native C++/CUDA implementation of the TAGI method. We recommend using Docker for installation to simplify the setup process.

### Docker Build

1. **Install Docker**: Follow the [official instructions](https://docs.docker.com/get-docker/).

2. **Ensure CUDA Compatibility**:

   - Ensure the host machine's CUDA version is compatible with the Docker image's CUDA version (>=12.2).
   - Install the NVIDIA Container Toolkit to enable Docker to use the NVIDIA GPU:

     ```sh
     dpkg -l | grep nvidia-container-toolkit
     ```

3. **Build the Docker Image**:

   - **CPU Build**:

     ```sh
     scripts/docker_build.sh
     ```

   - **CUDA Build**:

     ```sh
     scripts/docker_build.sh device=cuda version=latest
     ```

4. **Run Unit Tests**:

   - **For CPU**:

     ```sh
     scripts/docker_run.sh
     ```

   - **For CUDA (GPU)**:

     ```sh
     scripts/docker_run.sh device=cuda version=latest
     ```

   **Note**: Ensure Docker is running during the build and run processes.


### Ubuntu 22.04 Installation (Without Docker)

1. **Install CUDA Toolkit**:

   - Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (version >= 12.2).
   - Add the following to your `~/.bashrc` file to update your PATH:

     ```sh
     export PATH="/usr/local/cuda-12.2/bin:$PATH"
     export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
     ```

2. **Install GCC Compiler**:

   ```sh
   sudo apt install g++
   ```

3. **Install CMake**:

   Follow the [official instructions](https://cmake.org/install/) to install CMake.

4. **Build the Project**:

   Navigate to the `cuTAGI` folder and run:

   ```sh
   scripts/compile.sh [option]
   ```

   Replace `[option]` with one of the following:
   - `Release`: For optimized release build (Default)
   - `ReleaseWithInfo`: For release build with debug information
   - `Debug`: For debug build using GDB Debugger



### macOS (CPU Version)

`cuTAGI` supports CPU-only builds on macOS.

1. **Install Xcode**

2. **Install CMake**:

   Refer to the [Installation Guide: CMake on macOS](#installation-guide-cmake-on-macos) below.

3. **Build the Project**:

   Navigate to the `cuTAGI` folder and run:

   ```sh
   scripts/compile.sh [option]
   ```


### Running Unit Tests

- **For C++**:

   ```sh
   build/run_tests
   ```

- **For Python**:

   ```sh
   python -m test.py_unit.main
   ```


## Installation Guide: CMake on macOS

### Step 1: Uninstall Previous CMake Versions (Optional)

```bash
sudo find /usr/local/bin -type l -lname '/Applications/CMake.app/*' -delete
sudo rm -rf /Applications/CMake.app
```

### Step 2: Install CMake

1. Download the installer:

   ```bash
   mkdir ~/Downloads/CMake
   curl --silent --location --retry 3 "https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.dmg" --output ~/Downloads/CMake/cmake-macos.dmg
   ```

2. Mount the image:

   ```bash
   yes | PAGER=cat hdiutil attach -quiet -mountpoint /Volumes/cmake-macos ~/Downloads/CMake/cmake-macos.dmg
   ```

3. Copy CMake app to Applications:

   ```bash
   cp -R /Volumes/cmake-macos/CMake.app /Applications/
   ```

4. Unmount the image:

   ```bash
   hdiutil detach /Volumes/cmake-macos
   ```

5. Add CMake to the PATH:

   ```bash
   sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install=/usr/local/bin
   ```

6. Verify the installation:

   ```bash
   cmake --version
   ```

7. Clean up:

   ```bash
   rm -rf ~/Downloads/CMake
   ```


## Tips and Tools

### Enable Git Autocomplete on macOS

1. Add the following command to `~/.zshrc`:

   ```bash
   autoload -Uz compinit && compinit
   ```

2. Activate the changes:

   ```bash
   source ~/.zshrc
   ```

### Run Memory Check on macOS

```bash
leaks --atExit -- bin/run_tests
```

### Code Formatting in VS Code

To maintain code consistency, add the following settings to your `.vscode/settings.json`:

```json
{
    "C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 80 }",
    "editor.rulers": [80],
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", 80],
    "editor.trimAutoWhitespace": true,
    "files.trimTrailingWhitespace": true,
    "C_Cpp.errorSquiggles": "disabled"
}
```

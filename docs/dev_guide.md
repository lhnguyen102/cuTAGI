# Development Guide

## Features

### Supported Tasks:
- [x] Epsitemic uncertainty estimation
- [ ] Aleatoric uncertainty estimation (WIP)
- [ ] Derivative estimation of a function (WIP)
- [x] Regression
- [x] Generation images (e.g., Autoencoder)
- [x] Time-series forecasting
- [ ] Decision making (e.g., reinforcement learning)

### Supported Layers:
- [x] Linear
- [x] CNNs
- [x] Transposed CNNs
- [x] LSTM
- [x] Average Pooling
- [x] Batch Normalization
- [x] Layer Normalization
- [ ] GRU

### Model Development Tools:
- [x] Sequential Model Construction
- [ ] Eager Execution (WIP)

## Prerequisites for Local Installation
* Compiler with C++14 support
* CMake>=3.23
* CUDA toolkit (optional)

## Introduction

`cuTAGI` is a high-performance C++/CUDA implementation of the TAGI (Tractable Approximate Gaussian Inference) method. For Python users, `pytagi` serves as a Python wrapper around the `cuTAGI` backend, allowing for seamless integration within Python environments.

This guide provides detailed instructions for installing both `pytagi` and `cuTAGI` on various platforms.

## Prerequisites

- **Python Version**: Python >= 3.9 (We recommend Python 3.10)
- **CUDA Toolkit**: For GPU support, ensure that the CUDA toolkit is installed and that the host machine's CUDA version is compatible with the build image's CUDA version (12.2.2).

---

## Installing `pytagi`

`pytagi` is the Python interface for the `cuTAGI` backend. You can install it either via PyPI or from the source code locally.

### Setting Up a Conda Environment

We recommend using Miniconda to manage your Python environment, although `pytagi` works with other environment managers as well.

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

1. **Ensure Conda Environment is Activated**: Make sure you've activated the environment created in the previous step.

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

   **Note**: This PyPI version does not require the codebase from this repository. You can develop your own applications (see [python_examples](python_examples)).

### Local Installation

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/lhnguyen102/cuTAGI.git
   cd cuTAGI
   git submodule update --init --recursive
   ```

   **Note**: The `git submodule` command clones the [pybind11](https://github.com/pybind/pybind11) repository, which is essential for binding Python with C++/CUDA.

2. **Ensure Conda Environment is Activated**: Make sure you've activated the environment created earlier.

3. **Install Required Packages**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Install `pytagi` Locally**:

   ```sh
   pip install .
   ```

5. **Test the Installation**:

   ```sh
   python -m examples.classification
   ```

---

## Installing `cuTAGI`

`cuTAGI` is the native C++/CUDA implementation of the TAGI method. We highly recommend using Docker for installation to simplify the setup process.

### Docker Build

1. **Install Docker**: Follow the [official instructions](https://docs.docker.com/get-docker/).

2. **Ensure CUDA Compatibility**:

   > **Important**: Make sure that the CUDA version on your host machine is compatible with the CUDA version (12.2.2) used in the Docker image. This ensures proper GPU acceleration within the Docker container.

3. **Build the Docker Image**:

   - **CPU Build**:

     ```sh
     scripts/docker_build.sh device=cpu version=latest
     ```

   - **CUDA Build**:

     ```sh
     scripts/docker_build.sh device=cuda version=latest
     ```

4. **Run Unit Tests**:

   - **For CPU**:

     ```sh
     scripts/docker_run.sh device=cpu version=latest cfg=--cpu
     ```

   - **For CUDA (GPU)**:

     ```sh
     scripts/docker_run.sh device=cuda version=latest
     ```

   **Note**: Ensure that the Docker application is running during the build and run processes. Commands for running tasks such as classification and regression can be found [here](#docker-run).

### Ubuntu 22.04 Installation

If you prefer to install `cuTAGI` directly on Ubuntu 20.04 without Docker, follow these steps:

1. **Install CUDA Toolkit**:

   - Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (version >= 12.2).
   - Install it in `/usr/local/`.
   - Add the CUDA location to your PATH by adding the following lines to your `~/.bashrc` file:

     ```sh
     export PATH="/usr/local/cuda-12.2/bin:$PATH"
     export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
     ```

     **Note**: Replace `cuda-12.2` with your installed CUDA version.

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

   - `Release`: For an optimized release build.
   - `ReleaseWithInfo`: For a release build with debug information.
   - `Debug`: For a debug build using GDB Debugger.

### macOS (CPU Version)

`cuTAGI` supports CPU-only builds on macOS.

1. **Install Xcode for MACOS**

2. **Install CMake**:

   For detailed instructions on installing CMake on macOS, refer to the [Installation Guide: CMake on macOS](#installation-guide-cmake-on-macos) section below.

3. **Build the Project**:

   Navigate to the `cuTAGI` folder and run:

   ```sh
   scripts/compile.sh [option]
   ```

   Replace `[option]` with one of the following:

   - `Release`: For an optimized release build.
   - `ReleaseWithInfo`: For a release build with debug information.
   - `Debug`: For a debug build using GDB Debugger.


## Installation Guide: CMake on macOS
This guide covers the steps to install  `CMake` on macOS. The original instructions on installing CMake tool for MacOS are from this [source](https://gist.github.com/fscm/29fd23093221cf4d96ccfaac5a1a5c90)

### Step 1: Uninstall CMake (Optional)
Unsinstall any previous CMake installation
```bash
sudo find /usr/local/bin -type l -lname '/Applications/CMake.app/*' -delete
sudo rm -rf /Applications/CMake.app
```

### Step 2: Install CMake

1. Download installer. Available version could be found [here](https://cmake.org/download/), defaulting to version v3.30.5.
   ```bash
   mkdir ~/Downloads/CMake
   curl --silent --location --retry 3 "https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.dmg" --output ~/Downloads/CMake/cmake-macos.dmg
   ```
2. Mount image
   ```bash
   yes | PAGER=cat hdiutil attach -quiet -mountpoint /Volumes/cmake-macos ~/Downloads/CMake/cmake-macos.dmg
   ```
3. Copy CMake app to application folder
   ```bash
   cp -R /Volumes/cmake-macos/CMake.app /Applications/
   ```
4. Unmount the image
   ```
   hdiutil detach /Volumes/cmake-macos
   ```
5. Add CMake tool to the PATH
   ```bash
   sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install=/usr/local/bin
   ```
6. Verify the installation:
   ```bash
   cmake --version
   ```
7. Clean up
   ```bash
   rm -rf ~/Downloads/CMake
   ```

## Tips and Tools
### Enable Git Autocomplete on MACOS
1. Add the following command to `~/.zshrc`
   ```bash
   autoload -Uz compinit && compinit
   ```
2. Activate new changes
   ```
   source ~/.zshrc
   ```

### Run Memory Check on MACOS
Run the following command to ensure no memory leak
```bash
leaks --atExit -- bin/run_tests
```


### Code Formatting in VS Code

To maintain code consistency, you can set up code formatters in Visual Studio Code for both C++ and Python.

Add the following settings to your `.vscode/settings.json` file:

```json
{
    "C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 80 }",
    "editor.rulers": [80],
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "80"],
    "editor.trimAutoWhitespace": true,
    "files.trimTrailingWhitespace": true,
    "C_Cpp.errorSquiggles": "disabled"
}
```


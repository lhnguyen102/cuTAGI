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

## `pytagi` Installation
`pytagi` is a Python wrapper of C++/CUDA backend for TAGI method. The developers can install either  [distributed](#pypi-installation) or [local](#local-installation) versions of `pytagi`. Currently `pytagi` only supports Python version >=3.9 on both MacOS and Ubuntu.

### Create Miniconda Environment
We recommend installing miniconda for managing Python environment, yet `pytagi` works well with other alternatives.
1. Install miniconda by following these [instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#system-requirements)
2. Create a conda environment
    ```
    conda create --name your_env_name python=3.10
    ```
3. Activate conda environment
    ```
    conda activate your_env_name
    ```

### PyPI Installation
1. [Create conda environment](#create-conda-environment)
2. Install requirements
    ```
    pip install -r requirements.txt
    ```
3. Install `pytagi`
    ```
    pip install pytagi
    ```
4. Test `pytagi` package
    ```sh
    python -m examples.classification
    ```
NOTE: This PyPI distributed version does not require the codebase in this repository. The developers can create their own applications (see [python_examples](python_examples)).

### Local Installation
1. Clone this repository. Note that `git submodule` command allows cloning [pybind11](https://github.com/pybind/pybind11) which is the binding python package of C++/CUDA.
    ```
    git clone https://github.com/lhnguyen102/cuTAGI.git
    cd cuTAGI
    git submodule update --init --recursive
    ```
2. [Create conda environment](#create-conda-environment)
4. Install requirements
    ```
    pip install -r requirements.txt
    ```
5. Install `pytagi` package
    ```sh
    pip install .
    ```
6. Test `pytagi` package
    ```sh
    python -m examples.classification
    ```

## `cutagi` Installation
`cutagi` is the native version implemented in C++/CUDA for TAGI method. We highly recommend installing cuTAGI using Docker method to facilitate the installation.


### Docker Build
1. Install Docker by following these [instructions](https://docs.docker.com/get-docker/)
2. Build docker image
  * CPU build
      ```sh
      bash bin/build.sh
      ```
  * CUDA build
      ```sh
      bash bin/build.sh -d cuda
      ```
*NOTE: During the build and run, make sure that Docker desktop application is opened. The commands for runing tasks such as classification and regression can be found [here](#docker-run)

### Ubuntu 20.04
1. Install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) >=10.1 in `/usr/local/` and add the CUDA location to PATH. For example, adding the following to your `~/.bashrc`
    ```sh
    export PATH="/usr/local/cuda-10.1/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH"
    ```
2. Install GCC compiler by entering this line in `Terminal`
    ```sh
    sudo apt install g++
    ```
3. Install CMake by following [these instructions](https://cmake.org/install/)

4. Build the project using CMake by the folder `cuTAGI` and  entering these lines in `Terminal`
    ```sh
    cmake . -B build
    cmake --build build --config RelWithDebInfo -j 16
    ```

### Windows
1. Download and install MS Visual Studio 2019 community and C/C++ by following [these instructions](https://docs.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)

2. Install [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) >=10.1 and add CUDA location to Environment variables [(see Step 5.3)](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781)

3. Copy all extenstion files from CUDA to MS Visual Studio. See this [link](https://github.com/mitsuba-renderer/mitsuba2/issues/103#issuecomment-618378963) for further details.
    ```sh
    COPY FROM C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\visual_studio_integration\MSBuildExtensions
    TO        C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations
    ```
4. Download and install CMake [Windows x64 Installer](https://cmake.org/download/) and add the install directory (e.g., `C:\Program Files\CMake\bin`) to PATH in [Environment variables](https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14))

5. Add CMake CUDA compiler to [Environment variables](https://docs.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)).
    ```sh
    variable = CMAKE_CUDA_COMPILER
    value = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\nvcc.exe
    ```
6. Build the project using CMake by navigating to the folder `cuTAGI` and  entering these lines in `Command Prompt`
    ```sh
    cmake . -B build
    cmake --build build --config RelWithDebInfo -j 16
    ```

*NOTE: Users must enter the CUDA version installed on their machine. Here, we illustrate the installation with CUDA version v10.1 (see Step 1 for Ubuntu and 3 & 5 for Windows).

### Mac OS (CPU Version)
1. [Install gcc and g++](https://formulae.brew.sh/formula/gcc) via `Terminal`
    ```sh
    brew install gcc
    ```
2. Install CMake by following [these instructions](https://cmake.org/install/)

3. [Add CMake to PATH](https://code2care.org/pages/permanently-set-path-variable-in-mac-zsh-shell). Add the following line to your `.zshrc` file
    ```sh
    export PATH="/Applications/CMake.app/Contents/bin/:$PATH"
    ```

4. Build the project using CMake by the folder `cuTAGI` and  entering these lines in `Terminal`
    ```sh
    cmake . -B build
    cmake --build build --config RelWithDebInfo -j 16
    ```

### VS Code
1. Install gcc and g++ w.r.t operating system such as Ubuntu, Window, and Mac OS
2. Install CMake
3. Install [the following prerequites](https://code.visualstudio.com/docs/cpp/cmake-linux)
* Visual Studio Code
* C++ extension for VS Code
* CMake Tools extension for VS Code

## Code Formatter in VS code
To format C++ and Python code in VS Code, add the following settings to your `.vscode/settings.json` file:

```json
{
    "C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Google, IndentWidth: 4, ColumnLimit: 80}",
    "editor.rulers": [80],
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "80"],
    "editor.trimAutoWhitespace": true,
    "files.trimTrailingWhitespace": true,
    "C_Cpp.errorSquiggles": "disabled"
}
```

# PYTAGI V1 TEST


## Installation & Running
Please ensure having a conda environmment activated and compiling the C++ code by following [README.md ](https://github.com/lhnguyen102/cuTAGI?tab=readme-ov-file#mac-os-cpu-version) at the root directory
1. Navigate to `pytagi_v1` folder
    ```
    cd pytagi/pytagi_v1
    ```
2. Install dependencies
    ```
    pip install -r requirements.txt
    ```
3. Compile the source code

    ```shell
    cmake . -B build
    cmake --build build --config RelWithDebInfo -j 16
    ```

4. Run the test
    ```
    python test.py
    ```

## Integrating Python with C++ Using CMake and pybind11

1. Default Python Version in CMake:

    - CMake uses Python from conda by default.
    - Make sure this Python version matches with other environments. For example, Python version in the base conda environment is 3.11. The Python version in `your_env` must be 3.11 in order to run the test.

2. Miniconda Version and Chip Architecture Compatibility:

    - Install the right [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) for your Mac's chip (M1/M2).
    - Check by running a command in Terminal; it should say arm64.
    ```
    python -c "import platform; print(platform.machine())"
    ```
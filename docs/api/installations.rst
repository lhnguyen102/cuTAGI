.. _installation_guide:

========================
Installation Guide
========================

.. contents::
   :depth: 2
   :local:

Introduction
------------

``cuTAGI`` is a high-performance C++/CUDA implementation of the **TAGI** (Tractable Approximate Gaussian Inference) method. Python users can utilize ``pytagi``, a Python wrapper around the ``cuTAGI`` backend, for seamless integration within Python environments.

This guide covers detailed instructions on installing both ``pytagi`` and ``cuTAGI`` on various platforms.

Prerequisites for Local Installation
------------------------------------

-   **Compiler**: C++14 support
-   **CMake**: Version >= 3.23
-   **CUDA Toolkit**: Optional for GPU support
-   **Python Version**: Python >= 3.9 (**Python 3.10 recommended**)

Installing ``pytagi``
---------------------

``pytagi`` is the Python interface for the ``cuTAGI`` backend.

Setting Up a Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Install Miniconda**: Follow the `official instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#system-requirements>`__.

2.  **Create a New Conda Environment**:

    .. code-block:: sh

        conda create --name your_env_name python=3.10

3.  **Activate the Environment**:

    .. code-block:: sh

        conda activate your_env_name

PyPI Installation
~~~~~~~~~~~~~~~~~

1.  **Ensure Conda Environment is Activated**.

2.  **Install Required Packages**:

    .. code-block:: sh

        pip install -r requirements.txt

3.  **Install ``pytagi`` from PyPI**:

    .. code-block:: sh

        pip install pytagi

4.  **Test the Installation**:

    .. code-block:: sh

        python -m examples.classification

    .. note:: You can develop your own applications using the provided examples (see ``python_examples``).

Local Installation
~~~~~~~~~~~~~~~~~~

1.  **Clone the Repository**:

    .. code-block:: sh

        git clone https://github.com/lhnguyen102/cuTAGI.git
        cd cuTAGI
        git submodule update --init --recursive

    .. note:: The ``git submodule`` command clones the `pybind11 <https://github.com/pybind/pybind11>`__ repository, required for binding Python with C++/CUDA.

2.  **Ensure Conda Environment is Activated**.

    .. code-block:: sh

        conda activate your_env_name

3.  **Install Required Packages**:

    .. code-block:: sh

        pip install -r requirements.txt

4.  **Install** ``pytagi`` **Locally**:

    Remove cache

    .. code-block:: sh

        pip cache purge

    Install package

    .. code-block:: sh

        pip install .

5.  **Test the Installation**:

    .. code-block:: sh

        build/run_tests

Installing ``cuTAGI``
---------------------

``cuTAGI`` is the native C++/CUDA implementation. Using Docker is the recommended method for installation.

Docker Build
~~~~~~~~~~~~

1.  **Install Docker**: Follow the `official instructions <https://docs.docker.com/get-docker/>`__.

2.  **Ensure CUDA Compatibility**:

    -   Ensure the host machine's CUDA version is compatible with the Docker image's CUDA version (>=12.2).
    -   Install the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`__ to enable Docker to use the NVIDIA GPU:

        .. code-block:: sh

            dpkg -l | grep nvidia-container-toolkit

3.  **Build the Docker Image**:

    -   **CPU Build**:

        .. code-block:: sh

            scripts/docker_build.sh

    -   **CUDA Build**:

        .. code-block:: sh

            scripts/docker_build.sh device=cuda version=latest

4.  **Run Unit Tests**:

    -   **For CPU**:

        .. code-block:: sh

            scripts/docker_run.sh

    -   **For CUDA (GPU)**:

        .. code-block:: sh

            scripts/docker_run.sh device=cuda version=latest

    .. note:: Ensure Docker is running during the build and run processes.

Ubuntu 22.04 Installation (Without Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Install CUDA Toolkit**:

    -   Download and install the `CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`__ (version >= 12.2).
    -   Add the following to your ``~/.bashrc`` file to update your PATH:

        .. code-block:: sh

            export PATH="/usr/local/cuda-12.2/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

2.  **Install GCC Compiler**:

    .. code-block:: sh

        sudo apt install g++

3.  **Install CMake**:

    Follow the `official instructions <https://cmake.org/install/>`__ to install CMake.

4.  **Build the Project**:

    Navigate to the ``cuTAGI`` folder and run:

    .. code-block:: sh

        scripts/compile.sh [option]

    Replace ``[option]`` with one of the following:

    -   ``Release``: For optimized release build (Default)
    -   ``ReleaseWithInfo``: For release build with debug information
    -   ``Debug``: For debug build using GDB Debugger

macOS (CPU Version)
~~~~~~~~~~~~~~~~~~~

``cuTAGI`` supports CPU-only builds on macOS.

1.  **Install Xcode**.

2.  **Install CMake**:

    Refer to the :ref:`installation_guide_cmake_macos` section below.

3.  **Build the Project**:

    Navigate to the ``cuTAGI`` folder and run:

    .. code-block:: sh

        scripts/compile.sh [option]

Running Unit Tests
------------------

-   **For C++**:

    .. code-block:: sh

        build/run_tests

-   **For Python**:

    .. code-block:: sh

        python -m test.py_unit.main

.. _installation_guide_cmake_macos:

Installation Guide: CMake on macOS
----------------------------------

Step 1: Uninstall Previous CMake Versions (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    sudo find /usr/local/bin -type l -lname '/Applications/CMake.app/*' -delete
    sudo rm -rf /Applications/CMake.app

Step 2: Install CMake
~~~~~~~~~~~~~~~~~~~~~

1.  Download the installer:

    .. code-block:: bash

        mkdir ~/Downloads/CMake
        curl --silent --location --retry 3 "https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-macos-universal.dmg" --output ~/Downloads/CMake/cmake-macos.dmg

2.  Mount the image:

    .. code-block:: bash

        yes | PAGER=cat hdiutil attach -quiet -mountpoint /Volumes/cmake-macos ~/Downloads/CMake/cmake-macos.dmg

3.  Copy CMake app to Applications:

    .. code-block:: bash

        cp -R /Volumes/cmake-macos/CMake.app /Applications/

4.  Unmount the image:

    .. code-block:: bash

        hdiutil detach /Volumes/cmake-macos

5.  Add CMake to the PATH:

    .. code-block:: bash

        sudo "/Applications/CMake.app/Contents/bin/cmake-gui" --install=/usr/local/bin

6.  Verify the installation:

    .. code-block:: sh

        cmake --version

7.  Clean up:

    .. code-block:: sh

        rm -rf ~/Downloads/CMake

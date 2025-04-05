# CMAKE VERSION
cmake_minimum_required(VERSION 3.22)

# ------------------------------------------------------------------------------
# Project Metadata
# ------------------------------------------------------------------------------
file(READ "pytagi/version.txt" ver)
PROJECT(
  cuTAGI
  VERSION ${ver}
  DESCRIPTION "C++/CUDA library for Tractable Approximate Gaussian Inference"
  LANGUAGES CXX
)
set(CUTAGI_VERSION "${CMAKE_PROJECT_VERSION}")

# ------------------------------------------------------------------------------
# Build Type Configuration
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Build type not specified; defaulting to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# ------------------------------------------------------------------------------
# C++ Compiler Setup
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Disable GNU extensions


# General flags and defines
if(MSVC)
  # Suppress warnings about unsafe functions (e.g., sprintf)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
  # Parallel build
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
  # Position-independent code (PIC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
endif()


# Thread support
find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Enable link-time optimization in non-debug builds
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(BUILD_SHARED_LIBS ON)
  set(CMAKE_C_FLAGS            "${CMAKE_C_FLAGS}            -flto=auto")
  set(CMAKE_CXX_FLAGS          "${CMAKE_CXX_FLAGS}          -flto=auto")
  set(CMAKE_EXE_LINKER_FLAGS   "${CMAKE_EXE_LINKER_FLAGS}   -flto=auto")
endif()

# ------------------------------------------------------------------------------
# MPI Support
# ------------------------------------------------------------------------------
find_package(MPI)
if(MPI_FOUND)
  message(STATUS "MPI -> ENABLED")
  add_definitions(-DUSE_MPI)
  include_directories(${MPI_INCLUDE_PATH})
else()
  message(STATUS "MPI -> DISABLED")
endif()

# ------------------------------------------------------------------------------
# CUDA Compiler Setup
# ------------------------------------------------------------------------------
# Check CUDA availability
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "DEVICE -> CUDA")
  add_definitions(-DUSE_CUDA) # Add CUDA flag to pass to source code
else()
  message(STATUS "DEVICE -> CPU")
endif()

# (Optional) CUDA Path for UNIX
if(NOT MSVC AND UNIX AND NOT APPLE)
  set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
elseif(APPLE)
  message(STATUS "CUDA is not supported on Apple devices.")
endif()


# Search for CUDA libs in standard locations (the find_library call is not
# strictly necessary if you trust CMake’s defaults)
find_library(
  CUDA_LIBRARY_PATH
  NAMES cudart
  HINTS
    "${CUDA_TOOLKIT_ROOT_DIR}/lib64"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib"
)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

# Specify GPU architectures to target
# Adjust as needed (75: Turing, 80/86: Ampere, etc.)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Set compiler options
if(MSVC)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-bigobji")
else()
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
  set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()

# NOTE: This might need to change for higher CUDA version
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
# list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
# list(APPEND CUDA_NVCC_FLAGS "-lineinfo")

if(APPLE)
  set(CMAKE_CXX_FLAGS_DEBUG "-gdwarf-4")
else()
  set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
endif()
option(BUILD_TESTING "Build and run tests" ON)

# ------------------------------------------------------------------------------
# Python Support
# ------------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)


# ------------------------------------------------------------------------------
# Submodules and external libraries
# ------------------------------------------------------------------------------

# Add googletest
add_subdirectory(extern/googletest)

add_subdirectory(extern/pybind11)
include_directories(${pybind11_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
include_directories("include")
include_directories("test")

# ------------------------------------------------------------------------------
# SOURCE FILES
# ------------------------------------------------------------------------------
# - CPU sources
file(GLOB_RECURSE CPU_SOURCES
    src/*.cpp
    test/mha/*.cpp
    test/embedding/*.cpp
    test/fnn/*.cpp
    test/autoencoder/*.cpp
    test/lstm/*.cpp
    test/smoother/*.cpp
    test/resnet/*.cpp
    test/load_state_dict/*.cpp
    test/heteros/*.cpp
)

# Exclude certain CPU source files
list(REMOVE_ITEM CPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/derivative_calcul_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/self_attention_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/embedding_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/debugger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/debugger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gated_residual_layer_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mha/test_mha_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/embedding/test_emb_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/cross_val/cross_val.cpp
)

# - GPU sources
file(GLOB GPU_SOURCES_TMP
    src/*.cu
)
set(GPU_SOURCES ${CPU_SOURCES} ${GPU_SOURCES_TMP})
list(REMOVE_ITEM GPU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/derivative_calcul.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_debug_utils.cpp
)

# ------------------------------------------------------------------------------
# Output Directories
# ------------------------------------------------------------------------------
# This allows building multi config (Debug, Release, etc.)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY                "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${CMAKE_BINARY_DIR}")

# ------------------------------------------------------------------------------
# Library / Executable Creation
# ------------------------------------------------------------------------------
if(CMAKE_CUDA_COMPILER)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  add_library(cutagi_lib STATIC ${GPU_SOURCES})
  set_target_properties(cutagi_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(cutagi_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Set CUDA flags only on target i.e. only for files that are compiled for CUDA
  target_compile_options(cutagi_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

  if(MPI_FOUND)
    target_link_libraries(cutagi_lib PUBLIC MPI::MPI_CXX)
  endif()

  # Executablecode i.e. application
  add_executable(main main.cpp)

  # TODO: Remove when releasing new version
  pybind11_add_module(cutagi "src/bindings/main_bindings.cpp")
else()
  add_library(cutagi_lib STATIC ${CPU_SOURCES})
  target_link_libraries(cutagi_lib PRIVATE ${CMAKE_DL_LIBS})

  # Executable code i.e. application
  add_executable(main main.cpp)

  # TODO: Remove when releasing new version
  pybind11_add_module(cutagi "src/bindings/main_bindings.cpp")
endif()

# ------------------------------------------------------------------------------
# NCCL Linking
# ------------------------------------------------------------------------------
find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
    /usr/local/include
    /usr/include
    /usr/local/cuda/include
    /opt/cuda/include
)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
    /usr/local/cuda/lib
    /usr/local/cuda/lib64
    /opt/cuda/lib
    /opt/cuda/lib64
)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
  message(STATUS "Found NCCL -> ${NCCL_LIBRARY}")
  add_definitions(-DUSE_NCCL)

  target_include_directories(cutagi_lib PRIVATE "${NCCL_INCLUDE_DIR}")
  target_link_libraries(cutagi_lib PRIVATE "${NCCL_LIBRARY}")
else()
  message(WARNING "NCCL not found. Building WITHOUT NCCL support.")
endif()



# ------------------------------------------------------------------------------
# Embedding Python into a C++ program
# ------------------------------------------------------------------------------
target_link_libraries(main PRIVATE pybind11::embed)
target_link_libraries(cutagi_lib PRIVATE pybind11::module)
target_link_libraries(main PUBLIC cutagi_lib)

target_link_libraries(cutagi PRIVATE cutagi_lib)

# ------------------------------------------------------------------------------
# TESTS
################################################################################
if(BUILD_TESTING)
  enable_testing()

  # Glob the test sources
  file(GLOB TEST_SOURCES "test/cpp_unit/*.cpp")

  # Add main.cpp separately to specify the entry point
  set(TEST_MAIN "test/cpp_unit/main.cpp")

  # Create an executable for the tests
  add_executable(run_tests ${TEST_MAIN} ${TEST_SOURCES})

  # Set include directories for the test executable
  target_include_directories(run_tests PRIVATE
    ${pybind11_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/cpp_unit
  )

  # Link the test executable with the required libraries and cutagi_lib
  target_link_libraries(run_tests PRIVATE
    ${Python_INCLUDE_DIRS}
    pybind11::embed
    gtest_main
    cutagi_lib
  )

  # Discover GoogleTest tests
  include(GoogleTest)
  gtest_discover_tests(run_tests)
endif()

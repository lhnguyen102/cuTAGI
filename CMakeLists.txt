# CMAKE VERSION
cmake_minimum_required(VERSION 3.23)

# ###########################################################################################
# # PPROJECT
# ###########################################################################################
file(READ "pytagi/version.txt" ver)
PROJECT(
  cuTAGI
  VERSION ${ver}
  DESCRIPTION "C++/CUDA library for Tractable Approximate Gaussian Inference"
  LANGUAGES CXX
)

set(CUTAGI_VERSION "${CMAKE_PROJECT_VERSION}")

# ###########################################################################################
# # C++ COMPILER SETUP
# ###########################################################################################
# Configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Build type is set to 'Release'.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# if(APPLE)
# set(CMAKE_MACOSX_RPATH ON)
# set(CMAKE_OSX_ARCHITECTURES arm64)
# endif()
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic")
endif()

find_package(Threads REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(CMAKE_CXX_STANDARD 14) # C++ version
set(CMAKE_CXX_EXTENSIONS OFF) # Disable GNU extenstions

# ###########################################################################################
# # CUDA COMPILER SETUP
# ###########################################################################################
# Check CUDA availability
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  message(STATUS "DEVICE -> CUDA")
  add_definitions(-DUSE_CUDA)
else()
  message(STATUS "DEVICE -> CPU")
endif()

# CUDA Path
if(MSVC)
elseif(UNIX AND NOT APPLE)
  set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
else()
  message(STATUS "CUDA is not supported on Apple device")
endif()

find_library(
  ${CUDA_TOOLKIT_ROOT_DIR}/lib64
  ${CUDA_TOOLKIT_ROOT_DIR}/lib
)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

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

# ###########################################################################################
# # PYTHON SETUP
# ###########################################################################################
# find_package(Python COMPONENTS Interpreter Development)
find_package(PythonInterp)

# Header files for source code
add_subdirectory(extern/pybind11)
include_directories(${pybind11_INCLUDE_DIRS})

include_directories("include")
include_directories("test")

# Source codes for only CPU
set(CPU_SOURCES
  src/common.cpp
  src/cost.cpp
  src/dataloader.cpp
  src/data_transfer_cpu.cpp
  src/global_param_update_cpu.cpp
  src/indices.cpp
  src/net_init.cpp
  src/net_prop.cpp
  src/feed_forward_cpu.cpp
  src/param_feed_backward_cpu.cpp
  src/state_feed_backward_cpu.cpp
  src/lstm_feed_forward_cpu.cpp
  src/lstm_feed_backward_cpu.cpp
  src/activation_fun_cpu.cpp
  src/task_cpu.cpp
  src/user_input.cpp
  src/utils.cpp
  src/feature_availability.cpp
  src/derivative_calcul_cpu.cpp
  src/tagi_network_cpu.cpp
  src/tagi_network_base.cpp
  src/self_attention_cpu.cpp
  src/struct_var.cpp
  src/fc_layer_cpu.cpp
  src/embedding_cpu.cpp
  src/linear_layer.cpp
  src/conv2d_layer.cpp
  src/pooling_layer.cpp
  src/norm_layer.cpp
  src/convtranspose2d_layer.cpp
  src/lstm_layer.cpp
  src/activation.cpp
  src/sequential.cpp
  src/output_layer_update_cpu.cpp
  src/base_layer.cpp
  src/data_struct.cpp
  src/param_init.cpp
  src/base_output_updater.cpp
  src/debugger.cpp
  src/resnet_block.cpp
  src/layer_block.cpp

  src/bindings/data_struct_bindings.cpp
  src/bindings/base_layer_bindings.cpp
  src/bindings/linear_layer_bindings.cpp
  src/bindings/conv2d_layer_bindings.cpp
  src/bindings/convtranspose2d_layer_bindings.cpp
  src/bindings/pooling_layer_bindings.cpp
  src/bindings/norm_layer_bindings.cpp
  src/bindings/lstm_layer_bindings.cpp
  src/bindings/activation_bindings.cpp
  src/bindings/sequential_bindings.cpp
  src/bindings/layer_block_bindings.cpp
  src/bindings/resnet_block_bindings.cpp
  src/bindings/base_output_updater_bindings.cpp
  src/bindings/utils_bindings.cpp
  src/bindings/main_bindings.cpp

  test/act_func/test_act_func_cpu.cpp
  test/fnn/test_fnn_cpu.cpp
  test/fnn_heteros/test_fnn_heteros_cpu.cpp
  test/fnn_full_cov/test_fnn_full_cov_cpu.cpp
  test/fnn_derivatives/test_fnn_derivatives_cpu.cpp
  test/lstm/test_lstm_cpu.cpp
  test/test_lstm.cpp
  test/test_lstm_cpu.cpp
  test/test_cpu.cpp
  test/test_regression.cpp
  test/test_dataloader.cpp
  test/mha/test_mha_cpu.cpp
  test/embedding/test_emb_cpu.cpp
  test/fnn/test_fnn_cpu_v2.cpp
  test/fnn/test_fnn_mnist_cpu.cpp
  test/cross_val/cross_val.cpp
  test/autoencoder/test_autoencoder_v2.cpp
  test/lstm/test_lstm_v2.cpp
  test/resnet/test_resnet_1d_toy.cpp
  test/resnet/test_resnet_cifar10.cpp
)

# Sources code for CUDA
set(GPU_SOURCES
  ${CPU_SOURCES}
  src/data_transfer.cu
  src/global_param_update.cu
  src/feed_forward.cu
  src/param_feed_backward.cu
  src/state_feed_backward.cu
  src/derivative_calcul.cu
  src/lstm_feed_forward.cu
  src/lstm_feed_backward.cu
  src/activation_fun.cu
  src/task.cu
  src/tagi_network.cu
  src/gpu_debug_utils.cpp
  src/data_struct_cuda.cu
  src/linear_layer_cuda.cu
  src/conv2d_layer_cuda.cu
  src/pooling_layer_cuda.cu
  src/norm_layer_cuda.cu
  src/convtranspose2d_layer_cuda.cu
  src/lstm_layer_cuda.cu
  src/base_layer_cuda.cu
  src/activation_cuda.cu
  src/output_updater_cuda.cu
  src/resnet_block_cuda.cu

  test/test_gpu.cu
  test/autoencoder/test_autoencoder_gpu.cu
  test/cnn/test_cnn_gpu.cu
  test/cnn_batch_norm/test_cnn_batch_norm_gpu.cu
  test/test_classification.cu
  test/test_autoencoder.cu
)

# Output binary folder der for different mode
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

if(CMAKE_CUDA_COMPILER)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  add_library(cutagi_lib STATIC ${GPU_SOURCES})
  set_target_properties(cutagi_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_target_properties(cutagi_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # Set CUDA flags only on target i.e. only for files that are compiled for CUDA
  target_compile_options(cutagi_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

  # Executablecode i.e. application
  add_executable(main main.cu)

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

# Embedding Python into a C++ program
target_link_libraries(main PRIVATE pybind11::embed)
target_link_libraries(cutagi_lib PRIVATE pybind11::module)
target_link_libraries(main PUBLIC cutagi_lib)

target_link_libraries(cutagi PRIVATE cutagi_lib)


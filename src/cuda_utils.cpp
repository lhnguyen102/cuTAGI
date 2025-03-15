#include "../include/cuda_utils.h"

#include "../include/custom_logger.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <iomanip>
#include <sstream>

bool is_cuda_available() {
#ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR,
            "CUDA runtime error: " + std::string(cudaGetErrorString(error)));
        return false;
    }

    return deviceCount > 0;
#else
    return false;
#endif
}

int get_cuda_device_count() {
#ifdef USE_CUDA
    if (!is_cuda_available()) {
        return 0;
    }

    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Failed to get CUDA device count: " +
                                 std::string(cudaGetErrorString(error)));
        return 0;
    }

    return deviceCount;
#else
    return 0;
#endif
}

int get_current_cuda_device() {
#ifdef USE_CUDA
    if (!is_cuda_available()) {
        return -1;
    }

    int device = -1;
    cudaError_t error = cudaGetDevice(&device);

    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Failed to get current CUDA device: " +
                                 std::string(cudaGetErrorString(error)));
        return -1;
    }

    return device;
#else
    return -1;
#endif
}

bool set_cuda_device(int device_index) {
#ifdef USE_CUDA
    if (!is_cuda_available()) {
        LOG(LogLevel::ERROR, "CUDA is not available");
        return false;
    }

    int deviceCount = get_cuda_device_count();
    if (device_index < 0 || device_index >= deviceCount) {
        LOG(LogLevel::ERROR,
            "Invalid CUDA device index: " + std::to_string(device_index));
        return false;
    }

    cudaError_t error = cudaSetDevice(device_index);
    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Failed to set CUDA device: " +
                                 std::string(cudaGetErrorString(error)));
        return false;
    }

    return true;
#else
    LOG(LogLevel::ERROR, "CUDA is not available");
    return false;
#endif
}

bool is_cuda_device_available(int device_index) {
#ifdef USE_CUDA
    if (!is_cuda_available()) {
        return false;
    }

    int deviceCount = get_cuda_device_count();
    if (device_index < 0 || device_index >= deviceCount) {
        return false;
    }

    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, device_index);

    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Failed to get CUDA device properties: " +
                                 std::string(cudaGetErrorString(error)));
        return false;
    }

    // Check if the device is active
    return deviceProp.computeMode != cudaComputeModeProhibited;
#else
    return false;
#endif
}

std::string get_cuda_device_properties(int device_index) {
#ifdef USE_CUDA
    if (!is_cuda_available()) {
        return "CUDA is not available";
    }

    int deviceCount = get_cuda_device_count();
    if (device_index < 0 || device_index >= deviceCount) {
        return "Invalid CUDA device index: " + std::to_string(device_index);
    }

    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, device_index);

    if (error != cudaSuccess) {
        return "Failed to get CUDA device properties: " +
               std::string(cudaGetErrorString(error));
    }

    std::stringstream ss;
    ss << "Device " << device_index << ": " << deviceProp.name << std::endl;
    ss << "  Compute capability: " << deviceProp.major << "."
       << deviceProp.minor << std::endl;
    ss << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024)
       << " MB" << std::endl;
    ss << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    ss << "  Clock rate: " << deviceProp.clockRate / 1000 << " MHz"
       << std::endl;
    ss << "  Memory clock rate: " << deviceProp.memoryClockRate / 1000 << " MHz"
       << std::endl;
    ss << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits"
       << std::endl;
    ss << "  L2 cache size: " << deviceProp.l2CacheSize / 1024 << " KB"
       << std::endl;
    ss << "  Max threads per multiprocessor: "
       << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    ss << "  Max threads per block: " << deviceProp.maxThreadsPerBlock
       << std::endl;
    ss << "  Max block dimensions: [" << deviceProp.maxThreadsDim[0] << ", "
       << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2]
       << "]" << std::endl;
    ss << "  Max grid dimensions: [" << deviceProp.maxGridSize[0] << ", "
       << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]"
       << std::endl;

    return ss.str();
#else
    return "CUDA is not available";
#endif
}

bool get_cuda_device_memory(int device_index, size_t& free_memory,
                            size_t& total_memory) {
#ifdef USE_CUDA
    if (!is_cuda_available()) {
        LOG(LogLevel::ERROR, "CUDA is not available");
        return false;
    }

    int deviceCount = get_cuda_device_count();
    if (device_index < 0 || device_index >= deviceCount) {
        LOG(LogLevel::ERROR,
            "Invalid CUDA device index: " + std::to_string(device_index));
        return false;
    }

    // Save current device
    int currentDevice;
    cudaGetDevice(&currentDevice);

    // Set device to query
    cudaError_t error = cudaSetDevice(device_index);
    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Failed to set CUDA device: " +
                                 std::string(cudaGetErrorString(error)));
        return false;
    }

    // Get memory info
    error = cudaMemGetInfo(&free_memory, &total_memory);

    // Restore previous device
    cudaSetDevice(currentDevice);

    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Failed to get CUDA memory info: " +
                                 std::string(cudaGetErrorString(error)));
        return false;
    }

    return true;
#else
    LOG(LogLevel::ERROR, "CUDA is not available");
    return false;
#endif
}
#pragma once
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

// Macro to check the last CUDA error
#define CHECK_LAST_CUDA_ERROR() check_cuda_last(__FILE__, __LINE__)

// Macro to check a specific CUDA error
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)

// Function to check CUDA error and print detailed information
inline void check_cuda(cudaError_t err, const char* const func,
                       const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Function to check the last CUDA error and print detailed information
inline void check_cuda_last(const char* const file, const int line) {
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

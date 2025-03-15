#pragma once
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#include "custom_logger.h"
#ifdef USE_NCCL
#include <nccl.h>
#endif

// Macro to check the last CUDA error
#define CHECK_LAST_CUDA_ERROR() check_cuda_last()

// Macro to check a specific CUDA error
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val)

// Function to check CUDA error and print detailed information
inline void check_cuda(cudaError_t err, const char* const func) {
    if (err != cudaSuccess) {
        std::string error_message =
            "CUDA Runtime Error: " + std::string(cudaGetErrorString(err)) +
            " " + std::string(func);
        LOG(LogLevel::ERROR, error_message);
        std::exit(EXIT_FAILURE);
    }
}

// Function to check the last CUDA error and print detailed information
inline void check_cuda_last() {
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string error_message =
            "CUDA Runtime Error: " + std::string(cudaGetErrorString(err));
        LOG(LogLevel::ERROR, error_message);
        std::exit(EXIT_FAILURE);
    }
}

#ifdef USE_NCCL
inline void check_cuda_nccl_async(ncclComm_t comm) {
    // 1) Check NCCL async error (only if comm is valid. These asynchronous
    // checks do a quick status query rather than a full-blown synchronization.
    // That helps keep performance high.
    if (comm != nullptr) {
        ncclResult_t async_err;
        ncclCommGetAsyncError(comm, &async_err);
        if (async_err != ncclSuccess) {
            std::string error_message =
                "NCCL async error: " +
                std::string(ncclGetErrorString(async_err));
            LOG(LogLevel::ERROR, error_message);
            std::exit(EXIT_FAILURE);
        }
    }

    // 2) Check CUDA async error (non-blocking) meaning does not force a device
    // sync. It merely reports the most recent error if any is pending.
    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        std::string error_message =
            "CUDA async error: " + std::string(cudaGetErrorString(cuda_err));
        LOG(LogLevel::ERROR, error_message);
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA_NCCL_ASYNC(comm) check_cuda_nccl_async(comm)
#endif
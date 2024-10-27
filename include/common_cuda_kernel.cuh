#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"

// TODO: Update dual sum reduction for batch normalization
template <typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y>
__global__ void dual_sum_reduction_v2(const T *delta_mu_in,
                                      const T *delta_var_in, size_t len_x,
                                      size_t len_y, T *delta_mu_out,
                                      T *delta_var_out)
/*
 */
{
    __shared__ T smem_mu[BLOCK_TILE_Y][BLOCK_TILE_X];
    __shared__ T smem_var[BLOCK_TILE_Y][BLOCK_TILE_X];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t col = blockIdx.x * BLOCK_TILE_X + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < len_x && row < len_y) {
        smem_mu[ty][tx] = delta_mu_in[row * len_x + col];
        smem_var[ty][tx] = delta_var_in[row * len_x + col];
    } else {
        smem_mu[ty][tx] = static_cast<T>(0);
        smem_var[ty][tx] = static_cast<T>(0);
    }

    __syncthreads();

    for (size_t i = BLOCK_TILE_X / 2; i > WARP_SIZE; i >>= 1) {
        if (tx < i) {
            smem_mu[ty][tx] += smem_mu[ty][tx + i];
            smem_var[ty][tx] += smem_var[ty][tx + i];
        }
        __syncthreads();
    }

    if (tx < WARP_SIZE) {
        float mu_x = smem_mu[ty][tx];
        float var_x = smem_var[ty][tx];

        if (blockDim.x >= WARP_SIZE * 2) {
            mu_x += smem_mu[ty][tx + WARP_SIZE];
            var_x += smem_var[ty][tx + WARP_SIZE];
            __syncwarp();
            smem_mu[ty][tx] = mu_x;
            smem_var[ty][tx] = var_x;
            __syncwarp();
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            mu_x += smem_mu[ty][tx + offset];
            var_x += smem_var[ty][tx + offset];
            __syncwarp();
            smem_mu[ty][tx] = mu_x;
            smem_var[ty][tx] = var_x;
            __syncwarp();
        }
    }

    if (tx == 0 && row < len_y) {
        delta_mu_out[row * gridDim.x + blockIdx.x] = smem_mu[ty][tx];
        delta_var_out[row * gridDim.x + blockIdx.x] = smem_var[ty][tx];
    }
}
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"

template <typename T>
__device__ void dual_warp_smem_reduction_v2(volatile T *smem_mu,
                                            volatile T *smem_var, size_t tx,
                                            size_t ty);

template <typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y>
__global__ void dual_sum_reduction_v2(const T *delta_mu_in,
                                      const T *delta_var_in, size_t len_x,
                                      size_t len_y, T *delta_mu_out,
                                      T *delta_var_out);
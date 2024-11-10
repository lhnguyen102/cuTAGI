#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "config.h"

#define BANK_SIZE 32U

////////////////////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////////////////////

__global__ void linear_fwd_mean_var(float const *mu_w, float const *var_w,
                                    float const *mu_b, float const *var_b,
                                    const float *mu_a, const float *var_a,
                                    size_t input_size, size_t output_size,
                                    int batch_size, bool bias, float *mu_z,
                                    float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            float mu_a_tmp = mu_a[input_size * col + i];
            float var_a_tmp = var_a[input_size * col + i];
            float mu_w_tmp = mu_w[row * input_size + i];
            float var_w_tmp = var_w[row * input_size + i];

            sum_mu += mu_w_tmp * mu_a_tmp;
            sum_var += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                       var_w_tmp * mu_a_tmp * mu_a_tmp;
        }

        if (bias) {
            mu_z[col * output_size + row] = sum_mu + mu_b[row];
            var_z[col * output_size + row] = sum_var + var_b[row];
        } else {
            mu_z[col * output_size + row] = sum_mu;
            var_z[col * output_size + row] = sum_var;
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K, size_t THREADS>
__global__ void linear_fwd_mean_var_v1(const T *mu_w, const T *var_w,
                                       const T *mu_b, const T *var_b,
                                       const T *mu_a, const T *var_a,
                                       size_t input_size, size_t output_size,
                                       int batch_size, bool bias, T *mu_z,
                                       T *var_z)
/*
 */
{
    __shared__ T smem_mu_a[BLOCK_TILE][BLOCK_TILE_K];
    __shared__ T smem_var_a[BLOCK_TILE][BLOCK_TILE_K];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE];
    __shared__ T smem_var_w[BLOCK_TILE_K][BLOCK_TILE];

    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (input_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    T sum_mu = static_cast<T>(0);
    T sum_var = static_cast<T>(0);

    for (int phase = 0; phase < num_tiles; phase++) {
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            size_t a_ty = (thread_linear_idx + l_i * THREADS) / BLOCK_TILE_K;
            size_t a_tx = (thread_linear_idx + l_i * THREADS) % BLOCK_TILE_K;

            // input matrix to shared mem (batch_size x input_size)
            size_t a_row_idx = blockIdx.y * BLOCK_TILE + a_ty;
            size_t a_col_idx = phase * BLOCK_TILE_K + a_tx;
            if (a_row_idx < batch_size && a_col_idx < input_size) {
                smem_mu_a[a_ty][a_tx] =
                    mu_a[a_row_idx * input_size + a_col_idx];
                smem_var_a[a_ty][a_tx] =
                    var_a[a_row_idx * input_size + a_col_idx];
            } else {
                smem_mu_a[a_ty][a_tx] = static_cast<T>(0);
                smem_var_a[a_ty][a_tx] = static_cast<T>(0);
            }

            // weight matrix to shared mem (output_size x input_size)
            size_t w_ty = (thread_linear_idx + l_i * THREADS) / BLOCK_TILE;
            size_t w_tx = (thread_linear_idx + l_i * THREADS) % BLOCK_TILE;

            size_t w_col_idx = phase * BLOCK_TILE_K + w_ty;
            size_t w_row_idx = blockIdx.x * BLOCK_TILE + w_tx;

            if (w_row_idx < output_size && w_col_idx < input_size) {
                smem_mu_w[w_ty][w_tx] =
                    mu_w[w_row_idx * input_size + w_col_idx];
                smem_var_w[w_ty][w_tx] =
                    var_w[w_row_idx * input_size + w_col_idx];
            } else {
                smem_mu_w[w_ty][w_tx] = static_cast<T>(0);
                smem_var_w[w_ty][w_tx] = static_cast<T>(0);
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < BLOCK_TILE_K; i++) {
            T mu_w_tmp = smem_mu_w[i][threadIdx.x];
            T var_w_tmp = smem_var_w[i][threadIdx.x];
            T mu_a_tmp = smem_mu_a[threadIdx.y][i];
            T var_a_tmp = smem_var_a[threadIdx.y][i];

            sum_mu = __fmaf_rn(mu_w_tmp, mu_a_tmp, sum_mu);
            sum_var =
                __fmaf_rn(mu_w_tmp * mu_w_tmp + var_w_tmp, var_a_tmp,
                          __fmaf_rn(var_w_tmp, mu_a_tmp * mu_a_tmp, sum_var));
        }
        __syncthreads();
    }
    if (row < batch_size && col < output_size) {
        mu_z[row * output_size + col] = bias ? sum_mu + mu_b[col] : sum_mu;
        var_z[row * output_size + col] = bias ? sum_var + var_b[col] : sum_var;
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t SMEM_PADDING>
__global__ void linear_fwd_mean_var_v2(const T *mu_w, const T *var_w,
                                       const T *mu_b, const T *var_b,
                                       const T *mu_a, const T *var_a,
                                       size_t input_size, size_t output_size,
                                       int batch_size, bool bias, T *mu_z,
                                       T *var_z)
/*
 */
{
    __shared__ T smem_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_var_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_var_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t thread_linear_idx = ty * blockDim.x + tx;

    unsigned int num_tiles = (input_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr int num_loads =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T mu_w_val[THREAD_TILE] = {static_cast<T>(0)};
    T var_w_val[THREAD_TILE] = {static_cast<T>(0)};
    T mu_a_val[THREAD_TILE] = {static_cast<T>(0)};
    T var_a_val[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (int l_i = 0; l_i < num_loads; l_i++) {
            // input matrix to shared mem (batch_size x input_size)
            size_t a_ty = (thread_linear_idx + l_i * THREADS) / BLOCK_TILE_K;
            size_t a_tx = (thread_linear_idx + l_i * THREADS) % BLOCK_TILE_K;

            size_t a_row_idx = blockIdx.y * BLOCK_TILE + a_ty;
            size_t a_col_idx = phase * BLOCK_TILE_K + a_tx;

            if (a_row_idx < batch_size && a_col_idx < input_size) {
                smem_mu_a[a_tx][a_ty] =
                    mu_a[a_row_idx * input_size + a_col_idx];

                smem_var_a[a_tx][a_ty] =
                    var_a[a_row_idx * input_size + a_col_idx];
            } else {
                smem_mu_a[a_tx][a_ty] = static_cast<T>(0);
                smem_var_a[a_tx][a_ty] = static_cast<T>(0);
            }

            // weight matrix to shared mem (output_size x input_size)
            size_t w_ty = (thread_linear_idx + l_i * THREADS) / BLOCK_TILE;
            size_t w_tx = (thread_linear_idx + l_i * THREADS) % BLOCK_TILE;

            size_t w_col_idx = phase * BLOCK_TILE_K + w_ty;
            size_t w_row_idx = blockIdx.x * BLOCK_TILE + w_tx;

            if (w_row_idx < output_size && w_col_idx < input_size) {
                smem_mu_w[w_ty][w_tx] =
                    mu_w[w_row_idx * input_size + w_col_idx];
                smem_var_w[w_ty][w_tx] =
                    var_w[w_row_idx * input_size + w_col_idx];
            } else {
                smem_mu_w[w_ty][w_tx] = static_cast<T>(0);
                smem_var_w[w_ty][w_tx] = static_cast<T>(0);
            }
        }
        __syncthreads();
#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                mu_a_val[j] = smem_mu_a[i][ty * THREAD_TILE + j];
                var_a_val[j] = smem_var_a[i][ty * THREAD_TILE + j];
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                mu_w_val[j] = smem_mu_w[i][tx * THREAD_TILE + j];
                var_w_val[j] = smem_var_w[i][tx * THREAD_TILE + j];
            }
#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] += mu_w_val[j] * mu_a_val[t];
                    tmp_var[t][j] +=
                        (mu_w_val[j] * mu_w_val[j] + var_w_val[j]) *
                            var_a_val[t] +
                        var_w_val[j] * mu_a_val[t] * mu_a_val[t];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (size_t t = 0; t < THREAD_TILE; t++) {
        size_t row = blockIdx.y * BLOCK_TILE + ty * THREAD_TILE + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            size_t col = blockIdx.x * BLOCK_TILE + tx * THREAD_TILE + j;
            if (row < batch_size && col < output_size) {
                if (bias) {
                    mu_z[row * output_size + col] = tmp_mu[t][j] + mu_b[col];
                    var_z[row * output_size + col] = tmp_var[t][j] + var_b[col];
                } else {
                    mu_z[row * output_size + col] = tmp_mu[t][j];
                    var_z[row * output_size + col] = tmp_var[t][j];
                }
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING>
__global__ void linear_fwd_mean_var_v3(const T *mu_w, const T *var_w,
                                       const T *mu_b, const T *var_b,
                                       const T *mu_a, const T *var_a,
                                       size_t input_size, size_t output_size,
                                       int batch_size, bool bias, T *mu_z,
                                       T *var_z)
/*
 */
{
    __shared__ T smem_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_var_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_var_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    // Thread block
    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (input_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    // Warp
    constexpr unsigned int WARPS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WARPS_X;
    const size_t warp_col = warp_id % WARPS_X;
    const size_t thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_in_warp = lane_id % THREADS_PER_WARP_X;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T mu_w_val[THREAD_TILE] = {static_cast<T>(0)};
    T var_w_val[THREAD_TILE] = {static_cast<T>(0)};
    T mu_a_val[THREAD_TILE] = {static_cast<T>(0)};
    T var_a_val[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t phase = 0; phase < num_tiles; phase++) {
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            // input matrix to shared mem (batch_size x input_size)
            size_t thread_load_idx = thread_linear_idx + l_i * THREADS;
            size_t a_ty = thread_load_idx / BLOCK_TILE_K;
            size_t a_tx = thread_load_idx % BLOCK_TILE_K;

            size_t a_row = blockIdx.y * BLOCK_TILE + a_ty;
            size_t a_col = phase * BLOCK_TILE_K + a_tx;

            if (a_row < batch_size && a_col < input_size) {
                smem_mu_a[a_tx][a_ty] = mu_a[a_row * input_size + a_col];
                smem_var_a[a_tx][a_ty] = var_a[a_row * input_size + a_col];
            } else {
                smem_mu_a[a_tx][a_ty] = static_cast<T>(0);
                smem_var_a[a_tx][a_ty] = static_cast<T>(0);
            }

            // weight matrix to shared mem (output_size x input_size)
            size_t w_ty = thread_load_idx / BLOCK_TILE;
            size_t w_tx = thread_load_idx % BLOCK_TILE;

            size_t w_row = blockIdx.x * BLOCK_TILE + w_tx;
            size_t w_col = phase * BLOCK_TILE_K + w_ty;

            if (w_row < output_size && w_col < input_size) {
                smem_mu_w[w_ty][w_tx] = mu_w[w_row * input_size + w_col];
                smem_var_w[w_ty][w_tx] = var_w[w_row * input_size + w_col];
            } else {
                smem_mu_w[w_ty][w_tx] = static_cast<T>(0);
                smem_var_w[w_ty][w_tx] = static_cast<T>(0);
            }
        }
        __syncthreads();
#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_in_warp * THREAD_TILE + j;
                mu_a_val[j] = smem_mu_a[i][idx];
                var_a_val[j] = smem_var_a[i][idx];
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_in_warp * THREAD_TILE + j;
                mu_w_val[j] = smem_mu_w[i][idx];
                var_w_val[j] = smem_var_w[i][idx];
            }
#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] =
                        __fmaf_rn(mu_w_val[j], mu_a_val[t], tmp_mu[t][j]);
                    tmp_var[t][j] = __fmaf_rn(
                        mu_w_val[j] * mu_w_val[j] + var_w_val[j], var_a_val[t],
                        __fmaf_rn(var_w_val[j], mu_a_val[t] * mu_a_val[t],
                                  tmp_var[t][j]));
                }
            }
        }
        // __syncwarp();
        __syncthreads();
    }
    // __syncthreads();

    const size_t base_row = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                            thread_row_in_warp * THREAD_TILE;
    const size_t base_col = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                            thread_col_in_warp * THREAD_TILE;
#pragma unroll
    for (size_t t = 0; t < THREAD_TILE; t++) {
        size_t row = base_row + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            size_t col = base_col + j;
            if (row < batch_size && col < output_size) {
                mu_z[row * output_size + col] =
                    bias ? tmp_mu[t][j] + mu_b[col] : tmp_mu[t][j];
                var_z[row * output_size + col] =
                    bias ? tmp_var[t][j] + var_b[col] : tmp_var[t][j];
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t PACK_SIZE, size_t SMEM_PADDING>
__global__ void linear_fwd_mean_var_v4(
    const T *__restrict__ mu_w, const T *__restrict__ var_w,
    const T *__restrict__ mu_b, const T *__restrict__ var_b,
    const T *__restrict__ mu_a, const T *__restrict__ var_a, size_t input_size,
    size_t output_size, int batch_size, bool bias, T *__restrict__ mu_z,
    T *__restrict__ var_z)
/*
 */
{
    __shared__ T smem_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_var_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_var_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    // Thread block
    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (input_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int K_PACKS = BLOCK_TILE_K / PACK_SIZE;
    constexpr unsigned int THREAD_PACKS = THREAD_TILE / PACK_SIZE;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * K_PACKS + THREADS - 1) / THREADS;

    // Warp
    constexpr unsigned int WARPS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WARPS_X;
    const size_t warp_col = warp_id % WARPS_X;
    const size_t thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_in_warp = lane_id % THREADS_PER_WARP_X;

    float4 tmp_mu[THREAD_TILE][THREAD_PACKS] = {{0.0f}};
    float4 tmp_var[THREAD_TILE][THREAD_PACKS] = {{0.0f}};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            // input matrix to shared mem (batch_size x input_size)
            size_t thread_load_idx = thread_linear_idx + l_i * THREADS;
            size_t a_ty = thread_load_idx / K_PACKS;
            size_t a_tx = thread_load_idx % K_PACKS * PACK_SIZE;

            size_t a_row = blockIdx.y * BLOCK_TILE + a_ty;
            size_t a_col = phase * BLOCK_TILE_K + a_tx;

            // Hardcoded 4 values
            float4 mu_a_row_val = {0, 0, 0, 0};
            float4 var_a_row_val = {0, 0, 0, 0};
            if (a_row < batch_size && a_col < input_size) {
                mu_a_row_val = *reinterpret_cast<const float4 *>(
                    &mu_a[a_row * input_size + a_col]);
                var_a_row_val = *reinterpret_cast<const float4 *>(
                    &var_a[a_row * input_size + a_col]);
            }
#pragma unroll
            for (size_t p = 0; p < PACK_SIZE; p++) {
                smem_mu_a[a_tx + p][a_ty] =
                    reinterpret_cast<const T *>(&mu_a_row_val)[p];
                smem_var_a[a_tx + p][a_ty] =
                    reinterpret_cast<const T *>(&var_a_row_val)[p];
            }
        }
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            // weight matrix to shared mem(output_size x input_size)
            size_t thread_load_idx = thread_linear_idx + l_i * THREADS;
            size_t w_ty = (thread_load_idx / BLOCK_TILE) * PACK_SIZE;
            size_t w_tx = thread_load_idx % BLOCK_TILE;

            size_t w_row = blockIdx.x * BLOCK_TILE + w_tx;
            size_t w_col = phase * BLOCK_TILE_K + w_ty;

            float4 mu_w_row_val = {0, 0, 0, 0};
            float4 var_w_row_val = {0, 0, 0, 0};
            if (w_row < output_size && w_col < input_size) {
                mu_w_row_val = *reinterpret_cast<const float4 *>(
                    &mu_w[w_row * input_size + w_col]);
                var_w_row_val = *reinterpret_cast<const float4 *>(
                    &var_w[w_row * input_size + w_col]);
            }

#pragma unroll
            for (size_t p = 0; p < PACK_SIZE; p++) {
                smem_mu_w[w_ty + p][w_tx] =
                    reinterpret_cast<const T *>(&mu_w_row_val)[p];
                smem_var_w[w_ty + p][w_tx] =
                    reinterpret_cast<const T *>(&var_w_row_val)[p];
            }
        }
        __syncthreads();
#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            float4 mu_w_val[THREAD_PACKS] = {0.0f};
            float4 var_w_val[THREAD_PACKS] = {0.0f};
            float4 mu_a_val[THREAD_PACKS] = {0.0f};
            float4 var_a_val[THREAD_PACKS] = {0.0f};
#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_in_warp * THREAD_TILE + j * PACK_SIZE;
                mu_a_val[j] =
                    *reinterpret_cast<const float4 *>(&smem_mu_a[i][idx]);
                var_a_val[j] =
                    *reinterpret_cast<const float4 *>(&smem_var_a[i][idx]);
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_in_warp * THREAD_TILE + j * PACK_SIZE;
                mu_w_val[j] =
                    *reinterpret_cast<const float4 *>(&smem_mu_w[i][idx]);
                var_w_val[j] =
                    *reinterpret_cast<const float4 *>(&smem_var_w[i][idx]);
            }

#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                float mu_a_val_t = reinterpret_cast<float *>(
                    &mu_a_val[t / PACK_SIZE])[t & (PACK_SIZE - 1)];
                float var_a_val_t = reinterpret_cast<float *>(
                    &var_a_val[t / PACK_SIZE])[t & (PACK_SIZE - 1)];
#pragma unroll
                for (size_t j = 0; j < THREAD_PACKS; j++) {
                    float4 mu_w_val_j = mu_w_val[j];
                    float4 var_w_val_j = var_w_val[j];

                    tmp_mu[t][j].x =
                        __fmaf_rn(mu_w_val_j.x, mu_a_val_t, tmp_mu[t][j].x);
                    tmp_mu[t][j].y =
                        __fmaf_rn(mu_w_val_j.y, mu_a_val_t, tmp_mu[t][j].y);
                    tmp_mu[t][j].z =
                        __fmaf_rn(mu_w_val_j.z, mu_a_val_t, tmp_mu[t][j].z);
                    tmp_mu[t][j].w =
                        __fmaf_rn(mu_w_val_j.w, mu_a_val_t, tmp_mu[t][j].w);

                    float4 var_term;
                    var_term.x = __fmaf_rn(
                        mu_w_val_j.x * mu_w_val_j.x + var_w_val_j.x,
                        var_a_val_t, mu_a_val_t * mu_a_val_t * var_w_val_j.x);
                    var_term.y = __fmaf_rn(
                        mu_w_val_j.y * mu_w_val_j.y + var_w_val_j.y,
                        var_a_val_t, mu_a_val_t * mu_a_val_t * var_w_val_j.y);
                    var_term.z = __fmaf_rn(
                        mu_w_val_j.z * mu_w_val_j.z + var_w_val_j.z,
                        var_a_val_t, mu_a_val_t * mu_a_val_t * var_w_val_j.z);
                    var_term.w = __fmaf_rn(
                        mu_w_val_j.w * mu_w_val_j.w + var_w_val_j.w,
                        var_a_val_t, mu_a_val_t * mu_a_val_t * var_w_val_j.w);

                    tmp_var[t][j].x += var_term.x;
                    tmp_var[t][j].y += var_term.y;
                    tmp_var[t][j].z += var_term.z;
                    tmp_var[t][j].w += var_term.w;
                }
            }
        }
        __syncthreads();
    }

    const size_t row_base = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                            thread_row_in_warp * THREAD_TILE;
    const size_t col_base = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                            thread_col_in_warp * THREAD_TILE;

#pragma unroll
    for (size_t t = 0; t < THREAD_TILE; t++) {
        size_t row = row_base + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_PACKS; j++) {
            const size_t col = col_base + j * PACK_SIZE;
            if (row < batch_size && col < output_size) {
                if (bias) {
                    const float4 mu_b_val =
                        *reinterpret_cast<const float4 *>(&mu_b[col]);
                    const float4 var_b_val =
                        *reinterpret_cast<const float4 *>(&var_b[col]);

                    tmp_mu[t][j].x = __fadd_rn(tmp_mu[t][j].x, mu_b_val.x);
                    tmp_mu[t][j].y = __fadd_rn(tmp_mu[t][j].y, mu_b_val.y);
                    tmp_mu[t][j].z = __fadd_rn(tmp_mu[t][j].z, mu_b_val.z);
                    tmp_mu[t][j].w = __fadd_rn(tmp_mu[t][j].w, mu_b_val.w);

                    tmp_var[t][j].x = __fadd_rn(tmp_var[t][j].x, var_b_val.x);
                    tmp_var[t][j].y = __fadd_rn(tmp_var[t][j].y, var_b_val.y);
                    tmp_var[t][j].z = __fadd_rn(tmp_var[t][j].z, var_b_val.z);
                    tmp_var[t][j].w = __fadd_rn(tmp_var[t][j].w, var_b_val.w);
                }
                __stcs(
                    reinterpret_cast<float4 *>(&mu_z[row * output_size + col]),
                    tmp_mu[t][j]);
                __stcs(
                    reinterpret_cast<float4 *>(&var_z[row * output_size + col]),
                    tmp_var[t][j]);
            }
        }
    }
}

__global__ void linear_fwd_full_cov(float const *mu_w, float const *var_a_f,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *var_z_fp)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tu = 0, k = 0;
    float sum = 0.0f;
    float var_a_in = 0.0f;

    if (col <= (row % output_size) && row < output_size * batch_size) {
        for (int i = 0; i < input_size * input_size; i++) {
            int row_in = i / input_size;
            int col_in = i % input_size;
            if (row_in > col_in)  // lower triangle
            {
                tu = (input_size * col_in - ((col_in * (col_in + 1)) / 2) +
                      row_in);
            } else {
                tu = (input_size * row_in - ((row_in * (row_in + 1)) / 2) +
                      col_in);
            }
            var_a_in = var_a_f[tu + (row / output_size) *
                                        (input_size * (input_size + 1)) / 2];

            sum += mu_w[i % input_size + (row % output_size) * input_size] *
                   var_a_in *
                   mu_w[i / input_size + (col % output_size) * input_size];
        }
        k = output_size * col - ((col * (col + 1)) / 2) + row % output_size +
            (row / output_size) * (((output_size + 1) * output_size) / 2);
        var_z_fp[k] = sum;
    }
}

__global__ void linear_fwd_full_var(float const *mu_w, float const *var_w,
                                    float const *var_b, float const *mu_a,
                                    float const *var_a, float const *var_z_fp,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *var_z,
                                    float *var_z_f)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    float final_sum = 0;
    int k;

    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            sum += var_w[row * input_size + i] * var_a[input_size * col + i] +
                   var_w[row * input_size + i] * mu_a[input_size * col + i] *
                       mu_a[input_size * col + i];
        }
        k = output_size * row - (row * (row - 1)) / 2 +
            col * (output_size * (output_size + 1)) / 2;

        final_sum = sum + var_b[row] + var_z_fp[k];

        var_z[col * output_size + row] = final_sum;
    }
}

////////////////////////////////////////////////////////////////////////////////
// State backward
////////////////////////////////////////////////////////////////////////////////
__global__ void linear_bwd_delta_z(float const *mu_w, float const *jcb,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_in,
                                   float *delta_var_in)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    if (col < batch_size && row < input_size) {
        for (int i = 0; i < output_size; i++) {
            sum_mu += mu_w[input_size * i + row] *
                      delta_mu_out[col * output_size + i];

            sum_var += mu_w[input_size * i + row] *
                       delta_var_out[col * output_size + i] *
                       mu_w[input_size * i + row];
        }
        delta_mu_in[col * input_size + row] =
            sum_mu * jcb[col * input_size + row];

        delta_var_in[col * input_size + row] =
            sum_var * jcb[col * input_size + row] * jcb[col * input_size + row];
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t SMEM_PADDING>
__global__ void linear_bwd_delta_z_v2(const T *mu_w, const T *jcb,
                                      const T *delta_mu_out,
                                      const T *delta_var_out, size_t input_size,
                                      size_t output_size, int batch_size,
                                      T *delta_mu_in, T *delta_var_in)
/*
 */
{
    __shared__ T smem_delta_mu[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_delta_var[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t thread_linear_idx = ty * blockDim.x + tx;

    unsigned int num_tiles = (output_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    constexpr int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T mu_w_val[THREAD_TILE] = {static_cast<T>(0)};
    T delta_mu_val[THREAD_TILE] = {static_cast<T>(0)};
    T delta_var_val[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            const int thread_load_idx = thread_linear_idx + l_i * THREADS;

            // delta matrix
            size_t delta_ty = thread_load_idx / BLOCK_TILE_K;
            size_t delta_tx = thread_load_idx % BLOCK_TILE_K;

            size_t delta_row = blockIdx.y * BLOCK_TILE + delta_ty;
            size_t delta_col = phase * BLOCK_TILE_K + delta_tx;

            if (delta_row < batch_size && delta_col < output_size) {
                smem_delta_mu[delta_tx][delta_ty] =
                    delta_mu_out[delta_row * output_size + delta_col];
                smem_delta_var[delta_tx][delta_ty] =
                    delta_var_out[delta_row * output_size + delta_col];
            } else {
                smem_delta_mu[delta_tx][delta_ty] = static_cast<T>(0);
                smem_delta_var[delta_tx][delta_ty] = static_cast<T>(0);
            }

            // weight matrix
            size_t w_ty = thread_load_idx / BLOCK_TILE;
            size_t w_tx = thread_load_idx % BLOCK_TILE;
            size_t w_row = phase * BLOCK_TILE_K + w_ty;
            size_t w_col = blockIdx.x * BLOCK_TILE + w_tx;

            if (w_row < output_size && w_col < input_size) {
                smem_mu_w[w_ty][w_tx] = mu_w[w_row * input_size + w_col];
            } else {
                smem_mu_w[w_ty][w_tx] = static_cast<T>(0);
            }
        }
        __syncthreads();
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                delta_mu_val[j] = smem_delta_mu[i][ty * THREAD_TILE + j];
                delta_var_val[j] = smem_delta_var[i][ty * THREAD_TILE + j];
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                mu_w_val[j] = smem_mu_w[i][tx * THREAD_TILE + j];
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] += mu_w_val[j] * delta_mu_val[t];
                    tmp_var[t][j] +=
                        mu_w_val[j] * delta_var_val[t] * mu_w_val[j];
                }
            }
        }
        __syncthreads();
    }

    for (size_t t = 0; t < THREAD_TILE; t++) {
        int row = blockIdx.y * BLOCK_TILE + ty * THREAD_TILE + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            int col = blockIdx.x * BLOCK_TILE + tx * THREAD_TILE + j;
            if (row < batch_size && col < input_size) {
                T jcb_val = jcb[row * input_size + col];

                delta_mu_in[row * input_size + col] = tmp_mu[t][j] * jcb_val;
                delta_var_in[row * input_size + col] =
                    tmp_var[t][j] * jcb_val * jcb_val;
            }
        }
    }
}
template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING>
__global__ void linear_bwd_delta_z_v3(const T *mu_w, const T *jcb,
                                      const T *delta_mu_out,
                                      const T *delta_var_out, size_t input_size,
                                      size_t output_size, int batch_size,
                                      T *delta_mu_in, T *delta_var_in)
/*
 */
{
    __shared__ T smem_delta_mu[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_delta_var[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (output_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int WARPS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    constexpr int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WARPS_X;
    const size_t warp_col = warp_id % WARPS_X;
    const size_t thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_in_warp = lane_id % THREADS_PER_WARP_X;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_delta_mu[THREAD_TILE] = {static_cast<T>(0)};
    T tmp_delta_var[THREAD_TILE] = {static_cast<T>(0)};
    T tmp_mu_w[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            const int thread_load_idx = thread_linear_idx + l_i * THREADS;

            // delta matrix
            size_t delta_ty = thread_load_idx / BLOCK_TILE_K;
            size_t delta_tx = thread_load_idx % BLOCK_TILE_K;

            size_t delta_row = blockIdx.y * BLOCK_TILE + delta_ty;
            size_t delta_col = phase * BLOCK_TILE_K + delta_tx;

            if (delta_row < batch_size && delta_col < output_size) {
                smem_delta_mu[delta_tx][delta_ty] =
                    delta_mu_out[delta_row * output_size + delta_col];
                smem_delta_var[delta_tx][delta_ty] =
                    delta_var_out[delta_row * output_size + delta_col];
            } else {
                smem_delta_mu[delta_tx][delta_ty] = static_cast<T>(0);
                smem_delta_var[delta_tx][delta_ty] = static_cast<T>(0);
            }

            // weight matrix
            size_t w_ty = thread_load_idx / BLOCK_TILE;
            size_t w_tx = thread_load_idx % BLOCK_TILE;
            size_t w_row = phase * BLOCK_TILE_K + w_ty;
            size_t w_col = blockIdx.x * BLOCK_TILE + w_tx;

            if (w_row < output_size && w_col < input_size) {
                smem_mu_w[w_ty][w_tx] = mu_w[w_row * input_size + w_col];
            } else {
                smem_mu_w[w_ty][w_tx] = static_cast<T>(0);
            }
        }
        __syncthreads();

        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_in_warp * THREAD_TILE + j;
                tmp_delta_mu[j] = smem_delta_mu[i][idx];
                tmp_delta_var[j] = smem_delta_var[i][idx];
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_in_warp * THREAD_TILE + j;
                tmp_mu_w[j] = smem_mu_w[i][idx];
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] =
                        __fmaf_rn(tmp_mu_w[j], tmp_delta_mu[t], tmp_mu[t][j]);
                    tmp_var[t][j] = __fmaf_rn(tmp_mu_w[j] * tmp_mu_w[j],
                                              tmp_delta_var[t], tmp_var[t][j]);
                }
            }
        }
        // __syncwarp();
        __syncthreads();
    }
    // __syncthreads();

    for (size_t t = 0; t < THREAD_TILE; t++) {
        int row = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                  thread_row_in_warp * THREAD_TILE + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            int col = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                      thread_col_in_warp * THREAD_TILE + j;
            if (row < batch_size && col < input_size) {
                delta_mu_in[row * input_size + col] =
                    __fmul_rn(tmp_mu[t][j], jcb[row * input_size + col]);
                delta_var_in[row * input_size + col] = __fmul_rn(
                    jcb[row * input_size + col] * jcb[row * input_size + col],
                    tmp_var[t][j]);
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t PACK_SIZE, size_t SMEM_PADDING>
__global__ void linear_bwd_delta_z_v4(const T *mu_w, const T *jcb,
                                      const T *delta_mu_out,
                                      const T *delta_var_out, size_t input_size,
                                      size_t output_size, int batch_size,
                                      T *delta_mu_in, T *delta_var_in)
/*
 */
{
    __shared__ T smem_delta_mu[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_delta_var[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (output_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int WARPS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    constexpr unsigned int XY_PACKS = BLOCK_TILE / PACK_SIZE;
    constexpr unsigned int K_PACKS = BLOCK_TILE_K / PACK_SIZE;
    constexpr unsigned int THREAD_PACKS = THREAD_TILE / PACK_SIZE;
    constexpr int NUM_LOADS = (K_PACKS * BLOCK_TILE + THREADS - 1) / THREADS;

    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WARPS_X;
    const size_t warp_col = warp_id % WARPS_X;
    const size_t thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_in_warp = lane_id % THREADS_PER_WARP_X;

    float4 tmp_mu[THREAD_TILE][THREAD_PACKS] = {{0.0f}};
    float4 tmp_var[THREAD_TILE][THREAD_PACKS] = {{0.0f}};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            const int thread_load_idx = thread_linear_idx + l_i * THREADS;

            // delta matrix
            size_t delta_ty = thread_load_idx / K_PACKS;
            size_t delta_tx = thread_load_idx % K_PACKS * PACK_SIZE;

            size_t delta_row = blockIdx.y * BLOCK_TILE + delta_ty;
            size_t delta_col = phase * BLOCK_TILE_K + delta_tx;

            // Hardcoded value
            float4 delta_mu_row_val = {0, 0, 0, 0};
            float4 delta_var_row_val = {0, 0, 0, 0};
            if (delta_row < batch_size && delta_col < output_size) {
                delta_mu_row_val = *reinterpret_cast<const float4 *>(
                    &delta_mu_out[delta_row * output_size + delta_col]);
                delta_var_row_val = *reinterpret_cast<const float4 *>(
                    &delta_var_out[delta_row * output_size + delta_col]);
            }

#pragma unroll
            for (size_t p = 0; p < PACK_SIZE; p++) {
                smem_delta_mu[delta_tx + p][delta_ty] =
                    reinterpret_cast<const T *>(&delta_mu_row_val)[p];
                smem_delta_var[delta_tx + p][delta_ty] =
                    reinterpret_cast<const T *>(&delta_var_row_val)[p];
            }

            // weight matrix
            size_t w_ty = thread_load_idx / XY_PACKS;
            size_t w_tx = thread_load_idx % XY_PACKS * PACK_SIZE;
            size_t w_row = phase * BLOCK_TILE_K + w_ty;
            size_t w_col = blockIdx.x * BLOCK_TILE + w_tx;

            // Hardcoded value
            float4 mu_w_row_val = {0, 0, 0, 0};
            if (w_row < output_size && w_col < input_size) {
                mu_w_row_val = *reinterpret_cast<const float4 *>(
                    &mu_w[w_row * input_size + w_col]);
            }

            *reinterpret_cast<float4 *>(&smem_mu_w[w_ty][w_tx]) = mu_w_row_val;
        }
        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            float4 tmp_delta_mu[THREAD_TILE] = {0.0f};
            float4 tmp_delta_var[THREAD_TILE] = {0.0f};
            float4 tmp_mu_w[THREAD_TILE] = {0.0f};

#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_in_warp * THREAD_TILE + j * PACK_SIZE;
                tmp_delta_mu[j] =
                    *reinterpret_cast<float4 *>(&smem_delta_mu[i][idx]);
                tmp_delta_var[j] =
                    *reinterpret_cast<float4 *>(&smem_delta_var[i][idx]);
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_in_warp * THREAD_TILE + j * PACK_SIZE;
                tmp_mu_w[j] = *reinterpret_cast<float4 *>(&smem_mu_w[i][idx]);
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
                float delta_mu_t = reinterpret_cast<T *>(
                    &tmp_delta_mu[t / PACK_SIZE])[t & (PACK_SIZE - 1)];
                float delta_var_t = reinterpret_cast<T *>(
                    &tmp_delta_var[t / PACK_SIZE])[t & (PACK_SIZE - 1)];
#pragma unroll
                for (size_t j = 0; j < THREAD_PACKS; j++) {
                    tmp_mu[t][j].x =
                        __fmaf_rn(tmp_mu_w[j].x, delta_mu_t, tmp_mu[t][j].x);
                    tmp_mu[t][j].y =
                        __fmaf_rn(tmp_mu_w[j].y, delta_mu_t, tmp_mu[t][j].y);
                    tmp_mu[t][j].z =
                        __fmaf_rn(tmp_mu_w[j].z, delta_mu_t, tmp_mu[t][j].z);
                    tmp_mu[t][j].w =
                        __fmaf_rn(tmp_mu_w[j].w, delta_mu_t, tmp_mu[t][j].w);

                    tmp_var[t][j].x = __fmaf_rn(tmp_mu_w[j].x * tmp_mu_w[j].x,
                                                delta_var_t, tmp_var[t][j].x);
                    tmp_var[t][j].y = __fmaf_rn(tmp_mu_w[j].y * tmp_mu_w[j].y,
                                                delta_var_t, tmp_var[t][j].y);
                    tmp_var[t][j].z = __fmaf_rn(tmp_mu_w[j].z * tmp_mu_w[j].z,
                                                delta_var_t, tmp_var[t][j].z);
                    tmp_var[t][j].w = __fmaf_rn(tmp_mu_w[j].w * tmp_mu_w[j].w,
                                                delta_var_t, tmp_var[t][j].w);
                }
            }
        }
        __syncthreads();
    }

    const int row_base = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                         thread_row_in_warp * THREAD_TILE;
    const int col_base = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                         thread_col_in_warp * THREAD_TILE;

#pragma unroll
    for (size_t t = 0; t < THREAD_TILE; t++) {
        const int row = row_base + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_PACKS; j++) {
            const int col = col_base + j * PACK_SIZE;

            if (row < batch_size && col < input_size) {
                float4 jcb_row_val = *reinterpret_cast<const float4 *>(
                    &jcb[row * input_size + col]);

                tmp_mu[t][j].x = __fmul_rn(tmp_mu[t][j].x, jcb_row_val.x);
                tmp_mu[t][j].y = __fmul_rn(tmp_mu[t][j].y, jcb_row_val.y);
                tmp_mu[t][j].z = __fmul_rn(tmp_mu[t][j].z, jcb_row_val.z);
                tmp_mu[t][j].w = __fmul_rn(tmp_mu[t][j].w, jcb_row_val.w);

                tmp_var[t][j].x =
                    __fmul_rn(tmp_var[t][j].x, jcb_row_val.x * jcb_row_val.x);
                tmp_var[t][j].y =
                    __fmul_rn(tmp_var[t][j].y, jcb_row_val.y * jcb_row_val.y);
                tmp_var[t][j].z =
                    __fmul_rn(tmp_var[t][j].z, jcb_row_val.z * jcb_row_val.z);
                tmp_var[t][j].w =
                    __fmul_rn(tmp_var[t][j].w, jcb_row_val.w * jcb_row_val.w);

                __stcs(reinterpret_cast<float4 *>(
                           &delta_mu_in[row * input_size + col]),
                       tmp_mu[t][j]);
                __stcs(reinterpret_cast<float4 *>(
                           &delta_var_in[row * input_size + col]),
                       tmp_var[t][j]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Param Backward
////////////////////////////////////////////////////////////////////////////////
__global__ void linear_bwd_delta_w(float const *var_w, float const *mu_a,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_w,
                                   float *delta_var_w)
/**/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < output_size && row < input_size) {
        for (int i = 0; i < batch_size; i++) {
            sum_mu += mu_a[input_size * i + row] *
                      delta_mu_out[output_size * i + col];

            sum_var += mu_a[input_size * i + row] * mu_a[input_size * i + row] *
                       delta_var_out[output_size * i + col];
        }

        delta_mu_w[col * input_size + row] =
            sum_mu * var_w[col * input_size + row];

        delta_var_w[col * input_size + row] = sum_var *
                                              var_w[col * input_size + row] *
                                              var_w[col * input_size + row];
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t SMEM_PADDING>
__global__ void linear_bwd_delta_w_v2(const T *var_w, const T *mu_a,
                                      const T *delta_mu_out,
                                      const T *delta_var_out, size_t input_size,
                                      size_t output_size, int batch_size,
                                      T *delta_mu_w, T *delta_var_w)
/**/
{
    __shared__ T smem_delta_mu[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_delta_var[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    const size_t ty = threadIdx.y;
    const size_t tx = threadIdx.x;
    const size_t thread_linear_idx = ty * blockDim.x + tx;

    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;
    unsigned int num_tiles = (batch_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_delta_mu[THREAD_TILE] = {static_cast<T>(0)};
    T tmp_delta_var[THREAD_TILE] = {static_cast<T>(0)};
    T tmp_mu_a[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (size_t l_i = 0; l_i < NUM_LOADS; l_i++) {
            const size_t thread_load_idx = thread_linear_idx + l_i * THREADS;

            // Update quantity matrix
            size_t delta_ty = thread_load_idx / BLOCK_TILE_K;
            size_t delta_tx = thread_load_idx % BLOCK_TILE_K;
            size_t delta_col = blockIdx.y * BLOCK_TILE + delta_ty;
            size_t delta_row = phase * BLOCK_TILE_K + delta_tx;

            smem_delta_mu[delta_tx][delta_ty] =
                (delta_row < batch_size && delta_col < output_size)
                    ? delta_mu_out[delta_row * output_size + delta_col]
                    : static_cast<T>(0);

            smem_delta_var[delta_tx][delta_ty] =
                (delta_row < batch_size && delta_col < output_size)
                    ? delta_var_out[delta_row * output_size + delta_col]
                    : static_cast<T>(0);

            // activation matrix
            size_t a_ty = thread_load_idx / BLOCK_TILE;
            size_t a_tx = thread_load_idx & BLOCK_TILE;
            size_t a_col = blockIdx.x * BLOCK_TILE + a_tx;
            size_t a_row = phase * BLOCK_TILE_K + a_ty;

            smem_mu_a[a_ty][a_tx] = (a_row < batch_size && a_col < input_size)
                                        ? smem_mu_a[a_ty][a_tx] =
                                              mu_a[a_row * input_size + a_col]
                                        : static_cast<T>(0);
        }
        __syncthreads();

        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                tmp_delta_mu[j] = smem_delta_mu[i][ty * THREAD_TILE + j];
                tmp_delta_var[j] = smem_delta_var[i][ty * THREAD_TILE + j];
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                tmp_mu_a[j] = smem_mu_a[i][tx * THREAD_TILE + j];
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] += tmp_mu_a[j] * tmp_delta_mu[t];
                    tmp_var[t][j] +=
                        tmp_mu_a[j] * tmp_mu_a[j] * tmp_delta_var[t];
                }
            }
        }
        __syncthreads();
    }

    for (size_t t = 0; t < THREAD_TILE; t++) {
        size_t row = blockIdx.y * BLOCK_TILE + ty * THREAD_TILE + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            size_t col = blockIdx.x * BLOCK_TILE + tx * THREAD_TILE + j;
            if (row < output_size && col < input_size) {
                T tmp_var_w = var_w[row * input_size + col];

                delta_mu_w[row * input_size + col] = tmp_mu[t][j] * tmp_var_w;
                delta_var_w[row * input_size + col] =
                    tmp_var[t][j] * tmp_var_w * tmp_var_w;
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING>
__global__ void linear_bwd_delta_w_v3(const T *var_w, const T *mu_a,
                                      const T *delta_mu_out,
                                      const T *delta_var_out, size_t input_size,
                                      size_t output_size, int batch_size,
                                      T *delta_mu_w, T *delta_var_w)
/*
 */
{
    __shared__ T smem_delta_mu[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_delta_var[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T smem_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    constexpr unsigned int WAPRS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WAPRS_X;
    const size_t warp_col = warp_id % WAPRS_X;
    const size_t thread_row_per_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_per_warp = lane_id % THREADS_PER_WARP_X;

    const unsigned int num_tiles =
        (batch_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_delta_mu[THREAD_TILE] = {static_cast<T>(0)};
    T tmp_delta_var[THREAD_TILE] = {static_cast<T>(0)};
    T tmp_mu_a[THREAD_TILE] = {static_cast<T>(0)};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (size_t l_i = 0; l_i < NUM_LOADS; l_i++) {
            const size_t thread_load_idx = thread_linear_idx + l_i * THREADS;

            // Update quantity matrix
            size_t delta_ty = thread_load_idx / BLOCK_TILE_K;
            size_t delta_tx = thread_load_idx % BLOCK_TILE_K;
            size_t delta_col = blockIdx.y * BLOCK_TILE + delta_ty;
            size_t delta_row = phase * BLOCK_TILE_K + delta_tx;

            smem_delta_mu[delta_tx][delta_ty] =
                (delta_row < batch_size && delta_col < output_size)
                    ? delta_mu_out[delta_row * output_size + delta_col]
                    : static_cast<T>(0);

            smem_delta_var[delta_tx][delta_ty] =
                (delta_row < batch_size && delta_col < output_size)
                    ? delta_var_out[delta_row * output_size + delta_col]
                    : static_cast<T>(0);

            // activation matrix
            size_t a_ty = thread_load_idx / BLOCK_TILE;
            size_t a_tx = thread_load_idx % BLOCK_TILE;
            size_t a_col = blockIdx.x * BLOCK_TILE + a_tx;
            size_t a_row = phase * BLOCK_TILE_K + a_ty;

            smem_mu_a[a_ty][a_tx] = (a_row < batch_size && a_col < input_size)
                                        ? smem_mu_a[a_ty][a_tx] =
                                              mu_a[a_row * input_size + a_col]
                                        : static_cast<T>(0);
        }
        __syncthreads();

        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_per_warp * THREAD_TILE + j;
                tmp_delta_mu[j] = smem_delta_mu[i][idx];
                tmp_delta_var[j] = smem_delta_var[i][idx];
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_per_warp * THREAD_TILE + j;
                tmp_mu_a[j] = smem_mu_a[i][idx];
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] =
                        __fmaf_rn(tmp_mu_a[j], tmp_delta_mu[t], tmp_mu[t][j]);
                    tmp_var[t][j] = __fmaf_rn(tmp_mu_a[j] * tmp_mu_a[j],
                                              tmp_delta_var[t], tmp_var[t][j]);
                }
            }
        }
        __syncthreads();
    }

    size_t row_base = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                      thread_row_per_warp * THREAD_TILE;
    size_t col_base = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                      thread_col_per_warp * THREAD_TILE;
    for (size_t t = 0; t < THREAD_TILE; t++) {
        size_t row = row_base + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            size_t col = col_base + j;

            if (row < output_size && col < input_size) {
                T tmp_var_w = var_w[row * input_size + col];

                delta_mu_w[row * input_size + col] =
                    __fmul_rn(tmp_mu[t][j], tmp_var_w);
                delta_var_w[row * input_size + col] =
                    __fmul_rn(tmp_var_w * tmp_var_w, tmp_var[t][j]);
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t PACK_SIZE, size_t SHEM_PADDING>
__global__ void linear_bwd_delta_w_v4(const T *var_w, const T *mu_a,
                                      const T *delta_mu_out,
                                      const T *delta_var_out, size_t input_size,
                                      size_t output_size, int batch_size,
                                      T *delta_mu_w, T *delta_var_w)
/*
 */
{
    __shared__ T smem_delta_mu[BLOCK_TILE_K][BLOCK_TILE + SHEM_PADDING];
    __shared__ T smem_delta_var[BLOCK_TILE_K][BLOCK_TILE + SHEM_PADDING];
    __shared__ T smem_mu_a[BLOCK_TILE_K][BLOCK_TILE + SHEM_PADDING];

    constexpr unsigned int WAPRS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    constexpr unsigned int K_PACKS = BLOCK_TILE_K / PACK_SIZE;
    constexpr unsigned int XY_PACKS = BLOCK_TILE / PACK_SIZE;
    constexpr unsigned int THREAD_PACKS = THREAD_TILE / PACK_SIZE;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * K_PACKS + THREADS - 1) / THREADS;

    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WAPRS_X;
    const size_t warp_col = warp_id % WAPRS_X;
    const size_t thread_row_per_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_per_warp = lane_id % THREADS_PER_WARP_X;

    const unsigned int num_tiles =
        (batch_size + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

    float4 tmp_mu[THREAD_TILE][THREAD_PACKS] = {{0.0f}};
    float4 tmp_var[THREAD_TILE][THREAD_PACKS] = {{0.0f}};

    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (size_t l_i = 0; l_i < NUM_LOADS; l_i++) {
            const size_t thread_load_idx = thread_linear_idx + l_i * THREADS;

            // Update quantity matrix
            size_t delta_tx = thread_load_idx / XY_PACKS;
            size_t delta_ty = thread_load_idx % XY_PACKS * PACK_SIZE;
            size_t delta_col = blockIdx.y * BLOCK_TILE + delta_ty;
            size_t delta_row = phase * BLOCK_TILE_K + delta_tx;

            // Hardcode values
            float4 delta_mu_val = {0, 0, 0, 0};
            float4 delta_var_val = {0, 0, 0, 0};
            if (delta_row < batch_size && delta_col < output_size) {
                delta_mu_val = *reinterpret_cast<const float4 *>(
                    &delta_mu_out[delta_row * output_size + delta_col]);
                delta_var_val = *reinterpret_cast<const float4 *>(
                    &delta_var_out[delta_row * output_size + delta_col]);
            }

            *reinterpret_cast<float4 *>(&smem_delta_mu[delta_tx][delta_ty]) =
                delta_mu_val;
            *reinterpret_cast<float4 *>(&smem_delta_var[delta_tx][delta_ty]) =
                delta_var_val;

            // activation matrix
            size_t a_ty = thread_load_idx / XY_PACKS;
            size_t a_tx = thread_load_idx % XY_PACKS * PACK_SIZE;
            size_t a_col = blockIdx.x * BLOCK_TILE + a_tx;
            size_t a_row = phase * BLOCK_TILE_K + a_ty;

            // Harded code values
            float4 mu_a_val = {0, 0, 0, 0};
            if (a_row < batch_size && a_col < input_size) {
                mu_a_val = *reinterpret_cast<const float4 *>(
                    &mu_a[a_row * input_size + a_col]);
            }
            *reinterpret_cast<float4 *>(&smem_mu_a[a_ty][a_tx]) = mu_a_val;
        }
        __syncthreads();

        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            float4 tmp_delta_mu[THREAD_TILE / PACK_SIZE];
            float4 tmp_delta_var[THREAD_TILE / PACK_SIZE];
            float4 tmp_mu_a[THREAD_TILE / PACK_SIZE];

#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_per_warp * THREAD_TILE + j * PACK_SIZE;
                tmp_delta_mu[j] =
                    *reinterpret_cast<const float4 *>(&smem_delta_mu[i][idx]);
                tmp_delta_var[j] =
                    *reinterpret_cast<const float4 *>(&smem_delta_var[i][idx]);
            }
#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_per_warp * THREAD_TILE + j * PACK_SIZE;
                tmp_mu_a[j] =
                    *reinterpret_cast<const float4 *>(&smem_mu_a[i][idx]);
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
                T delta_mu_t = reinterpret_cast<T *>(
                    &tmp_delta_mu[t / PACK_SIZE])[t & (PACK_SIZE - 1)];
                T delta_var_t = reinterpret_cast<T *>(
                    &tmp_delta_var[t / PACK_SIZE])[t & (PACK_SIZE - 1)];
#pragma unroll
                for (size_t j = 0; j < THREAD_PACKS; j++) {
                    float4 mu_a_j = tmp_mu_a[j];
                    float4 mu_a_squared = {
                        mu_a_j.x * mu_a_j.x, mu_a_j.y * mu_a_j.y,
                        mu_a_j.z * mu_a_j.z, mu_a_j.w * mu_a_j.w};

                    tmp_mu[t][j].x =
                        __fmaf_rn(mu_a_j.x, delta_mu_t, tmp_mu[t][j].x);
                    tmp_mu[t][j].y =
                        __fmaf_rn(mu_a_j.y, delta_mu_t, tmp_mu[t][j].y);
                    tmp_mu[t][j].z =
                        __fmaf_rn(mu_a_j.z, delta_mu_t, tmp_mu[t][j].z);
                    tmp_mu[t][j].w =
                        __fmaf_rn(mu_a_j.w, delta_mu_t, tmp_mu[t][j].w);

                    tmp_var[t][j].x =
                        __fmaf_rn(mu_a_squared.x, delta_var_t, tmp_var[t][j].x);
                    tmp_var[t][j].y =
                        __fmaf_rn(mu_a_squared.y, delta_var_t, tmp_var[t][j].y);
                    tmp_var[t][j].z =
                        __fmaf_rn(mu_a_squared.z, delta_var_t, tmp_var[t][j].z);
                    tmp_var[t][j].w =
                        __fmaf_rn(mu_a_squared.w, delta_var_t, tmp_var[t][j].w);
                }
            }
        }
        __syncthreads();
    }
    size_t row_base = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                      thread_row_per_warp * THREAD_TILE;
    size_t col_base = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                      thread_col_per_warp * THREAD_TILE;

    for (size_t t = 0; t < THREAD_TILE; t++) {
        size_t row = row_base + t;
        if (row >= output_size) break;
#pragma unroll
        for (size_t j = 0; j < THREAD_PACKS; j++) {
            size_t col = col_base + j * PACK_SIZE;
            if (col >= input_size) break;

            float4 tmp_var_w = *reinterpret_cast<const float4 *>(
                &var_w[row * input_size + col]);
            float4 mu_acc_tmp = tmp_mu[t][j];
            float4 var_acc_tmp = tmp_var[t][j];

            mu_acc_tmp.x = __fmul_rn(mu_acc_tmp.x, tmp_var_w.x);
            mu_acc_tmp.y = __fmul_rn(mu_acc_tmp.y, tmp_var_w.y);
            mu_acc_tmp.z = __fmul_rn(mu_acc_tmp.z, tmp_var_w.z);
            mu_acc_tmp.w = __fmul_rn(mu_acc_tmp.w, tmp_var_w.w);

            var_acc_tmp.x = __fmul_rn(tmp_var_w.x * tmp_var_w.x, var_acc_tmp.x);
            var_acc_tmp.y = __fmul_rn(tmp_var_w.y * tmp_var_w.y, var_acc_tmp.y);
            var_acc_tmp.z = __fmul_rn(tmp_var_w.z * tmp_var_w.z, var_acc_tmp.z);
            var_acc_tmp.w = __fmul_rn(tmp_var_w.w * tmp_var_w.w, var_acc_tmp.w);

            __stcs(
                reinterpret_cast<float4 *>(&delta_mu_w[row * input_size + col]),
                mu_acc_tmp);
            __stcs(reinterpret_cast<float4 *>(
                       &delta_var_w[row * input_size + col]),
                   var_acc_tmp);
        }
    }
}

__global__ void linear_bwd_delta_b(float const *var_b,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_b,
                                   float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < 1 && row < output_size) {
        for (int i = 0; i < batch_size; i++) {
            sum_mu += delta_mu_out[output_size * i + row];
            sum_var += delta_var_out[output_size * i + row];
        }

        delta_mu_b[col * output_size + row] =
            sum_mu * var_b[col * output_size + row];

        delta_var_b[col * output_size + row] = sum_var *
                                               var_b[col * output_size + row] *
                                               var_b[col * output_size + row];
    }
}
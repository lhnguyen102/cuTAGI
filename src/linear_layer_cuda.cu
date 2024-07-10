///////////////////////////////////////////////////////////////////////////////
// File:         linear_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 03, 2023
// Updated:      March 09, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/linear_layer.h"
#include "../include/linear_layer_cuda.cuh"

#define WARP_SIZE 32U
#define BANK_SIZE 32U

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

            sum_mu += mu_w_tmp * mu_a_tmp;
            sum_var += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                       var_w_tmp * mu_a_tmp * mu_a_tmp;
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
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] += mu_w_val[j] * mu_a_val[t];
                    tmp_var[t][j] +=
                        (mu_w_val[j] * mu_w_val[j] + var_w_val[j]) *
                            var_a_val[t] +
                        var_w_val[j] * mu_a_val[t] * mu_a_val[t];
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();

    const size_t base_row = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                            thread_row_in_warp * THREAD_TILE;
    const size_t base_col = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                            thread_col_in_warp * THREAD_TILE;

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
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_row * WARP_TILE_Y +
                             thread_row_in_warp * THREAD_TILE + j;
                tmp_delta_mu[j] = smem_delta_mu[i][idx];
                tmp_delta_var[j] = smem_delta_var[i][idx];
            }
            for (size_t j = 0; j < THREAD_TILE; j++) {
                size_t idx = warp_col * WARP_TILE_X +
                             thread_col_in_warp * THREAD_TILE + j;
                tmp_mu_w[j] = smem_mu_w[i][idx];
            }

            for (size_t t = 0; t < THREAD_TILE; t++) {
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] += tmp_mu_w[j] * tmp_delta_mu[t];
                    tmp_var[t][j] +=
                        tmp_mu_w[j] * tmp_delta_var[t] * tmp_mu_w[j];
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();

    for (size_t t = 0; t < THREAD_TILE; t++) {
        int row = blockIdx.y * BLOCK_TILE + warp_row * WARP_TILE_Y +
                  thread_row_in_warp * THREAD_TILE + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            int col = blockIdx.x * BLOCK_TILE + warp_col * WARP_TILE_X +
                      thread_col_in_warp * THREAD_TILE + j;
            if (row < batch_size && col < input_size) {
                T jcb_val = jcb[row * input_size + col];
                delta_mu_in[row * input_size + col] = tmp_mu[t][j] * jcb_val;
                delta_var_in[row * input_size + col] =
                    tmp_var[t][j] * jcb_val * jcb_val;
            }
        }
    }
}

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
                    tmp_mu[t][j] += tmp_mu_a[j] * tmp_delta_mu[t];
                    tmp_var[t][j] +=
                        tmp_mu_a[j] * tmp_mu_a[j] * tmp_delta_var[t];
                }
            }
        }
        __syncwarp();
    }
    __syncthreads();
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

                delta_mu_w[row * input_size + col] = tmp_mu[t][j] * tmp_var_w;
                delta_var_w[row * input_size + col] =
                    tmp_var[t][j] * tmp_var_w * tmp_var_w;
            }
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

////////////////////////////////////////////////////////////////////////////////
// Fully Connected Layer
////////////////////////////////////////////////////////////////////////////////
void linear_forward_cuda(HiddenStateCuda *&cu_input_states,
                         HiddenStateCuda *&cu_output_states,
                         const float *d_mu_w, const float *d_var_w,
                         const float *d_mu_b, const float *d_var_b,
                         size_t input_size, size_t output_size, int batch_size,
                         bool bias) {
    if (output_size > 128 && input_size > 1024) {
        // TODO: remove hardcoded kernel config
        constexpr unsigned int BLOCK_SIZE = 128U;
        constexpr unsigned int THREAD_TILE = 8U;
        constexpr unsigned int WARPS_X = 4U;
        constexpr unsigned int WARPS_Y = 2U;
        constexpr unsigned int BLOCK_TILE_K = 16U;
        constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WARPS_X;
        constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WARPS_Y;
        constexpr unsigned int THREADS_X =
            WARPS_X * (WARP_TILE_X / THREAD_TILE);
        constexpr unsigned int THREADS_Y =
            WARPS_Y * (WARP_TILE_Y / THREAD_TILE);
        constexpr unsigned int THREADS = THREADS_X * THREADS_Y;
        constexpr unsigned int SMEM_PADDING = BANK_SIZE / THREADS_X;

        dim3 block_dim(THREADS_X, THREADS_Y, 1U);
        dim3 grid_dim(
            (static_cast<unsigned int>(output_size) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            (static_cast<unsigned int>(batch_size) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            1U);

        linear_fwd_mean_var_v3<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                               THREADS, WARP_TILE_X, WARP_TILE_Y, SMEM_PADDING>
            <<<grid_dim, block_dim>>>(
                d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                cu_input_states->d_var_a, input_size, output_size, batch_size,
                bias, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        constexpr unsigned int BLOCK_SIZE = 16U;
        unsigned int grid_rows = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1U);
        dim3 grid_dim(grid_cols, grid_rows, 1U);
        constexpr size_t THREADS = BLOCK_SIZE * BLOCK_SIZE;

        linear_fwd_mean_var_v1<float, BLOCK_SIZE, BLOCK_SIZE, THREADS>
            <<<grid_dim, block_dim>>>(
                d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                cu_input_states->d_var_a, input_size, output_size, batch_size,
                bias, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }
}

void linear_state_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                DeltaStateCuda *&cu_output_delta_states,
                                BackwardStateCuda *&cu_next_bwd_states,
                                const float *d_mu_w, size_t input_size,
                                size_t output_size, int batch_size)
/*
 */
{
    constexpr unsigned int BLOCK_SIZE = 128U;
    constexpr unsigned int THREAD_TILE = 8U;
    constexpr unsigned int WARPS_X = 4U;
    constexpr unsigned int WARPS_Y = 2U;
    constexpr unsigned int BLOCK_TILE_K = 16U;
    constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WARPS_X;
    constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WARPS_Y;
    constexpr unsigned int THREADS_X = WARPS_X * (WARP_TILE_X / THREAD_TILE);
    constexpr unsigned int THREADS_Y = WARPS_Y * (WARP_TILE_Y / THREAD_TILE);
    constexpr unsigned int THREADS = THREADS_X * THREADS_Y;
    constexpr unsigned int SMEM_PADDING = BANK_SIZE / THREADS_X;

    dim3 block_dim(THREADS_X, THREADS_Y, 1U);
    dim3 grid_dim(
        (static_cast<unsigned int>(input_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        (static_cast<unsigned int>(batch_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        1U);
    linear_bwd_delta_z_v3<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE, THREADS,
                          WARP_TILE_X, WARP_TILE_Y, SMEM_PADDING>
        <<<grid_dim, block_dim>>>(d_mu_w, cu_next_bwd_states->d_jcb,
                                  cu_input_delta_states->d_delta_mu,
                                  cu_input_delta_states->d_delta_var,
                                  input_size, output_size, batch_size,
                                  cu_output_delta_states->d_delta_mu,
                                  cu_output_delta_states->d_delta_var);
}

void linear_weight_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                 DeltaStateCuda *&cu_output_delta_states,
                                 BackwardStateCuda *&cu_next_bwd_states,
                                 const float *d_var_w, size_t input_size,
                                 size_t output_size, int batch_size,
                                 float *d_delta_mu_w, float *d_delta_var_w)
/*
 */
{
    constexpr unsigned int BLOCK_SIZE = 128U;
    constexpr unsigned int THREAD_TILE = 8U;
    constexpr unsigned int WARPS_X = 4U;
    constexpr unsigned int WARPS_Y = 2U;
    constexpr unsigned int BLOCK_TILE_K = 16U;
    constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WARPS_X;
    constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WARPS_Y;
    constexpr unsigned int THREADS_X = WARPS_X * (WARP_TILE_X / THREAD_TILE);
    constexpr unsigned int THREADS_Y = WARPS_Y * (WARP_TILE_Y / THREAD_TILE);
    constexpr unsigned int THREADS = THREADS_X * THREADS_Y;
    constexpr unsigned int SMEM_PADDING = BANK_SIZE / THREADS_X;

    dim3 block_dim(THREADS_X, THREADS_Y, 1U);
    dim3 grid_dim(
        (static_cast<unsigned int>(input_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        (static_cast<unsigned int>(output_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        1U);

    linear_bwd_delta_w_v3<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE, THREADS,
                          WARP_TILE_X, WARP_TILE_Y, SMEM_PADDING>
        <<<grid_dim, block_dim>>>(d_var_w, cu_next_bwd_states->d_mu_a,
                                  cu_input_delta_states->d_delta_mu,
                                  cu_input_delta_states->d_delta_var,
                                  input_size, output_size, batch_size,
                                  d_delta_mu_w, d_delta_var_w);
}

LinearCuda::LinearCuda(size_t ip_size, size_t op_size, bool bias,
                       float gain_weight, float gain_bias, std::string method)
    : gain_w(gain_weight),
      gain_b(gain_bias),
      init_method(method)
/*
 */
{
    this->input_size = ip_size;
    this->output_size = op_size;
    this->bias = bias;
    this->num_weights = this->input_size * this->output_size;
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = this->output_size;
    }

    // Initalize weights and bias
    this->init_weight_bias();
    if (this->training) {
        // TODO: to be removed
        this->bwd_states = std::make_unique<BackwardStateCuda>();
        this->allocate_param_delta();
    }
}

LinearCuda::~LinearCuda() {}

std::string LinearCuda::get_layer_info() const
/*
 */
{
    return "Linear(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LinearCuda::get_layer_name() const
/*
 */
{
    return "LinearCuda";
}

LayerType LinearCuda::get_layer_type() const
/*
 */
{
    return LayerType::Linear;
}

void LinearCuda::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->input_size, this->output_size,
                                this->num_weights, this->num_biases);

    this->allocate_param_memory();
    this->params_to_device();
}

void LinearCuda::forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int batch_size = input_states.block_size;

    this->set_cap_factor_udapte(batch_size);

    linear_forward_cuda(cu_input_states, cu_output_states, this->d_mu_w,
                        this->d_var_w, this->d_mu_b, this->d_var_b,
                        this->input_size, this->output_size,
                        input_states.block_size, this->bias);

    // Update number of actual states.
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    // Lazy initialization
    BackwardStateCuda *cu_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    if (cu_bwd_states->size == 0 && this->training) {
        cu_bwd_states->size = cu_input_states->actual_size * batch_size;
        cu_bwd_states->allocate_memory();
    }

    // Update backward state for inferring parameters
    if (this->training) {
        BackwardStateCuda *cu_bwd_states =
            dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states, *cu_bwd_states);
    }
}

void LinearCuda::backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states, bool state_udapte)
/**/
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;

    // Compute inovation vector
    unsigned int grid_row = (this->input_size + threads - 1) / threads;
    unsigned int grid_col = (batch_size + threads - 1) / threads;

    dim3 grid_dim(grid_col, grid_row);
    dim3 block_dim(threads, threads);

    if (state_udapte) {
        linear_state_backward_cuda(
            cu_input_delta_states, cu_output_delta_states, cu_next_bwd_states,
            this->d_mu_w, this->input_size, this->output_size, batch_size);
    }

    // Updated values for weights
    if (this->param_update) {
        linear_weight_backward_cuda(
            cu_input_delta_states, cu_output_delta_states, cu_next_bwd_states,
            this->d_var_w, this->input_size, this->output_size, batch_size,
            this->d_delta_mu_w, this->d_delta_var_w);

        // Updated values for biases
        if (this->bias) {
            unsigned int grid_row_b =
                (this->output_size + threads - 1) / threads;
            dim3 grid_dim_b(1, grid_row_b);

            linear_bwd_delta_b<<<grid_dim_b, block_dim>>>(
                this->d_var_b, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->input_size,
                this->output_size, batch_size, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

std::unique_ptr<BaseLayer> LinearCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_linear = std::make_unique<Linear>(
        this->input_size, this->output_size, this->bias, this->gain_w,
        this->gain_b, this->init_method);
    host_linear->mu_w = this->mu_w;
    host_linear->var_w = this->var_w;
    host_linear->mu_b = this->mu_b;
    host_linear->var_b = this->var_b;

    return host_linear;
}

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "common_cuda_kernel.cuh"
#include "config.h"

__global__ void conv2d_fwd_mean_var_cuda(float const *mu_w, float const *var_w,
                                         float const *mu_b, float const *var_b,
                                         float const *mu_a, float const *var_a,
                                         int const *aidx, int woho, int fo,
                                         int wihi, int fi, int ki, int B,
                                         int pad_idx, bool bias, float *mu_z,
                                         float *var_z)
/*Compute mean of product WA for convolutional layer

Args:
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int aidx_tmp;
    float mu_a_tmp;
    float var_a_tmp;
    float mu_w_tmp;
    float var_w_tmp;
    int ki2 = ki * ki;
    int n = ki2 * fi;
    if (col < woho * B && row < fo) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[(col % woho) * ki2 + i % ki2];

            if (aidx_tmp > -1) {
                // aidx's lowest value starts at 1
                aidx_tmp += (i / ki2) * wihi + (col / woho) * wihi * fi;
                mu_a_tmp = mu_a[aidx_tmp - 1];
                var_a_tmp = var_a[aidx_tmp - 1];

                mu_w_tmp = mu_w[row * n + i];
                var_w_tmp = var_w[row * n + i];

                sum_mu += mu_w_tmp * mu_a_tmp;
                sum_var += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                           var_w_tmp * mu_a_tmp * mu_a_tmp;
            }
        }

        if (bias) {
            mu_z[woho * (col / woho) * fo + col % woho + row * woho] =
                sum_mu + mu_b[row];
            var_z[woho * (col / woho) * fo + col % woho + row * woho] =
                sum_var + var_b[row];
        } else {
            mu_z[woho * (col / woho) * fo + col % woho + row * woho] = sum_mu;
            var_z[woho * (col / woho) * fo + col % woho + row * woho] = sum_var;
        }
    }
}

__global__ void conv2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    const float *delta_var_out, int const *zw_idx, int const *zud_idx, int woho,
    int fo, int wihi, int fi, int ki, int nr, int n, int B, int pad_idx,
    float *delta_mu, float *delta_var)
/* Compute updated quantities of the mean of hidden states for convolutional
 layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    zwidx: Weight indices for covariance Z|WA i.e. FCzwa_1
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaMz: Updated quantities for the mean of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    zwidxpos: Position of weight indices for covariance Z|WA
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    ki2: ki x ki
    nr: Number of rows of weight indices for covariance Z|WA i.e. row_zw
    n: nr x fo
    k: wihi x B
    padIdx: Size of the hidden state vector for this layer + 1
 */
// TODO: remove jposIn
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int widx_tmp;
    int aidx_tmp;
    float mu_w_tmp;
    int k = wihi * B;
    int ki2 = ki * ki;
    if (col < k && row < fi)  // k = wihi * B
    {
        for (int i = 0; i < n; i++) {
            // indices for mw. Note that nr = n / fo. Indices's lowest value
            // starts at 1
            widx_tmp = zw_idx[(col % wihi) * nr + i % nr] +
                       (i / nr) * ki2 * fi + row * ki2 - 1;

            // indices for deltaM
            aidx_tmp = zud_idx[col % wihi + wihi * (i % nr)];

            if (aidx_tmp > -1) {
                aidx_tmp += (i / nr) * woho + (col / wihi) * woho * fo;
                mu_w_tmp = mu_w[widx_tmp];

                sum_mu += delta_mu_out[aidx_tmp - 1] * mu_w_tmp;
                sum_var += mu_w_tmp * delta_var_out[aidx_tmp - 1] * mu_w_tmp;
            }
        }

        delta_mu[wihi * (col / wihi) * fi + col % wihi + row * wihi] =
            sum_mu * jcb[row * k + col];

        delta_var[wihi * (col / wihi) * fi + col % wihi + row * wihi] =
            sum_var * jcb[row * k + col] * jcb[row * k + col];
    }
}

__global__ void permmute_jacobian_cuda(float const *jcb_0, int wihi, int fi,
                                       int batch_size, float *jcb)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < wihi * fi && row < batch_size) {
        // Note that (col/(w * h)) equivalent to floorf((col/(w * h)))
        // because of interger division
        jcb[wihi * (col / wihi) * batch_size + col % wihi + row * wihi] =
            jcb_0[row * wihi * fi + col];
    }
}

__global__ void conv2d_bwd_delta_w_cuda(float const *var_w, float const *mu_a,
                                        float const *delta_mu_out,
                                        float const *delta_var_out,
                                        int const *aidx, int B, int k, int woho,
                                        int wihi, int fi, int ki,
                                        float *delta_mu_w, float *delta_var_w)
/* Compute update quantities for the mean of weights for convolutional layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    aidx: Activation indices for computing the mean of the product WA
    wpos: Weight position for this layer in the weight vector of network
    apos: Input-hidden-state position for this layer in the weight vector
          of network
    aidxpos: Position of the activation indices in its vector of the network
    m: ki x ki x fi
    n: wo x ho x B
    k: fo
    woho: Width x height of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    ki2: ki x ki
    padIdx: Size of the hidden state vector for this layer + 1
    deltaMw: Updated quantities for the mean of weights
*/
// TODO: remove the duplicate in the input variables
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    float mu_a_tmp;
    int aidx_tmp;
    int ki2 = ki * ki;
    int m = ki2 * fi;
    int n = woho * B;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[ki2 * (i % woho) + row % ki2];

            if (aidx_tmp > -1) {
                // Indices's lowest value starts at 1
                aidx_tmp += (row / ki2) * wihi + (i / woho) * wihi * fi;
                mu_a_tmp = mu_a[aidx_tmp - 1];
                sum_mu += mu_a_tmp * delta_mu_out[col * n + i];
                sum_var += mu_a_tmp * delta_var_out[col * n + i] * mu_a_tmp;
            }
        }
        float var_w_tmp = var_w[col * m + row];
        delta_mu_w[col * m + row] = sum_mu * var_w_tmp;
        delta_var_w[col * m + row] = sum_var * var_w_tmp * var_w_tmp;
    }
}

__global__ void conv2d_bwd_delta_b_cuda(float const *var_b,
                                        float const *delta_mu_out,
                                        const float *delta_var_out, int n,
                                        int k, float *delta_mu_b,
                                        float *delta_var_b)
/* Compute update quantities for the mean of biases for convolutional layer.

Args:
    Cbz: Covariance b|Z+
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    bpos: Bias position for this layer in the bias vector of network
    n: wo x ho xB
    k: fo
    deltaMb: Updated quantities for the mean of biases

*/
{
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    if (col < k) {
        for (int i = 0; i < n; i++) {
            sum_mu += delta_mu_out[col * n + i];
            sum_var += delta_var_out[col * n + i];
        }
        delta_mu_b[col] = sum_mu * var_b[col];
        delta_var_b[col] = sum_var * var_b[col] * var_b[col];
    }
}

__global__ void permute_delta_cuda(float const *delta_mu_0,
                                   float const *delta_var_0, int woho, int kp,
                                   int batch_size, float *delta_mu,
                                   float *delta_var) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < kp && row < batch_size)  // kp = woho * fo
    {
        // Note that (col/(w * h)) equvalent to floorf((col/(w * h)))
        // because of interger division
        delta_mu[woho * (col / woho) * batch_size + col % woho + row * woho] =
            delta_mu_0[row * kp + col];
        delta_var[woho * (col / woho) * batch_size + col % woho + row * woho] =
            delta_var_0[row * kp + col];
    }
}

// TILED
template <size_t TILE_SIZE, size_t SMEM_PADDING>
__global__ void conv2d_fwd_mean_var_cuda_v1(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a, int const *aidx,
    int woho, int fo, int wihi, int fi, int ki, int B, bool bias, float *mu_z,
    float *var_z)
/*Compute mean of product WA for convolutional layer

Args:
*/
{
    const int PADDED_TILE_SIZE = TILE_SIZE + SMEM_PADDING;
    __shared__ float s_mu_a[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_var_a[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_mu_w[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_var_w[TILE_SIZE * PADDED_TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int aidx_tmp;
    float mu_a_tmp;
    float var_a_tmp;
    float mu_w_tmp;
    float var_w_tmp;
    int ki2 = ki * ki;
    int n = ki2 * fi;
    int index, tile_idx_y, tile_idx_x;
    int col_div_woho = col / woho;
    int col_mod_woho = col % woho;
    float inv_ki2 = 1.0 / ki2;
    int aidx_sub = col_div_woho * wihi * fi;
    for (int phase = 0; phase < ceil((float)n / TILE_SIZE); phase++) {
        index = phase * TILE_SIZE + ty;
        int index_div_ki2 = __float2int_rd(index * inv_ki2);
        int index_mod_ki2 = index - index_div_ki2 * ki2;

        tile_idx_y = tx * PADDED_TILE_SIZE + ty;
        tile_idx_x = ty * TILE_SIZE + tx;

        s_mu_a[tile_idx_y] = 0.0;
        s_var_a[tile_idx_y] = 0.0;
        s_mu_w[tile_idx_x] = 0.0;
        s_var_w[tile_idx_x] = 0.0;

        if (col < woho * B && index < n) {
            aidx_tmp = __ldg(&aidx[col_mod_woho * ki2 + index_mod_ki2]);
            if (aidx_tmp > -1) {
                aidx_tmp += index_div_ki2 * wihi + aidx_sub - 1;
                s_mu_a[tile_idx_y] = mu_a[aidx_tmp];
                s_var_a[tile_idx_y] = var_a[aidx_tmp];
            }
        }
        if (row < fo && tx + phase * TILE_SIZE < n) {
            s_mu_w[tile_idx_x] = mu_w[row * n + tx + phase * TILE_SIZE];
            s_var_w[tile_idx_x] = var_w[row * n + tx + phase * TILE_SIZE];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            mu_w_tmp = s_mu_w[ty * TILE_SIZE + i];
            var_w_tmp = s_var_w[ty * TILE_SIZE + i];
            mu_a_tmp = s_mu_a[tx * PADDED_TILE_SIZE + i];
            var_a_tmp = s_var_a[tx * PADDED_TILE_SIZE + i];

            float mu_w_tmp_2 = __fmul_rn(mu_w_tmp, mu_w_tmp);

            sum_mu += __fmul_rn(mu_w_tmp, mu_a_tmp);
            sum_var += (mu_w_tmp_2 + var_w_tmp) * var_a_tmp +
                       var_w_tmp * mu_a_tmp * mu_a_tmp;
        }

        __syncthreads();
    }
    if (col < woho * B && row < fo) {
        int idx_out = woho * col_div_woho * fo + col_mod_woho + row * woho;
        if (bias) {
            mu_z[idx_out] = sum_mu + mu_b[row];
            var_z[idx_out] = sum_var + var_b[row];
        } else {
            mu_z[idx_out] = sum_mu;
            var_z[idx_out] = sum_var;
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING>
__global__ void conv2d_fwd_mean_var_cuda_v2(
    const T *__restrict__ mu_w, const T *__restrict__ var_w,
    const T *__restrict__ mu_b, const T *__restrict__ var_b,
    const T *__restrict__ mu_a, const T *__restrict__ var_a,
    const int *__restrict__ aidx, int woho, int fo, int wihi, int fi, int ki,
    int B, bool bias, T *__restrict__ mu_z, T *__restrict__ var_z)
/*
 */
{
    __shared__ T s_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_var_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_var_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    int ki2 = ki * ki;
    int n = ki2 * fi;
    float inv_ki2 = __frcp_rn(static_cast<float>(ki2));

    // Thread block
    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (n + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
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
    const size_t warp_row_coord =
        warp_row * WARP_TILE_Y + thread_row_in_warp * THREAD_TILE;
    const size_t warp_col_coord =
        warp_col * WARP_TILE_X + thread_col_in_warp * THREAD_TILE;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};

    for (int phase = 0; phase < num_tiles; phase++) {
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            const size_t thread_load_idx = (thread_linear_idx + l_i * THREADS);
            const size_t ty = thread_load_idx / BLOCK_TILE;
            const size_t tx = thread_load_idx % BLOCK_TILE;

            // Activation indices
            const size_t aw_col = phase * BLOCK_TILE_K + ty;
            const size_t a_idx_row = blockIdx.x * BLOCK_TILE + tx;
            const int aidx_sub = (a_idx_row / woho) * wihi * fi;

            const size_t index_div_ki2 = __float2int_rd(aw_col * inv_ki2);
            const size_t index_mod_ki2 = aw_col - index_div_ki2 * ki2;

            T mu_a_val = 0.0f, var_a_val = 0.0f;
            int aidx_tmp =
                (a_idx_row < woho * B && aw_col < n)
                    ? __ldg(&aidx[(a_idx_row % woho) * ki2 + index_mod_ki2])
                    : -1;
            if (aidx_tmp > -1) {
                aidx_tmp += index_div_ki2 * wihi + aidx_sub - 1;
                mu_a_val = __ldg(&mu_a[aidx_tmp]);
                var_a_val = __ldg(&var_a[aidx_tmp]);
            }
            s_mu_a[ty][tx] = mu_a_val;
            s_var_a[ty][tx] = var_a_val;

            // Weight
            const size_t w_row = blockIdx.y * BLOCK_TILE + tx;

            s_mu_w[ty][tx] = (w_row < fo && aw_col < n)
                                 ? __ldg(&mu_w[w_row * n + aw_col])
                                 : 0.0f;
            s_var_w[ty][tx] = (w_row < fo && aw_col < n)
                                  ? __ldg(&var_w[w_row * n + aw_col])
                                  : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            T mu_w_val[THREAD_TILE] = {static_cast<T>(0)},
              var_w_val[THREAD_TILE] = {static_cast<T>(0)};
            T mu_a_val[THREAD_TILE] = {static_cast<T>(0)},
              var_a_val[THREAD_TILE] = {static_cast<T>(0)};

#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                mu_w_val[j] = s_mu_w[i][warp_row_coord + j];
                var_w_val[j] = s_var_w[i][warp_row_coord + j];
                mu_a_val[j] = s_mu_a[i][warp_col_coord + j];
                var_a_val[j] = s_var_a[i][warp_col_coord + j];
            }

#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                T mu_a_val_sqrt = __fmul_rn(mu_a_val[t], mu_a_val[t]);
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] =
                        __fmaf_rn(mu_w_val[j], mu_a_val[t], tmp_mu[t][j]);
                    T var_term_1 =
                        __fmaf_rn(mu_w_val[j], mu_w_val[j], var_w_val[j]);
                    T var_term_2 =
                        __fmaf_rn(mu_a_val_sqrt, var_w_val[j], tmp_var[t][j]);
                    tmp_var[t][j] =
                        __fmaf_rn(var_term_1, var_a_val[t], var_term_2);
                }
            }
        }
        __syncthreads();
    }
    const size_t row_base = blockIdx.y * BLOCK_TILE + warp_row_coord;
    const size_t col_base = blockIdx.x * BLOCK_TILE + warp_col_coord;

#pragma unroll
    for (size_t j = 0; j < THREAD_TILE; j++) {
        const size_t col = col_base + j;
        const int col_div_woho = col / woho;
        const int col_mod_woho = col % woho;
#pragma unroll
        for (size_t t = 0; t < THREAD_TILE; t++) {
            const size_t row = row_base + t;
            if (row < fo && col < B * woho) {
                const int idx_out =
                    woho * col_div_woho * fo + col_mod_woho + row * woho;
                T bias_mu = bias ? __ldg(&mu_b[row]) : static_cast<T>(0);
                T bias_var = bias ? __ldg(&var_b[row]) : static_cast<T>(0);
                mu_z[idx_out] = __fadd_rn(tmp_mu[j][t], bias_mu);
                var_z[idx_out] = __fadd_rn(tmp_var[j][t], bias_var);
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t PACK_SIZE_T, size_t SMEM_PADDING>
__global__ void conv2d_fwd_mean_var_cuda_v3(const T *mu_w, const T *var_w,
                                            const T *mu_b, const T *var_b,
                                            const T *mu_a, const T *var_a,
                                            const int *aidx, int woho, int fo,
                                            int wihi, int fi, int ki, int B,
                                            bool bias, T *mu_z, T *var_z)
/*
 */
{
    __shared__ T s_mu_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_var_w[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_var_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    int ki2 = ki * ki;
    int n = ki2 * fi;
    float inv_ki2 = __frcp_rn(static_cast<float>(ki2));

    // Thread block
    const size_t thread_linear_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (n + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int XY_PACKS = BLOCK_TILE / PACK_SIZE_T;
    constexpr unsigned int THREAD_PACKS = THREAD_TILE / PACK_SIZE_T;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;
    constexpr unsigned int NUM_LOADS_2 =
        (XY_PACKS * BLOCK_TILE_K + THREADS - 1) / THREADS;

    // Warp
    constexpr unsigned int WARPS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WARPS_X;
    const size_t warp_col = warp_id % WARPS_X;
    const size_t thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_in_warp = lane_id % THREADS_PER_WARP_X;
    const size_t warp_row_coord =
        warp_row * WARP_TILE_Y + thread_row_in_warp * THREAD_TILE;
    const size_t warp_col_coord =
        warp_col * WARP_TILE_X + thread_col_in_warp * THREAD_TILE;

    T tmp_mu[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};

    for (int phase = 0; phase < num_tiles; phase++) {
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            const size_t thread_load_idx = (thread_linear_idx + l_i * THREADS);
            const size_t ty = thread_load_idx / BLOCK_TILE;
            const size_t tx = thread_load_idx % BLOCK_TILE;

            // Activation indices
            const size_t aw_col = phase * BLOCK_TILE_K + ty;
            const size_t a_idx_row = blockIdx.x * BLOCK_TILE + tx;
            const int aidx_sub = (a_idx_row / woho) * wihi * fi;

            const size_t index_div_ki2 = __float2int_rd(aw_col * inv_ki2);
            const size_t index_mod_ki2 = aw_col - index_div_ki2 * ki2;

            T mu_a_val = 0.0f, var_a_val = 0.0f;
            int aidx_tmp =
                (a_idx_row < woho * B && aw_col < n)
                    ? __ldg(&aidx[(a_idx_row % woho) * ki2 + index_mod_ki2])
                    : -1;
            if (aidx_tmp > -1) {
                aidx_tmp += index_div_ki2 * wihi + aidx_sub - 1;
                mu_a_val = __ldg(&mu_a[aidx_tmp]);
                var_a_val = __ldg(&var_a[aidx_tmp]);
            }
            s_mu_a[ty][tx] = mu_a_val;
            s_var_a[ty][tx] = var_a_val;
        }
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS_2; l_i++) {
            const size_t thread_load_idx = (thread_linear_idx + l_i * THREADS);
            const size_t w_ty = (thread_load_idx / BLOCK_TILE) * PACK_SIZE_T;
            const size_t w_tx = thread_load_idx % BLOCK_TILE;
            const size_t w_row = blockIdx.y * BLOCK_TILE + w_tx;
            const size_t w_col = phase * BLOCK_TILE_K + w_ty;

            float4 mu_w_row_val = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 var_w_row_val = {0.0f, 0.0f, 0.0f, 0.0f};
            if (w_row < fo && w_col < n) {
                mu_w_row_val =
                    *reinterpret_cast<const float4 *>(&mu_w[w_row * n + w_col]);
                var_w_row_val = *reinterpret_cast<const float4 *>(
                    &var_w[w_row * n + w_col]);
            }
#pragma unroll
            for (size_t p = 0; p < PACK_SIZE_T; p++) {
                s_mu_w[w_ty + p][w_tx] =
                    reinterpret_cast<const T *>(&mu_w_row_val)[p];
                s_var_w[w_ty + p][w_tx] =
                    reinterpret_cast<const T *>(&var_w_row_val)[p];
            }
        }
        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            T mu_w_val[THREAD_TILE] = {static_cast<T>(0)},
              var_w_val[THREAD_TILE] = {static_cast<T>(0)};
            T mu_a_val[THREAD_TILE] = {static_cast<T>(0)},
              var_a_val[THREAD_TILE] = {static_cast<T>(0)};

#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                *reinterpret_cast<float4 *>(&mu_w_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_mu_w[i][warp_row_coord + j * PACK_SIZE_T]);
                *reinterpret_cast<float4 *>(&var_w_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_var_w[i][warp_row_coord + j * PACK_SIZE_T]);
                *reinterpret_cast<float4 *>(&mu_a_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_mu_a[i][warp_col_coord + j * PACK_SIZE_T]);
                *reinterpret_cast<float4 *>(&var_a_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_var_a[i][warp_col_coord + j * PACK_SIZE_T]);
            }

#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                T mu_a_val_sqrt = __fmul_rn(mu_a_val[t], mu_a_val[t]);
#pragma unroll
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] =
                        __fmaf_rn(mu_w_val[j], mu_a_val[t], tmp_mu[t][j]);
                    T var_term_1 =
                        __fmaf_rn(mu_w_val[j], mu_w_val[j], var_w_val[j]);
                    T var_term_2 =
                        __fmaf_rn(mu_a_val_sqrt, var_w_val[j], tmp_var[t][j]);
                    tmp_var[t][j] =
                        __fmaf_rn(var_term_1, var_a_val[t], var_term_2);
                }
            }
        }
        __syncthreads();
    }
    const size_t row_base = blockIdx.y * BLOCK_TILE + warp_row_coord;
    const size_t col_base = blockIdx.x * BLOCK_TILE + warp_col_coord;

#pragma unroll
    for (size_t j = 0; j < THREAD_TILE; j++) {
        const size_t col = col_base + j;
        const int col_div_woho = col / woho;
        const int col_mod_woho = col % woho;
#pragma unroll
        for (size_t t = 0; t < THREAD_TILE; t++) {
            const size_t row = row_base + t;
            if (row < fo && col < B * woho) {
                const int idx_out =
                    woho * col_div_woho * fo + col_mod_woho + row * woho;
                T bias_mu = bias ? __ldg(&mu_b[row]) : static_cast<T>(0);
                T bias_var = bias ? __ldg(&var_b[row]) : static_cast<T>(0);
                mu_z[idx_out] = __fadd_rn(tmp_mu[j][t], bias_mu);
                var_z[idx_out] = __fadd_rn(tmp_var[j][t], bias_var);
            }
        }
    }
}

__global__ void conv2d_bwd_delta_w_cuda_tiled(
    float const *var_w, float const *mu_a, float const *delta_mu_out,
    float const *delta_var_out, int const *aidx, int B, int k, int woho,
    int wihi, int fi, int ki, int pad_idx, float *delta_mu_w,
    float *delta_var_w)
/**/
{
    const int TILE_SIZE = 16;
    const int PADDED_TILE_SIZE = TILE_SIZE + 2;
    __shared__ float s_mu_a[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_delta_mu_out[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_delta_var_out[TILE_SIZE * PADDED_TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int ki2 = ki * ki;
    int m = ki2 * fi;
    int n = woho * B;
    int index, tile_idx, tile_idx_x;

    int row_div_ki2 = row / ki2;
    int row_mod_ki2 = row % ki2;
    int wihi_fi = wihi * fi;
    int row_wihi = row_div_ki2 * wihi;

    // Precompute reciprocal of woho
    float inv_woho = 1.0f / woho;

    for (int phase = 0; phase < ceil((float)n / TILE_SIZE); phase++) {
        index = phase * TILE_SIZE + tx;

        int index_div_woho = __float2int_rd(index * inv_woho);
        int index_mod_woho = index - index_div_woho * woho;
        int aidx_base = ki2 * index_mod_woho;
        int aidx_offset = row_wihi + index_div_woho * wihi_fi - 1;

        tile_idx = ty * TILE_SIZE + tx;
        tile_idx_x = tx * PADDED_TILE_SIZE + ty;

        int aidx_tmp =
            (row < m && index < n) ? __ldg(&aidx[aidx_base + row_mod_ki2]) : -1;
        float mu_a_val = 0.0f;

        if (aidx_tmp > -1) {
            aidx_tmp += aidx_offset;
            mu_a_val = __ldg(&mu_a[aidx_tmp]);
        }

        s_mu_a[tile_idx] = mu_a_val;

        float delta_mu_out_val =
            (ty + phase * TILE_SIZE < n && col < k)
                ? __ldg(&delta_mu_out[col * n + ty + phase * TILE_SIZE])
                : 0.0f;
        float delta_var_out_val =
            (ty + phase * TILE_SIZE < n && col < k)
                ? __ldg(&delta_var_out[col * n + ty + phase * TILE_SIZE])
                : 0.0f;

        s_delta_mu_out[tile_idx_x] = delta_mu_out_val;
        s_delta_var_out[tile_idx_x] = delta_var_out_val;

        __syncthreads();

        // Perform computation using shared memory
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            float mu_a_tmp = s_mu_a[ty * TILE_SIZE + i];
            float delta_mu_tmp = s_delta_mu_out[tx * PADDED_TILE_SIZE + i];
            float delta_var_tmp = s_delta_var_out[tx * PADDED_TILE_SIZE + i];

            sum_mu += mu_a_tmp * delta_mu_tmp;
            sum_var += mu_a_tmp * delta_var_tmp * mu_a_tmp;
        }

        __syncthreads();
    }
    if (col < k && row < m) {
        float var_w_tmp = var_w[col * m + row];
        delta_mu_w[col * m + row] = sum_mu * var_w_tmp;
        delta_var_w[col * m + row] = sum_var * var_w_tmp * var_w_tmp;
    }
}

template <typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y>
__global__ void conv2d_bwd_delta_b_cuda_v1(const T *var_b,
                                           const T *delta_mu_out,
                                           const T *delta_var_out, int woho,
                                           int fo, int batch_size,
                                           T *delta_mu_b, T *delta_var_b)
/*
 */
{
    __shared__ T smem_mu[BLOCK_TILE_Y][BLOCK_TILE_X];
    __shared__ T smem_var[BLOCK_TILE_Y][BLOCK_TILE_X];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t col = blockIdx.x * BLOCK_TILE_X + tx;
    const size_t row = blockIdx.y * BLOCK_TILE_Y + ty;

    const size_t idx = row * woho * batch_size + col;

    if (col < woho * batch_size && row < fo) {
        smem_mu[ty][tx] = delta_mu_out[idx];
        smem_var[ty][tx] = delta_var_out[idx];
    } else {
        smem_mu[ty][tx] = static_cast<T>(0);
        smem_var[ty][tx] = static_cast<T>(0);
    }

    __syncthreads();
    for (size_t i = BLOCK_TILE_Y / 2; i > WARP_SIZE; i >>= 1) {
        if (tx < i) {
            smem_mu[ty][tx] += smem_mu[ty][tx + i];
            smem_var[ty][tx] += smem_var[ty][tx + i];
        }
        __syncthreads();
    }

    if (tx < WARP_SIZE) {
        T mu_x = smem_mu[ty][tx];
        T var_x = smem_var[ty][tx];

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

    if (tx == 0 && row < fo) {
        delta_mu_b[row * gridDim.x + blockIdx.x] = smem_mu[ty][tx] * var_b[row];
        delta_var_b[row * gridDim.x + blockIdx.x] =
            var_b[row] * smem_var[ty][tx] * var_b[row];
    }
}

template <typename T>
void conv2d_bwd_delta_b_dual_sum_reduction(T *&var_b, T *&delta_mu_out,
                                           T *&delta_var_out, int batch_size,
                                           int woho, int fo, T *&buf_mu_in,
                                           T *&buf_var_in, T *&buf_mu_out,
                                           T *&buf_var_out, T *&delta_mu,
                                           T *&delta_var)
/*
 */
{
    // Kernel config TODO: remove this hard code
    constexpr unsigned int BLOCK_SIZE_X = 64U;
    constexpr unsigned int BLOCK_SIZE_Y = 16U;
    const dim3 block_dim_rd(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1U);
    unsigned int grid_size_y = (fo + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    unsigned int grid_size_x =
        (batch_size * woho + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    dim3 grid_dim_rd(grid_size_x, grid_size_y, 1U);
    size_t reduced_size = grid_size_x;

    // Stage 1: Perform 1st custom sum reduction
    conv2d_bwd_delta_b_cuda_v1<T, BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_rd, block_dim_rd>>>(var_b, delta_mu_out, delta_var_out,
                                        woho, fo, batch_size, buf_mu_out,
                                        buf_var_out);

    // Stage 2: Perform recursive reduction sum
    while (grid_size_x > BLOCK_SIZE_X) {
        grid_size_x = (grid_size_x + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
        grid_dim_rd.x = grid_size_x;

        dual_sum_reduction_v2<T, BLOCK_SIZE_X, BLOCK_SIZE_Y>
            <<<grid_dim_rd, block_dim_rd>>>(buf_mu_out, buf_var_out,
                                            reduced_size, fo, buf_mu_in,
                                            buf_var_in);

        // Swap buffers
        std::swap(buf_mu_in, buf_mu_out);
        std::swap(buf_var_in, buf_var_out);

        reduced_size = grid_size_x;
    }

    // Stage 3: Perform final reduction sum
    dim3 grid_dim_1b(1, grid_size_y);
    dual_sum_reduction_v2<T, BLOCK_SIZE_X, BLOCK_SIZE_Y>
        <<<grid_dim_1b, block_dim_rd>>>(buf_mu_out, buf_var_out, reduced_size,
                                        fo, delta_mu, delta_var);
}
////////////////////////////////////////////////////////////////////////////////
// STATE BACKWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
template <typename T, size_t TILE_SIZE, size_t SMEM_PADDING>
__global__ void conv2d_bwd_delta_z_cuda_v1(
    const T *__restrict__ mu_w, const T *__restrict__ jcb,
    const T *__restrict__ delta_mu_out, const T *__restrict__ delta_var_out,
    const int *__restrict__ zw_idx, const int *__restrict__ zud_idx, int woho,
    int fo, int wihi, int fi, int ki, int nr, int n, int B, int pad_param_idx,
    T *__restrict__ delta_mu, T *__restrict__ delta_var)
/* Compute updated quantities of the mean of hidden states for convolutional
 layer.
 */

{
    constexpr unsigned int PADDED_TILE_SIZE = TILE_SIZE + SMEM_PADDING;
    __shared__ T s_delta_mu_out[TILE_SIZE][PADDED_TILE_SIZE];
    __shared__ T s_delta_var_out[TILE_SIZE][PADDED_TILE_SIZE];
    __shared__ int s_zw_idx[TILE_SIZE][PADDED_TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int widx_tmp, aidx_tmp;
    int k = wihi * B;
    int ki2 = ki * ki;
    int col_mod_wihi = col % wihi;
    int col_div_wihi = col / wihi;
    int aidx_sup = col_div_wihi * woho * fo;
    float inv_nr = 1.0 / nr;

    for (int phase = 0; phase < (n - 1) / TILE_SIZE + 1; phase++) {
        const unsigned int index_y = phase * TILE_SIZE + ty;
        const unsigned int index_y_div_nr = __float2int_rd(index_y * inv_nr);
        const unsigned int index_y_mod_nr = index_y - index_y_div_nr * nr;

        s_delta_mu_out[tx][ty] = 0.0;
        s_delta_var_out[tx][ty] = 0.0;
        s_zw_idx[tx][ty] = 0.0;

        if (col < k && index_y < n) {
            // NOTE: all index vectors starts with 1 not 0
            aidx_tmp = __ldg(&zud_idx[col_mod_wihi + wihi * index_y_mod_nr]);
            widx_tmp = __ldg(&zw_idx[col_mod_wihi * nr + index_y_mod_nr]);
            widx_tmp += index_y_div_nr * ki2 * fi;
            if (widx_tmp < pad_param_idx) {
                s_zw_idx[tx][ty] = widx_tmp;
            }
            if (aidx_tmp > -1) {
                aidx_tmp += index_y_div_nr * woho + aidx_sup - 1;
                s_delta_mu_out[tx][ty] = __ldg(&delta_mu_out[aidx_tmp]);
                s_delta_var_out[tx][ty] = __ldg(&delta_var_out[aidx_tmp]);
            }
        }
        __syncthreads();

        float tmp_mu_w = 0;
        for (int i = 0; i < TILE_SIZE; i++) {
            int tmp_zw_idx = s_zw_idx[tx][i];

            // TODO: mu_w must be loaded in shared memory for futher speedup
            if (tmp_zw_idx != 0) {
                tmp_zw_idx += row * ki2;
                tmp_mu_w = tmp_zw_idx < pad_param_idx ? mu_w[tmp_zw_idx - 1]
                                                      : static_cast<T>(0);
            } else {
                tmp_mu_w = 0.0;
            }

            sum_mu += tmp_mu_w * s_delta_mu_out[tx][i];
            sum_var += tmp_mu_w * tmp_mu_w * s_delta_var_out[tx][i];
        }
        __syncthreads();
    }

    if (col < k && row < fi) {
        int idx_out = wihi * col_div_wihi * fi + col_mod_wihi + row * wihi;
        delta_mu[idx_out] = sum_mu * jcb[row * k + col];
        delta_var[idx_out] = sum_var * jcb[row * k + col] * jcb[row * k + col];
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING>
__global__ void conv2d_bwd_delta_z_cuda_v2(
    const T *__restrict__ mu_w, const T *__restrict__ jcb,
    const T *__restrict__ delta_mu_out, const T *__restrict__ delta_var_out,
    const int *__restrict__ zw_idx, const int *__restrict__ zud_idx, int woho,
    int fo, int wihi, int fi, int ki, int nr, int n, int B, int pad_param_idx,
    T *__restrict__ delta_mu, T *__restrict__ delta_var)
/* Compute updated quantities of the mean of hidden states for convolutional
 layer.
 */

{
    __shared__ T s_delta_mu_out[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_delta_var_out[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ int s_zw_idx[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    int k = wihi * B;
    int wihifi = wihi * fi;
    int wohofo = woho * fo;
    int ki2 = ki * ki;

    // Thread block
    const unsigned int thread_linear_idx =
        threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (n + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    // WARPS
    constexpr unsigned int WARPS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    const size_t warp_id = thread_linear_idx / WARP_SIZE;
    const size_t lane_id = thread_linear_idx % WARP_SIZE;
    const size_t warp_row = warp_id / WARPS_X;
    const size_t warp_col = warp_id % WARPS_X;
    const size_t thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const size_t thread_col_in_warp = lane_id % THREADS_PER_WARP_X;
    const size_t warp_row_coord =
        warp_row * WARP_TILE_Y + thread_row_in_warp * THREAD_TILE;
    const size_t warp_col_coord =
        warp_col * WARP_TILE_X + thread_col_in_warp * THREAD_TILE;
    const size_t row_base = blockIdx.y * BLOCK_TILE + warp_row_coord;
    const size_t col_base = blockIdx.x * BLOCK_TILE + warp_col_coord;

    // Register storage
    T tmp_mu[THREAD_TILE][THREAD_TILE] = {{static_cast<T>(0)}};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {{static_cast<T>(0)}};

    for (int phase = 0; phase < num_tiles; phase++) {
// zud_idx[fo x wihi] and zw_idx [wihi x nr]
#pragma unroll
        for (size_t l_i = 0; l_i < NUM_LOADS; l_i++) {
            // Delta mu and delta var
            const unsigned int thread_load_idx =
                thread_linear_idx + l_i * THREADS;
            const unsigned int d_ty = thread_load_idx / BLOCK_TILE;
            const unsigned int d_tx = thread_load_idx % BLOCK_TILE;
            const unsigned int idx_row = phase * BLOCK_TILE_K + d_ty;
            const unsigned int idx_col = blockIdx.x * BLOCK_TILE + d_tx;
            const unsigned int idx_row_mod_nr = idx_row % nr;
            const unsigned int idx_col_mod_wihi = idx_col % wihi;

            bool valid = idx_col < k && idx_row < n;
            int aidx_tmp =
                valid ? zud_idx[wihi * idx_row_mod_nr + idx_col_mod_wihi] : -1;
            int widx_tmp =
                valid ? zw_idx[idx_col_mod_wihi * nr + idx_row_mod_nr] : 0;

            bool valid_widx = valid && widx_tmp < pad_param_idx;
            s_zw_idx[d_ty][d_tx] =
                valid_widx ? widx_tmp + idx_row / nr * ki2 * fi : 0;

            bool valid_aidx = aidx_tmp > -1;
            aidx_tmp += valid_aidx *
                        ((idx_col / wihi) * wohofo + (idx_row / nr) * woho - 1);

            T delta_mu =
                valid_aidx ? __ldg(&delta_mu_out[aidx_tmp]) : static_cast<T>(0);
            T delta_var = valid_aidx ? __ldg(&delta_var_out[aidx_tmp])
                                     : static_cast<T>(0);

            s_delta_mu_out[d_ty][d_tx] = delta_mu;
            s_delta_var_out[d_ty][d_tx] = delta_var;
        }
        __syncthreads();

        for (int i = 0; i < BLOCK_TILE_K; i++) {
            T delta_mu_val[THREAD_TILE] = {static_cast<T>(0)};
            T delta_var_val[THREAD_TILE] = {static_cast<T>(0)};
            T mu_w_val[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};

#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j += 4) {
                float4 delta_mu_float4 = *reinterpret_cast<const float4 *>(
                    &s_delta_mu_out[i][warp_col_coord + j]);
                float4 delta_var_float4 = *reinterpret_cast<const float4 *>(
                    &s_delta_var_out[i][warp_col_coord + j]);

                reinterpret_cast<float4 &>(delta_mu_val[j]) = delta_mu_float4;
                reinterpret_cast<float4 &>(delta_var_val[j]) = delta_var_float4;
            }

            const int row_base_ki2 = row_base * ki2;
#pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                const int j_ki2 = j * ki2;
                const bool row_condition = row_base + j < fi;

#pragma unroll
                for (size_t t = 0; t < THREAD_TILE; t++) {
                    const int idx_val_t = s_zw_idx[i][warp_col_coord + t];
                    const int tmp_zw_idx = idx_val_t + row_base_ki2;

                    const bool condition = idx_val_t != 0 && row_condition &&
                                           (tmp_zw_idx + j_ki2 < pad_param_idx);
                    const int address =
                        condition ? tmp_zw_idx + j * ki2 - 1 : 0;
                    mu_w_val[t][j] =
                        condition ? __ldg(&mu_w[address]) : static_cast<T>(0);
                }
            }

#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    tmp_mu[t][j] = __fmaf_rn(mu_w_val[t][j], delta_mu_val[t],
                                             tmp_mu[t][j]);
                    tmp_var[t][j] = __fmaf_rn(mu_w_val[t][j] * mu_w_val[t][j],
                                              delta_var_val[t], tmp_var[t][j]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (size_t j = 0; j < THREAD_TILE; j++) {
        const size_t col = col_base + j;
        const size_t col_div_wihi = col / wihi;
        const size_t col_mod_wihi = col % wihi;
        const size_t idx_out_partial = col_div_wihi * wihifi + col_mod_wihi;

#pragma unroll
        for (size_t t = 0; t < THREAD_TILE; t++) {
            const size_t row = row_base + t;
            if (col < k && row < fi) {
                T jcb_val = jcb[row * k + col];
                const unsigned int idx_out = idx_out_partial + row * wihi;
                delta_mu[idx_out] = __fmul_rn(tmp_mu[j][t], jcb_val);
                delta_var[idx_out] =
                    __fmul_rn(jcb_val * jcb_val, tmp_var[j][t]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// PARAM BACKWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
template <size_t TILE_SIZE, size_t SMEM_PADDING>
__global__ void conv2d_bwd_delta_w_cuda_v1(
    float const *var_w, float const *mu_a, float const *delta_mu_out,
    float const *delta_var_out, int const *aidx, int B, int k, int woho,
    int wihi, int fi, int ki, float *delta_mu_w, float *delta_var_w)
/**/
{
    const int PADDED_TILE_SIZE = TILE_SIZE + SMEM_PADDING;
    __shared__ float s_mu_a[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_delta_mu_out[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_delta_var_out[TILE_SIZE * PADDED_TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int ki2 = ki * ki;
    int m = ki2 * fi;
    int n = woho * B;
    int index, tile_idx, tile_idx_x;

    int col_div_ki2 = col / ki2;
    int col_mod_ki2 = col % ki2;
    int wihi_fi = wihi * fi;
    int col_wihi = col_div_ki2 * wihi;

    // Precompute reciprocal of woho
    float inv_woho = 1.0f / woho;

    for (int phase = 0; phase < ceil((float)n / TILE_SIZE); phase++) {
        index = phase * TILE_SIZE + ty;

        int index_div_woho = __float2int_rd(index * inv_woho);
        int index_mod_woho = index - index_div_woho * woho;
        int aidx_base = ki2 * index_mod_woho;
        int aidx_offset = col_wihi + index_div_woho * wihi_fi - 1;

        tile_idx = tx * PADDED_TILE_SIZE + ty;
        tile_idx_x = ty * TILE_SIZE + tx;

        int aidx_tmp =
            (col < m && index < n) ? __ldg(&aidx[aidx_base + col_mod_ki2]) : -1;
        float mu_a_val = 0.0f;

        if (aidx_tmp > -1) {
            aidx_tmp += aidx_offset;
            mu_a_val = __ldg(&mu_a[aidx_tmp]);
        }

        s_mu_a[tile_idx] = mu_a_val;

        float delta_mu_out_val =
            (tx + phase * TILE_SIZE < n && row < k)
                ? __ldg(&delta_mu_out[row * n + tx + phase * TILE_SIZE])
                : 0.0f;
        float delta_var_out_val =
            (tx + phase * TILE_SIZE < n && row < k)
                ? __ldg(&delta_var_out[row * n + tx + phase * TILE_SIZE])
                : 0.0f;

        s_delta_mu_out[tile_idx_x] = delta_mu_out_val;
        s_delta_var_out[tile_idx_x] = delta_var_out_val;

        __syncthreads();

        // Perform computation using shared memory
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            float mu_a_tmp = s_mu_a[tx * PADDED_TILE_SIZE + i];
            float delta_mu_tmp = s_delta_mu_out[ty * TILE_SIZE + i];
            float delta_var_tmp = s_delta_var_out[ty * TILE_SIZE + i];
            float mu_a_tmp_squared = mu_a_tmp * mu_a_tmp;

            sum_mu += mu_a_tmp * delta_mu_tmp;
            sum_var += mu_a_tmp_squared * delta_var_tmp;
        }

        __syncthreads();
    }
    if (row < k && col < m) {
        int out_idx = row * m + col;
        float var_w_tmp = var_w[out_idx];
        delta_mu_w[out_idx] = sum_mu * var_w_tmp;
        delta_var_w[out_idx] = sum_var * var_w_tmp * var_w_tmp;
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING>
__global__ void conv2d_bwd_delta_w_cuda_v2(
    const T *__restrict__ var_w, const T *__restrict__ mu_a,
    const T *__restrict__ delta_mu_out, const T *__restrict__ delta_var_out,
    const int *__restrict__ aidx, int B, int k, int woho, int wihi, int fi,
    int ki, T *__restrict__ delta_mu_w, T *__restrict__ delta_var_w)
/**/
{
    __shared__ T s_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_delta_mu_out[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_delta_var_out[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    int ki2 = ki * ki;
    int m = ki2 * fi;
    int n = woho * B;

    // Thread block
    const unsigned int thread_linear_idx =
        threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (n + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    // WARPS
    constexpr unsigned int WAPRS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    const unsigned int warp_id = thread_linear_idx / WARP_SIZE;
    const unsigned int lane_id = thread_linear_idx % WARP_SIZE;
    const unsigned int warp_row = warp_id / WAPRS_X;
    const unsigned int warp_col = warp_id % WAPRS_X;
    const unsigned int thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const unsigned int thread_col_in_warp = lane_id % THREADS_PER_WARP_X;
    const unsigned int warp_row_coord =
        warp_row * WARP_TILE_Y + thread_row_in_warp * THREAD_TILE;
    const unsigned int warp_col_coord =
        warp_col * WARP_TILE_X + thread_col_in_warp * THREAD_TILE;

    // Register storage
    T tmp_mu[THREAD_TILE][THREAD_TILE] = {{static_cast<T>(0)}};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {{static_cast<T>(0)}};

    float inv_woho = __frcp_rn(static_cast<float>(woho));

    for (int phase = 0; phase < num_tiles; phase++) {
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            const int thread_load_idx = thread_linear_idx + l_i * THREADS;
            const int a_ty = thread_load_idx / BLOCK_TILE;
            const int a_tx = thread_load_idx & (BLOCK_TILE - 1);

            const int a_idx_row = phase * BLOCK_TILE_K + a_ty;
            const int a_idx_col = blockIdx.x * BLOCK_TILE + a_tx;

            const int index_div_woho = __float2int_rd(a_idx_row * inv_woho);
            const int index_mod_woho = a_idx_row - index_div_woho * woho;
            const int a_idx_offset =
                a_idx_col / ki2 * wihi + index_div_woho * wihi * fi - 1;

            int aidx_tmp =
                (a_idx_col < m && a_idx_row < n)
                    ? __ldg(&aidx[index_mod_woho * ki2 + a_idx_col % ki2])
                    : -1;
            T mu_a_val =
                (aidx_tmp > -1) ? __ldg(&mu_a[aidx_tmp + a_idx_offset]) : 0.0f;

            s_mu_a[a_ty][a_tx] = mu_a_val;

            // Weights
            const int d_ty = thread_load_idx / BLOCK_TILE;
            const int d_tx = thread_load_idx & (BLOCK_TILE - 1);
            const int d_row = blockIdx.y * BLOCK_TILE + d_tx;
            const int d_col = phase * BLOCK_TILE_K + d_ty;

            float delta_mu_out_val =
                (d_col < n && d_row < k)
                    ? __ldg(&delta_mu_out[d_row * n + d_col])
                    : 0.0f;
            float delta_var_out_val =
                (d_col < n && d_row < k)
                    ? __ldg(&delta_var_out[d_row * n + d_col])
                    : 0.0f;

            s_delta_mu_out[d_ty][d_tx] = delta_mu_out_val;
            s_delta_var_out[d_ty][d_tx] = delta_var_out_val;
        }
        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            T delta_mu_val[THREAD_TILE] = {static_cast<T>(0)},
              delta_var_val[THREAD_TILE] = {static_cast<T>(0)};
            T mu_a_val[THREAD_TILE] = {static_cast<T>(0)};

#pragma unroll
            for (size_t j = 0; j < THREAD_TILE; j++) {
                mu_a_val[j] = s_mu_a[i][warp_col_coord + j];
                delta_mu_val[j] = s_delta_mu_out[i][warp_row_coord + j];
                delta_var_val[j] = s_delta_var_out[i][warp_row_coord + j];
            }

#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    const T mu_a_val_j = mu_a_val[j];
                    tmp_mu[t][j] =
                        __fmaf_rn(mu_a_val_j, delta_mu_val[t], tmp_mu[t][j]);
                    tmp_var[t][j] = __fmaf_rn(mu_a_val_j * mu_a_val_j,
                                              delta_mu_val[t], tmp_var[t][j]);
                }
            }
        }
        __syncthreads();
    }

    const size_t row_base = blockIdx.y * BLOCK_TILE + warp_row_coord;
    const size_t col_base = blockIdx.x * BLOCK_TILE + warp_col_coord;

#pragma unroll
    for (size_t t = 0; t < THREAD_TILE; t++) {
        const size_t row = row_base + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_TILE; j++) {
            const size_t col = col_base + j;
            if (row < k && col < m) {
                const T var_w_tmp = var_w[row * m + col];
                delta_mu_w[row * m + col] = __fmul_rn(tmp_mu[t][j], var_w_tmp);
                delta_var_w[row * m + col] =
                    __fmul_rn(var_w_tmp * var_w_tmp, tmp_var[t][j]);
            }
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t PACK_SIZE_T, size_t SMEM_PADDING>
__global__ void conv2d_bwd_delta_w_cuda_v3(
    const T *__restrict__ var_w, const T *__restrict__ mu_a,
    const T *__restrict__ delta_mu_out, const T *__restrict__ delta_var_out,
    const int *__restrict__ aidx, int B, int k, int woho, int wihi, int fi,
    int ki, T *__restrict__ delta_mu_w, T *__restrict__ delta_var_w)
/**/
{
    __shared__ T s_mu_a[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_delta_mu_out[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];
    __shared__ T s_delta_var_out[BLOCK_TILE_K][BLOCK_TILE + SMEM_PADDING];

    int ki2 = ki * ki;
    int m = ki2 * fi;
    int n = woho * B;
    float inv_woho = __frcp_rn(static_cast<float>(woho));

    // Thread block
    const unsigned int thread_linear_idx =
        threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (n + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int THREAD_PACKS = THREAD_TILE / PACK_SIZE_T;
    constexpr unsigned int XY_PACKS = BLOCK_TILE / PACK_SIZE_T;
    constexpr unsigned int K_PACKS = BLOCK_TILE_K / PACK_SIZE_T;
    constexpr unsigned int NUM_LOADS =
        (XY_PACKS * BLOCK_TILE_K + THREADS - 1) / THREADS;
    constexpr unsigned int NUM_LOADS_A =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    // WARPS
    constexpr unsigned int WAPRS_X = BLOCK_TILE / WARP_TILE_X;
    constexpr unsigned int THREADS_PER_WARP_X = WARP_TILE_X / THREAD_TILE;
    const unsigned int warp_id = thread_linear_idx / WARP_SIZE;
    const unsigned int lane_id = thread_linear_idx % WARP_SIZE;
    const unsigned int warp_row = warp_id / WAPRS_X;
    const unsigned int warp_col = warp_id % WAPRS_X;
    const unsigned int thread_row_in_warp = lane_id / THREADS_PER_WARP_X;
    const unsigned int thread_col_in_warp = lane_id % THREADS_PER_WARP_X;
    const unsigned int warp_row_coord =
        warp_row * WARP_TILE_Y + thread_row_in_warp * THREAD_TILE;
    const unsigned int warp_col_coord =
        warp_col * WARP_TILE_X + thread_col_in_warp * THREAD_TILE;

    // Register storage
    T tmp_mu[THREAD_TILE][THREAD_TILE] = {{static_cast<T>(0)}};
    T tmp_var[THREAD_TILE][THREAD_TILE] = {{static_cast<T>(0)}};

    for (int phase = 0; phase < num_tiles; phase++) {
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS_A; l_i++) {
            const int thread_load_idx = thread_linear_idx + l_i * THREADS;
            const int a_ty = thread_load_idx / BLOCK_TILE;
            const int a_tx = thread_load_idx & (BLOCK_TILE - 1);

            const int a_idx_row = phase * BLOCK_TILE_K + a_ty;
            const int a_idx_col = blockIdx.x * BLOCK_TILE + a_tx;

            const int index_div_woho = __float2int_rd(a_idx_row * inv_woho);
            const int index_mod_woho = a_idx_row - index_div_woho * woho;
            const int a_idx_offset =
                a_idx_col / ki2 * wihi + index_div_woho * wihi * fi - 1;

            int aidx_tmp =
                (a_idx_col < m && a_idx_row < n)
                    ? __ldg(&aidx[index_mod_woho * ki2 + a_idx_col % ki2])
                    : -1;
            T mu_a_val =
                (aidx_tmp > -1) ? __ldg(&mu_a[aidx_tmp + a_idx_offset]) : 0.0f;

            s_mu_a[a_ty][a_tx] = mu_a_val;
        }
#pragma unroll
        for (int l_i = 0; l_i < NUM_LOADS; l_i++) {
            // Weights
            const int thread_load_idx = thread_linear_idx + l_i * THREADS;
            const int d_ty = thread_load_idx / K_PACKS;
            const int d_tx = (thread_load_idx % K_PACKS) * PACK_SIZE_T;
            const int d_row = blockIdx.y * BLOCK_TILE + d_ty;
            const int d_col = phase * BLOCK_TILE_K + d_tx;

            float4 delta_mu_out_row_val = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 delta_var_out_row_val = {0.0f, 0.0f, 0.0f, 0.0f};
            if (d_col < n && d_row < k) {
                delta_mu_out_row_val = *reinterpret_cast<const float4 *>(
                    &delta_mu_out[d_row * n + d_col]);
                delta_var_out_row_val = *reinterpret_cast<const float4 *>(
                    &delta_var_out[d_row * n + d_col]);
            }
#pragma unroll
            for (size_t p = 0; p < PACK_SIZE_T; p++) {
                s_delta_mu_out[d_tx + p][d_ty] =
                    reinterpret_cast<const T *>(&delta_mu_out_row_val)[p];
                s_delta_var_out[d_tx + p][d_ty] =
                    reinterpret_cast<const T *>(&delta_var_out_row_val)[p];
            }
        }
        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            T delta_mu_val[THREAD_TILE] = {static_cast<T>(0)},
              delta_var_val[THREAD_TILE] = {static_cast<T>(0)};
            T mu_a_val[THREAD_TILE] = {static_cast<T>(0)};

#pragma unroll
            for (size_t j = 0; j < THREAD_PACKS; j++) {
                *reinterpret_cast<float4 *>(&mu_a_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_mu_a[i][warp_col_coord + j * PACK_SIZE_T]);
                *reinterpret_cast<float4 *>(&delta_mu_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_delta_mu_out[i][warp_row_coord + j * PACK_SIZE_T]);
                *reinterpret_cast<float4 *>(&delta_var_val[j * PACK_SIZE_T]) =
                    *reinterpret_cast<const float4 *>(
                        &s_delta_var_out[i][warp_row_coord + j * PACK_SIZE_T]);
            }

#pragma unroll
            for (size_t t = 0; t < THREAD_TILE; t++) {
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    const T mu_a_val_j = mu_a_val[j];
                    tmp_mu[t][j] =
                        __fmaf_rn(mu_a_val_j, delta_mu_val[t], tmp_mu[t][j]);
                    tmp_var[t][j] = __fmaf_rn(mu_a_val_j * mu_a_val_j,
                                              delta_mu_val[t], tmp_var[t][j]);
                }
            }
        }
        __syncthreads();
    }

    const size_t row_base = blockIdx.y * BLOCK_TILE + warp_row_coord;
    const size_t col_base = blockIdx.x * BLOCK_TILE + warp_col_coord;

#pragma unroll
    for (size_t t = 0; t < THREAD_TILE; t++) {
        const size_t row = row_base + t;
#pragma unroll
        for (size_t j = 0; j < THREAD_PACKS; j++) {
            const size_t col = col_base + j * PACK_SIZE_T;
            if (row < k && col < m) {
                const float4 var_w_tmp =
                    *reinterpret_cast<const float4 *>(&var_w[row * m + col]);
                float4 tmp_mu_row_val = *reinterpret_cast<const float4 *>(
                    &tmp_mu[t][j * PACK_SIZE_T]);
                float4 tmp_var_row_val = *reinterpret_cast<const float4 *>(
                    &tmp_var[t][j * PACK_SIZE_T]);

                reinterpret_cast<float4 *>(&delta_mu_w[row * m + col])[0] =
                    make_float4(__fmul_rn(tmp_mu_row_val.x, var_w_tmp.x),
                                __fmul_rn(tmp_mu_row_val.y, var_w_tmp.y),
                                __fmul_rn(tmp_mu_row_val.z, var_w_tmp.z),
                                __fmul_rn(tmp_mu_row_val.w, var_w_tmp.w));

                reinterpret_cast<float4 *>(&delta_var_w[row * m + col])[0] =
                    make_float4(
                        __fmul_rn(var_w_tmp.x * var_w_tmp.x, tmp_var_row_val.x),
                        __fmul_rn(var_w_tmp.y * var_w_tmp.y, tmp_var_row_val.y),
                        __fmul_rn(var_w_tmp.z * var_w_tmp.z, tmp_var_row_val.z),
                        __fmul_rn(var_w_tmp.w * var_w_tmp.w,
                                  tmp_var_row_val.w));
            }
        }
    }
}
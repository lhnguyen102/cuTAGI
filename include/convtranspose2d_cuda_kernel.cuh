#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "common_cuda_kernel.cuh"
#include "config.h"

////////////////////////////////////////////////////////////////////////////////
// FORWARD
////////////////////////////////////////////////////////////////////////////////
__global__ void convtranspose2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a, int const *widx,
    int const *aidx, int woho, int fo, int wihi, int fi, int ki, int rf,
    int batch_size, bool bias, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < woho * fo && row < batch_size)  // k = woho * fo
    {
        float sum_mu = 0;
        float sum_var = 0;
        int aidx_tmp = 0;
        int widx_tmp = 0;
        int div_idx = col / woho;
        int mod_idx = col % woho;

        for (int i = 0; i < rf * fi; i++)  // n = ? * fi
        {
            int i_div_rf = i / rf;

            // minus 1 due to the index starting at 1
            int tmp_idx = mod_idx * rf + i % rf;
            widx_tmp = widx[tmp_idx];
            aidx_tmp = aidx[tmp_idx];

            if (aidx_tmp > -1 && widx_tmp > -1) {
                widx_tmp += div_idx * ki * ki + i_div_rf * ki * ki * fo - 1;
                aidx_tmp += row * wihi * fi + i_div_rf * wihi - 1;

                sum_mu += mu_w[widx_tmp] * mu_a[aidx_tmp];

                sum_var += (mu_w[widx_tmp] * mu_w[widx_tmp] + var_w[widx_tmp]) *
                               var_a[aidx_tmp] +
                           var_w[widx_tmp] * mu_a[aidx_tmp] * mu_a[aidx_tmp];
            }
        }

        mu_z[col + row * woho * fo] = sum_mu;
        var_z[col + row * woho * fo] = sum_var;
        if (bias) {
            mu_z[col + row * woho * fo] += mu_b[div_idx];
            var_z[col + row * woho * fo] += var_b[div_idx];
        }
    }
}

template <typename T, size_t BLOCK_TILE, size_t SMEM_PADDING>
__global__ void convtranspose2d_fwd_mean_var_cuda_v1(
    const T *mu_w, const T *var_w, const T *mu_b, const T *var_b, const T *mu_a,
    const T *var_a, const int *widx, const int *aidx, int woho, int fo,
    int wihi, int fi, int ki, int rf, int batch_size, bool bias, float *mu_z,
    float *var_z)
/**/
{
    constexpr size_t PADDED_BLOCK_SIZE = BLOCK_TILE + SMEM_PADDING;
    __shared__ T s_mu_w[BLOCK_TILE][PADDED_BLOCK_SIZE];
    __shared__ T s_var_w[BLOCK_TILE][PADDED_BLOCK_SIZE];
    __shared__ int s_aidx[BLOCK_TILE][PADDED_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_TILE + threadIdx.y;
    int col = blockIdx.x * BLOCK_TILE + threadIdx.x;

    const size_t num_tiles = (rf * fi + BLOCK_TILE - 1) / BLOCK_TILE;
    const size_t col_div_woho = col / woho;
    const size_t col_mod_woho = col % woho;
    const size_t idx_pos_base = col_mod_woho * rf;
    const size_t col_div_woho_ki2 = col_div_woho * ki * ki;
    const size_t ki2_fo = ki * ki * fo;

    T sum_mu = static_cast<T>(0);
    T sum_var = static_cast<T>(0);
#pragma unroll
    for (size_t phase = 0; phase < num_tiles; phase++) {
        const size_t tile_index_y = phase * BLOCK_TILE + ty;
        const bool valid_tile = (tile_index_y < rf * fi) && (col < woho * fo);
        const size_t idx_pos = idx_pos_base + tile_index_y % rf;
        int widx_tmp = valid_tile ? widx[idx_pos] : -1;
        const bool valid_widx = widx_tmp > -1;

        const size_t rf_factor = tile_index_y / rf;
        const size_t widx_tmp_offset =
            valid_widx ? widx_tmp + col_div_woho_ki2 + rf_factor * ki2_fo - 1
                       : 0;

        s_mu_w[ty][tx] = valid_widx ? mu_w[widx_tmp_offset] : static_cast<T>(0);
        s_var_w[ty][tx] =
            valid_widx ? var_w[widx_tmp_offset] : static_cast<T>(0);

        const int aidx_tmp = valid_tile ? aidx[idx_pos] : -1;
        s_aidx[ty][tx] = (aidx_tmp > -1) ? aidx_tmp + rf_factor * wihi : 0;

        __syncthreads();
#pragma unroll
        for (size_t i = 0; i < BLOCK_TILE; i++) {
            int aidx_tmp_i = s_aidx[i][tx];

            bool valid_idx = aidx_tmp_i != 0 && row < batch_size;
            aidx_tmp_i = aidx_tmp_i + row * wihi * fi - 1;

            T tmp_mu_a = valid_idx ? mu_a[aidx_tmp_i] : static_cast<T>(0);
            T tmp_var_a = valid_idx ? var_a[aidx_tmp_i] : static_cast<T>(0);

            sum_mu = __fmaf_rn(s_mu_w[i][tx], tmp_mu_a, sum_mu);
            T var_term_1 =
                __fmaf_rn(s_mu_w[i][tx], s_mu_w[i][tx], s_var_w[i][tx]);
            T var_term_2 =
                __fmaf_rn(tmp_mu_a * tmp_mu_a, s_var_w[i][tx], sum_var);
            sum_var = __fmaf_rn(var_term_1, tmp_var_a, var_term_2);
        }
        __syncthreads();
    }
    if (col < woho * fo && row < batch_size) {
        T mu_b_val = bias ? mu_b[col_div_woho] : static_cast<T>(0);
        T var_b_val = bias ? var_b[col_div_woho] : static_cast<T>(0);
        mu_z[col + row * woho * fo] = __fadd_rn(sum_mu, mu_b_val);
        var_z[col + row * woho * fo] = __fadd_rn(sum_var, var_b_val);
    }
}

template <typename T, size_t BLOCK_TILE, size_t BLOCK_TILE_K,
          size_t THREAD_TILE, size_t THREADS, size_t WARP_TILE_X,
          size_t WARP_TILE_Y, size_t SMEM_PADDING, size_t PACK_SIZE>
__global__ void convtranspose2d_fwd_mean_var_cuda_v2(
    const T *mu_w, const T *var_w, const T *mu_b, const T *var_b, const T *mu_a,
    const T *var_a, const int *widx, const int *aidx, int woho, int fo,
    int wihi, int fi, int ki, int rf, int batch_size, bool bias, float *mu_z,
    float *var_z)
/**/
{
    constexpr size_t PADDED_BLOCK_SIZE = BLOCK_TILE + SMEM_PADDING;
    __shared__ T s_mu_w[BLOCK_TILE_K][PADDED_BLOCK_SIZE];
    __shared__ T s_var_w[BLOCK_TILE_K][PADDED_BLOCK_SIZE];
    __shared__ int s_aidx[BLOCK_TILE_K][PADDED_BLOCK_SIZE];

    size_t ki2 = ki * ki;

    // Thread block
    const unsigned int thread_linear_idx =
        threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int num_tiles = (rf * fi + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
    constexpr unsigned int NUM_LOADS =
        (BLOCK_TILE * BLOCK_TILE_K + THREADS - 1) / THREADS;

    // Warps
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

    // Loop over tiles
    for (size_t phase = 0; phase < num_tiles; phase++) {
        for (size_t l_i = 0; l_i < NUM_LOADS; l_i++) {
            const size_t thread_load_idx = l_i * THREADS + thread_linear_idx;
            const size_t tx = thread_load_idx % BLOCK_TILE;
            const size_t ty = thread_load_idx / BLOCK_TILE;
            const size_t tile_idx_y = phase * BLOCK_TILE_K + ty;
            const size_t idx_col = BLOCK_TILE * blockIdx.x + tx;
            const size_t idx_row_mod_rf = tile_idx_y % rf;
            const size_t idx_col_mod_woho = idx_col % woho;

            s_mu_w[ty][tx] = static_cast<T>(0);
            s_var_w[ty][tx] = static_cast<T>(0);
            s_aidx[ty][tx] = 0;

            if (tile_idx_y < rf * fi && idx_col < woho * fo) {
                const size_t idx_pos = idx_row_mod_rf + idx_col_mod_woho * rf;
                const int widx_tmp = widx[idx_pos];
                if (widx_tmp > -1) {
                    const size_t idx_w = (idx_col / woho) * ki2 +
                                         (tile_idx_y / rf) * ki2 * fo - 1;
                    s_mu_w[ty][tx] = mu_w[widx_tmp + idx_w];
                    s_var_w[ty][tx] = var_w[widx_tmp + idx_w];
                }

                const int aidx_tmp = aidx[idx_pos];
                if (aidx_tmp > -1) {
                    const size_t idx_a = (tile_idx_y / rf) * wihi;
                    s_aidx[ty][tx] = aidx_tmp + idx_a;
                }
            }
        }
        __syncthreads();
        const size_t row_base_wihifi = row_base * wihi * fi;
        for (size_t i = 0; i < BLOCK_TILE_K; i++) {
            T mu_w_val[THREAD_TILE] = {static_cast<T>(0)};
            T var_w_val[THREAD_TILE] = {static_cast<T>(0)};
            T mu_a_val[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
            T var_a_val[THREAD_TILE][THREAD_TILE] = {static_cast<T>(0)};
            for (size_t j = 0; j < THREAD_TILE; j += PACK_SIZE) {
                float4 mu_w_val_f4 = *reinterpret_cast<const float4 *>(
                    &s_mu_w[i][warp_col_coord + j]);
                float4 var_w_val_f4 = *reinterpret_cast<const float4 *>(
                    &s_var_w[i][warp_col_coord + j]);

                reinterpret_cast<float4 &>(mu_w_val[j]) = mu_w_val_f4;
                reinterpret_cast<float4 &>(var_w_val[j]) = var_w_val_f4;
            }
            for (size_t j = 0; j < THREAD_TILE; j++) {
                bool valid_row = (row_base + j < batch_size);
                for (size_t t = 0; t < THREAD_TILE; t++) {
                    const size_t aidx_tmp_t = s_aidx[i][warp_col_coord + t];
                    if (aidx_tmp_t != 0 && valid_row) {
                        const size_t aidx_tmp_j =
                            aidx_tmp_t + row_base_wihifi + j * wihi * fi;
                        mu_a_val[t][j] = mu_a[aidx_tmp_j - 1];
                        var_a_val[t][j] = var_a[aidx_tmp_j - 1];
                    }
                }
            }
            for (size_t t = 0; t < THREAD_TILE; t++) {
                for (size_t j = 0; j < THREAD_TILE; j++) {
                    const T mu_w_val_t = mu_w_val[t];
                    const T var_w_val_t = var_w_val[t];
                    const T mu_a_val_j = mu_a_val[t][j];
                    const T var_a_val_j = var_a_val[t][j];

                    tmp_mu[t][j] =
                        __fmaf_rn(mu_w_val_t, mu_a_val_j, tmp_mu[t][j]);

                    T var_term_1 =
                        __fmaf_rn(mu_w_val_t, mu_w_val_t, var_w_val_t);
                    T var_term_2 = __fmaf_rn(
                        var_w_val_t, mu_a_val_j * mu_a_val_j, tmp_var[t][j]);
                    tmp_var[t][j] =
                        __fmaf_rn(var_term_1, var_a_val_j, var_term_2);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (size_t j = 0; j < THREAD_TILE; j++) {
        const size_t row = row_base + j;
#pragma unroll
        for (size_t t = 0; t < THREAD_TILE; t++) {
            const size_t col = col_base + t;
            const size_t div_idx = col / woho;
            if (col < woho * fo && row < batch_size) {
                mu_z[col + row * woho * fo] = tmp_mu[t][j];
                var_z[col + row * woho * fo] = tmp_var[t][j];
                if (bias) {
                    mu_z[col + row * woho * fo] += mu_b[div_idx];
                    var_z[col + row * woho * fo] += var_b[div_idx];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// STATE BACKWARD
////////////////////////////////////////////////////////////////////////////////
__global__ void convtranspose2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    float const *delta_var_out, int const *widx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int rf, int batch_size, float *delta_mu,
    float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int K = wihi * fi;

    if (col < K && row < batch_size)  // k = wihi * fi, m = B
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        int widx_tmp;
        int zidx_tmp;                      // updated index (idxSzzUd)
        for (int i = 0; i < rf * fo; i++)  // n = ki2 * fo
        {
            // minus 1 due to the index starting at 1
            int tmp_idx = (col % wihi) * ki * ki + i % rf;
            widx_tmp = widx[tmp_idx];
            zidx_tmp = zidx[tmp_idx];

            if (zidx_tmp > -1 && widx_tmp > -1) {
                widx_tmp +=
                    (i / rf) * ki * ki + (col / wihi) * ki * ki * fo - 1;
                zidx_tmp += (i / rf) * woho + row * woho * fo - 1;

                sum_mu += delta_mu_out[zidx_tmp] * mu_w[widx_tmp];
                sum_var +=
                    mu_w[widx_tmp] * delta_var_out[zidx_tmp] * mu_w[widx_tmp];
            }
        }
        // TODO: Double check the definition zposIn
        delta_mu[col + row * K] = sum_mu * jcb[col + row * K];
        delta_var[col + row * K] =
            sum_var * jcb[col + row * K] * jcb[col + row * K];
    }
}

template <typename T, size_t BLOCK_TILE, size_t SMEM_PADDING>
__global__ void convtranspose2d_bwd_delta_z_cuda_v1(
    const T *mu_w, const T *jcb, const T *delta_mu_out, const T *delta_var_out,
    int const *widx, int const *zidx, int woho, int fo, int wihi, int fi,
    int ki, int rf, int batch_size, T *delta_mu, T *delta_var)
/*
 */
{
    constexpr size_t PADDED_TILE_SIZE = BLOCK_TILE + SMEM_PADDING;
    __shared__ T s_mu_w[BLOCK_TILE][PADDED_TILE_SIZE];
    __shared__ int s_zidx[BLOCK_TILE][PADDED_TILE_SIZE];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t row = blockIdx.y * BLOCK_TILE + threadIdx.y;  // B
    const size_t col = blockIdx.x * BLOCK_TILE + threadIdx.x;  // wihi *fi

    const size_t num_tiles = (rf * fo + BLOCK_TILE - 1) / BLOCK_TILE;
    const size_t col_div_wihi = col / wihi;
    const size_t col_mod_wihi = col % wihi;

    const size_t wihi_fi = wihi * fi;
    const size_t woho_fo = woho * fo;
    const size_t ki2 = ki * ki;
    T sum_mu = static_cast<T>(0);
    T sum_var = static_cast<T>(0);

    for (size_t phase = 0; phase < num_tiles; phase++) {
        const size_t tile_idx_y = phase * BLOCK_TILE + ty;
        const bool valid_tile_idx_y = (tile_idx_y < rf * fo) && (col < wihi_fi);
        const size_t idx_pos = col_mod_wihi * ki2 + tile_idx_y % rf;

        const int widx_tmp = valid_tile_idx_y ? widx[idx_pos] : -1;
        const int zidx_tmp = valid_tile_idx_y ? zidx[idx_pos] : -1;
        const bool valid_zidx = zidx_tmp > -1;
        const bool valid_widx = widx_tmp > -1;

        const size_t rf_factor = tile_idx_y / rf;
        const size_t widx_tmp_offset =
            valid_widx
                ? widx_tmp + rf_factor * ki2 + col_div_wihi * ki2 * fo - 1
                : 0;

        s_mu_w[ty][tx] = valid_widx ? mu_w[widx_tmp_offset] : static_cast<T>(0);
        s_zidx[ty][tx] = valid_zidx ? zidx_tmp + rf_factor * woho : 0;

        __syncthreads();

        for (size_t i = 0; i < BLOCK_TILE; i++) {
            int zidx_i = s_zidx[i][tx];
            bool valid_idx = zidx_i != 0 && row < batch_size;
            if (valid_idx) {
                sum_mu +=
                    delta_mu_out[zidx_i + row * woho_fo - 1] * s_mu_w[i][tx];
                sum_var += s_mu_w[i][tx] *
                           delta_var_out[zidx_i + row * woho_fo - 1] *
                           s_mu_w[i][tx];
            }
        }
        __syncthreads();
    }
    if (col < wihi_fi && row < batch_size) {
        delta_mu[col + row * wihi_fi] = sum_mu * jcb[col + row * wihi_fi];
        delta_var[col + row * wihi_fi] =
            sum_var * jcb[col + row * wihi_fi] * jcb[col + row * wihi_fi];
    }
}
////////////////////////////////////////////////////////////////////////////////
// PARAM BACKWARD
////////////////////////////////////////////////////////////////////////////////

__global__ void convtranspose2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *delta_mu_out,
    float const *delta_var_out, int const *aidx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/**/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int K = ki * ki * fo;
    int ki2 = ki * ki;
    if (col < K && row < fi)  // m = fi, k = ki2 * fo
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        int zidx_tmp;  // updated index
        int aidx_tmp;
        int col_div_ki2 = col / ki2;
        int col_mod_ki2 = col % ki2;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi * B
        {
            int i_div_wihi = i / wihi;
            int i_mod_wihi = i % wihi;

            // minus 1 due to the index starting at 1
            aidx_tmp = aidx[col_mod_ki2 * wihi + i_mod_wihi];

            if (aidx_tmp > -1) {
                // minus 1 due to the index starting at 1
                zidx_tmp = zidx[col_mod_ki2 * wihi + i_mod_wihi] +
                           col_div_ki2 * woho + i_div_wihi * woho * fo - 1;
                aidx_tmp += row * wihi + i_div_wihi * wihi * fi - 1;

                sum_mu += mu_a[aidx_tmp] * delta_mu_out[zidx_tmp];
                sum_var +=
                    mu_a[aidx_tmp] * mu_a[aidx_tmp] * delta_var_out[zidx_tmp];
            }
        }

        delta_mu_w[col + row * K] = sum_mu * var_w[col + row * K];
        delta_var_w[col + row * K] =
            sum_var * var_w[col + row * K] * var_w[col + row * K];
    }
}

template <typename T, size_t BLOCK_TILE, size_t SMEM_PADDING>
__global__ void convtranspose2d_bwd_delta_w_cuda_v1(
    const T *var_w, const T *mu_a, const T *delta_mu_out,
    const T *delta_var_out, const int *aidx, const int *zidx, int woho, int fo,
    int wihi, int fi, int ki, int batch_size, T *delta_mu_w, T *delta_var_w)
/**/
{
    constexpr size_t PADDED_BLOCK_TILE = BLOCK_TILE + SMEM_PADDING;
    __shared__ T s_delta_mu_out[BLOCK_TILE][PADDED_BLOCK_TILE];
    __shared__ T s_delta_var_out[BLOCK_TILE][PADDED_BLOCK_TILE];
    __shared__ int s_aidx[BLOCK_TILE][PADDED_BLOCK_TILE];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t row = blockIdx.y * BLOCK_TILE + threadIdx.y;  // fi
    const size_t col = blockIdx.x * BLOCK_TILE + threadIdx.x;  // ki2 * fo

    const size_t num_tiles = (wihi * batch_size + BLOCK_TILE - 1) / BLOCK_TILE;
    const size_t ki2 = ki * ki;
    const size_t ki2_fo = ki2 * fo;
    int col_div_ki2 = col / ki2;
    int col_mod_ki2 = col % ki2;

    T sum_mu = static_cast<T>(0);
    T sum_var = static_cast<T>(0);

    for (size_t phase = 0; phase < num_tiles; phase++) {
        const size_t tile_idx_y = phase * BLOCK_TILE + ty;
        const bool valid_tile =
            (tile_idx_y < wihi * batch_size) && (col < ki2_fo);
        const size_t idx_pos = col_mod_ki2 * wihi + tile_idx_y % wihi;
        const int zidx_tmp = valid_tile ? zidx[idx_pos] : -1;
        const int aidx_tmp = valid_tile ? aidx[idx_pos] : -1;
        const bool valid_zidx = zidx_tmp > -1;
        const bool valid_aidx = aidx_tmp > -1;

        const size_t wihi_factor = tile_idx_y / wihi;
        const size_t zidx_tmp_offset =
            valid_zidx
                ? zidx_tmp + col_div_ki2 * woho + wihi_factor * woho * fo - 1
                : 0;

        s_aidx[ty][tx] = valid_aidx ? aidx_tmp + wihi_factor * wihi * fi : 0;
        s_delta_mu_out[ty][tx] =
            valid_zidx ? delta_mu_out[zidx_tmp_offset] : static_cast<T>(0);
        s_delta_var_out[ty][tx] =
            valid_zidx ? delta_var_out[zidx_tmp_offset] : static_cast<T>(0);

        __syncthreads();

        for (size_t i = 0; i < BLOCK_TILE; i++) {
            int aidx_i = s_aidx[i][tx];
            const bool valid_aidx = aidx_i != 0 && row < fi;

            if (valid_aidx) {
                aidx_i += row * wihi - 1;
                sum_mu += mu_a[aidx_i] * s_delta_mu_out[i][tx];
                sum_var += mu_a[aidx_i] * mu_a[aidx_i] * s_delta_var_out[i][tx];
            }
        }

        __syncthreads();
    }
    if (col < ki2_fo && row < fi) {
        delta_mu_w[col + row * ki2_fo] = sum_mu * var_w[col + row * ki2_fo];
        delta_var_w[col + row * ki2_fo] =
            sum_var * var_w[col + row * ki2_fo] * var_w[col + row * ki2_fo];
    }
}

__global__ void convtranspose2d_bwd_delta_b_cuda(
    float const *var_b, float const *delta_mu_out, float const *delta_var_out,
    int woho, int fo, int batch_size, float *delta_mu_b, float *delta_var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < fo)  // k = fo, m = 1
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < woho * batch_size; i++)  // n = woho * B
        {
            int idx = col * woho + (i % woho) + (i / woho) * woho * fo;

            sum_mu += delta_mu_out[idx];
            sum_var += delta_var_out[idx];
        }

        delta_mu_b[col] = sum_mu * var_b[col];
        delta_var_b[col] = var_b[col] * sum_var * var_b[col];
    }
}

template <typename T, size_t BLOCK_TILE_X, size_t BLOCK_TILE_Y>
__global__ void convtranspose2d_bwd_delta_b_cuda_v1(
    const T *var_b, const T *delta_mu_out, const T *delta_var_out, int woho,
    int fo, int batch_size, T *delta_mu_b, T *delta_var_b)
/*
 */
{
    __shared__ T smem_mu[BLOCK_TILE_Y][BLOCK_TILE_X];
    __shared__ T smem_var[BLOCK_TILE_Y][BLOCK_TILE_X];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t col = blockIdx.x * BLOCK_TILE_X + tx;
    const size_t row = blockIdx.y * BLOCK_TILE_Y + ty;

    const size_t idx = (col / woho) * woho * fo + row * woho + (col % woho);

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
void convtranspose2d_bwd_delta_b_dual_sum_reduction(
    T *&var_b, T *&delta_mu_out, T *&delta_var_out, int batch_size, int woho,
    int fo, T *&buf_mu_in, T *&buf_var_in, T *&buf_mu_out, T *&buf_var_out,
    T *&delta_mu, T *&delta_var)
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
    convtranspose2d_bwd_delta_b_cuda_v1<T, BLOCK_SIZE_X, BLOCK_SIZE_Y>
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
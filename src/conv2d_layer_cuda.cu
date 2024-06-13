///////////////////////////////////////////////////////////////////////////////
// File:         conv2d_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 04, 2024
// Updated:      March 11, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../include/base_layer.h"
#include "../include/conv2d_layer.h"
#include "../include/conv2d_layer_cuda.cuh"
#include "../include/param_init.h"

////////////////////////////////////////////////////////////////////////////////
// CUDA Kernels
////////////////////////////////////////////////////////////////////////////////
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
                                        int wihi, int fi, int ki, int pad_idx,
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

__global__ void conv2d_fwd_mean_var_cuda_tiled(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a, int const *aidx,
    int woho, int fo, int wihi, int fi, int ki, int B, int pad_idx, bool bias,
    float *mu_z, float *var_z)
/*Compute mean of product WA for convolutional layer

Args:
*/
{
    const int TILE_SIZE = 16;
    const int PADDED_TILE_SIZE = TILE_SIZE + 2;
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

__global__ void conv2d_bwd_delta_z_cuda_tiled(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    const float *delta_var_out, int const *zw_idx, int const *zud_idx, int woho,
    int fo, int wihi, int fi, int ki, int nr, int n, int B, int pad_param_idx,
    float *delta_mu, float *delta_var)
/* Compute updated quantities of the mean of hidden states for convolutional
 layer.
 */

{
    const int TILE_SIZE = 16;
    const int PADDED_TILE_SIZE = TILE_SIZE + 2;
    __shared__ float s_delta_mu_out[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ float s_delta_var_out[TILE_SIZE * PADDED_TILE_SIZE];
    __shared__ int s_zw_idx[TILE_SIZE * PADDED_TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    int widx_tmp, aidx_tmp;
    int k = wihi * B;
    int ki2 = ki * ki;
    int index_y, tile_idx_y;
    int col_mod_wihi = col % wihi;
    int col_div_wihi = col / wihi;
    int aidx_sup = col_div_wihi * woho * fo;
    float inv_nr = 1.0 / nr;

    for (int phase = 0; phase < (n - 1) / TILE_SIZE + 1; phase++) {
        // index_x = phase * TILE_SIZE + tx;
        // tile_idx_x = ty * TILE_SIZE + tx;
        index_y = phase * TILE_SIZE + ty;
        tile_idx_y = tx * PADDED_TILE_SIZE + ty;
        int index_y_div_nr = __float2int_rd(index_y * inv_nr);
        int index_y_mod_nr = index_y - index_y_div_nr * nr;

        s_delta_mu_out[tile_idx_y] = 0.0;
        s_delta_var_out[tile_idx_y] = 0.0;
        s_zw_idx[tile_idx_y] = 0.0;

        if (col < k && index_y < n) {
            // NOTE: all index vectors starts with 1 not 0
            aidx_tmp = __ldg(&zud_idx[col_mod_wihi + wihi * index_y_mod_nr]);
            widx_tmp = __ldg(&zw_idx[col_mod_wihi * nr + index_y_mod_nr]);
            widx_tmp += index_y_div_nr * ki2 * fi;
            if (widx_tmp < pad_param_idx) {
                s_zw_idx[tile_idx_y] = widx_tmp;
            }
            if (aidx_tmp > -1) {
                aidx_tmp += index_y_div_nr * woho + aidx_sup - 1;
                s_delta_mu_out[tile_idx_y] = __ldg(&delta_mu_out[aidx_tmp]);
                s_delta_var_out[tile_idx_y] = __ldg(&delta_var_out[aidx_tmp]);
            }
        }

        __syncthreads();
#pragma unroll
        float tmp_mu_w = 0;
        for (int i = 0; i < TILE_SIZE; i++) {
            int tmp_zw_idx = s_zw_idx[tx * PADDED_TILE_SIZE + i];

            // TODO: mu_w must be loaded in shared memory for futher speedup
            if (tmp_zw_idx != 0) {
                tmp_zw_idx += row * ki2 - 1;
                tmp_mu_w = mu_w[tmp_zw_idx];
            } else {
                tmp_mu_w = 0.0;
            }

            sum_mu += tmp_mu_w * s_delta_mu_out[tx * PADDED_TILE_SIZE + i];
            sum_var += tmp_mu_w * tmp_mu_w *
                       s_delta_var_out[tx * PADDED_TILE_SIZE + i];
        }
        __syncthreads();
    }
    if (col < k && row < fi) {
        int idx_out = wihi * col_div_wihi * fi + col_mod_wihi + row * wihi;
        delta_mu[idx_out] = sum_mu * jcb[row * k + col];
        delta_var[idx_out] = sum_var * jcb[row * k + col] * jcb[row * k + col];
    }
}

__global__ void conv2d_bwd_delta_w_cuda_tiled_v3(
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

////////////////////////////////////////////////////////////////////////////////
// Conv2d Cuda Layer
////////////////////////////////////////////////////////////////////////////////

Conv2dCuda::Conv2dCuda(size_t in_channels, size_t out_channels,
                       size_t kernel_size, bool bias, int stride, int padding,
                       int padding_type, size_t in_width, size_t in_height,
                       float gain_w, float gain_b, std::string init_method)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/*
 */
{
    this->in_width = in_width;
    this->in_height = in_height;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->bias = bias;
}

Conv2dCuda::~Conv2dCuda() {
    cudaFree(d_idx_mwa_2);
    cudaFree(d_idx_cov_zwa_1);
    cudaFree(d_idx_var_z_ud);
}

std::string Conv2dCuda::get_layer_info() const {
    return "Conv2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

std::string Conv2dCuda::get_layer_name() const { return "Conv2dCuda"; }

LayerType Conv2dCuda::get_layer_type() const { return LayerType::Conv2d; };

void Conv2dCuda::compute_input_output_size(const InitArgs &args)
/*
 */
{
    if (this->in_height == 0 || this->in_height == 0) {
        this->in_width = args.width;
        this->in_height = args.height;
    }
    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride,
                                       this->in_width, this->in_height,
                                       this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void Conv2dCuda::get_number_param()

/* Get the number of parameters for conv. and tconv. layer.
 *
 * Args:
 *    kernel: Size of the receptive field
 *    fi: Number of filters for input image
 *    fo: Number of filters for output image
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = this->kernel_size * this->kernel_size * this->in_channels *
          this->out_channels;
    if (this->bias) {
        n_b = this->out_channels;
    } else {
        n_b = 0;
    }
    this->num_weights = n_w;
    this->num_biases = n_b;
}

void Conv2dCuda::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void Conv2dCuda::allocate_conv_index()
/*
 */
{
    cudaMalloc(&this->d_idx_mwa_2, this->idx_mwa_2.size() * sizeof(int));
    cudaMalloc(&this->d_idx_cov_zwa_1,
               this->idx_cov_zwa_1.size() * sizeof(int));
    cudaMalloc(&this->d_idx_var_z_ud, this->idx_var_z_ud.size() * sizeof(int));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void Conv2dCuda::conv_index_to_device()
/*
 */
{
    cudaMemcpy(this->d_idx_mwa_2, this->idx_mwa_2.data(),
               this->idx_mwa_2.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_cov_zwa_1, this->idx_cov_zwa_1.data(),
               this->idx_cov_zwa_1.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_var_z_ud, this->idx_var_z_ud.data(),
               this->idx_var_z_ud.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Host to device.");
    }
}

void Conv2dCuda::lazy_index_init()
/*
 */
{
    // Get precomputed conv indices
    int param_pad_idx =
        pow(this->kernel_size, 2) * this->in_channels * this->out_channels + 1;

    auto conv_idx = get_conv2d_idx(
        this->kernel_size, this->stride, this->in_width, this->in_height,
        this->out_width, this->out_height, this->padding, this->padding_type,
        -1, -1, param_pad_idx);

    this->idx_mwa_2 = conv_idx.Fmwa_2_idx;
    this->idx_cov_zwa_1 = conv_idx.FCzwa_1_idx;
    this->idx_var_z_ud = conv_idx.Szz_ud_idx;

    this->row_zw = conv_idx.h;
    this->col_z_ud = conv_idx.h;

    this->allocate_conv_index();
    this->conv_index_to_device();
}

void Conv2dCuda::forward(BaseHiddenStates &input_states,
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

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_2.size() == 0) {
        this->lazy_index_init();
    }

    // Assign output dimensions
    cu_output_states->width = this->out_width;
    cu_output_states->height = this->out_height;
    cu_output_states->depth = this->out_channels;
    cu_output_states->block_size = batch_size;
    cu_output_states->actual_size = this->output_size;

    // Launch kernel
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int woho_batch = woho * batch_size;
    int pad_idx = wihi * this->in_channels * batch_size + 1;

    int threads = this->num_cuda_threads;
    unsigned int grid_row = (this->out_channels + threads - 1) / threads;
    unsigned int grid_col = (woho_batch + threads - 1) / threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(threads, threads);

    conv2d_fwd_mean_var_cuda_tiled<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_idx_mwa_2,
        woho, this->out_channels, wihi, this->in_channels, this->kernel_size,
        batch_size, pad_idx, this->bias, cu_output_states->d_mu_a,
        cu_output_states->d_var_a);

    // Update backward state for inferring parameters
    if (this->training) {
        BackwardStateCuda *cu_bwd_states =
            dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states, *cu_bwd_states);
    }
}

void Conv2dCuda::state_backward(BaseBackwardStates &next_bwd_states,
                                BaseDeltaStates &input_delta_states,
                                BaseDeltaStates &output_delta_states,
                                BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(&next_bwd_states);
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;

    // Launch kernel
    int wihi = this->in_width * this->in_height;
    int woho = this->out_width * this->out_height;
    int row_zw_fo = this->row_zw * this->out_channels;
    int pad_idx = woho * this->out_channels * batch_size + 1;

    unsigned int grid_row_p = (batch_size + threads - 1) / threads;
    unsigned int grid_col_p =
        (wihi * this->in_channels + threads - 1) / threads;
    dim3 dim_grid_p(grid_col_p, grid_row_p);

    unsigned int grid_row = (this->in_channels + threads - 1) / threads;
    unsigned int grid_col = (wihi * batch_size + threads - 1) / threads;
    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(threads, threads);

    permmute_jacobian_cuda<<<dim_grid_p, dim_block>>>(
        cu_next_bwd_states->d_jcb, wihi, this->in_channels, batch_size,
        cu_temp_states->d_tmp_1);

    conv2d_bwd_delta_z_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, cu_temp_states->d_tmp_1,
        cu_input_delta_states->d_delta_mu, cu_input_delta_states->d_delta_var,
        this->d_idx_cov_zwa_1, this->d_idx_var_z_ud, woho, this->out_channels,
        wihi, this->in_channels, this->kernel_size, this->row_zw, row_zw_fo,
        batch_size, pad_idx, cu_output_delta_states->d_delta_mu,
        cu_output_delta_states->d_delta_var);
}

void Conv2dCuda::param_backward(BaseBackwardStates &next_bwd_states,
                                BaseDeltaStates &delta_states,
                                BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(&next_bwd_states);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    // Initalization
    int batch_size = delta_states.block_size;
    int threads = this->num_cuda_threads;

    // Lauch kernel
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int woho_batch = woho * batch_size;
    int wohofo = woho * this->out_channels;
    int pad_idx = wihi * this->in_channels * batch_size + 1;
    int ki2_fi = this->kernel_size * this->kernel_size * this->in_channels;

    unsigned int grid_row = (batch_size + threads - 1) / threads;
    unsigned int grid_col = (wohofo + threads - 1) / threads;
    unsigned int grid_row_w = (ki2_fi + threads - 1) / threads;
    unsigned int grid_col_w = (this->out_channels + threads - 1) / threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_grid_w(grid_col_w, grid_row_w);
    dim3 dim_block(threads, threads);

    permute_delta_cuda<<<dim_grid, dim_block>>>(
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var, woho, wohofo,
        batch_size, cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2);

    conv2d_bwd_delta_w_cuda<<<dim_grid_w, dim_block>>>(
        this->d_var_w, cu_next_bwd_states->d_mu_a, cu_temp_states->d_tmp_1,
        cu_temp_states->d_tmp_2, this->d_idx_mwa_2, batch_size,
        this->out_channels, woho, wihi, this->in_channels, this->kernel_size,
        pad_idx, this->d_delta_mu_w, this->d_delta_var_w);

    if (this->bias) {
        unsigned int grid_col_bias =
            (this->out_channels + threads - 1) / threads;
        // dim3 dim_grid_bias(grid_col_bias, 1);

        conv2d_bwd_delta_b_cuda<<<grid_col_bias, threads>>>(
            this->d_var_b, cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2,
            woho_batch, this->out_channels, this->d_delta_mu_b,
            this->d_delta_var_b);
    }
}

void Conv2dCuda::backward(BaseDeltaStates &input_delta_states,
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
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;
    dim3 dim_block(threads, threads);

    // Launch kernel
    int wihi = this->in_width * this->in_height;
    int woho = this->out_width * this->out_height;
    int row_zw_fo = this->row_zw * this->out_channels;
    int pad_param_idx = this->num_weights + 1;

    if (state_udapte) {
        int pad_idx = woho * this->out_channels * batch_size + 1;

        unsigned int grid_row_p = (batch_size + threads - 1) / threads;
        unsigned int grid_col_p =
            (wihi * this->in_channels + threads - 1) / threads;
        dim3 dim_grid_p(grid_col_p, grid_row_p);

        unsigned int grid_row = (this->in_channels + threads - 1) / threads;
        unsigned int grid_col = (wihi * batch_size + threads - 1) / threads;
        dim3 dim_grid(grid_col, grid_row);

        permmute_jacobian_cuda<<<dim_grid_p, dim_block>>>(
            cu_next_bwd_states->d_jcb, wihi, this->in_channels, batch_size,
            cu_temp_states->d_tmp_1);

        conv2d_bwd_delta_z_cuda_tiled<<<dim_grid, dim_block>>>(
            this->d_mu_w, cu_temp_states->d_tmp_1,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->d_idx_cov_zwa_1,
            this->d_idx_var_z_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, this->row_zw, row_zw_fo,
            batch_size, pad_param_idx, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    }
    if (param_update) {
        int woho_batch = woho * batch_size;
        int wohofo = woho * this->out_channels;
        int param_pad_idx = wihi * this->in_channels * batch_size + 1;
        int ki2_fi = this->kernel_size * this->kernel_size * this->in_channels;

        unsigned int grid_row_pp = (batch_size + threads - 1) / threads;
        unsigned int grid_col_pp = (wohofo + threads - 1) / threads;
        unsigned int grid_row_w = (ki2_fi + threads - 1) / threads;
        unsigned int grid_col_w = (this->out_channels + threads - 1) / threads;

        dim3 dim_grid_pp(grid_col_pp, grid_row_pp);
        dim3 dim_grid_w(grid_row_w, grid_col_w);
        dim3 dim_block(threads, threads);

        permute_delta_cuda<<<dim_grid_pp, dim_block>>>(
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, woho, wohofo, batch_size,
            cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2);

        conv2d_bwd_delta_w_cuda_tiled_v3<<<dim_grid_w, dim_block>>>(
            this->d_var_w, cu_next_bwd_states->d_mu_a, cu_temp_states->d_tmp_1,
            cu_temp_states->d_tmp_2, this->d_idx_mwa_2, batch_size,
            this->out_channels, woho, wihi, this->in_channels,
            this->kernel_size, param_pad_idx, this->d_delta_mu_w,
            this->d_delta_var_w);

        if (this->bias) {
            unsigned int grid_col_bias =
                (this->out_channels + threads - 1) / threads;

            conv2d_bwd_delta_b_cuda<<<grid_col_bias, threads>>>(
                this->d_var_b, cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2,
                woho_batch, this->out_channels, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

std::unique_ptr<BaseLayer> Conv2dCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_linear = std::make_unique<Conv2d>(
        this->in_channels, this->out_channels, this->kernel_size, this->bias,
        this->stride, this->padding, this->padding_type, this->in_width,
        this->in_height, this->gain_w, this->gain_b, this->init_method);

    host_linear->mu_w = this->mu_w;
    host_linear->var_w = this->var_w;
    host_linear->mu_b = this->mu_b;
    host_linear->var_b = this->var_b;

    return host_linear;
}

void Conv2dCuda::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_2.size() == 0) {
        this->lazy_index_init();
    }
}

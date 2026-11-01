#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// Mirrors the CPU helpers in src/rmsnorm_layer.cpp. States are [batch_size, ni]
// row-major, where batch_size counts every token (block_size * seq_len).

constexpr int RMSNORM_STAT_THREADS = 256;
constexpr int RMSNORM_DELTA_W_COLS = 32;
constexpr int RMSNORM_DELTA_W_ROW_LANES = 8;
constexpr int RMSNORM_DELTA_W_ROWS_PER_TILE = 32;

__device__ __forceinline__ float rmsnorm_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// rms_ra[row] = (1/ni) * sum_i (mu_a[row,i]^2 + var_a[row,i]).
// One block of RMSNORM_STAT_THREADS per row; warp shuffle then shared reduce.
__global__ void rmsnorm_stat_rms_cuda(const float *mu_a, const float *var_a,
                                      int ni, float *rms_ra) {
    __shared__ float warp_sums[RMSNORM_STAT_THREADS / 32];
    size_t row_offset = static_cast<size_t>(blockIdx.x) * ni;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < ni; i += RMSNORM_STAT_THREADS) {
        float m = mu_a[row_offset + i];
        sum += m * m + var_a[row_offset + i];
    }
    sum = rmsnorm_warp_sum(sum);
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < RMSNORM_STAT_THREADS / 32 ? warp_sums[lane] : 0.0f;
        sum = rmsnorm_warp_sum(sum);
        if (lane == 0) rms_ra[blockIdx.x] = sum / ni;
    }
}

// One thread per element.
__global__ void rmsnorm_fwd_mean_var_cuda(const float *mu_w, const float *var_w,
                                          const float *mu_a, const float *var_a,
                                          const float *rms_ra, float epsilon,
                                          int ni, int batch_size, float *mu_z,
                                          float *var_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * ni) return;
    int row = idx / ni;
    int col = idx % ni;

    float inv_rms_sq = 1.0f / (rms_ra[row] + epsilon);
    float mu_a_v = mu_a[idx];

    mu_z[idx] = mu_a_v * sqrtf(inv_rms_sq) * mu_w[col];
    var_z[idx] =
        inv_rms_sq * (var_a[idx] * (var_w[col] + mu_w[col] * mu_w[col]) +
                      var_w[col] * mu_a_v * mu_a_v);
}

// One thread per element.
__global__ void rmsnorm_bwd_delta_z_cuda(const float *mu_w, const float *rms_ra,
                                         const float *delta_mu_out,
                                         const float *delta_var_out,
                                         float epsilon, int ni, int batch_size,
                                         float *delta_mu, float *delta_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * ni) return;
    int row = idx / ni;
    int col = idx % ni;

    float tmp = mu_w[col] / sqrtf(rms_ra[row] + epsilon);
    delta_mu[idx] = tmp * delta_mu_out[idx];
    delta_var[idx] = tmp * delta_var_out[idx] * tmp;
}

// Stage 1 of the delta_w reduction over rows: each block sums one tile of
// RMSNORM_DELTA_W_ROWS_PER_TILE rows for RMSNORM_DELTA_W_COLS columns, with
// RMSNORM_DELTA_W_ROW_LANES threads per column folded in shared memory.
// Grid (column tiles, row tiles); partials are [row tiles, ni].
__global__ void rmsnorm_bwd_delta_w_partial_cuda(
    const float *mu_a, const float *rms_ra, const float *delta_mu_out,
    const float *delta_var_out, float epsilon, int ni, int batch_size,
    float *partial_mu, float *partial_var) {
    __shared__ float lane_mu[RMSNORM_DELTA_W_ROW_LANES][RMSNORM_DELTA_W_COLS];
    __shared__ float lane_var[RMSNORM_DELTA_W_ROW_LANES][RMSNORM_DELTA_W_COLS];
    int col = blockIdx.x * RMSNORM_DELTA_W_COLS + threadIdx.x;
    int row_begin = blockIdx.y * RMSNORM_DELTA_W_ROWS_PER_TILE;
    int row_end = min(row_begin + RMSNORM_DELTA_W_ROWS_PER_TILE, batch_size);
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    if (col < ni) {
        for (int row = row_begin + threadIdx.y; row < row_end;
             row += RMSNORM_DELTA_W_ROW_LANES) {
            size_t idx = static_cast<size_t>(row) * ni + col;
            float tmp = mu_a[idx] / sqrtf(rms_ra[row] + epsilon);
            sum_mu += tmp * delta_mu_out[idx];
            sum_var += tmp * delta_var_out[idx] * tmp;
        }
    }
    lane_mu[threadIdx.y][threadIdx.x] = sum_mu;
    lane_var[threadIdx.y][threadIdx.x] = sum_var;
    __syncthreads();
    if (threadIdx.y == 0 && col < ni) {
        for (int lane = 1; lane < RMSNORM_DELTA_W_ROW_LANES; lane++) {
            sum_mu += lane_mu[lane][threadIdx.x];
            sum_var += lane_var[lane][threadIdx.x];
        }
        partial_mu[blockIdx.y * ni + col] = sum_mu;
        partial_var[blockIdx.y * ni + col] = sum_var;
    }
}

// Stage 2: sum the row-tile partials in tile order. One thread per column.
__global__ void rmsnorm_bwd_delta_w_cuda(const float *partial_mu,
                                         const float *partial_var,
                                         const float *var_w, int ni,
                                         int num_row_tiles, float *delta_mu_w,
                                         float *delta_var_w) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ni) return;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    for (int tile = 0; tile < num_row_tiles; tile++) {
        sum_mu += partial_mu[tile * ni + col];
        sum_var += partial_var[tile * ni + col];
    }
    delta_mu_w[col] = sum_mu * var_w[col];
    delta_var_w[col] = sum_var * var_w[col] * var_w[col];
}

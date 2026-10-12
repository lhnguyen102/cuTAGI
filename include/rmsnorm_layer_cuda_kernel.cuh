#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// Mirrors the CPU helpers in src/rmsnorm_layer.cpp. States are [batch_size, ni]
// row-major, where batch_size counts every token (block_size * seq_len).

// rms_ra[row] = (1/ni) * sum_i (mu_a[row,i]^2 + var_a[row,i]).
// One thread per row.
__global__ void rmsnorm_stat_rms_cuda(const float *mu_a, const float *var_a,
                                      int ni, int batch_size, float *rms_ra) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;
    float sum = 0.0f;
    for (int i = 0; i < ni; i++) {
        float m = mu_a[row * ni + i];
        sum += m * m + var_a[row * ni + i];
    }
    rms_ra[row] = sum / ni;
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

// Reduces over rows. One thread per feature column.
__global__ void rmsnorm_bwd_delta_w_cuda(
    const float *mu_a, const float *var_w, const float *rms_ra,
    const float *delta_mu_out, const float *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu_w, float *delta_var_w) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ni) return;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    for (int row = 0; row < batch_size; row++) {
        float tmp = mu_a[col + row * ni] / sqrtf(rms_ra[row] + epsilon);
        sum_mu += tmp * delta_mu_out[col + row * ni];
        sum_var += tmp * delta_var_out[col + row * ni] * tmp;
    }
    delta_mu_w[col] = sum_mu * var_w[col];
    delta_var_w[col] = sum_var * var_w[col] * var_w[col];
}

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// Kernels mirror the CPU helpers in src/rmsnorm_layer.cpp exactly.
// Layout convention: input/output are [batch_size, ni] row-major where
// `ni` is the feature axis (= input_size).

// rms_ra[row] = (1/ni) * sum_i (mu_a[row,i]^2 + var_a[row,i])
// One thread per batch row.
__global__ void rmsnorm_stat_rms_kernel(const float *mu_a, const float *var_a,
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

// Forward: mu_z = (mu_a / sqrt(rms+eps)) * mu_w
//          var_z = (1/(rms+eps)) *
//                  (var_a * (var_w + mu_w^2) + var_w * mu_a^2)
// One thread per output element.
__global__ void rmsnorm_fwd_mean_var_kernel(
    const float *mu_w, const float *var_w, const float *mu_a,
    const float *var_a, const float *rms_ra, float epsilon, int ni,
    int batch_size, float *mu_z, float *var_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * ni;
    if (idx >= total) return;
    int row = idx / ni;
    int col = idx % ni;

    float inv_rms = rsqrtf(rms_ra[row] + epsilon);
    float inv_rms_sq = inv_rms * inv_rms;
    float mu_a_v = mu_a[idx];
    float var_a_v = var_a[idx];
    float mu_w_v = mu_w[col];
    float var_w_v = var_w[col];

    mu_z[idx] = mu_a_v * inv_rms * mu_w_v;
    var_z[idx] = inv_rms_sq * (var_a_v * (var_w_v + mu_w_v * mu_w_v) +
                               var_w_v * mu_a_v * mu_a_v);
}

// State backward: delta_mu = (inv_rms * mu_w) * delta_mu_out;
//                 delta_var = (inv_rms * mu_w)^2 * delta_var_out.
// One thread per element.
__global__ void rmsnorm_bwd_delta_z_kernel(
    const float *mu_w, const float *rms_ra, const float *delta_mu_out,
    const float *delta_var_out, float epsilon, int ni, int batch_size,
    float *delta_mu, float *delta_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * ni;
    if (idx >= total) return;
    int row = idx / ni;
    int col = idx % ni;

    float inv_rms = rsqrtf(rms_ra[row] + epsilon);
    float tmp = inv_rms * mu_w[col];
    delta_mu[idx] = tmp * delta_mu_out[idx];
    delta_var[idx] = tmp * tmp * delta_var_out[idx];
}

// Weight backward: reduce over batch.
// One thread per feature column.
__global__ void rmsnorm_bwd_delta_w_kernel(
    const float *mu_a, const float *var_w, const float *rms_ra,
    const float *delta_mu_out, const float *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu_w, float *delta_var_w) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= ni) return;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    for (int row = 0; row < batch_size; row++) {
        float inv_rms = rsqrtf(rms_ra[row] + epsilon);
        float tmp = inv_rms * mu_a[col + row * ni];
        sum_mu += tmp * delta_mu_out[col + row * ni];
        sum_var += tmp * tmp * delta_var_out[col + row * ni];
    }
    float var_w_v = var_w[col];
    delta_mu_w[col] = sum_mu * var_w_v;
    delta_var_w[col] = sum_var * var_w_v * var_w_v;
}

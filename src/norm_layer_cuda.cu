///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      January 24, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "../include/norm_layer_cuda.cuh"

__global__ void layernorm_stat_mean_var_cuda(float const *mu_a,
                                             float const *var_a, int ni,
                                             int batch_size, float *mu_s,
                                             float *var_s)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < batch_size) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < ni; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col * ni + i];
            sum_var += var_a[col * ni + i];
        }
        mu_s[col] = sum_mu / ni;
        var_s[col] = sum_var;
    }
}

__global__ void layernorm_sample_var_cuda(float const *mu_a, float const *mu_s,
                                          float const *var_s, int ni,
                                          int batch_size, float *var_sample)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < batch_size) {
        float sum = 0.0f;
        for (int i = 0; i < ni; i++) {
            sum += (mu_a[col * ni + i] - mu_s[col]) *
                   (mu_a[col * ni + i] - mu_s[col]);
        }
        var_sample[col] = (sum + var_s[col]) / (ni - 1);
    }
}

__global__ void layernorm_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, float epsilon, int ni, int B,
    float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < ni && row < B) {
        mu_z[col + row * ni] = (1 / sqrtf(var_ra[row] + epsilon)) *
                                   (mu_a[col + row * ni] - mu_ra[row]) *
                                   mu_w[col] +
                               mu_b[col];
        var_z[col + row * ni] =
            (1.0f / (var_ra[row] + epsilon)) *
                (var_a[col + row * ni] * mu_w[col] * mu_w[col] +
                 var_w[col] *
                     (mu_a[col + row * ni] * mu_a[col + row * ni] -
                      mu_ra[row] * mu_ra[row] + var_a[col + row * ni])) +
            var_b[col];
    }
}

__global__ void layernorm2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, float epsilon, int wihi, int m,
    int k, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi * fi, m = B
    {
        mu_z[col + row * k] = (1.0f / sqrtf(var_ra[row] + epsilon)) *
                                  (mu_a[col + row * k] - mu_ra[row]) *
                                  mu_w[col / wihi] +
                              mu_b[col / wihi];
        var_z[col + row * k] =
            (1.0f / (var_ra[row] + epsilon)) *
                (var_a[col + row * k] * mu_w[col / wihi] * mu_w[col / wihi] +
                 var_w[col / wihi] *
                     (mu_a[col + row * k] * mu_a[col + row * k] -
                      mu_ra[row] * mu_ra[row] + var_a[col + row * k])) +
            var_b[col / wihi];
    }
}

////
// Layer Norm's backward
////
__global__ void layernorm_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float tmp = (1.0f / sqrtf(var_hat[row] + epsilon)) * mu_w[col] *
                    jcb[col + row * ni];

        delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];
        delta_var[col + row * ni] = tmp * delta_var_out[col + row * ni] * tmp;
    }
}

__global__ void layernorm_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_hat,
    float const *var_hat, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int ni, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = (1.0f / sqrtf(var_hat[i] + epsilon)) *
                      (mu_a[col + i * ni] - mu_hat[i]) * var_w[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

__global__ void layernorm_bwd_delta_b_cuda(float const *var_b,
                                           float const *delta_mu_out,
                                           float const *delta_var_out,
                                           float epsilon, int ni,
                                           int batch_size, float *delta_mu_b,
                                           float *delta_var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float A = var_b[col];
            sum_mu += A * delta_mu_out[col + i * ni];
            sum_var += A * delta_var_out[col + i * ni] * A;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

__global__ void layernorm2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int wihi, int fi, int m, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi * fi, m = B
    {
        float tmp = (1.0f / sqrtf(var_hat[row % fi] + epsilon)) *
                    mu_w[row % fi] * jcb[col + row * wihi];

        delta_mu[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
        delta_var[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp;
    }
}

__global__ void layernorm2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_hat,
    float const *var_hat, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int wihi, int m, int k, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        float A = (1.0f / sqrtf(var_hat[row] + epsilon)) *
                  (mu_a[col + row * k] - mu_hat[row]) * var_w[col / wihi];
        delta_mu_w[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_w[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

__global__ void layernorm2d_bwd_delta_b_cuda(float const *var_b,
                                             float const *delta_mu_out,
                                             float const *delta_var_out,
                                             float epsilon, int wihi, int m,
                                             int k, float *delta_mu_b,
                                             float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        float A = var_b[col / wihi];
        delta_mu_b[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_b[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

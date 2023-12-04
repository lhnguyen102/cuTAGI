///////////////////////////////////////////////////////////////////////////////
// File:         activation_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 04, 2023
// Updated:      December 04, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/activation_cuda.cuh"

__global__ void relu_mean_var(float const *mu_z, float const *var_z,
                              int num_states, float *mu_a, float *jcb,
                              float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float one_pad = 1.0f;
    if (col < num_states) {
        mu_a[col] = mu_z[col];
        jcb[col] = one_pad;
        var_a[col] = var_z[col];
    }
}

__global__ void sigmoid_mean_var(float const *mu_z, float const *var_z,
                                 int num_states, float *mu_a, float *jcb,
                                 float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;

    if (col < num_states) {
        tmp = 1.0f / (1.0f + expf(-mu_z[col]));
        mu_a[col] = tmp;
        jcb[col] = tmp * (1 - tmp);
        var_a[col] = tmp * (1 - tmp) * var_z[col] * tmp * (1 - tmp);
    }
}

__global__ void tanh_mean_var(float const *mu_z, float const *var_z,
                              int num_states, float *mu_a, float *jcb,
                              float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = tanhf(mu_z[col]);
        mu_a[col] = tmp;
        jcb[col] = (1 - powf(tmp, 2));
        var_a[col] = (1 - powf(tmp, 2)) * var_z[col] * (1 - powf(tmp, 2));
    }
}

__global__ void mixture_relu(float const *mu_z, float const *var_z,
                             float omega_tol, int num_states, float *mu_a,
                             float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha, beta, omega, kappa, mu_z_til, var_z_til;
    float pi = 3.141592;  // pi number
    if (col < num_states) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mu_z[col] / powf(var_z[col], 0.5);
        omega = max(1.0f - normcdff(alpha), omega_tol);
        beta = (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha, 2) / 2.0f) /
               omega;
        kappa = 1.0f + alpha * beta - powf(beta, 2);

        // Gaussian mixture's parameters
        mu_z_til = mu_z[col] + beta * powf(var_z[col], 0.5);
        var_z_til = kappa * var_z[col];

        // Activation distribution
        if (omega * mu_z_til > omega_tol) {
            mu_a[col] = omega * mu_z_til;
            var_a[col] =
                omega * var_z_til + omega * (1.0f - omega) * powf(mu_z_til, 2);
            jcb[col] = powf(omega * kappa, 0.5);
        } else {
            mu_a[col] = omega_tol;
            var_a[col] =
                omega * var_z_til + omega * (1.0f - omega) * powf(omega_tol, 2);
            jcb[col] = 0.0f;  // TODO replace by 1.0f
        }
    }
}

__global__ void mixture_sigmoid(float const *mu_z, float const *var_z,
                                float omega_tol, int num_states, float *mu_a,
                                float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        alpha_upper = (1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                (pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[col] - beta * powf(var_z[col], 0.5);
        var_z_til = kappa * var_z[col];

        // Activation distribution
        mu_a[col] = omega * mu_z_til - cdf_lower + (1 - cdf_upper);

        var_a[col] = omega * var_z_til + omega * powf(mu_z_til - mu_a[col], 2) +
                     cdf_lower * powf(1 + mu_a[col], 2) +
                     (1 - cdf_upper) * powf(1 - mu_a[col], 2);

        jcb[col] = powf(omega * kappa, 0.5);
    }
}

__global__ void mixture_tanh(float const *mu_z, float const *var_z,
                             float omega_tol, int num_states, float *mu_a,
                             float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        alpha_upper = (1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                (pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[col] - beta * powf(var_z[col], 0.5);
        var_z_til = kappa * var_z[col];

        // Activation distribution
        mu_a[col] = omega * mu_z_til - cdf_lower + (1 - cdf_upper);
        var_a[col] = omega * var_z_til + omega * powf(mu_z_til - mu_a[col], 2) +
                     cdf_lower * powf(1 + mu_a[col], 2) +
                     (1 - cdf_upper) * powf(1 - mu_a[col], 2);
        jcb[col] = powf(omega * kappa, 0.5);
    }
}

__global__ void softplus(float const *mu_z, float const *var_z, int num_states,
                         float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < num_states) {
        mu_a[col] = logf(1 + expf(mu_z[col]));
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = tmp * var_z[col] * tmp;
    }
}

__global__ void leakyrelu(float const *mu_z, float const *var_z, float alpha,
                          int num_states, float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0.0f;
    float one_pad = 1.0f;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = max(mu_z[col], zero_pad);
        if (tmp == 0) {
            mu_a[col] = alpha * mu_z[col];
            jcb[col] = alpha;
            var_a[col] = alpha * var_z[col] * alpha;

        } else {
            mu_a[col] = tmp;
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

__global__ void softmax(float const *mu_z, float *var_z, size_t output_size,
                        int batch_size, float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float max_mu = mu_z[0];
    float max_var = var_z[0];

    for (int j = 1; j < output_size; j++) {
        if (mu_z[j + i * output_size] > max_mu) {
            max_mu = mu_z[j + i * output_size];
            max_var = var_z[j + i * output_size];
        }
    }

    float sum_mu = 0.0f;
    for (int j = 0; j < output_size; j++) {
        sum_mu += expf(mu_z[j + i * output_size] - max_mu);
    }

    float tmp_mu;
    for (int j = 0; j < output_size; j++) {
        tmp_mu = expf(mu_z[j + output_size * i] - max_mu) / sum_mu;

        mu_a[j + i * output_size] = tmp_mu;

        jcb[j + output_size * i] = tmp_mu * (1 - tmp_mu);

        var_a[j + output_size * i] = jcb[j + output_size * i] *
                                     (var_z[j + output_size * i] + max_var) *
                                     jcb[j + output_size * i];
    }
}
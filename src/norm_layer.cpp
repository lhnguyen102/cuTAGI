///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      January 31, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/norm_layer.h"

void layernorm_state_mean_var(const std::vector<float> &mu_a,
                              const std::vector<float> &var_a, int ni,
                              int batch_size, std::vector<float> &mu_s,
                              std::vector<float> &var_s)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    for (int col = 0; col < batch_size; col++) {
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

void layernorm_sample_var(const std::vector<float> &mu_a,
                          const std::vector<float> &mu_s,
                          const std::vector<float> &var_s, int ni,
                          int batch_size, std::vector<float> &var_sample)
/*
 */
{
    // ni in the case of conv2d will be wihi * fi
    for (int col = 0; col < batch_size; col++) {
        float sum = 0.0f;
        for (int i = 0; i < ni; i++) {
            sum += (mu_a[col * ni + i] - mu_s[col]) *
                   (mu_a[col * ni + i] - mu_s[col]);
        }
        var_sample[col] = (sum + var_s[col]) / (ni - 1);
    }
}

void running_mean_var(const std::vector<float> &mu_s,
                      const std::vector<float> &var_s,
                      const std::vector<float> &mu_ra_prev,
                      const std::vector<float> &var_ra_prev, float momentum,
                      int num_states, std::vector<float> &mu_ra,
                      std::vector<float> &var_ra)
/*Copute the running average for the normalization layers.

Args:
    ms: New statistical mean of samples
    Ss: New statistical variance of samples
    mraprev: Previous mean for the normalization layers
    Sraprev: Previous statistical variance for the normalization layers
    momentum: Running average factor
    mra: Statistical mean for the normalization layers
    Sra: Statistical variance for the normalization layers
    N: Size of mra
 */
{
    for (int col = 0; col < num_states; col++) {
        float tmp = mu_ra_prev[col] * momentum + mu_s[col] * (1 - momentum);
        var_ra[col] = var_ra_prev[col] * momentum + var_s[col] * (1 - momentum);
        mu_ra[col] = tmp;
    }
}

void layernorm_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    float epsilon, int ni, int batch_size, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*
 */
{
    for (int col = 0; col < ni; col++) {
        for (int row = 0; row < batch_size; row++) {
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
}

void layernorm2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    float epsilon, int wihi, int m, int k, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*
 */
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            mu_z[col + row * k] = (1.0f / sqrtf(var_ra[row] + epsilon)) *
                                      (mu_a[col + row * k] - mu_ra[row]) *
                                      mu_w[col / wihi] +
                                  mu_b[col / wihi];
            var_z[col + row * k] =
                (1.0f / (var_ra[row] + epsilon)) *
                    (var_a[col + row * k] * mu_w[col / wihi] *
                         mu_w[col / wihi] +
                     var_w[col / wihi] *
                         (mu_a[col + row * k] * mu_a[col + row * k] -
                          mu_ra[row] * mu_ra[row] + var_a[col + row * k])) +
                var_b[col / wihi];
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// Layer Norm's backward
////////////////////////////////////////////////////////////////////////////////
void layernorm_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_hat, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, std::vector<float> &delta_mu, std::vector<float> &delta_var)
/*
 */
{
    for (int row = 0; row < batch_size; row++) {
        for (int col = 0; col < ni; col++) {
            float tmp = (1.0f / sqrtf(var_hat[row] + epsilon)) * mu_w[col] *
                        jcb[col + row * ni];

            delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];
            delta_var[col + row * ni] =
                tmp * delta_var_out[col + row * ni] * tmp;
        }
    }
}

void layernorm_bwd_delta_w(const std::vector<float> &var_w,
                           const std::vector<float> &mu_a,
                           const std::vector<float> &mu_hat,
                           const std::vector<float> &var_hat,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int batch_size,
                           std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w)
/*
 */
{
    for (int col = 0; col < ni; col++) {
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

void layernorm_bwd_delta_b(const std::vector<float> &var_b,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int batch_size,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b)
/*
 */
{
    for (int col = 0; col < ni; col++) {
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

void layernorm2d_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_hat, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int m, std::vector<float> &delta_mu, std::vector<float> &delta_var)
/*
 */
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < wihi; col++) {
            float tmp = (1.0f / sqrtf(var_hat[row % fi] + epsilon)) *
                        mu_w[row % fi] * jcb[col + row * wihi];

            delta_mu[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
            delta_var[col + row * wihi] =
                tmp * delta_var_out[col + row * wihi] * tmp;
        }
    }
}

void layernorm2d_bwd_delta_w(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_hat, const std::vector<float> &var_hat,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int m,
    int k, std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/*
 */
{
    // k = wihi, m = fi*B
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float A = (1.0f / sqrtf(var_hat[row] + epsilon)) *
                      (mu_a[col + row * k] - mu_hat[row]) * var_w[col / wihi];
            delta_mu_w[col + row * k] = A * delta_mu_out[col + row * k];
            delta_var_w[col + row * k] = A * delta_var_out[col + row * k] * A;
        }
    }
}
void layernorm2d_bwd_delta_b(const std::vector<float> &var_b,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int m, int k,
                             std::vector<float> &delta_mu_b,
                             std::vector<float> &delta_var_b)
/*
 */
{
    // k = wihi, m = fi*B
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            float A = var_b[col / wihi];
            delta_mu_b[col + row * k] = A * delta_mu_out[col + row * k];
            delta_var_b[col + row * k] = A * delta_var_out[col + row * k] * A;
        }
    }
}

void delta_param_sum(const std::vector<float> delta_mu_e,
                     const std::vector<float> delta_var_e, int wihi, int fi,
                     int n, std::vector<float> delta_mu,
                     std::vector<float> delta_var) {
    for (int col = 0; col < fi; col++) {
        float sum_delta_mu = 0.0f;
        float sum_delta_var = 0.0f;
        for (int i = 0; i < n; i++)  // n = wihi * fi
        {
            sum_delta_mu +=
                delta_mu_e[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sum_delta_var +=
                delta_var_e[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        delta_mu[col] = sum_delta_mu;
        delta_var[col] = sum_delta_var;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

std::tuple<int, int> get_number_params_layer_norm(
    const std::vector<int> &normalized_shape)
/*
 */
{
    int num_elements = normalized_shape.size();
    int num_weights, num_biases;
    if (num_elements == 1 || num_elements == 3) {
        num_weights = normalized_shape[0];
        num_biases = normalized_shape[0];
    } else {
        throw std::runtime_error(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Normalized shape provided are not supported.");
    }
}
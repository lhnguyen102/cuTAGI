///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      February 04, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/norm_layer.h"

////////////////////////////////////////////////////////////////////////////////
/// CPU kernels for Layer Norm
////////////////////////////////////////////////////////////////////////////////
void layernorm_stat_mean_var(const std::vector<float> &mu_a,
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
                      const std::vector<float> &var_s, float momentum,
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
        mu_ra[col] = mu_ra[col] * momentum + mu_s[col] * (1 - momentum);
        var_ra[col] = var_ra[col] * momentum + var_s[col] * (1 - momentum);
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
    const std::vector<float> &var_ra, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int m, std::vector<float> &delta_mu, std::vector<float> &delta_var)
/*
 */
{
    // k = wihi * fi, m = B
    int k = wihi * fi;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < wihi * fi; col++) {
            float tmp = (1 / sqrtf(var_ra[row] + epsilon)) * mu_w[col / wihi] *
                        jcb[col + row * k];

            delta_mu[col + row * k] = tmp * delta_mu_out[col + row * k];
            delta_var[col + row * k] = tmp * delta_var_out[col + row * k] * tmp;
        }
    }
}

void layernorm2d_bwd_delta_w(const std::vector<float> &var_w,
                             const std::vector<float> &mu_a,
                             const std::vector<float> &mu_hat,
                             const std::vector<float> &var_hat,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int batch_size,
                             std::vector<float> &delta_mu_w,
                             std::vector<float> &delta_var_w)
/*
 */
{
    // k = wihi, m = B
    int k = wihi * fi;
    for (int row = 0; row < batch_size; row++) {
        for (int col = 0; col < k; col++) {
            float tmp = (1.0f / sqrtf(var_hat[row] + epsilon)) *
                        (mu_a[col + row * k] - mu_hat[row]) * var_w[col / wihi];

            delta_mu_w[col + row * k] = tmp * delta_mu_out[col + row * k];
            delta_var_w[col + row * k] =
                tmp * delta_var_out[col + row * k] * tmp;
        }
    }
}
void layernorm2d_bwd_delta_b(const std::vector<float> &var_b,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int batch_size,
                             std::vector<float> &delta_mu_b,
                             std::vector<float> &delta_var_b)
/*
 */
{
    // k = wihi*fi, m = B
    int k = wihi * fi;
    for (int row = 0; row < batch_size; row++) {
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
/// CPU kernels for Batch Norm
////////////////////////////////////////////////////////////////////////////////
void batchnorm_stat_mean_var(const std::vector<float> &mu_a,
                             const std::vector<float> &var_a, int ni,
                             int batch_size, std::vector<float> &mu_s,
                             std::vector<float> &var_s)
/*Compute sample mean and variance of activation units of full-connected layer
for each batch.
*/
{
    for (int col = 0; col < ni; col++) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col + i * ni];
            sum_var += var_a[col + i * ni];
        }
        mu_s[col] = sum_mu / batch_size;
        var_s[col] = sum_var;
    }
}

void batchnorm_sample_var(const std::vector<float> &mu_a,
                          const std::vector<float> &mu_s,
                          const std::vector<float> &var_s, int ni,
                          int batch_size, std::vector<float> &var)
/*Compute statistical mean and variance of activation units for full-connected
layer for each batch.
*/
{
    for (int col = 0; col < ni; col++) {
        float sum = 0;
        for (int i = 0; i < batch_size; i++) {
            sum += (mu_a[col + i * ni] - mu_s[col]) *
                   (mu_a[col + i * ni] - mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (batch_size - 1);
    }
}

void batchnorm_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    float epsilon, int ni, int batch_size, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*Compute pmean of product WA of batch-normalization layer.
 */
{
    for (int row = 0; row < batch_size; row++) {
        for (int col = 0; col < ni; col++) {
            mu_z[col + row * ni] = (1 / sqrtf(var_ra[col] + epsilon)) *
                                       (mu_a[col + row * ni] - mu_ra[col]) *
                                       mu_w[col] +
                                   mu_b[col];
            var_z[col + row * ni] =
                (1 / (var_ra[col] + epsilon)) *
                    (var_a[col + row * ni] * mu_w[col] * mu_w[col] +
                     var_w[col] *
                         (mu_a[col + row * ni] * mu_a[col + row * ni] -
                          mu_ra[col] * mu_ra[col] + var_a[col + row * ni])) +
                var_b[col];
        }
    }
}

void batchnorm2d_stat_mean_var(const std::vector<float> &mu_a,
                               const std::vector<float> &var_a, int wihi,
                               int fi, int batch_size, std::vector<float> &mu_s,
                               std::vector<float> &var_s)
/*Compute sample mean and variance of activation units for batch-normalization
layer.
*/
{
    for (int col = 0; col < fi; col++) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sum_var += var_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        mu_s[col] = sum_mu / (wihi * batch_size);
        var_s[col] = sum_var;
    }
}

void batchnorm2d_sample_var(const std::vector<float> &mu_a,
                            const std::vector<float> &mu_s,
                            const std::vector<float> &var_s, int wihi, int fi,
                            int batch_size, std::vector<float> &var)
/*Compute statistical mean and variance of activation units for
batch-normalization layer.
*/
{
    for (int col = 0; col < fi; col++) {
        float sum = 0;
        for (int i = 0; i < wihi * batch_size; i++) {
            sum += (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]) *
                   (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (wihi * batch_size - 1);
    }
}

void batchnorm2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    float epsilon, int wihi, int fi, int m, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*Compute mean of product WA of batch-normalization. Note that the previous
layer is a convolutional layer.
*/
{
    int k = wihi;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++)  // k = wihi, m = fi*B
        {
            mu_z[col + row * k] = (1 / sqrtf(var_ra[row % fi] + epsilon)) *
                                      (mu_a[col + row * k] - mu_ra[row % fi]) *
                                      mu_w[row % fi] +
                                  mu_b[row % fi];

            var_z[col + row * k] =
                (1 / (var_ra[row % fi] + epsilon)) *
                    (var_a[col + row * k] * mu_w[row % fi] * mu_w[row % fi] +
                     var_w[row % fi] *
                         (mu_a[col + row * k] * mu_a[col + row * k] -
                          mu_ra[row % fi] * mu_ra[row % fi] +
                          var_a[col + row * k])) +
                var_b[row % fi];
        }
    }
}

void batchnorm_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_hat, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, std::vector<float> &delta_mu, std::vector<float> &delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is full-connected layer.
*/
{
    for (int row = 0; row < batch_size; row++) {
        for (int col = 0; col < ni; col++) {
            float tmp = (1 / sqrtf(var_hat[col] + epsilon)) * mu_w[col] *
                        jcb[col + row * ni];

            delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];

            delta_var[col + row * ni] =
                tmp * delta_var_out[col + row * ni] * tmp;
        }
    }
}

void batchnorm2d_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_hat, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int m, std::vector<float> &delta_mu, std::vector<float> &delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is convolutional layer.
 */
{
    for (int row = 0; row < m; row++)  // k = wihi * fi, m = B
        for (int col = 0; col < wihi; col++) {
            float tmp = (1 / sqrtf(var_hat[row % fi] + epsilon)) *
                        mu_w[row % fi] * jcb[col + row * wihi];

            delta_mu[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];

            delta_var[col + row * wihi] =
                tmp * delta_var_out[col + row * wihi] * tmp;
        }
}

void batchnorm_bwd_delta_w(const std::vector<float> &var_w,
                           const std::vector<float> &mu_a,
                           const std::vector<float> &mu_hat,
                           const std::vector<float> &var_hat,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int batch_size,
                           std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to full-connected layer.
*/
{
    for (int col = 0; col < ni; col++) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < batch_size; i++) {
            float tmp = (1 / sqrtf(var_hat[col] + epsilon)) *
                        (mu_a[col + i * ni] - mu_hat[col]) * var_w[col];
            sum_var += tmp * delta_mu_out[col + i * ni];
            sum_var += tmp * delta_var_out[col + i * ni] * tmp;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

void batchnorm_bwd_delta_b(const std::vector<float> &var_b,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int batch_size,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to full-connected layer.
*/
{
    for (int col = 0; col < ni; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float tmp = var_b[col];
            sum_mu += tmp * delta_mu_out[col + i * ni];
            sum_var += tmp * delta_var_out[col + i * ni] * tmp;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

void batchnorm2d_bwd_delta_w(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_hat, const std::vector<float> &var_hat,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int m, std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to convolutional layer.
*/
{
    for (int row = 0; row < m; row++)
        for (int col = 0; col < wihi; col++)  // k = wihi, m = fi*B
        {
            float tmp = (1 / sqrtf(var_hat[row % fi] + epsilon)) *
                        (mu_a[col + row * wihi] - mu_hat[row % fi]) *
                        var_w[row % fi];

            delta_mu_w[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
            delta_var_w[col + row * wihi] =
                tmp * delta_var_out[col + row * wihi] * tmp;
        }
}

void batchnorm2d_bwd_delta_b(const std::vector<float> &var_b,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int m,
                             std::vector<float> &delta_mu_b,
                             std::vector<float> &delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to convolutional layer.
*/
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < wihi; col++)  // k = wihi, m = fi*B
        {
            float tmp = var_b[row % fi];

            delta_mu_b[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
            delta_var_b[col + row * wihi] =
                tmp * delta_var_out[col + row * wihi] * tmp;
        }
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
    return {num_weights, num_biases};
}

////////////////////////////////////////////////////////////////////////////////
// Layer Norm
////////////////////////////////////////////////////////////////////////////////

LayerNorm::LayerNorm(const std::vector<int> &normalized_shape, float eps,
                     float momentum, bool bias)
    : normalized_shape(normalized_shape),
      epsilon(eps),
      momentum(momentum),
      bias(bias)
/*
 */
{
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
    if (this->normalized_shape.size() == 1) {
        this->input_size = this->normalized_shape[0];
        this->output_size = normalized_shape[0];
    } else if (this->normalized_shape.size() == 3) {
        this->in_channels = this->normalized_shape[0];
        this->in_width = this->normalized_shape[1];
        this->in_height = this->normalized_shape[2];
        this->out_channels = this->normalized_shape[0];
        this->out_width = this->normalized_shape[1];
        this->out_height = this->normalized_shape[2];
        this->input_size = this->in_channels * this->in_width * this->in_height;
        this->output_size =
            this->out_channels * this->out_width * this->out_height;
    } else {
        throw std::runtime_error(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Normalized shape provided are not supported.");
    }
}

std::string LayerNorm::get_layer_info() const
/*
 */
{
    return "LayerNorm()";
}

std::string LayerNorm::get_layer_name() const
/*
 */
{
    return "LayerNorm";
}

LayerType LayerNorm::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void LayerNorm::init_weight_bias()
/*
 */
{
    std::tie(this->num_weights, this->num_biases) =
        get_number_params_layer_norm(this->normalized_shape);

    this->mu_w.resize(this->num_weights, 1.0f);
    this->var_w.resize(this->num_weights, 1.0f);
    if (this->bias) {
        this->mu_b.resize(this->num_weights, 0.0f);
        this->var_b.resize(this->num_weights, 0.0001f);

    } else {
        this->num_biases = 0;
    }
}

void LayerNorm::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);
    this->delta_mu_b.resize(this->num_biases, 0.0f);
    this->delta_var_b.resize(this->num_biases, 0.0f);
}

void LayerNorm::allocate_running_mean_var(int batch_size)
/*
 */
{
    this->mu_ra.resize(batch_size, 0.0f);
    this->var_ra.resize(batch_size, 0.0f);
}

void LayerNorm::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/**/
{
    int batch_size = input_states.block_size;
    // Lazy intialization
    if (this->mu_ra.size() == 0) {
        this->allocate_running_mean_var(batch_size);
    }

    layernorm_stat_mean_var(input_states.mu_a, input_states.var_a,
                            this->input_size, batch_size, temp_states.tmp_1,
                            temp_states.tmp_2);

    layernorm_sample_var(input_states.mu_a, temp_states.tmp_1,
                         temp_states.tmp_2, this->input_size, batch_size,
                         temp_states.tmp_2);

    running_mean_var(temp_states.tmp_1, temp_states.tmp_2, this->momentum,
                     batch_size, this->mu_ra, this->var_ra);

    if (this->normalized_shape.size() == 1) {
        layernorm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               input_states.mu_a, input_states.var_a,
                               this->mu_ra, this->var_ra, this->epsilon,
                               this->input_size, batch_size, output_states.mu_a,
                               output_states.var_a);
    } else {
        int wihi = this->in_height * this->in_width;
        layernorm2d_fwd_mean_var(
            this->mu_w, this->var_w, this->mu_b, this->var_b, input_states.mu_a,
            input_states.var_a, this->mu_ra, this->var_ra, this->epsilon, wihi,
            batch_size, this->input_size, output_states.mu_a,
            output_states.var_a);
    }
    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void LayerNorm::state_backward(BaseBackwardStates &next_bwd_states,
                               BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_delta_states,
                               BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_delta_states.block_size;
    if (this->normalized_shape.size() == 1) {
        layernorm_bwd_delta_z(
            this->mu_w, next_bwd_states.jcb, this->var_ra,
            input_delta_states.delta_mu, input_delta_states.delta_var,
            this->epsilon, this->input_size, batch_size,
            output_delta_states.delta_mu, output_delta_states.delta_var);
    } else {
        int wihi = this->in_height * this->in_width;

        layernorm2d_bwd_delta_z(
            this->mu_w, next_bwd_states.jcb, this->var_ra,
            input_delta_states.delta_mu, input_delta_states.delta_var,
            this->epsilon, wihi, this->in_channels, batch_size,
            output_delta_states.delta_mu, output_delta_states.delta_var);
    }
}

void LayerNorm::param_backward(BaseBackwardStates &next_bwd_states,
                               BaseDeltaStates &delta_states,
                               BaseTempStates &temp_states)
/*
 */
{
    int batch_size = delta_states.block_size;
    if (this->normalized_shape.size() == 1) {
        layernorm_bwd_delta_w(
            this->var_w, next_bwd_states.mu_a, this->mu_ra, this->var_ra,
            delta_states.delta_mu, delta_states.delta_var, this->epsilon,
            this->input_size, batch_size, this->delta_mu_w, this->delta_var_w);

        if (this->bias) {
            layernorm_bwd_delta_b(this->var_b, delta_states.delta_mu,
                                  delta_states.delta_var, this->epsilon,
                                  this->input_size, batch_size,
                                  this->delta_mu_b, this->delta_var_b);
        }
    } else {
        int wihi = this->in_height * this->in_width;

        layernorm2d_bwd_delta_w(this->var_w, next_bwd_states.mu_a, this->mu_ra,
                                this->var_ra, delta_states.delta_mu,
                                delta_states.delta_var, this->epsilon, wihi,
                                this->in_channels, batch_size,
                                temp_states.tmp_1, temp_states.tmp_2);

        delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                        this->in_channels, batch_size, this->delta_mu_w,
                        this->delta_var_w);

        if (this->bias) {
            layernorm2d_bwd_delta_b(this->var_b, delta_states.delta_mu,
                                    delta_states.delta_var, this->epsilon, wihi,
                                    this->in_channels, batch_size,
                                    temp_states.tmp_1, temp_states.tmp_2);

            delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                            this->in_channels, batch_size, this->delta_mu_b,
                            this->delta_var_b);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//// Batch Norm
////////////////////////////////////////////////////////////////////////////////
BatchNorm::BatchNorm(const int num_features, float eps, float momentum,
                     bool bias)
    : num_features(num_features),
      epsilon(eps),
      momentum(momentum),
      bias(bias)
/*
 */
{}

BatchNorm::~BatchNorm()
/*
 */
{}

std::string BatchNorm::get_layer_info() const
/*
 */
{
    return "BatchNorm()";
}

std::string BatchNorm::get_layer_name() const
/*
 */
{
    return "BatchNorm";
}

LayerType BatchNorm::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void BatchNorm::init_weight_bias()
/*
 */
{
    if (this->in_channels == 0) {
        this->num_weights = this->input_size;
        this->num_biases = this->input_size;
    } else {
        this->num_weights = this->in_channels;
        this->num_biases = this->in_channels;
    }

    this->mu_w.resize(this->num_weights, 1.0f);
    this->var_w.resize(this->num_weights, 1.0f);
    if (this->bias) {
        this->mu_b.resize(this->num_weights, 0.0f);
        this->var_b.resize(this->num_weights, 0.0001f);

    } else {
        this->num_biases = 0;
    }
}

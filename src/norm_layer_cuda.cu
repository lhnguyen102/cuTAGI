///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      March 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/norm_layer.h"
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

__global__ void norm_sum_reduced(float *vec, float *result, int N)
/*Reduced sum of a vector*/
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) atomicAdd(result, vec[index] / N);
}
__global__ void layernorm_divide_by_interger(float *value, int batch_size) {
    *value = *value / batch_size;
}

__global__ void running_mean_var_cuda(float const *mu_s, float const *var_s,
                                      float momentum, int num_states,
                                      float *mu_ra, float *var_ra)
/*Copute the running average for the normalization layers.
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_states) {
        mu_ra[col] = mu_ra[col] * momentum + mu_s[col] * (1 - momentum);
        var_ra[col] = var_ra[col] * momentum + var_s[col] * (1 - momentum);
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
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[0] + epsilon);
        int idx = col + row * ni;
        float mu_w_term = mu_w[col];
        float mu_a_term = mu_a[idx];
        float mu_ra_term = mu_ra[0];

        mu_z[idx] =
            inv_sqrt_var_ra * (mu_a_term - mu_ra_term) * mu_w_term + mu_b[col];
        var_z[idx] = inv_sqrt_var_ra * inv_sqrt_var_ra *
                         (var_a[idx] * mu_w_term * mu_w_term +
                          var_w[col] * (mu_a_term * mu_a_term -
                                        mu_ra_term * mu_ra_term + var_a[idx])) +
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
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[0] + epsilon);
        float mu_ra_term = mu_ra[0];
        int idx = col + row * k;
        int div_idx = col / wihi;
        float mu_w_term = mu_w[div_idx];
        float mu_a_term = mu_a[idx];

        mu_z[idx] = inv_sqrt_var_ra * (mu_a_term - mu_ra_term) * mu_w_term +
                    mu_b[div_idx];
        var_z[idx] =
            inv_sqrt_var_ra * inv_sqrt_var_ra *
                (var_a[idx] * mu_w_term * mu_w_term +
                 var_w[div_idx] * (mu_a_term * mu_a_term -
                                   mu_ra_term * mu_a_term + var_a[idx])) +
            var_b[div_idx];
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
        float tmp = (1.0f / sqrtf(var_hat[0] + epsilon)) * mu_w[col] *
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
            float A = (1.0f / sqrtf(var_hat[0] + epsilon)) *
                      (mu_a[col + i * ni] - mu_hat[0]) * var_w[col];
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
    int wihi, int fi, int batch_size, float *delta_mu, float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi * fi;
    if (col < k && row < batch_size)  // k = wihi * fi, m = B
    {
        float tmp = (1 / sqrtf(var_hat[0] + epsilon)) * mu_w[col / wihi] *
                    jcb[col + row * k];

        delta_mu[col + row * k] = tmp * delta_mu_out[col + row * k];
        delta_var[col + row * k] = tmp * delta_var_out[col + row * k] * tmp;
    }
}

__global__ void layernorm2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_ra,
    float const *var_ra, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int wihi, int fi, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi * fi;
    if (col < k && row < batch_size)  // k = wihi*fi, m = B
    {
        float A = (1.0f / sqrtf(var_ra[0] + epsilon)) *
                  (mu_a[col + row * k] - mu_ra[0]) * var_w[col / wihi];
        delta_mu_w[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_w[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

__global__ void layernorm2d_bwd_delta_b_cuda(float const *var_b,
                                             float const *delta_mu_out,
                                             float const *delta_var_out,
                                             float epsilon, int wihi, int fi,
                                             int m, float *delta_mu_b,
                                             float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi * fi;
    if (col < k && row < m)  // k = wihi*f, m = B
    {
        float A = var_b[col / wihi];
        delta_mu_b[col + row * k] = A * delta_mu_out[col + row * k];
        delta_var_b[col + row * k] = A * delta_var_out[col + row * k] * A;
    }
}

__global__ void delta_param_sum(float const *delta_mu_e,
                                float const *delta_var_e, int wihi, int fi,
                                int batch_size, float *delta_mu,
                                float *delta_var) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < fi) {
        float sum_delta_mu = 0.0f;
        float sum_delta_var = 0.0f;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi * B
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

__global__ void batchnorm_stat_mean_var_cuda(float const *mu_a,
                                             float const *var_a, int ni,
                                             int batch_size, float *mu_s,
                                             float *var_s)
/*Compute sample mean and variance of activation units of full-connected layer
for each batch.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0;
    float sum_var = 0;
    if (col < ni) {
        for (int i = 0; i < batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col + i * ni];
            sum_var += var_a[col + i * ni];
        }
        mu_s[col] = sum_mu / batch_size;
        var_s[col] = sum_var;
    }
}

__global__ void batchnorm_sample_var_cuda(float const *mu_a, float const *mu_s,
                                          float const *var_s, int ni,
                                          int batch_size, float *var)
/*Compute statistical mean and variance of activation units for full-connected
layer for each batch.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < ni) {
        for (int i = 0; i < batch_size; i++) {
            sum += (mu_a[col + i * ni] - mu_s[col]) *
                   (mu_a[col + i * ni] - mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (batch_size - 1);
    }
}

__global__ void batchnorm_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, float epsilon, int ni,
    int batch_size, float *mu_z, float *var_z)
/*Compute pmean of product WA of batch-normalization layer.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[col] + epsilon);
        int idx = col + row * ni;

        mu_z[idx] =
            inv_sqrt_var_ra * (mu_a[idx] - mu_ra[col]) * mu_w[col] + mu_b[col];

        var_z[idx] = inv_sqrt_var_ra * inv_sqrt_var_ra *
                         (var_a[idx] * mu_w[col] * mu_w[col] +
                          var_w[col] * (mu_a[idx] * mu_a[idx] -
                                        mu_ra[col] * mu_ra[col] + var_a[idx])) +
                     var_b[col];
    }
}

__global__ void batchnorm2d_stat_mean_var_cuda(float const *mu_a,
                                               float const *var_a, int wihi,
                                               int fi, int batch_size,
                                               float *mu_s, float *var_s)
/*Compute sample mean and variance of activation units for batch-normalization
layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0;
    float sum_var = 0;
    if (col < fi) {
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sum_var += var_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        mu_s[col] = sum_mu / (wihi * batch_size);
        var_s[col] = sum_var;
    }
}

__global__ void batchnorm2d_sample_var_cuda(float const *mu_a,
                                            float const *mu_s,
                                            float const *var_s, int wihi,
                                            int fi, int batch_size, float *var)
/*Compute statistical mean and variance of activation units for
batch-normalization layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < fi) {
        for (int i = 0; i < wihi * batch_size; i++) {
            sum += (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]) *
                   (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (wihi * batch_size - 1);
    }
}

__global__ void batchnorm2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a,
    float const *mu_ra, float const *var_ra, float epsilon, int wihi, int fi,
    int m, float *mu_z, float *var_z)
/*Compute mean of product WA of batch-normalization. Note that the previous
layer is a convolutional layer.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k = wihi;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        int div_idx = row % fi;
        float inv_sqrt_var_ra = 1.0f / sqrtf(var_ra[div_idx] + epsilon);

        int idx = col + row * k;

        mu_z[idx] =
            inv_sqrt_var_ra * (mu_a[idx] - mu_ra[div_idx]) * mu_w[div_idx] +
            mu_b[div_idx];

        var_z[idx] = inv_sqrt_var_ra * inv_sqrt_var_ra *
                         (var_a[idx] * mu_w[div_idx] * mu_w[div_idx] +
                          var_w[div_idx] *
                              (mu_a[idx] * mu_a[idx] -
                               mu_ra[div_idx] * mu_ra[div_idx] + var_a[idx])) +
                     var_b[div_idx];
    }
}

__global__ void batchnorm_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int ni, int batch_size, float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is full-connected layer.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < batch_size) {
        float tmp = (1 / sqrtf(var_hat[col] + epsilon)) * mu_w[col] *
                    jcb[col + row * ni];

        delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];

        delta_var[col + row * ni] = tmp * delta_var_out[col + row * ni] * tmp;
    }
}

__global__ void batchnorm2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *var_hat,
    float const *delta_mu_out, float const *delta_var_out, float epsilon,
    int wihi, int fi, int m, float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is convolutional layer.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi * fi, m = B
    {
        float tmp = (1 / sqrtf(var_hat[row % fi] + epsilon)) * mu_w[row % fi] *
                    jcb[col + row * wihi];

        delta_mu[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];

        delta_var[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp;
    }
}

__global__ void batchnorm_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_ra,
    float const *var_ra, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int ni, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to full-connected layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < batch_size; i++) {
            float tmp = (1 / sqrtf(var_ra[col] + epsilon)) *
                        (mu_a[col + i * ni] - mu_ra[col]) * var_w[col];
            sum_mu += tmp * delta_mu_out[col + i * ni];
            sum_var += tmp * delta_var_out[col + i * ni] * tmp;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

__global__ void batchnorm_bwd_delta_b_cuda(float const *var_b,
                                           float const *delta_mu_out,
                                           float const *delta_var_out,
                                           float epsilon, int ni,
                                           int batch_size, float *delta_mu_b,
                                           float *delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to full-connected layer.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < ni) {
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

__global__ void batchnorm2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *mu_ra,
    float const *var_ra, float const *delta_mu_out, float const *delta_var_out,
    float epsilon, int wihi, int fi, int m, float *delta_mu_w,
    float *delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to convolutional layer.
*/
// TODO: remove the duplicates
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi, m = fi*B
    {
        float tmp = (1 / sqrtf(var_ra[row % fi] + epsilon)) *
                    (mu_a[col + row * wihi] - mu_ra[row % fi]) *
                    var_w[row % fi];

        delta_mu_w[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
        delta_var_w[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp;
    }
}

__global__ void batchnorm2d_bwd_delta_b_cuda(float const *var_b,
                                             float const *delta_mu_out,
                                             float const *delta_var_out,
                                             float epsilon, int wihi, int fi,
                                             int m, float *delta_mu_b,
                                             float *delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to convolutional layer.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi && row < m)  // k = wihi, m = fi*B
    {
        float tmp = var_b[row % fi];

        delta_mu_b[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
        delta_var_b[col + row * wihi] =
            tmp * delta_var_out[col + row * wihi] * tmp;
    }
}

////////////////////////////////////////////////////////////////////////////////
//// Layer Norm
////////////////////////////////////////////////////////////////////////////////
LayerNormCuda::LayerNormCuda(const std::vector<int> &normalized_shape,
                             float eps, float momentum, bool bias)
/*
 */
{
    this->normalized_shape = normalized_shape;
    this->epsilon = eps;
    this->momentum = momentum;
    this->bias = bias;
    this->init_weight_bias();
    this->allocate_running_mean_var();
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

LayerNormCuda::~LayerNormCuda() {
    cudaFree(d_mu_ra);
    cudaFree(d_var_ra);
    cudaFree(d_mu_norm_batch);
    cudaFree(d_var_norm_batch);
}

std::string LayerNormCuda::get_layer_info() const
/*
 */
{
    return "LayerNorm()";
}

std::string LayerNormCuda::get_layer_name() const
/*
 */
{
    return "LayerNormCuda";
}

LayerType LayerNormCuda::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void LayerNormCuda::init_weight_bias()
/*
 */
{
    this->num_weights = this->normalized_shape[0];
    float scale = 1.0f / this->num_weights;
    this->mu_w.resize(this->num_weights, 1.0f);
    this->var_w.resize(this->num_weights, scale);
    if (this->bias) {
        this->num_biases = normalized_shape[0];
        this->mu_b.resize(this->num_biases, 0.0f);
        this->var_b.resize(this->num_biases, scale);
    }
    this->allocate_param_memory();
    this->params_to_device();
}

void LayerNormCuda::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);
    this->delta_mu_b.resize(this->num_biases, 0.0f);
    this->delta_var_b.resize(this->num_biases, 0.0f);
    cudaMalloc(&this->d_delta_mu_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_delta_var_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_delta_mu_b, this->num_biases * sizeof(float));
    cudaMalloc(&this->d_delta_var_b, this->num_biases * sizeof(float));
}

void LayerNormCuda::allocate_running_mean_var()
/*
 */
{
    if (this->mu_ra.size() == 0) {
        this->mu_ra.resize(1, 0.0f);
        this->var_ra.resize(1, 1.0f);
        cudaMalloc(&this->d_mu_ra, sizeof(float));
        cudaMalloc(&this->d_var_ra, sizeof(float));
    }

    this->mu_norm_batch.resize(1, 0.0f);
    this->var_norm_batch.resize(1, 1.0f);
    cudaMalloc(&this->d_mu_norm_batch, sizeof(float));
    cudaMalloc(&this->d_var_norm_batch, sizeof(float));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var memory allocation.");
    }
    this->running_mean_var_to_device();
}

void LayerNormCuda::running_mean_var_to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_ra, this->mu_ra.data(),
               this->mu_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_ra, this->var_ra.data(),
               this->var_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_norm_batch, this->mu_norm_batch.data(),
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_norm_batch, this->var_norm_batch.data(),
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var host to device.");
    }
}

void LayerNormCuda::running_mean_var_to_host()
/*
 */
{
    cudaMemcpy(this->mu_ra.data(), this->d_mu_ra,
               this->mu_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ra.data(), this->d_var_ra,
               this->var_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mu_norm_batch.data(), this->d_mu_norm_batch,
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_norm_batch.data(), this->d_var_norm_batch,
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var device to host.");
    }
}

void LayerNormCuda::reset_norm_mean_var()
/*
 */
{
    this->mu_norm_batch.assign({0});
    this->var_norm_batch.assign({0});
    cudaMemcpy(this->d_mu_norm_batch, this->mu_norm_batch.data(),
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_norm_batch, this->var_norm_batch.data(),
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var host to device.");
    }
}

void LayerNormCuda::forward(BaseHiddenStates &input_states,
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
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    int batch_size = input_states.block_size;
    int num_threads = this->num_cuda_threads;
    unsigned int grid_size_ra = (batch_size + num_threads - 1) / num_threads;
    dim3 block_dim(num_threads, num_threads);

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    // Lazy intialization
    float _momentum = this->momentum;
    if (this->first_batch) {
        if (this->training) {
            _momentum = 0.0f;
        }
        this->first_batch = false;
    }
    unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
    unsigned int grid_col = (this->input_size + num_threads - 1) / num_threads;
    dim3 grid_size(grid_col, grid_row);
    auto d_mu_target = this->training ? this->d_mu_norm_batch : this->d_mu_ra;
    auto d_var_target =
        this->training ? this->d_var_norm_batch : this->d_var_ra;
    if (this->training) {
        layernorm_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->input_size,
            batch_size, cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2);

        layernorm_sample_var_cuda<<<grid_size_ra, num_threads>>>(
            cu_input_states->d_mu_a, cu_temp_states->d_tmp_1,
            cu_temp_states->d_tmp_2, this->input_size, batch_size,
            cu_temp_states->d_tmp_2);

        // Compute the sum over the batch size
        this->reset_norm_mean_var();
        norm_sum_reduced<<<grid_row, num_threads>>>(
            cu_temp_states->d_tmp_1, this->d_mu_norm_batch, batch_size);
        norm_sum_reduced<<<grid_row, num_threads>>>(
            cu_temp_states->d_tmp_2, this->d_var_norm_batch, batch_size);

        running_mean_var_cuda<<<1, 1>>>(
            this->d_mu_norm_batch, this->d_var_norm_batch, _momentum,
            batch_size, this->d_mu_ra, this->d_var_ra);
    }
    if (this->normalized_shape.size() == 1) {
        layernorm_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, d_mu_target,
            d_var_target, this->epsilon, this->input_size, batch_size,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        int wihi = this->in_height * this->in_width;
        layernorm2d_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, d_mu_target,
            d_var_target, this->epsilon, wihi, batch_size, this->input_size,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }

    // Update backward state for inferring parameters
    if (this->training) {
        BackwardStateCuda *cu_bwd_states =
            dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states, *cu_bwd_states);
    }
}

void LayerNormCuda::state_backward(BaseBackwardStates &next_bwd_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
    unsigned int grid_col = (this->input_size + num_threads - 1) / num_threads;
    dim3 grid_size(grid_col, grid_row);

    if (this->normalized_shape.size() == 1) {
        layernorm_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_norm_batch,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->epsilon, this->input_size,
            batch_size, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    } else {
        int wihi = this->in_height * this->in_width;

        layernorm2d_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_norm_batch,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->epsilon, wihi,
            this->in_channels, batch_size, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    }
}

void LayerNormCuda::param_backward(BaseBackwardStates &next_bwd_states,
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
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    unsigned int grid_col = (this->input_size + num_threads - 1) / num_threads;

    if (this->normalized_shape.size() == 1) {
        layernorm_bwd_delta_w_cuda<<<grid_col, num_threads>>>(
            this->d_var_w, cu_next_bwd_states->d_mu_a, this->d_mu_norm_batch,
            this->d_var_norm_batch, cu_delta_states->d_delta_mu,
            cu_delta_states->d_delta_var, this->epsilon, this->input_size,
            batch_size, this->d_delta_mu_w, this->d_delta_var_w);

        if (this->bias) {
            layernorm_bwd_delta_b_cuda<<<grid_col, num_threads>>>(
                this->d_var_b, cu_delta_states->d_delta_mu,
                cu_delta_states->d_delta_var, this->epsilon, this->input_size,
                batch_size, this->d_delta_mu_b, this->d_delta_var_b);
        }

    } else {
        int wihi = this->in_height * this->in_width;
        unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);
        unsigned int sum_grid_size =
            (this->in_channels + num_threads - 1) / num_threads;

        // Weights
        // TODO: Not sure if it should be batch_size or batch_size * fi
        layernorm2d_bwd_delta_w_cuda<<<grid_size, block_dim>>>(
            this->d_var_w, cu_next_bwd_states->d_mu_a, this->d_mu_norm_batch,
            this->d_var_norm_batch, cu_delta_states->d_delta_mu,
            cu_delta_states->d_delta_var, this->epsilon, wihi,
            this->in_channels, batch_size, cu_temp_states->d_tmp_1,
            cu_temp_states->d_tmp_2);

        delta_param_sum<<<sum_grid_size, num_threads>>>(
            cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2, wihi,
            this->in_channels, batch_size, this->d_delta_mu_w,
            this->d_delta_var_w);

        // Biases
        if (this->bias) {
            layernorm2d_bwd_delta_b_cuda<<<grid_size, block_dim>>>(
                this->d_var_b, cu_delta_states->d_delta_mu,
                cu_delta_states->d_delta_var, this->epsilon, wihi,
                this->in_channels, batch_size, cu_temp_states->d_tmp_1,
                cu_temp_states->d_tmp_2);

            delta_param_sum<<<sum_grid_size, num_threads>>>(
                cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2, wihi,
                this->in_channels, batch_size, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

std::unique_ptr<BaseLayer> LayerNormCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<LayerNorm>(
        this->normalized_shape, this->epsilon, this->momentum, this->bias);

    host_layer->mu_w = this->mu_w;
    host_layer->var_w = this->var_w;
    host_layer->mu_b = this->mu_b;
    host_layer->var_b = this->var_b;

    return host_layer;
}

std::tuple<std::vector<float>, std::vector<float>>
LayerNormCuda::get_running_mean_var()
/*
 */
{
    return {this->mu_ra, this->var_ra};
}

void LayerNormCuda::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }
    // Transfer data to host
    this->params_to_host();
    this->running_mean_var_to_host();

    // Save the name length and name
    auto layer_name = this->get_layer_name();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (const auto &m_ra : this->mu_ra) {
        file.write(reinterpret_cast<const char *>(&m_ra), sizeof(m_ra));
    }
    for (const auto &v_ra : this->var_ra) {
        file.write(reinterpret_cast<const char *>(&v_ra), sizeof(v_ra));
    }
}

void LayerNormCuda::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_name();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (auto &m_ra : this->mu_ra) {
        file.read(reinterpret_cast<char *>(&m_ra), sizeof(m_ra));
    }
    for (auto &v_ra : this->var_ra) {
        file.read(reinterpret_cast<char *>(&v_ra), sizeof(v_ra));
    }

    // It wont set momentum to zero for running average of norm's mean & var
    this->first_batch = false;

    // Transfer data to device
    this->params_to_device();
    this->running_mean_var_to_device();
}

////////////////////////////////////////////////////////////////////////////////
//// Batch Norm
////////////////////////////////////////////////////////////////////////////////
BatchNorm2dCuda::BatchNorm2dCuda(int num_features, float eps, float momentum,
                                 bool bias)
    : num_features(num_features),
      epsilon(eps),
      momentum(momentum)
/*
 */
{
    this->bias = bias;
    this->init_weight_bias();
    this->allocate_running_mean_var();
    if (this->training) {
        this->allocate_param_delta();
    }
}

BatchNorm2dCuda::~BatchNorm2dCuda()
/*
 */
{
    cudaFree(d_mu_ra);
    cudaFree(d_var_ra);
}

std::string BatchNorm2dCuda::get_layer_info() const
/*
 */
{
    return "BatchNorm()";
}

std::string BatchNorm2dCuda::get_layer_name() const
/*
 */
{
    return "BatchNorm2dCuda";
}

LayerType BatchNorm2dCuda::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void BatchNorm2dCuda::init_weight_bias()
/*
 */
{
    this->num_weights = this->num_features;
    this->num_biases = this->num_features;

    float scale = 1.0f / this->num_weights;
    this->mu_w.resize(this->num_weights, 1.0f);
    this->var_w.resize(this->num_weights, scale);
    if (this->bias) {
        this->mu_b.resize(this->num_weights, 0.0f);
        this->var_b.resize(this->num_weights, scale);

    } else {
        this->num_biases = 0;
    }
    this->allocate_param_memory();
    this->params_to_device();
}

void BatchNorm2dCuda::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);
    this->delta_mu_b.resize(this->num_biases, 0.0f);
    this->delta_var_b.resize(this->num_biases, 0.0f);
    cudaMalloc(&this->d_delta_mu_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_delta_var_w, this->num_weights * sizeof(float));
    cudaMalloc(&this->d_delta_mu_b, this->num_biases * sizeof(float));
    cudaMalloc(&this->d_delta_var_b, this->num_biases * sizeof(float));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void BatchNorm2dCuda::allocate_running_mean_var()
/*
 */
{
    this->mu_ra.resize(this->num_features, 0.0f);
    this->var_ra.resize(this->num_features, 1.0f);
    this->mu_norm_batch.resize(this->num_features, 0.0f);
    this->var_norm_batch.resize(this->num_features, 1.0f);
    cudaMalloc(&this->d_mu_ra, this->num_features * sizeof(float));
    cudaMalloc(&this->d_var_ra, this->num_features * sizeof(float));
    cudaMalloc(&this->d_mu_norm_batch, this->num_features * sizeof(float));
    cudaMalloc(&this->d_var_norm_batch, this->num_features * sizeof(float));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var memory allocation.");
    }
    this->running_mean_var_to_device();
}

void BatchNorm2dCuda::running_mean_var_to_device()
/*
 */
{
    cudaMemcpy(this->d_mu_ra, this->mu_ra.data(),
               this->mu_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_ra, this->var_ra.data(),
               this->var_ra.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_mu_norm_batch, this->mu_norm_batch.data(),
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_var_norm_batch, this->var_norm_batch.data(),
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var host to device.");
    }
}

void BatchNorm2dCuda::running_mean_var_to_host()
/*
 */
{
    cudaMemcpy(this->mu_ra.data(), this->d_mu_ra,
               this->mu_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_ra.data(), this->d_var_ra,
               this->var_ra.size() * sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(this->mu_norm_batch.data(), this->d_mu_norm_batch,
               this->mu_norm_batch.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->var_norm_batch.data(), this->d_var_norm_batch,
               this->var_norm_batch.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Running mean var device to host.");
    }
}

void BatchNorm2dCuda::forward(BaseHiddenStates &input_states,
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
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    int batch_size = input_states.block_size;
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    if (this->input_size == 0 || this->output_size == 0) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }
    float _momentum = this->momentum;
    if (this->first_batch) {
        if (this->training) {
            _momentum = 0.0f;
        }
        this->first_batch = false;
    }

    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    auto d_mu_target = this->training ? this->d_mu_norm_batch : this->d_mu_ra;
    auto d_var_target =
        this->training ? this->d_var_norm_batch : this->d_var_ra;

    if (this->num_features != this->in_channels) {
        unsigned int grid_size_ra =
            (this->input_size + num_threads - 1) / num_threads;

        if (this->training) {
            batchnorm_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                cu_input_states->d_mu_a, cu_input_states->d_var_a,
                this->input_size, batch_size, this->d_mu_norm_batch,
                cu_temp_states->d_tmp_2);

            batchnorm_sample_var_cuda<<<grid_size_ra, num_threads>>>(
                cu_input_states->d_mu_a, this->d_mu_norm_batch,
                cu_temp_states->d_tmp_2, this->input_size, batch_size,
                this->d_var_norm_batch);

            running_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                this->d_mu_norm_batch, this->d_var_norm_batch, _momentum,
                this->input_size, this->d_mu_ra, this->d_var_ra);
        }
        unsigned int grid_col =
            (this->input_size + num_threads - 1) / num_threads;
        unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, d_mu_target,
            d_var_target, this->epsilon, this->input_size, batch_size,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        int wihi = this->in_height * this->in_width;
        unsigned int grid_size_ra =
            (this->in_channels + num_threads - 1) / num_threads;

        if (this->training) {
            batchnorm2d_stat_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                cu_input_states->d_mu_a, cu_input_states->d_var_a, wihi,
                this->in_channels, batch_size, this->d_mu_norm_batch,
                cu_temp_states->d_tmp_2);

            batchnorm2d_sample_var_cuda<<<grid_size_ra, num_threads>>>(
                cu_input_states->d_mu_a, this->d_mu_norm_batch,
                cu_temp_states->d_tmp_2, wihi, this->in_channels, batch_size,
                this->d_var_norm_batch);

            running_mean_var_cuda<<<grid_size_ra, num_threads>>>(
                this->d_mu_norm_batch, this->d_var_norm_batch, _momentum,
                this->in_channels, this->d_mu_ra, this->d_var_ra);
        }

        int fi_batch = this->in_channels * batch_size;
        unsigned int grid_row = (fi_batch + num_threads - 1) / num_threads;
        unsigned int grid_col = (wihi + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm2d_fwd_mean_var_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a, d_mu_target,
            d_var_target, this->epsilon, wihi, this->in_channels, fi_batch,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }

    // Update backward state for inferring parameters
    if (this->training) {
        BackwardStateCuda *cu_bwd_states =
            dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states, *cu_bwd_states);
    }
}

void BatchNorm2dCuda::state_backward(BaseBackwardStates &next_bwd_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int batch_size = cu_input_delta_states->block_size;
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    if (this->in_channels == 0) {
        unsigned int grid_row = (batch_size + num_threads - 1) / num_threads;
        unsigned int grid_col =
            (this->input_size + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_norm_batch,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->epsilon, this->input_size,
            batch_size, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);

    } else {
        int fi_batch = this->in_channels * batch_size;
        int wihi = this->in_width * this->in_height;

        unsigned int grid_row = (fi_batch + num_threads - 1) / num_threads;
        unsigned int grid_col = (wihi + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm2d_bwd_delta_z_cuda<<<grid_size, block_dim>>>(
            this->d_mu_w, cu_next_bwd_states->d_jcb, this->d_var_norm_batch,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->epsilon, wihi,
            this->in_channels, fi_batch, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    }
}

void BatchNorm2dCuda::param_backward(BaseBackwardStates &next_bwd_states,
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
    int num_threads = this->num_cuda_threads;
    dim3 block_dim(num_threads, num_threads);

    if (this->in_channels == 0) {
        unsigned int grid_size =
            (this->input_size + num_threads - 1) / num_threads;

        batchnorm_bwd_delta_w_cuda<<<grid_size, num_threads>>>(
            this->d_var_w, cu_next_bwd_states->d_mu_a, this->d_mu_norm_batch,
            this->d_var_norm_batch, cu_delta_states->d_delta_mu,
            cu_delta_states->d_delta_var, this->epsilon, this->input_size,
            batch_size, this->d_delta_mu_w, this->d_delta_var_w);

        if (this->num_biases > 0) {
            batchnorm_bwd_delta_b_cuda<<<grid_size, num_threads>>>(
                this->d_var_b, cu_delta_states->d_delta_mu,
                cu_delta_states->d_delta_var, this->epsilon, this->input_size,
                batch_size, this->d_delta_mu_b, this->d_delta_var_b);
        }

    } else {
        int wihi = this->in_width * this->in_height;
        int fi_batch = this->in_channels * batch_size;

        unsigned int grid_row = (fi_batch + num_threads - 1) / num_threads;
        unsigned int grid_col = (wihi + num_threads - 1) / num_threads;
        unsigned int sum_grid_size =
            (this->in_channels + num_threads - 1) / num_threads;
        dim3 grid_size(grid_col, grid_row);

        batchnorm2d_bwd_delta_w_cuda<<<grid_size, block_dim>>>(
            this->d_var_w, cu_next_bwd_states->d_mu_a, this->d_mu_norm_batch,
            this->d_var_norm_batch, cu_delta_states->d_delta_mu,
            cu_delta_states->d_delta_var, this->epsilon, wihi,
            this->in_channels, fi_batch, cu_temp_states->d_tmp_1,
            cu_temp_states->d_tmp_2);

        delta_param_sum<<<sum_grid_size, num_threads>>>(
            cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2, wihi,
            this->in_channels, batch_size, this->d_delta_mu_w,
            this->d_delta_var_w);

        if (this->num_biases > 0) {
            batchnorm2d_bwd_delta_b_cuda<<<grid_size, block_dim>>>(
                this->d_var_b, cu_delta_states->d_delta_mu,
                cu_delta_states->d_delta_var, this->epsilon, wihi,
                this->in_channels, fi_batch, cu_temp_states->d_tmp_1,
                cu_temp_states->d_tmp_2);

            delta_param_sum<<<sum_grid_size, num_threads>>>(
                cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2, wihi,
                this->in_channels, batch_size, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

std::unique_ptr<BaseLayer> BatchNorm2dCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<BatchNorm2d>(
        this->num_features, this->epsilon, this->momentum, this->bias);

    host_layer->mu_w = this->mu_w;
    host_layer->var_w = this->var_w;
    host_layer->mu_b = this->mu_b;
    host_layer->var_b = this->var_b;

    return host_layer;
}

void BatchNorm2dCuda::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }
    // Transfer data to host
    this->params_to_host();
    this->running_mean_var_to_host();

    // Save the name length and name
    auto layer_name = this->get_layer_name();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (const auto &m_ra : this->mu_ra) {
        file.write(reinterpret_cast<const char *>(&m_ra), sizeof(m_ra));
    }
    for (const auto &v_ra : this->var_ra) {
        file.write(reinterpret_cast<const char *>(&v_ra), sizeof(v_ra));
    }
}

void BatchNorm2dCuda::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_name();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (auto &m_ra : this->mu_ra) {
        file.read(reinterpret_cast<char *>(&m_ra), sizeof(m_ra));
    }
    for (auto &v_ra : this->var_ra) {
        file.read(reinterpret_cast<char *>(&v_ra), sizeof(v_ra));
    }

    // It wont set momentum to zero for running average of norm's mean & var
    this->first_batch = false;

    // Transfer data to device
    this->params_to_device();
    this->running_mean_var_to_device();
}
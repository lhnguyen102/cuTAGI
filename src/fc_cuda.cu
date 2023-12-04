///////////////////////////////////////////////////////////////////////////////
// File:         fc_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 03, 2023
// Updated:      December 03, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/fc_cuda.cuh"

__global__ void fwd_mean_var(float const *mu_w, float const *var_w,
                             float const *mu_b, float const *var_b,
                             const float *mu_a, const float *var_a,
                             size_t input_size, size_t output_size,
                             int batch_size, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    float mu_a_tmp = 0.0f;
    float var_a_tmp = 0.0f;
    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            mu_a_tmp = mu_a[input_size * col + i];
            var_a_tmp = var_a[input_size * col + i];

            if (mu_a_tmp != 0) {
                sum_mu += mu_w[row * input_size + i] * mu_a_tmp;
                sum_var +=
                    (mu_w[row * input_size + i] * mu_w[row * input_size + i] +
                     var_w[row * input_size + i]) *
                        var_a_tmp +
                    var_w[row * input_size + i] * mu_a_tmp * mu_a_tmp;
            }
        }
        mu_z[col * output_size + row] = sum_mu + mu_b[row];
        var_z[col * output_size + row] = sum_var + var_b[row];
    }
}

__global__ void fwd_full_cov(float const *mu_w, float const *var_a_f,
                             size_t input_size, size_t output_size,
                             int batch_size, float *var_z_fp)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tu = 0, k = 0;
    float sum = 0.0f;
    float var_a_in = 0.0f;

    if (col <= (row % output_size) && row < output_size * batch_size) {
        for (int i = 0; i < input_size * input_size; i++) {
            int row_in = i / input_size;
            int col_in = i % input_size;
            if (row_in > col_in)  // lower triangle
            {
                tu = (input_size * col_in - ((col_in * (col_in + 1)) / 2) +
                      row_in);
            } else {
                tu = (input_size * row_in - ((row_in * (row_in + 1)) / 2) +
                      col_in);
            }
            var_a_in = var_a_f[tu + (row / output_size) *
                                        (input_size * (input_size + 1)) / 2];

            sum += mu_w[i % input_size + (row % output_size) * input_size] *
                   var_a_in *
                   mu_w[i / input_size + (col % output_size) * input_size];
        }
        k = output_size * col - ((col * (col + 1)) / 2) + row % output_size +
            (row / output_size) * (((output_size + 1) * output_size) / 2);
        var_z_fp[k] = sum;
    }
}

__global__ void fwd_full_var(float const *mu_w, float const *var_w,
                             float const *var_b, float const *mu_a,
                             float const *var_a, float const *var_z_fp,
                             size_t input_size, size_t output_size,
                             int batch_size, float *var_z, float *var_z_f)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    float final_sum = 0;
    int k;

    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            sum += var_w[row * input_size + i] * var_a[input_size * col + i] +
                   var_w[row * input_size + i] * mu_a[input_size * col + i] *
                       mu_a[input_size * col + i];
        }
        k = output_size * row - (row * (row - 1)) / 2 +
            col * (output_size * (output_size + 1)) / 2;

        final_sum = sum + var_b[row] + var_z_fp[k];

        var_z[col * output_size + row] = final_sum;
    }
}

__global__ void bwd_delta_z(float const *mu_w, float const *jcb,
                            float const *delta_mu_out,
                            float const *delta_var_out, size_t input_size,
                            size_t output_size, int batch_size,
                            float *delta_mu_in, float *delta_var_in)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;
    if (col < batch_size && row < input_size) {
        for (int i = 0; i < output_size; i++) {
            sum_mu += mu_w[input_size * i + row] *
                      delta_mu_out[col * output_size + i];

            sum_var += mu_w[input_size * i + row] *
                       delta_var_out[col * output_size + i] *
                       mu_w[input_size * i + row];
        }
        delta_mu_in[col * input_size + row] =
            sum_mu * jcb[col * input_size + row];

        delta_var_in[col * input_size + row] =
            sum_var * jcb[col * input_size + row] * jcb[col * input_size + row];
    }
}

__global__ void bwd_delta_w(float const *var_w, float const *mu_a,
                            float const *delta_mu_out,
                            float const *delta_var_out, size_t input_size,
                            size_t output_size, int batch_size,
                            float *delta_mu_w, float *delta_var_w)
/**/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < output_size && row < input_size) {
        for (int i = 0; i < batch_size; i++) {
            sum_mu += mu_a[input_size * i + row] *
                      delta_mu_out[output_size * i + col];

            sum_var += mu_a[input_size * i + row] * mu_a[input_size * i + row] *
                       delta_var_out[output_size * i + col];
        }

        delta_mu_w[col * input_size + row] =
            sum_mu * var_w[col * input_size + row];

        delta_var_w[col * input_size + row] = sum_var *
                                              var_w[col * input_size + row] *
                                              var_w[col * input_size + row];
    }
}

__global__ void bwd_delta_b(float const *var_b, float const *delta_mu_out,
                            float const *delta_var_out, size_t input_size,
                            size_t output_size, int batch_size,
                            float *delta_mu_b, float *delta_var_b)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < output_size && row < input_size) {
        for (int i = 0; i < batch_size; i++) {
            sum_mu += delta_mu_out[input_size * i + row];
            sum_var += delta_var_out[input_size * i + row];
        }

        delta_mu_b[col * input_size + row] =
            sum_mu * var_b[col * input_size + row];

        delta_var_b[col * input_size + row] = sum_var *
                                              var_b[col * input_size + row] *
                                              var_b[col * input_size + row];
    }
}
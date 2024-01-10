///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 08, 2024
// Updated:      January 09, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/conv2d_layer.h"
#include "../include/pooling_layer.h"
#include "../include/pooling_layer_cuda.cuh"

AvgPool2dCuda::AvgPool2dCuda(size_t kernel_size, int stride, int padding,
                             int padding_type)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type) {}

AvgPool2dCuda::~AvgPool2dCuda() {
    cudaFree(d_pool_idx);
    cudaFree(d_z_ud_idx);
}

std::string AvgPool2dCuda::get_layer_info() const {
    return "AvgPool2d(" + std::to_string(this->kernel_size) + ")";
    ;
}

std::string AvgPool2dCuda::get_layer_name() const { return "AvgPool2dCuda"; }

LayerType AvgPool2dCuda::get_layer_type() const { return LayerType::AvgPool2d; }

void AvgPool2dCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states)
/*
 */
{}

void AvgPool2dCuda::state_backward(BaseBackwardStates &next_bwd_states,
                                   BaseDeltaStates &input_delta_states,
                                   BaseDeltaStates &output_hidden_states,
                                   BaseTempStates &temp_states)
/*
 */
{}

void AvgPool2dCuda::param_backward(BaseBackwardStates &next_bwd_states,
                                   BaseDeltaStates &delta_states,
                                   BaseTempStates &temp_states)
/*
 */
{}

void AvgPool2dCuda::lazy_init(size_t width, size_t height, size_t depth,
                              int batch_size)
/*
 */
{
    this->in_width = width;
    this->in_height = height;
    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride, width,
                                       height, this->padding,
                                       this->padding_type);

    // int pad_idx_in =
    //     net.widths[j] * net.heights[j] * net.filters[j] * net.batch_size + 1;
    // int pad_idx_out = net.widths[j + 1] * net.heights[j + 1] *
    //                       net.filters[j + 1] * net.batch_size +
    //                   1;
    // auto pool_idx = get_pool_index(
    //     this->kernel_size, this->stride, this->in_width, this->in_height,
    //     this->out_width, this->out_height, this->padding,
    //     this->padding_type);
}

////////////////////////////////////////////////////////////////////////////////
// CUDA Kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void avgpool2d_fwd_overlapped_mean_var(
    float const *mu_a, float const *var_a, int const *a_idx, int woho, int wihi,
    int ki2, int k, int pad_idx, float *mu_z, float *var_z)
/*Compute product mean & variance WA for average pooling for the case where
there is the overlap when sliding kernel size.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu_z = 0;
    float sum_var_z = 0;
    int a_idx_tmp = 0;

    if (col < k && row < 1) {
        for (int i = 0; i < ki2; i++) {
            a_idx_tmp = a_idx[col % woho + woho * i] + (col / woho) * wihi;
            if (a_idx_tmp < pad_idx) {
                // index in a_idx starts at 1
                sum_mu_z += mu_a[a_idx_tmp - 1];
                sum_var_z += var_a[a_idx_tmp - 1];
            }
        }
        mu_z[col] = sum_mu_z / ki2;
        var_z[col] = sum_var_z / (ki2 * ki2);
    }
}

__global__ void avgpool2d_fwd_mean_var(float const *mu_a, float const *var_a,
                                       int const *a_idx, int woho, int wihi,
                                       int ki2, int k, float *mu_z,
                                       float *var_z)
/* Compute product mean & variance WA for average pooling for the case there
is no overlap when sliding kernel size.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu_z = 0;
    float sum_var_z = 0;
    int a_idx_tmp = 0;
    if (col < k && row < 1) {
        for (int i = 0; i < ki2; i++) {
            // index in a_idx starts at 1
            a_idx_tmp = a_idx[col % woho + woho * i] + (col / woho) * wihi - 1;
            sum_mu_z += mu_a[a_idx_tmp];
            sum_var_z += var_a[a_idx_tmp];
        }
        mu_z[col] = sum_mu_z / ki2;
        var_z[col] = sum_var_z / (ki2 * ki2);
    }
}

__global__ void avgpool2d_bwd_overlapped_delta_z(
    float const *var_z, float const *jcb, float const *delta_mu_out,
    float const *delta_var_out, int const *z_ud_idx, int woho, int wihi,
    int ki2, int n, int k, int pad_idx, float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
 average pooling layer. Note that this case the kernel size overlap each other
 when scaning images.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_delta_mu = 0;
    float sum_delta_var = 0;
    int z_idx_tmp = 0;
    if (col < k && row < 1) {
        for (int i = 0; i < n; i++) {
            z_idx_tmp = z_ud_idx[col % wihi + wihi * i] + (col / wihi) * woho;
            if (z_idx_tmp < pad_idx) {
                sum_delta_mu += delta_mu_out[z_idx_tmp - 1];
                sum_delta_var += delta_var_out[z_idx_tmp - 1];
            }
        }
        delta_mu[col] = sum_delta_mu * jcb[col] / ki2;
        delta_var[col] = sum_delta_var * jcb[col] * jcb[col] / (ki2 * ki2);
    }
}

__global__ void avgpool2d_bwd_delta_z(float const *var_z, float const *jcb,
                                      float const *delta_mu_out,
                                      float const *delta_var_out, int wo,
                                      int ki, int ki2, int m, int k,
                                      float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
 average pooling layer.

 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi * fi * B / (k*wo); m = k*wo
    {
        delta_mu[row + col * m] =
            delta_mu_out[row / ki + (col / ki) * wo] * jcb[row + col * m] / ki2;

        delta_var[row + col * m] = delta_var_out[row / ki + (col / ki) * wo] *
                                   jcb[row + col * m] * jcb[row + col * m] /
                                   (ki2 * ki2);
    }
}

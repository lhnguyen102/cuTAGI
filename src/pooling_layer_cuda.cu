#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/config.h"
#include "../include/conv2d_layer.h"
#include "../include/cuda_error_checking.cuh"
#include "../include/pooling_layer.h"
#include "../include/pooling_layer_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// CUDA Kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void avgpool2d_fwd_overlapped_mean_var_cuda(
    float const *mu_a, float const *var_a, int const *a_idx, int woho, int wihi,
    int ki, int k, int pad_idx, float *mu_z, float *var_z)
/*Compute product mean & variance WA for average pooling for the case where
there is the overlap when sliding kernel size.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu_z = 0;
    float sum_var_z = 0;
    int a_idx_tmp = 0;
    int ki2 = ki * ki;
    if (col < k) {
        for (int i = 0; i < ki2; i++) {
            a_idx_tmp = a_idx[col % woho + woho * i];
            if (a_idx_tmp > -1) {
                a_idx_tmp += (col / woho) * wihi;
                // index in a_idx starts at 1
                sum_mu_z += mu_a[a_idx_tmp - 1];
                sum_var_z += var_a[a_idx_tmp - 1];
            }
        }
        mu_z[col] = sum_mu_z / ki2;
        var_z[col] = sum_var_z / (ki2 * ki2);
    }
}

__global__ void avgpool2d_fwd_mean_var_cuda(float const *mu_a,
                                            float const *var_a,
                                            int const *a_idx, int woho,
                                            int wihi, int ki, int k,
                                            float *mu_z, float *var_z)
/* Compute product mean & variance WA for average pooling for the case there
is no overlap when sliding kernel size.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu_z = 0;
    float sum_var_z = 0;
    int a_idx_tmp = 0;
    int ki2 = ki * ki;
    if (col < k) {
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

__global__ void avgpool2d_bwd_overlapped_delta_z_cuda(
    float const *jcb, float const *delta_mu_out, float const *delta_var_out,
    int const *z_ud_idx, int woho, int wihi, int ki, int n, int k, int pad_idx,
    float *delta_mu, float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
 average pooling layer. Note that this case the kernel size overlap each other
 when scaning images.
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_delta_mu = 0;
    float sum_delta_var = 0;
    int z_idx_tmp = 0;
    int ki2 = ki * ki;
    if (col < k) {
        for (int i = 0; i < n; i++) {
            z_idx_tmp = z_ud_idx[col % wihi + wihi * i];
            if (z_idx_tmp > -1) {
                z_idx_tmp += (col / wihi) * woho;
                sum_delta_mu += delta_mu_out[z_idx_tmp - 1];
                sum_delta_var += delta_var_out[z_idx_tmp - 1];
            }
        }
        delta_mu[col] = sum_delta_mu * jcb[col] / ki2;
        delta_var[col] = sum_delta_var * jcb[col] * jcb[col] / (ki2 * ki2);
    }
}

__global__ void avgpool2d_bwd_delta_z_cuda(float const *jcb,
                                           float const *delta_mu_out,
                                           float const *delta_var_out, int wo,
                                           int ki, int k, float *delta_mu,
                                           float *delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
 average pooling layer.

 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ki2 = ki * ki;
    int m = ki * wo;
    if (col < k && row < m)  // k = wihi * fi * B / (k*wo); m = k*wo
    {
        delta_mu[row + col * m] =
            delta_mu_out[row / ki + (col / ki) * wo] * jcb[row + col * m] / ki2;

        delta_var[row + col * m] = delta_var_out[row / ki + (col / ki) * wo] *
                                   jcb[row + col * m] * jcb[row + col * m] /
                                   (ki2 * ki2);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Avg. Pooling 2D
////////////////////////////////////////////////////////////////////////////////

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
    return "AvgPool2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

std::string AvgPool2dCuda::get_layer_name() const { return "AvgPool2dCuda"; }

LayerType AvgPool2dCuda::get_layer_type() const { return LayerType::Pool2d; }

void AvgPool2dCuda::compute_input_output_size(const InitArgs &args)
/*
 */
{
    this->in_width = args.width;
    this->in_height = args.height;
    this->in_channels = args.depth;
    this->out_channels = args.depth;

    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride,
                                       this->in_width, this->in_height,
                                       this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void AvgPool2dCuda::forward(BaseHiddenStates &input_states,
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

    int batch_size = input_states.block_size;
    unsigned int threads = this->num_cuda_threads;

    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
    }

    // Assign output dimensions
    cu_output_states->width = this->out_width;
    cu_output_states->height = this->out_height;
    cu_output_states->depth = this->out_channels;
    cu_output_states->block_size = batch_size;
    cu_output_states->actual_size = this->output_size;

    // Launch kernels
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int num_states = woho * this->out_channels * batch_size;
    int pad_idx_in = wihi * this->in_channels * batch_size + 1;

    unsigned int grid_size = (num_states + threads - 1) / threads;

    if (this->overlap) {
        avgpool2d_fwd_overlapped_mean_var_cuda<<<grid_size, threads>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_pool_idx,
            woho, wihi, this->kernel_size, num_states, pad_idx_in,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        avgpool2d_fwd_mean_var_cuda<<<grid_size, threads>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_pool_idx,
            woho, wihi, this->kernel_size, num_states, cu_output_states->d_mu_a,
            cu_output_states->d_var_a);
    }

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void AvgPool2dCuda::backward(BaseDeltaStates &input_delta_states,
                             BaseDeltaStates &output_delta_states,
                             BaseTempStates &temp_states, bool state_udapte)
/**/
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    unsigned int num_threads = this->num_cuda_threads;

    // Launch kernel
    if (state_udapte) {
        int woho = this->out_width * this->out_height;
        int wihi = this->in_width * this->in_height;
        int pad_out_idx = woho * this->out_channels * batch_size + 1;
        if (overlap) {
            int num_in_states = this->in_width * this->in_height *
                                this->in_channels * batch_size;
            unsigned int grid_size =
                (num_in_states + num_threads - 1) / num_threads;

            avgpool2d_bwd_overlapped_delta_z_cuda<<<grid_size, num_threads>>>(
                cu_next_bwd_states->d_jcb, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->d_z_ud_idx, woho,
                wihi, this->kernel_size, this->col_z_ud, num_in_states,
                pad_out_idx, cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);

        } else {
            int kiwo = this->kernel_size * this->out_width;
            int nums = wihi * this->in_channels * batch_size / kiwo;
            unsigned int grid_row = (kiwo + num_threads - 1) / num_threads;
            unsigned int grid_col = (nums + num_threads - 1) / num_threads;
            dim3 dim_grid(grid_col, grid_row);
            dim3 dim_block(num_threads, num_threads);

            avgpool2d_bwd_delta_z_cuda<<<dim_grid, dim_block>>>(
                cu_next_bwd_states->d_jcb, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->out_width,
                this->kernel_size, nums, cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
        }
    }
}

void AvgPool2dCuda::lazy_index_init()
/*
 */
{
    if (this->kernel_size == this->stride ||
        this->kernel_size == this->in_width) {
        this->overlap = false;
    }

    int pad_idx_in = -1;
    int pad_idx_out = -1;

    auto idx = get_pool_index(this->kernel_size, this->stride, this->in_width,
                              this->in_height, this->out_width,
                              this->out_height, this->padding,
                              this->padding_type, pad_idx_in, pad_idx_out);

    this->pool_idx = idx.pool_idx;
    this->z_ud_idx = idx.z_ud_idx;
    this->row_zw = idx.w;
    this->col_z_ud = idx.h;

    // Allocate memory for indices and send them to cuda device
    this->allocate_avgpool2d_index();
    this->avgpool2d_index_to_device();
}

void AvgPool2dCuda::allocate_avgpool2d_index()
/*
 */
{
    // Memory aligment
    unsigned int size_pool_idx =
        ((this->pool_idx.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    unsigned int size_z_ud_idx =
        ((this->z_ud_idx.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    cudaMalloc(&this->d_pool_idx, size_pool_idx * sizeof(int));
    cudaMalloc(&this->d_z_ud_idx, size_z_ud_idx * sizeof(int));
    CHECK_LAST_CUDA_ERROR();
}

void AvgPool2dCuda::avgpool2d_index_to_device()
/*
 */
{
    cudaMemcpy(this->d_pool_idx, this->pool_idx.data(),
               this->pool_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_z_ud_idx, this->z_ud_idx.data(),
               this->z_ud_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    CHECK_LAST_CUDA_ERROR();
}

void AvgPool2dCuda::preinit_layer() {
    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
    }
}

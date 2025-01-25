#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/config.h"
#include "../include/conv2d_layer.h"
#include "../include/cuda_error_checking.cuh"
#include "../include/custom_logger.h"
#include "../include/max_pooling_layer_cuda.cuh"
#include "../include/pooling_layer.h"
////////////////////////////////////////////////////////////////////////////////
// Kernels for MaxPool2dCuda
////////////////////////////////////////////////////////////////////////////////
__global__ void max2dpool_overlapped_mean_var_cuda(
    float const *mu_a, float const *var_a, int const *a_idx, int woho, int wihi,
    int ki, int k, int *max_pool_idx, float *mu_z, float *var_z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float max_mu_z = -1e9;
    float max_var_z = -1e9;
    int max_pool_idx_tmp = -1;
    int ki2 = ki * ki;
    if (col < k) {
        for (int i = 0; i < ki2; i++) {
            int a_idx_tmp = a_idx[col % woho + woho * i];
            if (a_idx_tmp > -1) {
                // index in a_idx starts at 1
                a_idx_tmp += (col / woho) * wihi - 1;
                if (mu_a[a_idx_tmp] > max_mu_z) {
                    max_mu_z = mu_a[a_idx_tmp];
                    max_var_z = var_a[a_idx_tmp];
                    max_pool_idx_tmp = a_idx_tmp;
                }
            }
        }
        mu_z[col] = max_mu_z;
        var_z[col] = max_var_z;
        max_pool_idx[col] = max_pool_idx_tmp;
    }
}

__global__ void max2dpool_mean_var_cuda(float const *mu_a, float const *var_a,
                                        int const *a_idx, int woho, int wihi,
                                        int ki, int k, int *max_pool_idx,
                                        float *mu_z, float *var_z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float max_mu_z = -1e9;
    float max_var_z = -1e9;
    int max_pool_idx_tmp = -1;
    int ki2 = ki * ki;

    if (col < k) {
        for (int i = 0; i < ki2; i++) {
            // index in a_idx starts at 1
            int a_idx_tmp =
                a_idx[col % woho + woho * i] + (col / woho) * wihi - 1;

            if (mu_a[a_idx_tmp] > max_mu_z) {
                max_mu_z = mu_a[a_idx_tmp];
                max_var_z = var_a[a_idx_tmp];
                max_pool_idx_tmp = a_idx_tmp;
            }
        }
        mu_z[col] = max_mu_z;
        var_z[col] = max_var_z;
        max_pool_idx[col] = max_pool_idx_tmp;
    }
}

__global__ void max2dpool_bwd_overlapped_delta_z_cuda(
    int const *max_pool_idx, float const *jcb, float const *delta_mu_out,
    float const *delta_var_out, int num_out_states, float *delta_mu,
    float *delta_var) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_out_states) {
        int idx = max_pool_idx[col];

        float delta_mu_tmp = delta_mu_out[col] * jcb[idx];
        float delta_var_tmp = delta_var_out[col] * jcb[idx] * jcb[idx];
        atomicAdd(&delta_mu[idx], delta_mu_tmp);
        atomicAdd(&delta_var[idx], delta_var_tmp);
    }
}

__global__ void max2dpool_bwd_delta_z_cuda(int const *max_pool_idx,
                                           float const *jcb,
                                           float const *delta_mu_out,
                                           float const *delta_var_out,
                                           int num_out_states, float *delta_mu,
                                           float *delta_var) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < num_out_states) {
        int idx = max_pool_idx[col];
        delta_mu[idx] = delta_mu_out[col] * jcb[idx];
        delta_var[idx] = delta_var_out[col] * jcb[idx] * jcb[idx];
    }
}

////////////////////////////////////////////////////////////////////////////////
// MaxPool2dCuda
////////////////////////////////////////////////////////////////////////////////
MaxPool2dCuda::MaxPool2dCuda(size_t kernel_size, int stride, int padding,
                             int padding_type)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type) {}

MaxPool2dCuda::~MaxPool2dCuda() {}

std::string MaxPool2dCuda::get_layer_info() const {
    return "MaxPool2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

std::string MaxPool2dCuda::get_layer_name() const { return "MaxPool2dCuda"; }

LayerType MaxPool2dCuda::get_layer_type() const { return LayerType::Pool2d; }

void MaxPool2dCuda::compute_input_output_size(const InitArgs &args) {
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

void MaxPool2dCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states) {
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = input_states.block_size;
    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->allocate_max_val_index(batch_size);
    }
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

    int THREADS_PER_BLOCK = 256;
    unsigned int grid_size =
        (num_states + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (this->overlap) {
        max2dpool_overlapped_mean_var_cuda<<<grid_size, THREADS_PER_BLOCK>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_pool_idx,
            woho, wihi, this->kernel_size, num_states, this->d_max_pool_idx,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    } else {
        max2dpool_mean_var_cuda<<<grid_size, THREADS_PER_BLOCK>>>(
            cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_pool_idx,
            woho, wihi, this->kernel_size, num_states, this->d_max_pool_idx,
            cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void MaxPool2dCuda::backward(BaseDeltaStates &input_delta_states,
                             BaseDeltaStates &output_delta_states,
                             BaseTempStates &temp_states, bool state_update) {
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int woho = this->out_width * this->out_height;
    int num_out_states = woho * this->out_channels * batch_size;
    unsigned int THREADS_PER_BLOCK = 256;
    unsigned int grid_size =
        (num_out_states + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (state_update) {
        cu_output_delta_states->reset_zeros();
        if (this->overlap) {
            max2dpool_bwd_overlapped_delta_z_cuda<<<grid_size,
                                                    THREADS_PER_BLOCK>>>(
                this->d_max_pool_idx, cu_next_bwd_states->d_jcb,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, num_out_states,
                cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
        } else {
            max2dpool_bwd_delta_z_cuda<<<grid_size, THREADS_PER_BLOCK>>>(
                this->d_max_pool_idx, cu_next_bwd_states->d_jcb,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, num_out_states,
                cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
        }
    }
}

void MaxPool2dCuda::lazy_index_init()
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

    // Allocate memory for indices and send them to cuda device
    this->allocate_maxpool2d_index();
    this->maxpool2d_index_to_device();
}

void MaxPool2dCuda::allocate_maxpool2d_index()
/*
 */
{
    // Memory aligment
    unsigned int size_pool_idx =
        ((this->pool_idx.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    cudaMalloc(&this->d_pool_idx, size_pool_idx * sizeof(int));
    CHECK_LAST_CUDA_ERROR();
}

void MaxPool2dCuda::maxpool2d_index_to_device()
/*
 */
{
    cudaMemcpy(this->d_pool_idx, this->pool_idx.data(),
               this->pool_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    CHECK_LAST_CUDA_ERROR();
}

void MaxPool2dCuda::preinit_layer() {
    if (this->pool_idx.size() == 0) {
        this->lazy_index_init();
    }
}

void MaxPool2dCuda::allocate_max_val_index(int batch_size) {
    // Memory aligment
    int wohofo = this->out_width * this->out_height * this->out_channels;
    unsigned int size_max_pool_idx =
        ((wohofo * batch_size + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    this->max_pool_idx.resize(size_max_pool_idx);

    cudaMalloc(&this->d_max_pool_idx, size_max_pool_idx * sizeof(int));
    CHECK_LAST_CUDA_ERROR();
    this->max_val_index_to_device();
}

void MaxPool2dCuda::max_val_index_to_device() {
    cudaMemcpy(this->d_max_pool_idx, this->max_pool_idx.data(),
               this->max_pool_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    CHECK_LAST_CUDA_ERROR();
}

void MaxPool2dCuda::max_val_index_to_host() {
    cudaMemcpy(this->max_pool_idx.data(), this->d_max_pool_idx,
               this->max_pool_idx.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
    CHECK_LAST_CUDA_ERROR();
}

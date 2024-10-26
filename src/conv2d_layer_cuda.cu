///////////////////////////////////////////////////////////////////////////////
// File:         conv2d_layer_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 04, 2024
// Updated:      July 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../include/base_layer.h"
#include "../include/config.h"
#include "../include/conv2d_cuda_kernel.cuh"
#include "../include/conv2d_layer.h"
#include "../include/conv2d_layer_cuda.cuh"
#include "../include/cuda_error_checking.cuh"
#include "../include/param_init.h"

////////////////////////////////////////////////////////////////////////////////
// Conv2d Cuda Layer
////////////////////////////////////////////////////////////////////////////////
void conv2d_forward_cuda(HiddenStateCuda *&cu_input_states, const float *d_mu_w,
                         const float *d_var_w, const float *d_mu_b,
                         const float *d_var_b, const int *d_idx_mwa_2,
                         int out_channels, int woho, int in_channels, int wihi,
                         int kernel_size, int batch_size, int pad_idx,
                         bool bias, HiddenStateCuda *&cu_output_states)
/**/
{
    int ki2_fi = kernel_size * kernel_size * in_channels;
    if (batch_size * woho > 1024) {
        constexpr unsigned int BLOCK_SIZE = 64U;
        constexpr unsigned int THREAD_TILE = 4U;
        constexpr unsigned int WAPRS_X = 4U;
        constexpr unsigned int WAPRS_Y = 2U;
        constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WAPRS_X;
        constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WAPRS_Y;
        constexpr unsigned int THREAD_X = WAPRS_X * (WARP_TILE_X / THREAD_TILE);
        constexpr unsigned int THREAD_Y = WAPRS_Y * (WARP_TILE_Y / THREAD_TILE);
        constexpr unsigned int BLOCK_TILE_K = THREAD_Y;
        constexpr size_t THREADS = THREAD_X * THREAD_Y;
        dim3 block_dim(THREAD_X, THREAD_Y, 1U);
        dim3 grid_dim(
            (static_cast<unsigned int>(batch_size * woho) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            (static_cast<unsigned int>(out_channels) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            1U);

        if (ki2_fi % PACK_SIZE == 0) {
            constexpr size_t SMEM_PADDING = 0;
            conv2d_fwd_mean_var_cuda_v3<float, BLOCK_SIZE, BLOCK_TILE_K,
                                        THREAD_TILE, THREADS, WARP_TILE_X,
                                        WARP_TILE_Y, PACK_SIZE, SMEM_PADDING>
                <<<grid_dim, block_dim>>>(
                    d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                    cu_input_states->d_var_a, d_idx_mwa_2, woho, out_channels,
                    wihi, in_channels, kernel_size, batch_size, bias,
                    cu_output_states->d_mu_a, cu_output_states->d_var_a);
        } else {
            constexpr size_t SMEM_PADDING = 2;
            conv2d_fwd_mean_var_cuda_v2<float, BLOCK_SIZE, BLOCK_TILE_K,
                                        THREAD_TILE, THREADS, WARP_TILE_X,
                                        WARP_TILE_Y, SMEM_PADDING>
                <<<grid_dim, block_dim>>>(
                    d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                    cu_input_states->d_var_a, d_idx_mwa_2, woho, out_channels,
                    wihi, in_channels, kernel_size, batch_size, bias,
                    cu_output_states->d_mu_a, cu_output_states->d_var_a);
        }
    } else {
        constexpr unsigned int BLOCK_TILE = 16;
        constexpr size_t SMEM_PADDING = WARP_SIZE / BLOCK_TILE;
        int woho_batch = woho * batch_size;
        unsigned int grid_row = (out_channels + BLOCK_TILE - 1) / BLOCK_TILE;
        unsigned int grid_col = (woho_batch + BLOCK_TILE - 1) / BLOCK_TILE;

        dim3 dim_grid(grid_col, grid_row);
        dim3 dim_block(BLOCK_TILE, BLOCK_TILE);

        conv2d_fwd_mean_var_cuda_v1<BLOCK_TILE, SMEM_PADDING>
            <<<dim_grid, dim_block>>>(
                d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                cu_input_states->d_var_a, d_idx_mwa_2, woho, out_channels, wihi,
                in_channels, kernel_size, batch_size, bias,
                cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }
}

void conv2d_state_backward_cuda(
    DeltaStateCuda *&cu_input_delta_states, TempStateCuda *&cu_temp_states,
    const float *d_mu_w, const int *d_idx_mwa_2, const int *d_idx_cov_zwa_1,
    const int *d_idx_var_z_ud, int out_channels, int woho, int in_channels,
    int wihi, int kernel_size, int row_zw, int row_zw_fo, int batch_size,
    int pad_param_idx, DeltaStateCuda *&cu_output_delta_states)
/*
 */
{
    if (batch_size * wihi > 1024) {
        constexpr unsigned int BLOCK_SIZE = 64U;
        constexpr unsigned int THREAD_TILE = 4U;
        constexpr unsigned int WAPRS_X = 4U;
        constexpr unsigned int WAPRS_Y = 2U;
        constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WAPRS_X;
        constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WAPRS_Y;
        constexpr unsigned int THREAD_X = WAPRS_X * (WARP_TILE_X / THREAD_TILE);
        constexpr unsigned int THREAD_Y = WAPRS_Y * (WARP_TILE_Y / THREAD_TILE);
        constexpr unsigned int BLOCK_TILE_K = 16;
        constexpr size_t THREADS = THREAD_X * THREAD_Y;
        constexpr size_t SMEM_PADDING = 0;
        dim3 block_dim(THREAD_X, THREAD_Y, 1U);
        dim3 grid_dim(
            (static_cast<unsigned int>(wihi * batch_size) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            (static_cast<unsigned int>(in_channels) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            1U);

        conv2d_bwd_delta_z_cuda_v2<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                                   THREADS, WARP_TILE_X, WARP_TILE_Y,
                                   SMEM_PADDING><<<grid_dim, block_dim>>>(
            d_mu_w, cu_temp_states->d_tmp_1, cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, d_idx_cov_zwa_1, d_idx_var_z_ud,
            woho, out_channels, wihi, in_channels, kernel_size, row_zw,
            row_zw_fo, batch_size, pad_param_idx,
            cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);

    } else {
        constexpr unsigned int BLOCK_TILE = 16;
        constexpr unsigned int SMEM_PADDING = 0;
        unsigned int grid_row = (in_channels + BLOCK_TILE - 1) / BLOCK_TILE;
        unsigned int grid_col =
            (wihi * batch_size + BLOCK_TILE - 1) / BLOCK_TILE;
        dim3 dim_grid(grid_col, grid_row);
        dim3 dim_block(BLOCK_TILE, BLOCK_TILE);
        conv2d_bwd_delta_z_cuda_v1<float, BLOCK_TILE, SMEM_PADDING>
            <<<dim_grid, dim_block>>>(
                d_mu_w, cu_temp_states->d_tmp_1,
                cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, d_idx_cov_zwa_1,
                d_idx_var_z_ud, woho, out_channels, wihi, in_channels,
                kernel_size, row_zw, row_zw_fo, batch_size, pad_param_idx,
                cu_output_delta_states->d_delta_mu,
                cu_output_delta_states->d_delta_var);
    }
}

void conv2d_param_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                TempStateCuda *&cu_temp_states,
                                BackwardStateCuda *&cu_next_bwd_states,
                                const float *d_var_w, const int *d_idx_mwa_2,
                                int out_channels, int woho, int in_channels,
                                int wihi, int kernel_size, int batch_size,
                                float *d_delta_mu_w, float *d_delta_var_w)
/*
 */
{
    int ki2_fi = kernel_size * kernel_size * in_channels;
    int wohofo = woho * out_channels;
    int woho_batch = woho * batch_size;

    // Permute delta
    constexpr unsigned int BLOCK_TILE_P = 16;
    unsigned int grid_row_pp = (batch_size + BLOCK_TILE_P - 1) / BLOCK_TILE_P;
    unsigned int grid_col_pp = (wohofo + BLOCK_TILE_P - 1) / BLOCK_TILE_P;
    dim3 dim_grid_pp(grid_col_pp, grid_row_pp);
    dim3 dim_block_pp(BLOCK_TILE_P, BLOCK_TILE_P);
    permute_delta_cuda<<<dim_grid_pp, dim_block_pp>>>(
        cu_input_delta_states->d_delta_mu, cu_input_delta_states->d_delta_var,
        woho, wohofo, batch_size, cu_temp_states->d_tmp_1,
        cu_temp_states->d_tmp_2);

    if (ki2_fi > 1024) {
        constexpr unsigned int BLOCK_SIZE = 64U;
        constexpr unsigned int THREAD_TILE = 4U;
        constexpr unsigned int WAPRS_X = 4U;
        constexpr unsigned int WAPRS_Y = 2U;
        constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WAPRS_X;
        constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WAPRS_Y;
        constexpr unsigned int THREAD_X = WAPRS_X * (WARP_TILE_X / THREAD_TILE);
        constexpr unsigned int THREAD_Y = WAPRS_Y * (WARP_TILE_Y / THREAD_TILE);
        constexpr unsigned int BLOCK_TILE_K = 16;
        constexpr size_t THREADS = THREAD_X * THREAD_Y;

        dim3 block_dim(THREAD_X, THREAD_Y, 1U);
        dim3 grid_dim(
            (static_cast<unsigned int>(ki2_fi) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
            (static_cast<unsigned int>(out_channels) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            1U);

        if (woho_batch % PACK_SIZE == 0) {
            constexpr size_t SMEM_PADDING = PACK_SIZE;
            conv2d_bwd_delta_w_cuda_v3<float, BLOCK_SIZE, BLOCK_TILE_K,
                                       THREAD_TILE, THREADS, WARP_TILE_X,
                                       WARP_TILE_Y, PACK_SIZE, SMEM_PADDING>
                <<<grid_dim, block_dim>>>(
                    d_var_w, cu_next_bwd_states->d_mu_a,
                    cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2,
                    d_idx_mwa_2, batch_size, out_channels, woho, wihi,
                    in_channels, kernel_size, d_delta_mu_w, d_delta_var_w);
        } else {
            constexpr size_t SMEM_PADDING = 2;
            conv2d_bwd_delta_w_cuda_v2<float, BLOCK_SIZE, BLOCK_TILE_K,
                                       THREAD_TILE, THREADS, WARP_TILE_X,
                                       WARP_TILE_Y, SMEM_PADDING>
                <<<grid_dim, block_dim>>>(
                    d_var_w, cu_next_bwd_states->d_mu_a,
                    cu_temp_states->d_tmp_1, cu_temp_states->d_tmp_2,
                    d_idx_mwa_2, batch_size, out_channels, woho, wihi,
                    in_channels, kernel_size, d_delta_mu_w, d_delta_var_w);
        }
    } else {
        constexpr unsigned int SMEM_PADDING = WARP_SIZE / BLOCK_TILE_P;
        unsigned int grid_row_w = (ki2_fi + BLOCK_TILE_P - 1) / BLOCK_TILE_P;
        unsigned int grid_col_w =
            (out_channels + BLOCK_TILE_P - 1) / BLOCK_TILE_P;
        dim3 dim_grid_w(grid_row_w, grid_col_w);
        conv2d_bwd_delta_w_cuda_v1<BLOCK_TILE_P, SMEM_PADDING>
            <<<dim_grid_w, dim_block_pp>>>(
                d_var_w, cu_next_bwd_states->d_mu_a, cu_temp_states->d_tmp_1,
                cu_temp_states->d_tmp_2, d_idx_mwa_2, batch_size, out_channels,
                woho, wihi, in_channels, kernel_size, d_delta_mu_w,
                d_delta_var_w);
    }
}

Conv2dCuda::Conv2dCuda(size_t in_channels, size_t out_channels,
                       size_t kernel_size, bool bias, int stride, int padding,
                       int padding_type, size_t in_width, size_t in_height,
                       float gain_w, float gain_b, std::string init_method)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/*
 */
{
    this->in_width = in_width;
    this->in_height = in_height;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->bias = bias;
}

Conv2dCuda::~Conv2dCuda() {
    cudaFree(d_idx_mwa_2);
    cudaFree(d_idx_cov_zwa_1);
    cudaFree(d_idx_var_z_ud);
}

std::string Conv2dCuda::get_layer_info() const {
    return "Conv2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

std::string Conv2dCuda::get_layer_name() const { return "Conv2dCuda"; }

LayerType Conv2dCuda::get_layer_type() const { return LayerType::Conv2d; };

void Conv2dCuda::compute_input_output_size(const InitArgs &args)
/*
 */
{
    if (this->in_height == 0 || this->in_height == 0) {
        this->in_width = args.width;
        this->in_height = args.height;
    }
    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride,
                                       this->in_width, this->in_height,
                                       this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void Conv2dCuda::get_number_param()

/* Get the number of parameters for conv. and tconv. layer.
 *
 * Args:
 *    kernel: Size of the receptive field
 *    fi: Number of filters for input image
 *    fo: Number of filters for output image
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = this->kernel_size * this->kernel_size * this->in_channels *
          this->out_channels;
    if (this->bias) {
        n_b = this->out_channels;
    } else {
        n_b = 0;
    }
    this->num_weights = n_w;
    this->num_biases = n_b;
}

void Conv2dCuda::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void Conv2dCuda::allocate_conv_index()
/*
 */
{
    // Memory alignment
    unsigned int size_idx_mwa_2 =
        ((this->idx_mwa_2.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    unsigned int size_idx_cov_zwa_1 =
        ((this->idx_cov_zwa_1.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    unsigned int size_idx_var_z_ud =
        ((this->idx_var_z_ud.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    cudaMalloc(&this->d_idx_mwa_2, size_idx_mwa_2 * sizeof(int));
    cudaMalloc(&this->d_idx_cov_zwa_1, size_idx_cov_zwa_1 * sizeof(int));
    cudaMalloc(&this->d_idx_var_z_ud, size_idx_var_z_ud * sizeof(int));

    CHECK_LAST_CUDA_ERROR();
}

void Conv2dCuda::conv_index_to_device()
/*
 */
{
    cudaMemcpy(this->d_idx_mwa_2, this->idx_mwa_2.data(),
               this->idx_mwa_2.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_cov_zwa_1, this->idx_cov_zwa_1.data(),
               this->idx_cov_zwa_1.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_var_z_ud, this->idx_var_z_ud.data(),
               this->idx_var_z_ud.size() * sizeof(int), cudaMemcpyHostToDevice);

    CHECK_LAST_CUDA_ERROR();
}

void Conv2dCuda::lazy_index_init()
/*
 */
{
    // Get precomputed conv indices
    int param_pad_idx =
        pow(this->kernel_size, 2) * this->in_channels * this->out_channels + 1;

    auto conv_idx = get_conv2d_idx(
        this->kernel_size, this->stride, this->in_width, this->in_height,
        this->out_width, this->out_height, this->padding, this->padding_type,
        -1, -1, param_pad_idx);

    this->idx_mwa_2 = conv_idx.Fmwa_2_idx;
    this->idx_cov_zwa_1 = conv_idx.FCzwa_1_idx;
    this->idx_var_z_ud = conv_idx.Szz_ud_idx;

    this->row_zw = conv_idx.h;
    this->col_z_ud = conv_idx.h;

    this->allocate_conv_index();
    this->conv_index_to_device();
}

void Conv2dCuda::forward(BaseHiddenStates &input_states,
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
    this->set_cap_factor_udapte(batch_size);

    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_2.size() == 0) {
        this->lazy_index_init();
    }

    // Assign output dimensions
    cu_output_states->width = this->out_width;
    cu_output_states->height = this->out_height;
    cu_output_states->depth = this->out_channels;
    cu_output_states->block_size = batch_size;
    cu_output_states->actual_size = this->output_size;

    // Launch kernel
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    int woho_batch = woho * batch_size;
    int pad_idx = wihi * this->in_channels * batch_size + 1;

    int threads = this->num_cuda_threads;
    unsigned int grid_row = (this->out_channels + threads - 1) / threads;
    unsigned int grid_col = (woho_batch + threads - 1) / threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(threads, threads);

    conv2d_forward_cuda(cu_input_states, this->d_mu_w, this->d_var_w,
                        this->d_mu_b, this->d_var_b, this->d_idx_mwa_2,
                        this->out_channels, woho, this->in_channels, wihi,
                        this->kernel_size, batch_size, pad_idx, this->bias,
                        cu_output_states);

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void Conv2dCuda::backward(BaseDeltaStates &input_delta_states,
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
    TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda *>(&temp_states);

    // Initialization
    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;
    dim3 dim_block(threads, threads);

    // Launch kernel
    int wihi = this->in_width * this->in_height;
    int woho = this->out_width * this->out_height;
    int row_zw_fo = this->row_zw * this->out_channels;
    int pad_param_idx = this->num_weights + 1;

    if (param_update) {
        conv2d_param_backward_cuda(
            cu_input_delta_states, cu_temp_states, cu_next_bwd_states,
            this->d_var_w, this->d_idx_mwa_2, this->out_channels, woho,
            this->in_channels, wihi, this->kernel_size, batch_size,
            this->d_delta_mu_w, this->d_delta_var_w);

        if (this->bias) {
            // Local pointer for swapping. Leverage the existing and
            // not-yet-used memory blocks defined in GPU device to reduce the
            // memory allocation
            float *buf_mu_out = cu_output_delta_states->d_delta_mu;
            float *buf_var_out = cu_output_delta_states->d_delta_var;
            float *buf_mu_in = cu_temp_states->d_tmp_1;
            float *buf_var_in = cu_temp_states->d_tmp_2;

            conv2d_bwd_delta_b_dual_sum_reduction<float>(
                this->d_var_b, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, batch_size, woho,
                this->out_channels, buf_mu_in, buf_var_in, buf_mu_out,
                buf_var_out, this->d_delta_mu_b, this->d_delta_var_b);
        }
    }

    // NOTE: state need to be updated after parameter update
    if (state_udapte) {
        unsigned int grid_row_p = (batch_size + threads - 1) / threads;
        unsigned int grid_col_p =
            (wihi * this->in_channels + threads - 1) / threads;
        dim3 dim_grid_p(grid_col_p, grid_row_p);

        permmute_jacobian_cuda<<<dim_grid_p, dim_block>>>(
            cu_next_bwd_states->d_jcb, wihi, this->in_channels, batch_size,
            cu_temp_states->d_tmp_1);

        conv2d_state_backward_cuda(
            cu_input_delta_states, cu_temp_states, this->d_mu_w,
            this->d_idx_mwa_2, this->d_idx_cov_zwa_1, this->d_idx_var_z_ud,
            this->out_channels, woho, this->in_channels, wihi,
            this->kernel_size, this->row_zw, row_zw_fo, batch_size,
            pad_param_idx, cu_output_delta_states);
    }
}

std::unique_ptr<BaseLayer> Conv2dCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_linear = std::make_unique<Conv2d>(
        this->in_channels, this->out_channels, this->kernel_size, this->bias,
        this->stride, this->padding, this->padding_type, this->in_width,
        this->in_height, this->gain_w, this->gain_b, this->init_method);

    host_linear->mu_w = this->mu_w;
    host_linear->var_w = this->var_w;
    host_linear->mu_b = this->mu_b;
    host_linear->var_b = this->var_b;

    return host_linear;
}

void Conv2dCuda::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_2.size() == 0) {
        this->lazy_index_init();
    }
}

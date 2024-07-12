///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      March 14, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/convtranspose2d_layer.h"
#include "../include/convtranspose2d_layer_cuda.cuh"
#include "../include/param_init.h"

__global__ void convtranspose2d_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, float const *mu_a, float const *var_a, int const *widx,
    int const *aidx, int woho, int fo, int wihi, int fi, int ki, int rf,
    int batch_size, bool bias, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < woho * fo && row < batch_size)  // k = woho * fo
    {
        float sum_mu = 0;
        float sum_var = 0;
        int aidx_tmp = 0;
        int widx_tmp = 0;
        int div_idx = col / woho;
        int mod_idx = col % woho;

        for (int i = 0; i < rf * fi; i++)  // n = ? * fi
        {
            int i_div_rf = i / rf;

            // minus 1 due to the index starting at 1
            int tmp_idx = mod_idx * rf + i % rf;
            widx_tmp = widx[tmp_idx];
            aidx_tmp = aidx[tmp_idx];

            if (aidx_tmp > -1 && widx_tmp > -1) {
                widx_tmp += div_idx * ki * ki + i_div_rf * ki * ki * fo - 1;
                aidx_tmp += row * wihi * fi + i_div_rf * wihi - 1;

                sum_mu += mu_w[widx_tmp] * mu_a[aidx_tmp];

                sum_var += (mu_w[widx_tmp] * mu_w[widx_tmp] + var_w[widx_tmp]) *
                               var_a[aidx_tmp] +
                           var_w[widx_tmp] * mu_a[aidx_tmp] * mu_a[aidx_tmp];
            }
        }

        mu_z[col + row * woho * fo] = sum_mu;
        var_z[col + row * woho * fo] = sum_var;
        if (bias) {
            mu_z[col + row * woho * fo] += mu_b[div_idx];
            var_z[col + row * woho * fo] += var_b[div_idx];
        }
    }
}

__global__ void convtranspose2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    float const *delta_var_out, int const *widx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int rf, int batch_size, float *delta_mu,
    float *delta_var)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int K = wihi * fi;

    if (col < K && row < batch_size)  // k = wihi * fi, m = B
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        int widx_tmp;
        int zidx_tmp;                      // updated index (idxSzzUd)
        for (int i = 0; i < rf * fo; i++)  // n = ki2 * fo
        {
            // minus 1 due to the index starting at 1
            int tmp_idx = (col % wihi) * ki * ki + i % rf;
            widx_tmp = widx[tmp_idx];
            zidx_tmp = zidx[tmp_idx];

            if (zidx_tmp > -1 && widx_tmp > -1) {
                widx_tmp +=
                    (i / rf) * ki * ki + (col / wihi) * ki * ki * fo - 1;
                zidx_tmp += (i / rf) * woho + row * woho * fo - 1;

                sum_mu += delta_mu_out[zidx_tmp] * mu_w[widx_tmp];
                sum_var +=
                    mu_w[widx_tmp] * delta_var_out[zidx_tmp] * mu_w[widx_tmp];
            }
        }
        // TODO: Double check the definition zposIn
        delta_mu[col + row * K] = sum_mu * jcb[col + row * K];
        delta_var[col + row * K] =
            sum_var * jcb[col + row * K] * jcb[col + row * K];
    }
}

__global__ void convtranspose2d_bwd_delta_w_cuda(
    float const *var_w, float const *mu_a, float const *delta_mu_out,
    float const *delta_var_out, int const *aidx, int const *zidx, int woho,
    int fo, int wihi, int fi, int ki, int batch_size, float *delta_mu_w,
    float *delta_var_w)
/**/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int K = ki * ki * fo;
    int ki2 = ki * ki;
    if (col < K && row < fi)  // m = fi, k = ki2 * fo
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        int zidx_tmp;  // updated index
        int aidx_tmp;
        int col_div_ki2 = col / ki2;
        int col_mod_ki2 = col % ki2;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi * B
        {
            int i_div_wihi = i / wihi;
            int i_mod_wihi = i % wihi;

            // minus 1 due to the index starting at 1
            aidx_tmp = aidx[col_mod_ki2 * wihi + i_mod_wihi];

            if (aidx_tmp > -1) {
                // minus 1 due to the index starting at 1
                zidx_tmp = zidx[col_mod_ki2 * wihi + i_mod_wihi] +
                           col_div_ki2 * woho + i_div_wihi * woho * fo - 1;
                aidx_tmp += row * wihi + i_div_wihi * wihi * fi - 1;

                sum_mu += mu_a[aidx_tmp] * delta_mu_out[zidx_tmp];
                sum_var +=
                    mu_a[aidx_tmp] * mu_a[aidx_tmp] * delta_var_out[zidx_tmp];
            }
        }

        delta_mu_w[col + row * K] = sum_mu * var_w[col + row * K];
        delta_var_w[col + row * K] =
            sum_var * var_w[col + row * K] * var_w[col + row * K];
    }
}

__global__ void convtranspose2d_bwd_delta_b_cuda(
    float const *var_b, float const *delta_mu_out, float const *delta_var_out,
    int woho, int fo, int batch_size, float *delta_mu_b, float *delta_var_b)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < fo)  // k = fo, m = 1
    {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < woho * batch_size; i++)  // n = woho * B
        {
            int idx = col * woho + (i % woho) + (i / woho) * woho * fo;

            sum_mu += delta_mu_out[idx];
            sum_var += delta_var_out[idx];
        }

        delta_mu_b[col] = sum_mu * var_b[col];
        delta_var_b[col] = var_b[col] * sum_var * var_b[col];
    }
}

ConvTranspose2dCuda::ConvTranspose2dCuda(
    size_t in_channels, size_t out_channels, size_t kernel_size, bool bias,
    int stride, int padding, int padding_type, size_t in_width,
    size_t in_height, float gain_w, float gain_b, std::string init_method)
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

ConvTranspose2dCuda::~ConvTranspose2dCuda() {
    cudaFree(d_idx_mwa_1);
    cudaFree(d_idx_mwa_2);
    cudaFree(d_idx_cov_wz_2);
    cudaFree(d_idx_var_wz_ud);
    cudaFree(d_idx_cov_z_wa_1);
    cudaFree(d_idx_var_z_ud);
}

std::string ConvTranspose2dCuda::get_layer_name() const {
    return "ConvTranspose2dCuda";
}

std::string ConvTranspose2dCuda::get_layer_info() const {
    return "ConvTranspose2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

LayerType ConvTranspose2dCuda::get_layer_type() const {
    return LayerType::ConvTranspose2d;
};

void ConvTranspose2dCuda::compute_input_output_size(const InitArgs &args)
/*
 */
{
    if (this->in_height == 0 || this->in_height == 0) {
        this->in_width = args.width;
        this->in_height = args.height;
    }
    std::tie(this->out_width, this->out_height) = compute_upsample_img_size_v2(
        this->kernel_size, this->stride, this->in_width, this->in_height,
        this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void ConvTranspose2dCuda::get_number_param()
/*
 */
{
    this->num_weights = this->kernel_size * this->kernel_size *
                        this->in_channels * this->out_channels;
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = this->out_channels;
    }
}

void ConvTranspose2dCuda::init_weight_bias()
/**/
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void ConvTranspose2dCuda::lazy_index_init()
/*
 */
{
    // int ki2 = this->kernel_size * this->kernel_size;
    // int param_pad_idx = ki2 * this->in_channels * this->out_channels + 1;

    auto conv_idx =
        get_conv2d_idx(this->kernel_size, this->stride, this->out_width,
                       this->out_height, this->in_width, this->in_height,
                       this->padding, this->padding_type, -1, -1, -1);

    auto conv_transpose_idx = get_tconv_idx(-1, -1, -1, conv_idx);

    this->idx_mwa_1 = conv_idx.FCzwa_1_idx;
    this->idx_mwa_2 =
        transpose_matrix(conv_idx.Szz_ud_idx, conv_idx.w, conv_idx.h);
    this->idx_cov_wz_2 = conv_transpose_idx.FCwz_2_idx;
    this->idx_var_wz_ud = conv_transpose_idx.Swz_ud_idx;
    this->idx_cov_z_wa_1 = conv_transpose_idx.FCzwa_1_idx;
    this->idx_var_z_ud = conv_transpose_idx.Szz_ud_idx;

    // Dimension
    this->row_zw = conv_transpose_idx.w_wz;
    this->col_z_ud = conv_transpose_idx.w_zz;
    this->col_cov_mwa_1 = conv_idx.h;

    this->allocate_convtranspose_index();
    this->convtranspose_index_to_device();
}

void ConvTranspose2dCuda::allocate_convtranspose_index()
/*
 */
{
    cudaMalloc(&this->d_idx_mwa_1, this->idx_mwa_1.size() * sizeof(int));
    cudaMalloc(&this->d_idx_mwa_2, this->idx_mwa_2.size() * sizeof(int));
    cudaMalloc(&this->d_idx_cov_wz_2, this->idx_cov_wz_2.size() * sizeof(int));

    cudaMalloc(&this->d_idx_var_wz_ud,
               this->idx_var_wz_ud.size() * sizeof(int));
    cudaMalloc(&this->d_idx_cov_z_wa_1,
               this->idx_cov_z_wa_1.size() * sizeof(int));
    cudaMalloc(&this->d_idx_var_z_ud, this->idx_var_z_ud.size() * sizeof(int));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Device memory allocation.");
    }
}

void ConvTranspose2dCuda::convtranspose_index_to_device()
/*
 */
{
    cudaMemcpy(this->d_idx_mwa_1, this->idx_mwa_1.data(),
               this->idx_mwa_1.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_mwa_2, this->idx_mwa_2.data(),
               this->idx_mwa_2.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_cov_wz_2, this->idx_cov_wz_2.data(),
               this->idx_cov_wz_2.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(this->d_idx_var_wz_ud, this->idx_var_wz_ud.data(),
               this->idx_var_wz_ud.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_cov_z_wa_1, this->idx_cov_z_wa_1.data(),
               this->idx_cov_z_wa_1.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_idx_var_z_ud, this->idx_var_z_ud.data(),
               this->idx_var_z_ud.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Host to device.");
    }
}

void ConvTranspose2dCuda::forward(BaseHiddenStates &input_states,
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
    int threads = this->num_cuda_threads;

    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
        this->allocate_param_delta();
    }

    if (this->idx_mwa_1.size() == 0) {
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
    unsigned int grid_row = (batch_size + threads - 1) / threads;
    unsigned int grid_col = (woho * this->out_channels + threads - 1) / threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(threads, threads);

    convtranspose2d_fwd_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->d_idx_mwa_1,
        this->d_idx_mwa_2, woho, this->out_channels, wihi, this->in_channels,
        this->kernel_size, this->col_cov_mwa_1, batch_size, this->bias,
        cu_output_states->d_mu_a, cu_output_states->d_var_a);

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void ConvTranspose2dCuda::backward(BaseDeltaStates &input_delta_states,
                                   BaseDeltaStates &output_delta_states,
                                   BaseTempStates &temp_states,
                                   bool state_udapte)
/*
 */
{
    // New poitner will point to the same memory location when casting
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;

    // Lauch kernel
    int wihi = this->in_height * this->in_width;
    int woho = this->out_width * this->out_height;
    unsigned int grid_row = (batch_size + threads - 1) / threads;
    unsigned int grid_col = (wihi * this->in_channels + threads - 1) / threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(threads, threads);

    convtranspose2d_bwd_delta_z_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, cu_next_bwd_states->d_jcb,
        cu_input_delta_states->d_delta_mu, cu_input_delta_states->d_delta_var,
        this->d_idx_cov_z_wa_1, this->d_idx_var_z_ud, woho, this->out_channels,
        wihi, this->in_channels, this->kernel_size, this->row_zw, batch_size,
        cu_output_delta_states->d_delta_mu,
        cu_output_delta_states->d_delta_var);

    // Parameter

    // Launch kernel
    int ki2 = this->kernel_size * this->kernel_size;
    unsigned int grid_row_w = (this->in_channels + threads - 1) / threads;
    unsigned int grid_col_w =
        (ki2 * this->out_channels + threads - 1) / threads;

    dim3 dim_grid_w(grid_col_w, grid_row_w);

    convtranspose2d_bwd_delta_w_cuda<<<dim_grid_w, dim_block>>>(
        this->d_var_w, cu_next_bwd_states->d_mu_a,
        cu_input_delta_states->d_delta_mu, cu_input_delta_states->d_delta_var,
        this->d_idx_cov_wz_2, this->d_idx_var_wz_ud, woho, this->out_channels,
        wihi, this->in_channels, this->kernel_size, batch_size,
        this->d_delta_mu_w, this->d_delta_var_w);

    if (this->bias) {
        unsigned int grid_size = (this->out_channels + threads - 1) / threads;

        convtranspose2d_bwd_delta_b_cuda<<<grid_size, threads>>>(
            this->d_var_b, cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, woho, this->out_channels,
            batch_size, this->d_delta_mu_b, this->d_delta_var_b);
    }
}

std::unique_ptr<BaseLayer> ConvTranspose2dCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_linear = std::make_unique<ConvTranspose2d>(
        this->in_channels, this->out_channels, this->kernel_size, this->bias,
        this->stride, this->padding, this->padding_type, this->in_width,
        this->in_height, this->gain_w, this->gain_b, this->init_method);

    host_linear->mu_w = this->mu_w;
    host_linear->var_w = this->var_w;
    host_linear->mu_b = this->mu_b;
    host_linear->var_b = this->var_b;

    return host_linear;
}

void ConvTranspose2dCuda::preinit_layer() {
    this->get_number_param();
    this->init_weight_bias();
    this->lazy_index_init();
    if (this->training) {
        this->allocate_param_delta();
    }
}

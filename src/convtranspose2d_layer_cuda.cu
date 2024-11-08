#include "../include/common.h"
#include "../include/config.h"
#include "../include/convtranspose2d_cuda_kernel.cuh"
#include "../include/convtranspose2d_layer.h"
#include "../include/convtranspose2d_layer_cuda.cuh"
#include "../include/custom_logger.h"
#include "../include/param_init.h"

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
    // Memory alignment
    unsigned int size_idx_mwa_1 =
        ((this->idx_mwa_1.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    unsigned int size_idx_mwa_2 =
        ((this->idx_mwa_2.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    unsigned int size_idx_cov_wz_2 =
        ((this->idx_cov_wz_2.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    unsigned int size_idx_var_wz_ud =
        ((this->idx_var_wz_ud.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    unsigned int size_idx_cov_z_wa_1 =
        ((this->idx_cov_z_wa_1.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    unsigned int size_idx_var_z_ud =
        ((this->idx_var_z_ud.size() + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;

    cudaMalloc(&this->d_idx_mwa_1, size_idx_mwa_1 * sizeof(int));
    cudaMalloc(&this->d_idx_mwa_2, size_idx_mwa_2 * sizeof(int));
    cudaMalloc(&this->d_idx_cov_wz_2, size_idx_cov_wz_2 * sizeof(int));

    cudaMalloc(&this->d_idx_var_wz_ud, size_idx_var_wz_ud * sizeof(int));
    cudaMalloc(&this->d_idx_cov_z_wa_1, size_idx_cov_z_wa_1 * sizeof(int));
    cudaMalloc(&this->d_idx_var_z_ud, size_idx_var_z_ud * sizeof(int));

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        LOG(LogLevel::ERROR, "Device memory allocation.");
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
        LOG(LogLevel::ERROR,
            std::string("Host to device. ") + cudaGetErrorString(error));
    }
}

void ConvTranspose2dCuda::forward(BaseHiddenStates &input_states,
                                  BaseHiddenStates &output_states,
                                  BaseTempStates &temp_states)
/*
 */
{
    // Checkout input size
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

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
    constexpr unsigned int THREADS = 16;
    constexpr size_t SMEM_PADDING = 0;
    int woho = this->out_width * this->out_height;
    int wihi = this->in_width * this->in_height;
    unsigned int grid_row = (batch_size + THREADS - 1) / THREADS;
    unsigned int grid_col = (woho * this->out_channels + THREADS - 1) / THREADS;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(THREADS, THREADS);

    convtranspose2d_fwd_mean_var_cuda_v1<float, THREADS, SMEM_PADDING>
        <<<dim_grid, dim_block>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            cu_input_states->d_mu_a, cu_input_states->d_var_a,
            this->d_idx_mwa_1, this->d_idx_mwa_2, woho, this->out_channels,
            wihi, this->in_channels, this->kernel_size, this->col_cov_mwa_1,
            batch_size, this->bias, cu_output_states->d_mu_a,
            cu_output_states->d_var_a);

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

    // Lauch kernel
    constexpr unsigned int THREADS = 16;
    constexpr size_t SMEM_PADDING = 0;
    dim3 dim_block(THREADS, THREADS);
    int wihi = this->in_height * this->in_width;
    int woho = this->out_width * this->out_height;

    // Parameters
    int ki2 = this->kernel_size * this->kernel_size;
    unsigned int grid_row_w = (this->in_channels + THREADS - 1) / THREADS;
    unsigned int grid_col_w =
        (ki2 * this->out_channels + THREADS - 1) / THREADS;

    dim3 dim_grid_w(grid_col_w, grid_row_w);
    convtranspose2d_bwd_delta_w_cuda_v1<float, THREADS, SMEM_PADDING>
        <<<dim_grid_w, dim_block>>>(
            this->d_var_w, cu_next_bwd_states->d_mu_a,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->d_idx_cov_wz_2,
            this->d_idx_var_wz_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, batch_size,
            this->d_delta_mu_w, this->d_delta_var_w);

    if (this->bias) {
        TempStateCuda *cu_temp_states =
            dynamic_cast<TempStateCuda *>(&temp_states);

        // Local pointer for swapping. Leverage the existing and
        // not-yet-used memory blocks defined in GPU device to reduce the
        // memory allocation
        float *buf_mu_out = cu_output_delta_states->d_delta_mu;
        float *buf_var_out = cu_output_delta_states->d_delta_var;
        float *buf_mu_in = cu_temp_states->d_tmp_1;
        float *buf_var_in = cu_temp_states->d_tmp_2;

        convtranspose2d_bwd_delta_b_dual_sum_reduction<float>(
            this->d_var_b, cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, batch_size, woho,
            this->out_channels, buf_mu_in, buf_var_in, buf_mu_out, buf_var_out,
            this->d_delta_mu_b, this->d_delta_var_b);
    }

    // State update
    unsigned int grid_row = (batch_size + THREADS - 1) / THREADS;
    unsigned int grid_col = (wihi * this->in_channels + THREADS - 1) / THREADS;
    dim3 dim_grid(grid_col, grid_row);

    convtranspose2d_bwd_delta_z_cuda_v1<float, THREADS, SMEM_PADDING>
        <<<dim_grid, dim_block>>>(
            this->d_mu_w, cu_next_bwd_states->d_jcb,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->d_idx_cov_z_wa_1,
            this->d_idx_var_z_ud, woho, this->out_channels, wihi,
            this->in_channels, this->kernel_size, this->row_zw, batch_size,
            cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
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

#include <cstdint>

#include "../include/config.h"
#include "../include/custom_logger.h"
#include "../include/linear_cuda_kernel.cuh"
#include "../include/linear_layer.h"
#include "../include/linear_layer_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// Fully Connected Layer
////////////////////////////////////////////////////////////////////////////////
void linear_forward_cuda(HiddenStateCuda *&cu_input_states,
                         HiddenStateCuda *&cu_output_states,
                         const float *d_mu_w, const float *d_var_w,
                         const float *d_mu_b, const float *d_var_b,
                         size_t input_size, size_t output_size, int batch_size,
                         bool bias) {
    if (output_size * input_size > 1024 * 128) {
        // TODO: remove hardcoded kernel config
        constexpr unsigned int BLOCK_SIZE = 64U;
        constexpr unsigned int THREAD_TILE = 4U;
        constexpr unsigned int WARPS_X = 4U;
        constexpr unsigned int WARPS_Y = 2U;
        constexpr unsigned int BLOCK_TILE_K = 16U;
        constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WARPS_X;
        constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WARPS_Y;
        constexpr unsigned int THREADS_X =
            WARPS_X * (WARP_TILE_X / THREAD_TILE);
        constexpr unsigned int THREADS_Y =
            WARPS_Y * (WARP_TILE_Y / THREAD_TILE);
        constexpr unsigned int THREADS = THREADS_X * THREADS_Y;

        dim3 block_dim(THREADS_X, THREADS_Y, 1U);
        dim3 grid_dim(
            (static_cast<unsigned int>(output_size) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            (static_cast<unsigned int>(batch_size) + BLOCK_SIZE - 1U) /
                BLOCK_SIZE,
            1U);

        if (output_size % PACK_SIZE == 0 && input_size % PACK_SIZE == 0) {
            constexpr unsigned int SMEM_PADDING = PACK_SIZE;
            linear_fwd_mean_var_v4<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                                   THREADS, WARP_TILE_X, WARP_TILE_Y, PACK_SIZE,
                                   SMEM_PADDING><<<grid_dim, block_dim>>>(
                d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                cu_input_states->d_var_a, input_size, output_size, batch_size,
                bias, cu_output_states->d_mu_a, cu_output_states->d_var_a);

        } else {
            constexpr unsigned int SMEM_PADDING = WARP_SIZE / THREADS_X;
            linear_fwd_mean_var_v3<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                                   THREADS, WARP_TILE_X, WARP_TILE_Y,
                                   SMEM_PADDING><<<grid_dim, block_dim>>>(
                d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                cu_input_states->d_var_a, input_size, output_size, batch_size,
                bias, cu_output_states->d_mu_a, cu_output_states->d_var_a);
        }
    } else {
        constexpr unsigned int BLOCK_SIZE = 16U;
        unsigned int grid_rows = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE, 1U);
        dim3 grid_dim(grid_cols, grid_rows, 1U);
        constexpr size_t THREADS = BLOCK_SIZE * BLOCK_SIZE;

        linear_fwd_mean_var_v1<float, BLOCK_SIZE, BLOCK_SIZE, THREADS>
            <<<grid_dim, block_dim>>>(
                d_mu_w, d_var_w, d_mu_b, d_var_b, cu_input_states->d_mu_a,
                cu_input_states->d_var_a, input_size, output_size, batch_size,
                bias, cu_output_states->d_mu_a, cu_output_states->d_var_a);
    }
}

void linear_state_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                DeltaStateCuda *&cu_output_delta_states,
                                BackwardStateCuda *&cu_next_bwd_states,
                                const float *d_mu_w, size_t input_size,
                                size_t output_size, int batch_size)
/*
 */
{
    constexpr unsigned int BLOCK_SIZE = 64U;
    constexpr unsigned int THREAD_TILE = 4U;
    constexpr unsigned int WARPS_X = 4U;
    constexpr unsigned int WARPS_Y = 2U;
    constexpr unsigned int BLOCK_TILE_K = 16U;
    constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WARPS_X;
    constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WARPS_Y;
    constexpr unsigned int THREADS_X = WARPS_X * (WARP_TILE_X / THREAD_TILE);
    constexpr unsigned int THREADS_Y = WARPS_Y * (WARP_TILE_Y / THREAD_TILE);
    constexpr unsigned int THREADS = THREADS_X * THREADS_Y;

    dim3 block_dim(THREADS_X, THREADS_Y, 1U);
    dim3 grid_dim(
        (static_cast<unsigned int>(input_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        (static_cast<unsigned int>(batch_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        1U);

    if (output_size % PACK_SIZE == 0 && input_size % PACK_SIZE == 0) {
        constexpr unsigned int SMEM_PADDING = PACK_SIZE;
        linear_bwd_delta_z_v4<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                              THREADS, WARP_TILE_X, WARP_TILE_Y, PACK_SIZE,
                              SMEM_PADDING><<<grid_dim, block_dim>>>(
            d_mu_w, cu_next_bwd_states->d_jcb,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, input_size, output_size,
            batch_size, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    } else {
        constexpr unsigned int SMEM_PADDING = BANK_SIZE / THREADS_X;
        linear_bwd_delta_z_v3<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                              THREADS, WARP_TILE_X, WARP_TILE_Y, SMEM_PADDING>
            <<<grid_dim, block_dim>>>(d_mu_w, cu_next_bwd_states->d_jcb,
                                      cu_input_delta_states->d_delta_mu,
                                      cu_input_delta_states->d_delta_var,
                                      input_size, output_size, batch_size,
                                      cu_output_delta_states->d_delta_mu,
                                      cu_output_delta_states->d_delta_var);
    }
}

void linear_weight_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                 DeltaStateCuda *&cu_output_delta_states,
                                 BackwardStateCuda *&cu_next_bwd_states,
                                 const float *d_var_w, size_t input_size,
                                 size_t output_size, int batch_size,
                                 float *d_delta_mu_w, float *d_delta_var_w)
/*
 */
{
    constexpr unsigned int BLOCK_SIZE = 64U;
    constexpr unsigned int THREAD_TILE = 4U;
    constexpr unsigned int WARPS_X = 4U;
    constexpr unsigned int WARPS_Y = 2U;
    constexpr unsigned int BLOCK_TILE_K = 16U;
    constexpr unsigned int WARP_TILE_X = BLOCK_SIZE / WARPS_X;
    constexpr unsigned int WARP_TILE_Y = BLOCK_SIZE / WARPS_Y;
    constexpr unsigned int THREADS_X = WARPS_X * (WARP_TILE_X / THREAD_TILE);
    constexpr unsigned int THREADS_Y = WARPS_Y * (WARP_TILE_Y / THREAD_TILE);
    constexpr unsigned int THREADS = THREADS_X * THREADS_Y;

    dim3 block_dim(THREADS_X, THREADS_Y, 1U);
    dim3 grid_dim(
        (static_cast<unsigned int>(input_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        (static_cast<unsigned int>(output_size) + BLOCK_SIZE - 1U) / BLOCK_SIZE,
        1U);

    if (output_size % PACK_SIZE == 0 && input_size % PACK_SIZE == 0) {
        constexpr unsigned int SMEM_PADDING = 0;
        linear_bwd_delta_w_v4<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                              THREADS, WARP_TILE_X, WARP_TILE_Y, PACK_SIZE,
                              SMEM_PADDING><<<grid_dim, block_dim>>>(
            d_var_w, cu_next_bwd_states->d_mu_a,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, input_size, output_size,
            batch_size, d_delta_mu_w, d_delta_var_w);
    } else {
        constexpr unsigned int SMEM_PADDING = BANK_SIZE / THREADS_X;
        linear_bwd_delta_w_v3<float, BLOCK_SIZE, BLOCK_TILE_K, THREAD_TILE,
                              THREADS, WARP_TILE_X, WARP_TILE_Y, SMEM_PADDING>
            <<<grid_dim, block_dim>>>(d_var_w, cu_next_bwd_states->d_mu_a,
                                      cu_input_delta_states->d_delta_mu,
                                      cu_input_delta_states->d_delta_var,
                                      input_size, output_size, batch_size,
                                      d_delta_mu_w, d_delta_var_w);
    }
}

LinearCuda::LinearCuda(size_t ip_size, size_t op_size, bool bias,
                       float gain_weight, float gain_bias, std::string method)
    : gain_w(gain_weight),
      gain_b(gain_bias),
      init_method(method)
/*
 */
{
    this->input_size = ip_size;
    this->output_size = op_size;
    this->bias = bias;
    this->num_weights = this->input_size * this->output_size;
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = this->output_size;
    }

    if (this->training) {
        // TODO: to be removed
        this->bwd_states = std::make_unique<BackwardStateCuda>();
        this->allocate_param_delta();
    }
}

LinearCuda::~LinearCuda() {}

std::string LinearCuda::get_layer_info() const
/*
 */
{
    return "Linear(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LinearCuda::get_layer_name() const
/*
 */
{
    return "LinearCuda";
}

LayerType LinearCuda::get_layer_type() const
/*
 */
{
    return LayerType::Linear;
}

void LinearCuda::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->input_size, this->output_size,
                                this->num_weights, this->num_biases);

    this->allocate_param_memory();
    this->params_to_device();
}

void LinearCuda::forward(BaseHiddenStates &input_states,
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
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Checkout input size
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    int batch_size = input_states.block_size;

    this->set_cap_factor_udapte(batch_size);

    linear_forward_cuda(cu_input_states, cu_output_states, this->d_mu_w,
                        this->d_var_w, this->d_mu_b, this->d_var_b,
                        this->input_size, this->output_size,
                        input_states.block_size, this->bias);

    // Update number of actual states.
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }
}

void LinearCuda::backward(BaseDeltaStates &input_delta_states,
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
    int threads = this->num_cuda_threads;

    // Compute inovation vector
    unsigned int grid_row = (this->input_size + threads - 1) / threads;
    unsigned int grid_col = (batch_size + threads - 1) / threads;

    dim3 grid_dim(grid_col, grid_row);
    dim3 block_dim(threads, threads);

    if (state_udapte) {
        linear_state_backward_cuda(
            cu_input_delta_states, cu_output_delta_states, cu_next_bwd_states,
            this->d_mu_w, this->input_size, this->output_size, batch_size);
    }

    // Updated values for weights
    if (this->param_update) {
        linear_weight_backward_cuda(
            cu_input_delta_states, cu_output_delta_states, cu_next_bwd_states,
            this->d_var_w, this->input_size, this->output_size, batch_size,
            this->d_delta_mu_w, this->d_delta_var_w);

        // Updated values for biases
        if (this->bias) {
            unsigned int grid_row_b =
                (this->output_size + threads - 1) / threads;
            dim3 grid_dim_b(1, grid_row_b);

            linear_bwd_delta_b<<<grid_dim_b, block_dim>>>(
                this->d_var_b, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->input_size,
                this->output_size, batch_size, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

std::unique_ptr<BaseLayer> LinearCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_linear = std::make_unique<Linear>(
        this->input_size, this->output_size, this->bias, this->gain_w,
        this->gain_b, this->init_method);
    host_linear->mu_w = this->mu_w;
    host_linear->var_w = this->var_w;
    host_linear->mu_b = this->mu_b;
    host_linear->var_b = this->var_b;

    return host_linear;
}

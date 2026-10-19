#include <cstdint>

#include "../include/config.h"
#include "../include/cuda_error_checking.cuh"
#include "../include/custom_logger.h"
#include "../include/linear_cuda_kernel.cuh"
#include "../include/linear_layer.h"
#include "../include/linear_layer_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// Fully Connected Layer
////////////////////////////////////////////////////////////////////////////////
// The v4 forward kernel rounds the variance term differently from v3, so the
// sparse readout must follow the same selection to stay bit-equal.
static bool linear_fwd_uses_v4(size_t input_size, size_t output_size) {
    return output_size * input_size > 1024 * 128 &&
           output_size % PACK_SIZE == 0 && input_size % PACK_SIZE == 0;
}

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

        if (linear_fwd_uses_v4(input_size, output_size)) {
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
                       float gain_weight, float gain_bias, std::string method,
                       int device_idx)
    : gain_w(gain_weight),
      gain_b(gain_bias),
      init_method(method)
/*
 */
{
    this->input_size = ip_size;
    this->output_size = op_size;
    this->bias = bias;
    this->device_idx = device_idx;
    this->num_weights = this->input_size * this->output_size;
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = this->output_size;
    }

    if (this->training) {
        this->allocate_param_delta();
    }
}

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
    int seq_len = input_states.seq_len;
    int effective_batch = batch_size * seq_len;
    this->set_cap_factor_udapte(effective_batch);

    this->sparse_pass = this->output_idx_set;
    this->output_idx_set = false;
    if (this->sparse_pass) {
        this->sparse_forward(*cu_input_states, *cu_output_states,
                             effective_batch);
    } else {
        linear_forward_cuda(cu_input_states, cu_output_states, this->d_mu_w,
                            this->d_var_w, this->d_mu_b, this->d_var_b,
                            this->input_size, this->output_size,
                            effective_batch, this->bias);
    }

    // Update number of actual states.
    cu_output_states->width = this->out_width;
    cu_output_states->height = this->out_height;
    cu_output_states->depth = this->out_channels;
    cu_output_states->block_size = batch_size;
    cu_output_states->seq_len = seq_len;
    cu_output_states->actual_size =
        this->sparse_pass ? this->n_obs : this->output_size;

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
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int batch_size = input_delta_states.block_size;
    int seq_len = input_delta_states.seq_len;
    int effective_batch = batch_size * seq_len;
    int threads = this->num_cuda_threads;

    if (this->sparse_pass) {
        this->sparse_backward(*cu_input_delta_states, *cu_output_delta_states,
                              *cu_next_bwd_states, effective_batch,
                              state_udapte);
        return;
    }

    unsigned int grid_row = (this->input_size + threads - 1) / threads;
    unsigned int grid_col = (effective_batch + threads - 1) / threads;

    dim3 grid_dim(grid_col, grid_row);
    dim3 block_dim(threads, threads);

    if (state_udapte) {
        linear_state_backward_cuda(
            cu_input_delta_states, cu_output_delta_states, cu_next_bwd_states,
            this->d_mu_w, this->input_size, this->output_size, effective_batch);
        cu_output_delta_states->seq_len = seq_len;
    }

    if (this->param_update) {
        linear_weight_backward_cuda(
            cu_input_delta_states, cu_output_delta_states, cu_next_bwd_states,
            this->d_var_w, this->input_size, this->output_size, effective_batch,
            this->d_delta_mu_w, this->d_delta_var_w);

        if (this->bias) {
            unsigned int grid_row_b =
                (this->output_size + threads - 1) / threads;
            dim3 grid_dim_b(1, grid_row_b);

            linear_bwd_delta_b<<<grid_dim_b, block_dim>>>(
                this->d_var_b, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->input_size,
                this->output_size, effective_batch, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

void LinearCuda::set_output_indices(const std::vector<int> &unit_idx, int n_obs)
/*
 */
{
    if (n_obs <= 0 || n_obs > SPARSE_READOUT_MAX_OBS) {
        LOG(LogLevel::ERROR,
            "set_output_indices: n_obs=" + std::to_string(n_obs) +
                " outside 1.." + std::to_string(SPARSE_READOUT_MAX_OBS));
    }
    if (unit_idx.size() % n_obs != 0) {
        LOG(LogLevel::ERROR,
            "set_output_indices: " + std::to_string(unit_idx.size()) +
                " indices are not a multiple of n_obs=" +
                std::to_string(n_obs));
    }
    std::vector<int> zero_based(unit_idx.size());
    for (size_t i = 0; i < unit_idx.size(); i++) {
        if (unit_idx[i] < 1 || unit_idx[i] > this->output_size) {
            LOG(LogLevel::ERROR,
                "set_output_indices: unit " + std::to_string(unit_idx[i]) +
                    " outside 1.." + std::to_string(this->output_size));
        }
        zero_based[i] = unit_idx[i] - 1;
    }

    cudaSetDevice(this->device_idx);
    int num = static_cast<int>(unit_idx.size());
    this->segments.reserve(num);
    cudaMemcpy(this->segments.d_row, zero_based.data(), num * sizeof(int),
               cudaMemcpyHostToDevice);
    this->num_output_idx = num;
    this->n_obs = n_obs;
    this->output_idx_set = true;
}

void LinearCuda::sparse_forward(HiddenStateCuda &input_states,
                                HiddenStateCuda &output_states, int num_tokens)
/*
 */
{
    if (this->num_output_idx != num_tokens * this->n_obs) {
        LOG(LogLevel::ERROR,
            "set_output_indices gave " + std::to_string(this->num_output_idx) +
                " indices for " + std::to_string(num_tokens) +
                " tokens x n_obs=" + std::to_string(this->n_obs));
    }
    constexpr unsigned int THREADS = 128;
    unsigned int blocks = (this->num_output_idx + THREADS - 1) / THREADS;

    if (linear_fwd_uses_v4(this->input_size, this->output_size)) {
        linear_sparse_fwd<true><<<blocks, THREADS>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            input_states.d_mu_a, input_states.d_var_a, this->segments.d_row,
            this->input_size, this->n_obs, this->num_output_idx, this->bias,
            output_states.d_mu_a, output_states.d_var_a);
    } else {
        linear_sparse_fwd<false><<<blocks, THREADS>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            input_states.d_mu_a, input_states.d_var_a, this->segments.d_row,
            this->input_size, this->n_obs, this->num_output_idx, this->bias,
            output_states.d_mu_a, output_states.d_var_a);
    }
}

void LinearCuda::sparse_backward(DeltaStateCuda &input_delta_states,
                                 DeltaStateCuda &output_delta_states,
                                 BackwardStateCuda &bwd_states, int num_tokens,
                                 bool state_udapte)
/*
 */
{
    if (state_udapte) {
        constexpr unsigned int THREADS = 256;
        dim3 grid((this->input_size + THREADS - 1) / THREADS, num_tokens);
        linear_sparse_bwd_delta_z<<<grid, THREADS>>>(
            this->d_mu_w, bwd_states.d_jcb, input_delta_states.d_delta_mu,
            input_delta_states.d_delta_var, this->segments.d_row,
            this->input_size, this->n_obs, output_delta_states.d_delta_mu,
            output_delta_states.d_delta_var);
        output_delta_states.seq_len = input_delta_states.seq_len;
    }

    if (this->param_update) {
        this->segments.group(this->num_output_idx);
        cudaMemsetAsync(this->d_delta_mu_w, 0,
                        this->num_weights * sizeof(float));
        cudaMemsetAsync(this->d_delta_var_w, 0,
                        this->num_weights * sizeof(float));
        linear_sparse_bwd_delta_w<<<this->segments.num_touched, 256>>>(
            this->d_var_w, bwd_states.d_mu_a, input_delta_states.d_delta_mu,
            input_delta_states.d_delta_var, this->segments.d_touched_row,
            this->segments.d_segment_offset, this->segments.d_segment_count,
            this->segments.d_sorted_item, this->input_size, this->n_obs,
            this->d_delta_mu_w, this->d_delta_var_w);

        if (this->bias) {
            cudaMemsetAsync(this->d_delta_mu_b, 0,
                            this->num_biases * sizeof(float));
            cudaMemsetAsync(this->d_delta_var_b, 0,
                            this->num_biases * sizeof(float));
            constexpr unsigned int THREADS = 256;
            unsigned int blocks =
                (this->segments.num_touched + THREADS - 1) / THREADS;
            linear_sparse_bwd_delta_b<<<blocks, THREADS>>>(
                this->d_var_b, input_delta_states.d_delta_mu,
                input_delta_states.d_delta_var, this->segments.d_touched_row,
                this->segments.d_segment_offset, this->segments.d_segment_count,
                this->segments.d_sorted_item, this->segments.num_touched,
                this->d_delta_mu_b, this->d_delta_var_b);
        }
    }
}

void LinearCuda::update_weights()
/*
 */
{
    if (this->sparse_pass) {
        this->update_weight_rows(this->segments.d_touched_row,
                                 this->segments.num_touched, this->input_size);
    } else {
        BaseLayerCuda::update_weights();
    }
}

void LinearCuda::update_biases()
/*
 */
{
    if (this->sparse_pass) {
        this->update_bias_rows(this->segments.d_touched_row,
                               this->segments.num_touched);
    } else {
        BaseLayerCuda::update_biases();
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

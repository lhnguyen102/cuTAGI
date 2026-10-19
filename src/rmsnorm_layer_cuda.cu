#include "../include/cuda_error_checking.cuh"
#include "../include/custom_logger.h"
#include "../include/data_struct_cuda.cuh"
#include "../include/rmsnorm_layer_cuda.cuh"
#include "../include/rmsnorm_layer_cuda_kernel.cuh"

namespace {
constexpr int NUM_THREADS = 256;
int num_blocks(int total) { return (total + NUM_THREADS - 1) / NUM_THREADS; }
int num_row_tiles(int batch_size) {
    return (batch_size + RMSNORM_DELTA_W_ROWS_PER_TILE - 1) /
           RMSNORM_DELTA_W_ROWS_PER_TILE;
}
}  // namespace

RMSNormCuda::RMSNormCuda(const std::vector<int> &normalized_shape, float eps,
                         float gain_w, int device_idx)
    : normalized_shape(normalized_shape), epsilon(eps), gain_w(gain_w) {
    this->bias = false;
    this->device_idx = device_idx;
    if (this->normalized_shape.size() != 1) {
        LOG(LogLevel::ERROR, "Normalized shape provided are not supported.");
    }
    this->input_size = this->normalized_shape[0];
    this->output_size = this->normalized_shape[0];
    this->num_weights = this->normalized_shape[0];
    this->num_biases = 0;
    if (this->training) {
        this->allocate_param_delta();
    }
}

RMSNormCuda::~RMSNormCuda() {
    cudaFree(d_rms_ra);
    cudaFree(d_delta_w_partial);
}

std::string RMSNormCuda::get_layer_info() const { return "RMSNorm()"; }

std::string RMSNormCuda::get_layer_name() const { return "RMSNormCuda"; }

LayerType RMSNormCuda::get_layer_type() const { return LayerType::Norm; }

void RMSNormCuda::allocate_batch_buffers() {
    cudaSetDevice(this->device_idx);
    cudaFree(d_rms_ra);
    CHECK_CUDA_ERROR(
        cudaMalloc((void **)&d_rms_ra, this->_batch_size * sizeof(float)));
    cudaFree(d_delta_w_partial);
    size_t partial_size =
        2 * static_cast<size_t>(num_row_tiles(this->_batch_size)) *
        this->input_size;
    CHECK_CUDA_ERROR(
        cudaMalloc((void **)&d_delta_w_partial, partial_size * sizeof(float)));
}

void RMSNormCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states) {
    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = cu_in->block_size;
    int seq_len = cu_in->seq_len;
    int effective_batch = batch_size * seq_len;
    int ni = this->input_size;
    this->set_cap_factor_udapte(effective_batch);

    if (this->_batch_size != effective_batch) {
        this->_batch_size = effective_batch;
        this->allocate_batch_buffers();
    }

    rmsnorm_stat_rms_cuda<<<effective_batch, RMSNORM_STAT_THREADS>>>(
        cu_in->d_mu_a, cu_in->d_var_a, ni, d_rms_ra);
    rmsnorm_fwd_mean_var_cuda<<<num_blocks(effective_batch * ni),
                                NUM_THREADS>>>(
        this->d_mu_w, this->d_var_w, cu_in->d_mu_a, cu_in->d_var_a, d_rms_ra,
        this->epsilon, ni, effective_batch, cu_out->d_mu_a, cu_out->d_var_a);
    CHECK_LAST_CUDA_ERROR();

    cu_out->width = this->out_width;
    cu_out->height = this->out_height;
    cu_out->depth = this->out_channels;
    cu_out->block_size = batch_size;
    cu_out->seq_len = seq_len;
    cu_out->actual_size = this->output_size;

    if (this->training) {
        this->store_states_for_training_cuda(*cu_in, *cu_out);
    }
}

void RMSNormCuda::backward(BaseDeltaStates &input_delta_states,
                           BaseDeltaStates &output_delta_states,
                           BaseTempStates &temp_states, bool state_udapte) {
    DeltaStateCuda *cu_in_delta =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_out_delta =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);
    BackwardStateCuda *cu_bwd =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

    int batch_size = cu_in_delta->block_size;
    int seq_len = cu_in_delta->seq_len;
    int effective_batch = batch_size * seq_len;
    int ni = this->input_size;

    if (state_udapte) {
        rmsnorm_bwd_delta_z_cuda<<<num_blocks(effective_batch * ni),
                                   NUM_THREADS>>>(
            this->d_mu_w, d_rms_ra, cu_in_delta->d_delta_mu,
            cu_in_delta->d_delta_var, this->epsilon, ni, effective_batch,
            cu_out_delta->d_delta_mu, cu_out_delta->d_delta_var);
        cu_out_delta->seq_len = seq_len;
    }
    if (this->param_update) {
        int row_tiles = num_row_tiles(effective_batch);
        float *partial_mu = d_delta_w_partial;
        float *partial_var = d_delta_w_partial + row_tiles * ni;
        dim3 partial_grid(
            (ni + RMSNORM_DELTA_W_COLS - 1) / RMSNORM_DELTA_W_COLS, row_tiles);
        dim3 partial_block(RMSNORM_DELTA_W_COLS, RMSNORM_DELTA_W_ROW_LANES);
        rmsnorm_bwd_delta_w_partial_cuda<<<partial_grid, partial_block>>>(
            cu_bwd->d_mu_a, d_rms_ra, cu_in_delta->d_delta_mu,
            cu_in_delta->d_delta_var, this->epsilon, ni, effective_batch,
            partial_mu, partial_var);
        rmsnorm_bwd_delta_w_cuda<<<num_blocks(ni), NUM_THREADS>>>(
            partial_mu, partial_var, this->d_var_w, ni, row_tiles,
            this->d_delta_mu_w, this->d_delta_var_w);
    }
    CHECK_LAST_CUDA_ERROR();
}

std::unique_ptr<BaseLayer> RMSNormCuda::to_host() {
    this->params_to_host();
    auto host = std::make_unique<RMSNorm>(this->normalized_shape, this->epsilon,
                                          this->gain_w, this->device_idx);
    host->mu_w = this->mu_w;
    host->var_w = this->var_w;
    return host;
}

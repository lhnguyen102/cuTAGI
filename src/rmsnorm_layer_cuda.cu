#include "../include/cuda_error_checking.cuh"
#include "../include/data_struct_cuda.cuh"
#include "../include/param_init.h"
#include "../include/rmsnorm_layer_cuda.cuh"
#include "../include/rmsnorm_layer_cuda_kernel.cuh"

namespace {
constexpr int THREADS = 256;
inline int blocks_for(int total) { return (total + THREADS - 1) / THREADS; }
}  // namespace

RMSNormCuda::RMSNormCuda(const std::vector<int> &normalized_shape, float eps,
                         float gain_w, int device_idx)
    : normalized_shape(normalized_shape), epsilon(eps), gain_w(gain_w) {
    this->bias = false;
    this->device_idx = device_idx;

    if (this->normalized_shape.size() == 1) {
        this->input_size = this->normalized_shape[0];
        this->output_size = this->normalized_shape[0];
    } else {
        LOG(LogLevel::ERROR, "Normalized shape provided are not supported.");
    }
    this->num_weights = this->normalized_shape[0];
    this->num_biases = 0;

    if (this->training) {
        this->allocate_param_delta();
    }
}

RMSNormCuda::~RMSNormCuda() { this->deallocate_running_rms(); }

std::string RMSNormCuda::get_layer_info() const { return "RMSNorm()"; }

std::string RMSNormCuda::get_layer_name() const { return "RMSNormCuda"; }

LayerType RMSNormCuda::get_layer_type() const { return LayerType::Norm; }

void RMSNormCuda::init_weight_bias() {
    int num_features = this->normalized_shape[0];
    this->num_weights = num_features;
    this->num_biases = 0;

    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_norm("", this->gain_w, 1.0f, num_features,
                              num_features, this->num_weights,
                              this->num_biases);

    this->allocate_param_memory();
    this->params_to_device();
}

void RMSNormCuda::allocate_running_rms() {
    this->deallocate_running_rms();
    this->rms_ra.assign(this->_batch_size, 1.0f);
    cudaSetDevice(this->device_idx);
    CHECK_CUDA_ERROR(
        cudaMalloc((void **)&d_rms_ra, this->_batch_size * sizeof(float)));
}

void RMSNormCuda::deallocate_running_rms() {
    if (d_rms_ra) {
        cudaFree(d_rms_ra);
        d_rms_ra = nullptr;
    }
}

void RMSNormCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states) {
    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = cu_in->block_size;
    int seq_len = cu_in->seq_len;
    int effective_batch = batch_size * seq_len;
    int ni = (int)this->input_size;

    if (effective_batch <= 0 || ni <= 0) return;

    if (this->_batch_size != effective_batch) {
        this->_batch_size = effective_batch;
        this->allocate_running_rms();
    }

    rmsnorm_stat_rms_kernel<<<blocks_for(effective_batch), THREADS>>>(
        cu_in->d_mu_a, cu_in->d_var_a, ni, effective_batch, d_rms_ra);

    int total = effective_batch * ni;
    rmsnorm_fwd_mean_var_kernel<<<blocks_for(total), THREADS>>>(
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
    int ni = (int)this->input_size;

    if (effective_batch <= 0 || ni <= 0) return;

    if (state_udapte) {
        int total = effective_batch * ni;
        rmsnorm_bwd_delta_z_kernel<<<blocks_for(total), THREADS>>>(
            this->d_mu_w, d_rms_ra, cu_in_delta->d_delta_mu,
            cu_in_delta->d_delta_var, this->epsilon, ni, effective_batch,
            cu_out_delta->d_delta_mu, cu_out_delta->d_delta_var);
    }

    if (this->param_update) {
        rmsnorm_bwd_delta_w_kernel<<<blocks_for(ni), THREADS>>>(
            cu_bwd->d_mu_a, this->d_var_w, d_rms_ra, cu_in_delta->d_delta_mu,
            cu_in_delta->d_delta_var, this->epsilon, ni, effective_batch,
            this->d_delta_mu_w, this->d_delta_var_w);
    }
    CHECK_LAST_CUDA_ERROR();
}

std::unique_ptr<BaseLayer> RMSNormCuda::to_host() {
    auto host = std::make_unique<RMSNorm>(this->normalized_shape, this->epsilon,
                                          this->gain_w, this->device_idx);
    this->params_to_host();
    host->mu_w = this->mu_w;
    host->var_w = this->var_w;
    host->mu_b = this->mu_b;
    host->var_b = this->var_b;
    return host;
}

#include "../include/rmsnorm_layer.h"

#include <cmath>
#include <thread>

#include "../include/custom_logger.h"
#include "../include/param_init.h"

////////////////////////////////////////////////////////////////////////////////
/// CPU kernels for RMS Norm
////////////////////////////////////////////////////////////////////////////////
void rmsnorm_stat_rms(const std::vector<float> &mu_a,
                      const std::vector<float> &var_a, int ni, int start_chunk,
                      int end_chunk, std::vector<float> &rms_ra)
/*
 */
{
    for (int col = start_chunk; col < end_chunk; col++) {  // batch size
        float sum = 0.0f;
        for (int i = 0; i < ni; i++) {  // hidden node
            float mu_sq = mu_a[col * ni + i] * mu_a[col * ni + i];
            sum += mu_sq;
        }
        rms_ra[col] = sum / ni;
    }
}

void rmsnorm_fwd_mean_var(const std::vector<float> &mu_w,
                          const std::vector<float> &var_w,
                          const std::vector<float> &mu_a,
                          const std::vector<float> &var_a,
                          const std::vector<float> &rms_ra, float epsilon,
                          int ni, int start_chunk, int end_chunk,
                          std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    for (int row = start_chunk; row < end_chunk; row++) {  // batch size
        float inv_rms = 1.0f / std::sqrt(rms_ra[row] + epsilon);
        float inv_rms_sq = inv_rms * inv_rms;

        for (int col = 0; col < ni; col++) {  // hidden node
            int index = col + row * ni;
            float normalized_mu = mu_a[index] * inv_rms;
            float normalized_var = var_a[index] * inv_rms_sq;

            mu_z[index] = normalized_mu * mu_w[col];
            var_z[index] =
                normalized_var *
                (var_a[index] * (var_w[col] + mu_w[col] * mu_w[col]) +
                 var_w[col] * mu_a[index] * mu_a[index]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// RMS Norm's backward
////////////////////////////////////////////////////////////////////////////////
void rmsnorm_bwd_delta_z(const std::vector<float> &mu_w,
                         const std::vector<float> &rms_ra,
                         const std::vector<float> &delta_mu_out,
                         const std::vector<float> &delta_var_out, float epsilon,
                         int ni, int start_chunk, int end_chunk,
                         std::vector<float> &delta_mu,
                         std::vector<float> &delta_var)
/*
 */
{
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_rms = 1.0f / std::sqrt(rms_ra[row] + epsilon);
        for (int col = 0; col < ni; col++) {
            float tmp = inv_rms * mu_w[col];

            delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];
            delta_var[col + row * ni] =
                tmp * delta_var_out[col + row * ni] * tmp;
        }
    }
}

void rmsnorm_bwd_delta_w(const std::vector<float> &mu_a,
                         const std::vector<float> &rms_ra,
                         const std::vector<float> &delta_mu_out,
                         const std::vector<float> &delta_var_out, float epsilon,
                         int ni, int batch_size, int start_chunk, int end_chunk,
                         std::vector<float> &delta_mu_w,
                         std::vector<float> &delta_var_w)
/*
 */
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            float inv_rms = 1.0f / std::sqrt(rms_ra[row] + epsilon);
            float tmp = inv_rms * mu_a[col + row * ni];

            sum_mu += tmp * delta_mu_out[col + row * ni];
            sum_var += tmp * delta_var_out[col + row * ni] * tmp;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Multiprocessing kernels for RMS norm
////////////////////////////////////////////////////////////////////////////////
void rmsnorm_stat_rms_mp(const std::vector<float> &mu_a,
                         const std::vector<float> &var_a, int ni,
                         int batch_size, const int num_threads,
                         std::vector<float> &rms_ra)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &var_a, &rms_ra] {
            rmsnorm_stat_rms(mu_a, var_a, ni, start_chunk, end_chunk, rms_ra);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void rmsnorm_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &rms_ra, float epsilon, int ni, int batch_size,
    const int num_threads, std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back(
            [=, &mu_w, &var_w, &mu_a, &var_a, &rms_ra, &mu_z, &var_z] {
                rmsnorm_fwd_mean_var(mu_w, var_w, mu_a, var_a, rms_ra, epsilon,
                                     ni, start_chunk, end_chunk, mu_z, var_z);
            });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void rmsnorm_bwd_delta_z_mp(const std::vector<float> &mu_w,
                            const std::vector<float> &rms_ra,
                            const std::vector<float> &delta_mu_out,
                            const std::vector<float> &delta_var_out,
                            float epsilon, int ni, int batch_size,
                            const int num_threads, std::vector<float> &delta_mu,
                            std::vector<float> &delta_var)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &rms_ra, &delta_mu_out, &delta_var_out,
                              &delta_mu, &delta_var] {
            rmsnorm_bwd_delta_z(mu_w, rms_ra, delta_mu_out, delta_var_out,
                                epsilon, ni, start_chunk, end_chunk, delta_mu,
                                delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void rmsnorm_bwd_delta_w_mp(const std::vector<float> &mu_a,
                            const std::vector<float> &rms_ra,
                            const std::vector<float> &delta_mu_out,
                            const std::vector<float> &delta_var_out,
                            float epsilon, int ni, int batch_size,
                            const int num_threads,
                            std::vector<float> &delta_mu_w,
                            std::vector<float> &delta_var_w)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &rms_ra, &delta_mu_out, &delta_var_out,
                              &delta_mu_w, &delta_var_w] {
            rmsnorm_bwd_delta_w(mu_a, rms_ra, delta_mu_out, delta_var_out,
                                epsilon, ni, batch_size, start_chunk, end_chunk,
                                delta_mu_w, delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// RMS Norm class
////////////////////////////////////////////////////////////////////////////////
std::tuple<int, int> get_number_params_rms_norm(
    const std::vector<int> &normalized_shape)
/*
 */
{
    int num_elements = normalized_shape.size();
    int num_weights, num_biases = 0;
    if (num_elements == 1) {
        num_weights = normalized_shape[0];
    } else {
        std::string message = "Normalized shape provided are not supported.";
        LOG(LogLevel::ERROR, message);
    }
    return {num_weights, num_biases};
}

RMSNorm::RMSNorm(const std::vector<int> &normalized_shape, float eps,
                 int device_idx)
    : normalized_shape(normalized_shape),
      epsilon(eps)
/*
 */
{
    this->bias = false;
    this->device_idx = device_idx;
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
    if (this->normalized_shape.size() == 1) {
        this->input_size = this->normalized_shape[0];
        this->output_size = normalized_shape[0];
    } else {
        std::string message = "Normalized shape provided are not supported.";
        LOG(LogLevel::ERROR, message);
    }
}

RMSNorm::~RMSNorm() {}

std::string RMSNorm::get_layer_info() const
/*
 */
{
    return "RMSNorm()";
}

std::string RMSNorm::get_layer_name() const
/*
 */
{
    return "RMSNorm";
}

LayerType RMSNorm::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void RMSNorm::init_weight_bias()
/*
 */
{
    int num_features = this->normalized_shape[0];
    this->num_weights = this->normalized_shape[0];
    this->num_biases = 0;
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_norm("", 1.0f, 1.0f, num_features, num_features,
                              this->num_weights, this->num_biases);
}

void RMSNorm::allocate_running_rms()
/*
 */
{
    this->rms_ra.resize(this->_batch_size, 1.0f);
}

void RMSNorm::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/**/
{
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    int batch_size = input_states.block_size;
    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->allocate_running_rms();
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    if (this->num_threads <= 1) {
        rmsnorm_stat_rms(input_states.mu_a, input_states.var_a,
                         this->input_size, 0, batch_size, this->rms_ra);

        rmsnorm_fwd_mean_var(this->mu_w, this->var_w, input_states.mu_a,
                             input_states.var_a, this->rms_ra, this->epsilon,
                             this->input_size, 0, batch_size,
                             output_states.mu_a, output_states.var_a);
    } else {
        rmsnorm_stat_rms_mp(input_states.mu_a, input_states.var_a,
                            this->input_size, batch_size, this->num_threads,
                            this->rms_ra);

        rmsnorm_fwd_mean_var_mp(this->mu_w, this->var_w, input_states.mu_a,
                                input_states.var_a, this->rms_ra, this->epsilon,
                                this->input_size, batch_size, this->num_threads,
                                output_states.mu_a, output_states.var_a);
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void RMSNorm::backward(BaseDeltaStates &input_delta_states,
                       BaseDeltaStates &output_delta_states,
                       BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    int batch_size = input_delta_states.block_size;

    if (state_udapte) {
        if (this->num_threads <= 1) {
            rmsnorm_bwd_delta_z(
                this->mu_w, this->rms_ra, input_delta_states.delta_mu,
                input_delta_states.delta_var, this->epsilon, this->input_size,
                0, batch_size, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        } else {
            rmsnorm_bwd_delta_z_mp(
                this->mu_w, this->rms_ra, input_delta_states.delta_mu,
                input_delta_states.delta_var, this->epsilon, this->input_size,
                batch_size, this->num_threads, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        }
    }
    if (this->param_update) {
        if (this->num_threads <= 1) {
            rmsnorm_bwd_delta_w(
                this->bwd_states->mu_a, this->rms_ra,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->epsilon, this->input_size, batch_size, 0,
                this->input_size, this->delta_mu_w, this->delta_var_w);
        } else {
            rmsnorm_bwd_delta_w_mp(
                this->bwd_states->mu_a, this->rms_ra,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->epsilon, this->input_size, batch_size, this->num_threads,
                this->delta_mu_w, this->delta_var_w);
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> RMSNorm::to_cuda(int device_idx) {
    std::string message = "CUDA version of RMSNorm is not implemented yet";
    LOG(LogLevel::ERROR, message);
    return nullptr;
}
#endif

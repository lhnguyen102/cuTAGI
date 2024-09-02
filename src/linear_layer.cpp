///////////////////////////////////////////////////////////////////////////////
// File:         linear_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      August 06, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/linear_layer.h"

#include "../include/common.h"

#ifdef USE_CUDA
#include "../include/linear_layer_cuda.cuh"
#endif

void linear_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                         std::vector<float> &mu_b, std::vector<float> &var_b,
                         std::vector<float> &mu_a, std::vector<float> &var_a,
                         int start_chunk, int end_chunk, size_t input_size,
                         size_t output_size, int batch_size, bool bias,
                         std::vector<float> &mu_z, std::vector<float> &var_z)
/*Compute mean of product WA for full connected layer

Args:
  mu_w: Mean of weights
  mu_b: Mean of the biases
  mu_a: Mean of activation unitsthis->mu_a.resize(this->size, 0.0f);
    this->var_a.resize(this->size, 0.0f);
    this->jcb.resize(this->size, 0.0f);

    this->deallocate_memory();
    this->allocate_memory();
  mu_z: Mean of hidden states
  start_chunk: Start index of the chunk
  end_chunk: End index of the chunk
  n: Input node
  m: Output node
  k: Number of batches
*/
{
    float mu_a_tmp;
    float var_a_tmp;
    int n = input_size;
    for (int i = start_chunk; i < end_chunk; i++) {
        int row = i / batch_size;
        int col = i % batch_size;
        float sum_mu_z = 0.0f;
        float sum_var_z = 0.0f;
        for (int j = 0; j < input_size; j++) {
            mu_a_tmp = mu_a[n * col + j];
            var_a_tmp = var_a[n * col + j];
            float mu_w_tmp = mu_w[row * n + j];
            float var_w_tmp = var_w[row * n + j];

            sum_mu_z += mu_w_tmp * mu_a_tmp;
            sum_var_z += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                         var_w_tmp * mu_a_tmp * mu_a_tmp;
        }
        if (bias) {
            mu_z[col * output_size + row] = sum_mu_z + mu_b[row];
            var_z[col * output_size + row] = sum_var_z + var_b[row];
        } else {
            mu_z[col * output_size + row] = sum_mu_z;
            var_z[col * output_size + row] = sum_var_z;
        }
    }
}

void linear_fwd_mean_var_mp(std::vector<float> &mu_w, std::vector<float> &var_w,
                            std::vector<float> &mu_b, std::vector<float> &var_b,
                            std::vector<float> &mu_a, std::vector<float> &var_a,
                            size_t input_size, size_t output_size,
                            int batch_size, bool bias, unsigned int num_threads,
                            std::vector<float> &mu_z, std::vector<float> &var_z)
/*Multi-processing verion of forward pass for fc layer
 */
{
    const int tot_ops = output_size * batch_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &input_size, &output_size, &batch_size, &bias,
                              &mu_z, &var_z] {
            linear_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a,
                                start_chunk, end_chunk, input_size, output_size,
                                batch_size, bias, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void linear_fwd_full_cov(std::vector<float> &mu_w, std::vector<float> &var_a_f,
                         size_t input_size, size_t output_size, int B,
                         int start_chunk, int end_chunk,
                         std::vector<float> &var_z_fp)
/* Add diagonal terms to the full covariance matrix.

Args:
    var_a_f: Full-covariance matrix of activation units for the previous layer
    B: Number of batches
    Sz_fp: Partial full-covariance matrix of hidden states of current
        layer

*/
{
    int tu, col, row, k;
    float Sa_in;
    int ni = input_size;
    int no = output_size;
    for (int j = start_chunk; j < end_chunk; j++) {
        row = j / no;
        col = j % no;
        if (col <= (row % no)) {
            float sum = 0.0f;
            for (int i = 0; i < ni * ni; i++) {
                if ((i / ni) > (i % ni))  // Upper triangle
                {
                    tu = (ni * (i % ni) - (((i % ni) * (i % ni + 1)) / 2) +
                          i / ni);
                } else {
                    tu = (ni * (i / ni) - (((i / ni) * (i / ni + 1)) / 2) +
                          i % ni);
                }
                Sa_in = var_a_f[tu + (row / no) * (ni * (ni + 1)) / 2];
                sum += mu_w[i % ni + (row % no) * ni] * Sa_in *
                       mu_w[i / ni + (col % no) * ni];
            }
            k = no * col - ((col * (col + 1)) / 2) + row % no +
                (row / no) * (((no + 1) * no) / 2);
            var_z_fp[k] = sum;
        }
    }
}

void linear_fwd_full_cov_mp(std::vector<float> &mu_w,
                            std::vector<float> &var_a_f, size_t input_size,
                            size_t output_size, int batch_size,
                            unsigned int num_threads,
                            std::vector<float> &var_z_fp) {
    const int tot_ops = output_size * batch_size * output_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_a_f, &input_size, &output_size,
                              &batch_size, &var_z_fp] {
            linear_fwd_full_cov(mu_w, var_a_f, input_size, output_size,
                                batch_size, start_chunk, end_chunk, var_z_fp);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void linear_fwd_fc_full_var(std::vector<float> &var_w,
                            std::vector<float> &var_b, std::vector<float> &mu_a,
                            std::vector<float> &var_a,
                            std::vector<float> &var_z_fp, size_t input_size,
                            size_t output_size, int B, int start_chunk,
                            int end_chunk, std::vector<float> &var_z,
                            std::vector<float> &var_z_f)
/**/
{
    int col, row, i, k;
    float final_sum;
    int ni = input_size;
    int no = output_size;
    for (int j = start_chunk; j < end_chunk; j++) {
        row = j / B;
        col = j % B;
        float sum = 0.0f;
        for (i = 0; i < ni; i++) {
            sum +=
                var_w[row * ni + i] * var_a[ni * col + i] +
                var_w[row * ni + i] * mu_a[ni * col + i] * mu_a[ni * col + i];
        }
        k = no * row - (row * (row - 1)) / 2 + col * (no * (no + 1)) / 2;
        final_sum = sum + var_b[row] + var_z_fp[k];
        var_z[col * no + row] = final_sum;
        var_z_f[k] = final_sum;
    }
}
void linear_fwd_fc_full_var_mp(
    std::vector<float> &var_w, std::vector<float> &var_b,
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z_fp, int input_size, int output_size,
    int batch_size, unsigned int num_threads, std::vector<float> &var_z,
    std::vector<float> &var_z_f)
/**/
{
    int no = output_size;
    const int tot_ops = no * batch_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int j = 0; j < (no * (no + 1) / 2) * batch_size; j++) {
        var_z_f[j] = var_z_fp[j];
    }
    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &var_b, &mu_a, &var_a, &var_z_fp,
                              &input_size, &output_size, &batch_size, &var_z,
                              &var_z_f] {
            linear_fwd_fc_full_var(var_w, var_b, mu_a, var_a, var_z_fp,
                                   input_size, output_size, batch_size,
                                   start_chunk, end_chunk, var_z, var_z_f);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void linear_bwd_fc_delta_z(std::vector<float> &mu_w, std::vector<float> &jcb,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t input_size,
                           size_t output_size, int B, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu_z,
                           std::vector<float> &delta_var_z)
/*
 */
{
    int ni = input_size;
    int no = output_size;
    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / B;
        int col = j % B;
        float sum_mu_z = 0.0f;
        float sum_var_z = 0.0f;
        for (int i = 0; i < no; i++) {
            sum_mu_z += mu_w[ni * i + row] * delta_mu[col * no + i];

            sum_var_z += mu_w[ni * i + row] * delta_var[col * no + i] *
                         mu_w[ni * i + row];
        }

        // NOTE: Compute directly inovation vector
        delta_mu_z[col * ni + row] = sum_mu_z * jcb[col * ni + row];
        delta_var_z[col * ni + row] =
            sum_var_z * jcb[col * ni + row] * jcb[col * ni + row];
    }
}

void linear_bwd_fc_delta_z_mp(std::vector<float> &mu_w, std::vector<float> &jcb,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t input_size,
                              size_t output_size, int batch_size,
                              unsigned int num_threads,
                              std::vector<float> &delta_mu_z,
                              std::vector<float> &delta_var_z)
/*
 */
{
    const int tot_ops = input_size * batch_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &jcb, &delta_mu, &delta_var,
                              &input_size, &output_size, &batch_size,
                              &delta_mu_z, &delta_var_z] {
            linear_bwd_fc_delta_z(mu_w, jcb, delta_mu, delta_var, input_size,
                                  output_size, batch_size, start_chunk,
                                  end_chunk, delta_mu_z, delta_var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void linear_bwd_fc_delta_w(std::vector<float> &var_w, std::vector<float> &mu_a,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t input_size,
                           size_t output_size, int batch_size, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w)
/* Compute update quantities for the mean of weights for full-connected layer.

Args:
    mu_a: Mean of activation units
    delta_mu: Inovation vector for mean i.e. (M_observation - M_prediction)
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    delta_mu_w: Updated quantities for the mean of weights
    delta_var_w: Updated quantities for the variance of weights
 */
{
    int k = output_size;
    int m = input_size;

    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mu_w = 0.0f;
        float sum_var_w = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum_mu_w += mu_a[m * i + row] * delta_mu[col + k * i];
            sum_var_w +=
                mu_a[m * i + row] * mu_a[m * i + row] * delta_var[col + k * i];
        }

        delta_mu_w[col * m + row] = sum_mu_w * var_w[col * m + row];
        delta_var_w[col * m + row] =
            sum_var_w * var_w[col * m + row] * var_w[col * m + row];
    }
}

void linear_bwd_fc_delta_w_mp(std::vector<float> &var_w,
                              std::vector<float> &mu_a,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t input_size,
                              size_t output_size, int batch_size,
                              unsigned int num_threads,
                              std::vector<float> &delta_mu_w,
                              std::vector<float> &delta_var_w)
/**/
{
    const int tot_ops = input_size * output_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &mu_a, &delta_mu, &delta_var,
                              &input_size, &output_size, &batch_size,
                              &delta_mu_w, &delta_var_w] {
            linear_bwd_fc_delta_w(var_w, mu_a, delta_mu, delta_var, input_size,
                                  output_size, batch_size, start_chunk,
                                  end_chunk, delta_mu_w, delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void linear_bwd_fc_delta_b(std::vector<float> &var_b,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t output_size,
                           int batch_size, int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b)
/* Compute update quantities for the variance of biases for full-connected
layer.

Args:
    C_bz: Covariance b|Z+
    delta_S: Inovation vector for variance i.e. (S_observation - S_prediction)
    m: Number of hidden units for outputs
    n: Number of batches
    k: 1
    deltaSb: Updated quantities for the variance of biases
*/
{
    int m = output_size;
    for (int j = start_chunk; j < end_chunk; j++) {
        int row = j / 1;
        int col = j % 1;
        float sum_mu_b = 0.0f;
        float sum_var_b = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum_mu_b += delta_mu[m * i + row];
            sum_var_b += delta_var[m * i + row];
        }

        delta_mu_b[col * m + row] = sum_mu_b * var_b[col * m + row];
        delta_var_b[col * m + row] =
            sum_var_b * var_b[col * m + row] * var_b[col * m + row];
    }
}

void linear_bwd_fc_delta_b_mp(std::vector<float> &var_b,
                              std::vector<float> &delta_mu,
                              std::vector<float> &delta_var, size_t output_size,
                              int batch_size, unsigned int num_threads,
                              std::vector<float> &delta_mu_b,
                              std::vector<float> &delta_var_b)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = output_size / num_threads;
    int extra = output_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_b, &delta_mu, &delta_var, &output_size,
                              &batch_size, &delta_mu_b, &delta_var_b] {
            linear_bwd_fc_delta_b(var_b, delta_mu, delta_var, output_size,
                                  batch_size, start_chunk, end_chunk,
                                  delta_mu_b, delta_var_b);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

Linear::Linear(size_t ip_size, size_t op_size, bool bias, float gain_weight,
               float gain_bias, std::string method)
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

    // Initalize weights and bias
    if (this->device.compare("cpu") == 0) {
        this->init_weight_bias();
    }

    // Allocate the update quantities for parameters
    if (this->training && this->device.compare("cpu") == 0) {
        this->allocate_param_delta();
    }
}

Linear::~Linear() {}

std::string Linear::get_layer_info() const
/*
 */
{
    return "Linear(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string Linear::get_layer_name() const
/*
 */
{
    return "Linear";
}

LayerType Linear::get_layer_type() const
/*
 */
{
    return LayerType::Linear;
}

void Linear::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_linear(this->init_method, this->gain_w, this->gain_b,
                                this->input_size, this->output_size,
                                this->num_weights, this->num_biases);
}

void Linear::forward(BaseHiddenStates &input_states,
                     BaseHiddenStates &output_states,
                     BaseTempStates &temp_states)
/*
 */
{
    // Initialization
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    // Forward pass
    if (this->num_threads > 1) {
        linear_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               input_states.mu_a, input_states.var_a,
                               this->input_size, this->output_size, batch_size,
                               this->bias, this->num_threads,
                               output_states.mu_a, output_states.var_a);
    } else {
        int start_chunk = 0;
        int end_chunk = this->output_size * batch_size;
        linear_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                            input_states.mu_a, input_states.var_a, start_chunk,
                            end_chunk, this->input_size, this->output_size,
                            batch_size, this->bias, output_states.mu_a,
                            output_states.var_a);
    }
    // Update number of actual states.
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void Linear::backward(BaseDeltaStates &input_delta_states,
                      BaseDeltaStates &output_delta_states,
                      BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    // Initialization
    int batch_size = input_delta_states.block_size;

    // Compute inovation vector
    if (state_udapte) {
        if (this->num_threads > 1) {
            linear_bwd_fc_delta_z_mp(
                this->mu_w, this->bwd_states->jcb, input_delta_states.delta_mu,
                input_delta_states.delta_var, this->input_size,
                this->output_size, batch_size, this->num_threads,
                output_delta_states.delta_mu, output_delta_states.delta_var);
        } else {
            int start_chunk = 0;
            int end_chunk = batch_size * this->input_size;
            linear_bwd_fc_delta_z(
                this->mu_w, this->bwd_states->jcb, input_delta_states.delta_mu,
                input_delta_states.delta_var, this->input_size,
                this->output_size, batch_size, start_chunk, end_chunk,
                output_delta_states.delta_mu, output_delta_states.delta_var);
        }
    }

    // Update values for weights & biases
    if (this->param_update) {
        if (this->num_threads > 1) {
            linear_bwd_fc_delta_w_mp(
                this->var_w, this->bwd_states->mu_a,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->input_size, this->output_size, batch_size,
                this->num_threads, this->delta_mu_w, this->delta_var_w);

            if (this->bias) {
                linear_bwd_fc_delta_b_mp(
                    this->var_b, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->output_size, batch_size,
                    this->num_threads, this->delta_mu_b, this->delta_var_b);
            }
        } else {
            int start_chunk = 0;
            int end_chunk = this->input_size * this->output_size;
            linear_bwd_fc_delta_w(
                this->var_w, this->bwd_states->mu_a,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->input_size, this->output_size, batch_size, start_chunk,
                end_chunk, this->delta_mu_w, this->delta_var_w);

            if (this->bias) {
                linear_bwd_fc_delta_b(this->var_b, input_delta_states.delta_mu,
                                      input_delta_states.delta_var,
                                      this->output_size, batch_size,
                                      start_chunk, this->output_size,
                                      this->delta_mu_b, this->delta_var_b);
            }
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Linear::to_cuda() {
    this->device = "cuda";
    return std::make_unique<LinearCuda>(this->input_size, this->output_size,
                                        this->bias, this->gain_w, this->gain_b,
                                        this->init_method);
}
#endif

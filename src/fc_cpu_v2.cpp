///////////////////////////////////////////////////////////////////////////////
// File:         fc_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      November 13, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/fc_cpu_v2.h"

FullyConnectedLayer::FullyConnectedLayer(size_t ip_size, size_t op_size,
                                         float gain_weight, float gain_bias,
                                         std::string method)
    : gain_w(gain_weight),
      gain_b(gain_bias),
      init_method(method)
/*
 */
{
    this->input_size = ip_size;
    this->output_size = op_size;

    // Initalize weights and bias
    this->init_weight_bias();
}

FullyConnectedLayer::~FullyConnectedLayer() {}

void FullyConnectedLayer::init_weight_bias()
/*
 */
{
    int num_weights = this->input_size * this->output_size;
    float scale = 0.1f;
    if (this->init_method.compare("Xavier") == 0 ||
        this->init_method.compare("xavier") == 0) {
        auto scale = xavier_init(this->input_size, this->output_size);
    } else if (this->init_method.compare("He") == 0 ||
               this->init_method.compare("he") == 0) {
        auto scale = he_init(this->input_size);
    } else {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Inital parameter method '" +
                                    init_method + "'is not supported.");
    }

    // Weights & biases
    std::tie(this->mu_w, this->var_w) =
        gaussian_param_init(scale, this->gain_w, num_weights);

    std::tie(this->mu_b, this->var_b) =
        gaussian_param_init(scale, this->gain_b, this->output_size);
}

void FullyConnectedLayer::fwd_mean_var(
    std::vector<float> &mu_w, std::vector<float> &var_w,
    std::vector<float> &mu_b, std::vector<float> &var_b,
    std::vector<float> &mu_a, std::vector<float> &var_a, int start_chunk,
    int end_chunk, size_t input_size, size_t output_size, int batch_size,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*Compute mean of product WA for full connected layer

Args:
  mu_w: Mean of weights
  mu_b: Mean of the biases
  mu_a: Mean of activation units
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
            sum_mu_z += mu_w[row * n + j] * mu_a_tmp;
            sum_var_z +=
                (mu_w[row * n + j] * mu_w[row * n + j] + var_w[row * n + j]) *
                    mu_a_tmp +
                var_w[row * n + j] * mu_a_tmp * mu_a_tmp;
        }
        mu_z[col * output_size + row] = sum_mu_z + mu_b[row];
        var_z[col * output_size + row] = sum_var_z + var_b[row];
    }
}

int FullyConnectedLayer::get_input_size()
/*
 */
{
    return this->input_size;
}

int FullyConnectedLayer::get_output_size()
/*
 */
{
    return this->output_size;
}

void FullyConnectedLayer::fwd_mean_var_mp(std::vector<float> &mu_a,
                                          std::vector<float> &var_a,
                                          int batch_size,
                                          unsigned int num_threads,
                                          std::vector<float> &mu_z,
                                          std::vector<float> &var_z)
/*Multi-processing verion of forward pass for fc layer
 */
{
    const int tot_ops = output_size * batch_size;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(&FullyConnectedLayer::fwd_mean_var, this,
                                 std::ref(this->mu_w), std::ref(this->var_w),
                                 std::ref(this->mu_b), std::ref(this->var_b),
                                 std::ref(mu_a), std::ref(var_a), start_chunk,
                                 end_chunk, this->input_size, this->output_size,
                                 batch_size, std::ref(mu_z), std::ref(var_z));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void FullyConnectedLayer::fwd_full_cov(std::vector<float> &mu_w,
                                       std::vector<float> &var_a_f,
                                       size_t input_size, size_t output_size,
                                       int B, int start_chunk, int end_chunk,
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

void FullyConnectedLayer::fwd_full_cov_mp(std::vector<float> &var_a_f, int B,
                                          unsigned int num_threads,
                                          std::vector<float> &var_z_fp) {
    const int tot_ops = this->output_size * B * this->output_size;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(&FullyConnectedLayer::fwd_full_cov, this,
                                 std::ref(this->mu_w), std::ref(var_a_f),
                                 this->input_size, this->output_size, B,
                                 start_chunk, end_chunk, std::ref(var_z_fp));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void FullyConnectedLayer::fwd_fc_full_var(
    std::vector<float> &var_w, std::vector<float> &var_b,
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z_fp, size_t input_size, size_t output_size, int B,
    int start_chunk, int end_chunk, std::vector<float> &var_z,
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

void FullyConnectedLayer::fwd_fc_full_var_mp(std::vector<float> &mu_a,
                                             std::vector<float> &var_a,
                                             std::vector<float> &var_z_fp,
                                             int B, unsigned int num_threads,
                                             std::vector<float> &var_z,
                                             std::vector<float> &var_z_f)
/**/
{
    int no = this->output_size;
    const int tot_ops = no * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_chunk, end_chunk;

    for (int j = 0; j < (no * (no + 1) / 2) * B; j++) {
        var_z_f[j] = var_z_fp[j];
    }
    std::thread threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            &FullyConnectedLayer::fwd_fc_full_var, this, std::ref(this->var_w),
            std::ref(this->var_b), std::ref(mu_a), std::ref(var_a),
            std::ref(var_z_fp), this->input_size, this->output_size, B,
            start_chunk, end_chunk, std::ref(var_z), std::ref(var_z_f));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void FullyConnectedLayer::bwd_fc_delta_z(std::vector<float> &mu_w,
                                         std::vector<float> &jcb,
                                         std::vector<float> &delta_mu,
                                         std::vector<float> &delta_var,
                                         size_t input_size, size_t output_size,
                                         int B, int start_chunk, int end_chunk,
                                         std::vector<float> &delta_mu_z,
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

void FullyConnectedLayer::bwd_fc_delta_z_mp(std::vector<float> &jcb,
                                            std::vector<float> &delta_mu,
                                            std::vector<float> &delta_var,
                                            int B, unsigned int num_threads,
                                            std::vector<float> &delta_mu_z,
                                            std::vector<float> &delta_var_z)
/*
 */
{
    const int ni = this->input_size;
    const int tot_ops = ni * B;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            &FullyConnectedLayer::bwd_fc_delta_z, this, std::ref(this->mu_w),
            std::ref(jcb), std::ref(delta_mu), std::ref(delta_var),
            this->input_size, this->output_size, B, start_chunk, end_chunk,
            std::ref(delta_mu_z), std::ref(delta_var_z));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void FullyConnectedLayer::bwd_fc_delta_w(
    std::vector<float> &var_w, std::vector<float> &mu_a,
    std::vector<float> &delta_mu, std::vector<float> &delta_var,
    size_t input_size, size_t output_size, int batch_size, int start_chunk,
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

void FullyConnectedLayer::bwd_fc_delta_w_mp(
    std::vector<float> &mu_a, std::vector<float> &delta_mu,
    std::vector<float> &delta_var, int batch_size, unsigned int num_threads,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/**/
{
    const int tot_ops = this->input_size * this->output_size;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            &FullyConnectedLayer::bwd_fc_delta_w, this, std::ref(this->var_w),
            std::ref(mu_a), std::ref(delta_mu), std::ref(delta_var),
            this->input_size, this->output_size, batch_size, start_chunk,
            end_chunk, std::ref(delta_mu_w), std::ref(delta_var_w));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void FullyConnectedLayer::bwd_fc_delta_b(std::vector<float> &var_b,
                                         std::vector<float> &delta_mu,
                                         std::vector<float> &delta_var,
                                         int batch_size, size_t output_size,
                                         int start_chunk, int end_chunk,
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
        float sum_mu_b = 0.0f;
        float sum_var_b = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum_mu_b += delta_mu[m * i + j];
            sum_var_b += delta_var[m * i + j];
        }

        delta_mu_b[j * m + j] = sum_mu_b * var_b[j * m + j];
        delta_var_b[j * m + j] =
            sum_var_b * var_b[j * m + j] * var_b[j * m + j];
    }
}

void FullyConnectedLayer::bwd_fc_delta_b_mp(std::vector<float> &delta_mu,
                                            std::vector<float> &delta_var,
                                            int batch_size,
                                            unsigned int num_threads,
                                            std::vector<float> &delta_mu_b,
                                            std::vector<float> &delta_var_b)
/*
 */
{
    const int tot_ops = this->output_size;
    const int n_batch = tot_ops / num_threads;
    const int rem_batch = tot_ops % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(&FullyConnectedLayer::bwd_fc_delta_b, this,
                                 std::ref(this->var_b), std::ref(delta_mu),
                                 std::ref(delta_var), this->output_size,
                                 batch_size, start_chunk, end_chunk,
                                 std::ref(delta_mu_b), std::ref(delta_var_b));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void FullyConnectedLayer::forward(HiddenStates &input_states,
                                  HiddenStates &output_states,
                                  TempStates &temp_states)
/*
 */
{
    // Initialization. TODO: figure out where to put batch size if mu_a is a
    // member of buffer
    int batch_size = input_states.block_size;
    int start_chunk = 0;
    int end_chunk = this->output_size * batch_size;

    // Forward pass
    this->fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                       input_states.mu_a, input_states.var_a, start_chunk,
                       end_chunk, this->input_size, this->output_size,
                       batch_size, output_states.mu_z, output_states.var_z);

    // Update number of actual states.
    output_states.size = this->output_size * batch_size;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;
}

void FullyConnectedLayer::state_backward(std::vector<float> &jcb,
                                         DeltaStates &input_delta_states,
                                         DeltaStates &output_delta_states,
                                         TempStates &temp_states)
/*
 */
{
    // Initialization
    int batch_size = input_delta_states.block_size;
    int start_chunk = 0;
    int end_chunk = batch_size * this->output_size;

    // Compute inovation vector
    this->bwd_fc_delta_z(this->mu_w, jcb, input_delta_states.delta_mu,
                         input_delta_states.delta_var, this->input_size,
                         this->output_size, batch_size, start_chunk, end_chunk,
                         output_delta_states.delta_mu,
                         output_delta_states.delta_var);
}

void FullyConnectedLayer::param_backward(DeltaStates &delta_states,
                                         TempStates &temp_states)
/*
...

Args:
    mu_a: Mean of input activations
 */
{
    // Initialization
    int batch_size = delta_states.block_size;
    int start_chunk = 0;
    int end_chunk = batch_size * this->output_size;

    // Update values for weights
    this->bwd_fc_delta_w(this->var_w, this->mu_a, delta_states.delta_mu,
                         delta_states.delta_var, this->input_size,
                         this->output_size, batch_size, start_chunk, end_chunk,
                         this->delta_mu_w, this->delta_var_w);

    // Update values for biases
    this->bwd_fc_delta_b(this->var_b, delta_states.delta_mu,
                         delta_states.delta_var, this->output_size, batch_size,
                         start_chunk, end_chunk, this->delta_mu_b,
                         this->delta_var_b);
}
///////////////////////////////////////////////////////////////////////////////
// File:         fc_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      September 20, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/fc_cpu_v2.h"

FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size,
                                         size_t batch_size)
    : input_size(input_size),
      output_size(output_size),
      batch_size(batch_size),
      mu_z(output_size * batch_size),
      var_z(output_size * batch_size),
      mu_a(output_size * batch_size),
      var_a(output_size * batch_size),
      jcb(output_size * batch_size) {}

void FullyConnectedLayer::fwd_mean_var(
    std::vector<float> &mu_w, std::vector<float> &var_w,
    std::vector<float> &mu_b, std::vector<float> &var_b,
    std::vector<float> &mu_a, std::vector<float> &var_a, int start_chunk,
    int end_chunk, int w_pos, int b_pos, int z_pos_in, int z_pos_out,
    int output_size, int input_size, int batch_size, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*Compute mean of product WA for full connected layer

Args:
  mu_w: Mean of weights
  mu_b: Mean of the biases
  mu_a: Mean of activation units
  mu_z: Mean of hidden states
  start_chunk: Start index of the chunk
  end_chunk: End index of the chunk
  w_pos: Weight position for this layer in the weight vector of network
  b_pos: Bias position for this layer in the bias vector of network
  z_pos_in: Input-hidden-state position for this layer in the hidden-state
      vector of network
  z_pos_out: Output-hidden-state position for this layer in the hidden-state
      vector of network
  n: Input node
  m: Output node
  k: Number of batches
*/
{
    float mu_a_tmp = 0;
    float var_a_tmp = 0;
    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < batch_size; col++) {
            float sum_mu = 0;
            float sum_var = 0;
            for (int i = 0; i < input_size; i++) {
                int idx_a = input_size * col + i + z_pos_in;
                int idx_w = row * input_size + i + w_pos;

                mu_a_tmp = mu_a[idx_a];
                var_a_tmp = var_a[idx_a];

                sum_mu += mu_w[idx_w] * mu_a_tmp;
                sum_var +=
                    (mu_w[idx_w] * mu_w[idx_w] + var_w[idx_w]) * var_a_tmp +
                    var_w[idx_w] * mu_a_tmp * mu_a_tmp;
            }
            mu_z[col * output_size + row + z_pos_out] =
                sum_mu + mu_b[row + b_pos];

            var_z[col * output_size + row + z_pos_out] =
                sum_var + var_b[row + b_pos];
        }
    }
}

void FullyConnectedLayer::fwd_mean_var_mp(
    std::vector<float> &mu_w, std::vector<float> &var_w,
    std::vector<float> &mu_b, std::vector<float> &var_b,
    std::vector<float> &mu_a, std::vector<float> &var_a, int w_pos, int b_pos,
    int z_pos_in, int z_pos_out, int output_size, int input_size,
    int batch_size, unsigned int NUM_THREADS, std::vector<float> &mu_z,
    std::vector<float> &var_z)
/*Multi-processing verion of forward pass for fc layer
 */
{
    const int tot_ops = output_size * batch_size;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_chunk, end_chunk;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            &FullyConnectedLayer::fwd_mean_var, this, std::ref(mu_w),
            std::ref(var_w), std::ref(mu_b), std::ref(var_b), std::ref(mu_a),
            std::ref(var_a), start_chunk, end_chunk, w_pos, b_pos, z_pos_in,
            z_pos_out, output_size, input_size, batch_size, std::ref(mu_z),
            std::ref(var_z));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}
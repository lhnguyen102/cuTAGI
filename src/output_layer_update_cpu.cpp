///////////////////////////////////////////////////////////////////////////////
// File:         output_layer_update_cpu.cpu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 22, 2023
// Updated:      October 23, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/output_layer_update_cpu.h"

void compute_delta_z_output(std::vector<float> &mu_a, std::vector<float> &var_a,
                            std::vector<float> &var_z, std::vector<float> &jcb,
                            std::vector<float> &obs, std::vector<float> &var_v,
                            int start_chunk, int end_chunk,
                            std::vector<float> &delta_mu,
                            std::vector<float> &delta_var)
/*
 */
{
    float zero_pad = 0;
    float tmp = 0;
    // We compute directely the inovation vector for output layer
    for (int col = start_chunk; col < end_chunk; col++) {
        tmp = jcb[col] / (var_a[col] + var_v[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[col] = zero_pad;
            delta_var[col] = zero_pad;
        } else {
            delta_mu[col] = tmp * (obs[col] - mu_a[col]);
            delta_var[col] = -tmp * jcb[col];
        }
    }
}

void compute_delta_z_output_mp(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z, std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_v, int n, unsigned int num_threads,
    std::vector<float> &delta_mu, std::vector<float> &delta_var)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::vector<std::thread> threads(num_threads);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            compute_delta_z_output, std::ref(mu_a), std::ref(var_a),
            std::ref(var_z), std::ref(jcb), std::ref(obs), std::ref(var_v),
            start_chunk, end_chunk, std::ref(delta_mu), std::ref(delta_var));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void compute_delta_z_output_with_indices(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z, std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_v, std::vector<int> &ud_idx, int n_obs, int n_enc,
    int start_chunk, int end_chunk, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/*
 */
{
    float zero_pad = 0;
    float tmp = 0;
    int idx = 0;
    // We compute directely the inovation vector for output layer
    for (int col = start_chunk; col < end_chunk; col++) {
        // minus 1 because we start index at 1 not zero. TODO: reset hierarchy
        // encoder index at 0
        idx = ud_idx[col] + (col / n_enc) * n_obs - 1;
        tmp = jcb[idx] / (var_a[idx] + var_v[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[idx] = zero_pad;
            delta_var[idx] = zero_pad;
        } else {
            delta_mu[idx] = tmp * (obs[col] - mu_a[idx]);
            delta_var[idx] = -tmp * jcb[idx];
        }
    }
}

void compute_delta_z_output_with_indices_mp(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &var_z, std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_v, std::vector<int> &ud_idx, int n_obs, int n_enc,
    int n, unsigned int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::vector<std::thread> threads(num_threads);

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            compute_delta_z_output_with_indices, std::ref(mu_a),
            std::ref(var_a), std::ref(var_z), std::ref(jcb), std::ref(obs),
            std::ref(var_v), std::ref(ud_idx), n_obs, n_enc, start_chunk,
            end_chunk, std::ref(delta_mu), std::ref(delta_var));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}
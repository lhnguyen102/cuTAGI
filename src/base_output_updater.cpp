///////////////////////////////////////////////////////////////////////////////
// File:         base_output_updater.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 27, 2023
// Updated:      December 27, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/base_output_updater.h"

#ifdef USE_CUDA
#include "../include/output_updater_cuda.cuh"
#endif

void compute_delta_z_output(std::vector<float> &mu_a, std::vector<float> &var_a,
                            std::vector<float> &jcb, std::vector<float> &obs,
                            std::vector<float> &var_obs, int start_chunk,
                            int end_chunk, std::vector<float> &delta_mu,
                            std::vector<float> &delta_var)
/*
 */
{
    float zero_pad = 0;
    float tmp = 0;
    // We compute directely the inovation vector for output layer
    for (int col = start_chunk; col < end_chunk; col++) {
        tmp = jcb[col] / (var_a[col] + var_obs[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[col] = zero_pad;
            delta_var[col] = zero_pad;
        } else {
            delta_mu[col] = tmp * (obs[col] - mu_a[col]);
            delta_var[col] = -tmp * jcb[col];
        }
    }
}

void compute_delta_z_output_mp(std::vector<float> &mu_a,
                               std::vector<float> &var_a,
                               std::vector<float> &jcb, std::vector<float> &obs,
                               std::vector<float> &var_v, int n,
                               unsigned int num_threads,
                               std::vector<float> &delta_mu,
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
        threads[i] = std::thread(compute_delta_z_output, std::ref(mu_a),
                                 std::ref(var_a), std::ref(jcb), std::ref(obs),
                                 std::ref(var_v), start_chunk, end_chunk,
                                 std::ref(delta_mu), std::ref(delta_var));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void compute_selected_delta_z_output(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_obs, std::vector<int> &selected_idx, int n_obs,
    int n_enc, int start_chunk, int end_chunk, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/*
It computes the selected delta hidden states for output layer e.g., hierarchical
binary tree for classification task.
*/
{
    float zero_pad = 0.0f;
    float tmp = 0.0f;
    int idx = 0;
    for (int col = start_chunk; col < end_chunk; col++) {
        // minus 1 because the encoding index start at 1
        idx = selected_idx[col] + (col / n_enc) * n_obs - 1;
        tmp = jcb[idx] / (var_a[idx] + var_obs[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu[idx] = zero_pad;
            delta_var[idx] = zero_pad;
        } else {
            delta_mu[idx] = tmp * (obs[col] - mu_a[idx]);
            delta_var[idx] = -tmp * jcb[idx];
        }
    }
}

void compute_selected_delta_z_output_mp(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &jcb, std::vector<float> &obs,
    std::vector<float> &var_obs, std::vector<int> &selected_idx, int n_obs,
    int n_enc, int n, unsigned int num_threads, std::vector<float> &delta_mu,
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
            compute_selected_delta_z_output, std::ref(mu_a), std::ref(var_a),
            std::ref(jcb), std::ref(obs), std::ref(var_obs),
            std::ref(selected_idx), n_obs, n_enc, start_chunk, end_chunk,
            std::ref(delta_mu), std::ref(delta_var));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Base Output Updater
////////////////////////////////////////////////////////////////////////////////
BaseOutputUpdater::BaseOutputUpdater() {}
BaseOutputUpdater::~BaseOutputUpdater() {}

void BaseOutputUpdater::update_output_delta_z(BaseHiddenStates &output_states,
                                              BaseObservation &obs,
                                              BaseDeltaStates &delta_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = obs.size;

    delta_states.reset_zeros();

    compute_delta_z_output(output_states.mu_a, output_states.var_a,
                           output_states.jcb, obs.mu_obs, obs.var_obs,
                           start_chunk, end_chunk, delta_states.delta_mu,
                           delta_states.delta_var);
}

void BaseOutputUpdater::update_selected_output_delta_z(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = obs.size;
    int n_enc = obs.idx_size / output_states.block_size;
    int n_obs = output_states.actual_size;

    delta_states.reset_zeros();

    compute_selected_delta_z_output(
        output_states.mu_a, output_states.var_a, output_states.jcb, obs.mu_obs,
        obs.var_obs, obs.selected_idx, n_obs, n_enc, start_chunk, end_chunk,
        delta_states.delta_mu, delta_states.delta_var);
}

////////////////////////////////////////////////////////////////////////////////
// Output Updater
////////////////////////////////////////////////////////////////////////////////
OutputUpdater::OutputUpdater(const std::string model_device)
    : device(model_device) {
#ifdef USE_CUDA
    if (this->device.compare("cuda") == 0) {
        this->updater = std::make_shared<OutputUpdaterCuda>();
        this->obs = std::make_shared<ObservationCuda>();
    } else
#endif
    {
        this->updater = std::make_shared<BaseOutputUpdater>();
        this->obs = std::make_shared<BaseObservation>();
    }
}

OutputUpdater::OutputUpdater() {}

OutputUpdater::~OutputUpdater() {}

void OutputUpdater::update(BaseHiddenStates &output_states,
                           std::vector<float> &mu_obs,
                           std::vector<float> &var_obs,
                           BaseDeltaStates &delta_states)
/*
 */
{
    this->obs->set_obs(mu_obs, var_obs);
    this->obs->block_size = output_states.block_size;
    this->obs->size = mu_obs.size();
    this->obs->actual_size = mu_obs.size() / output_states.block_size;

    this->updater->update_output_delta_z(output_states, *this->obs,
                                         delta_states);
}

void OutputUpdater::update_using_indices(BaseHiddenStates &output_states,
                                         std::vector<float> &mu_obs,
                                         std::vector<float> &var_obs,
                                         std::vector<int> &selected_idx,
                                         BaseDeltaStates &delta_states)
/*
 */
{
    this->obs->set_obs(mu_obs, var_obs);
    this->obs->set_selected_idx(selected_idx);
    if (this->obs->size != mu_obs.size()) {
        this->obs->block_size = output_states.block_size;
        this->obs->size = mu_obs.size();
        this->obs->actual_size = mu_obs.size() / output_states.block_size;
        this->obs->idx_size = selected_idx.size();
    }
    this->updater->update_selected_output_delta_z(output_states, *this->obs,
                                                  delta_states);
}

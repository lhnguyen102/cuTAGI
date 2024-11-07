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
        if (isinf(tmp) || isnan(tmp) || isnan(obs[col])) {
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

void compute_delta_z_heteros(std::vector<float> &mu_a,
                             std::vector<float> &var_a, std::vector<float> &jcb,
                             std::vector<float> &obs, int start_chunk,
                             int end_chunk, std::vector<float> &delta_mu,
                             std::vector<float> &delta_var)
/*
Compute delta hidden states for output layer with learned heteroscedastic
noise. This function receives a vector of observations and the twice output
hidden states. Using AGVI, we can infere the posterior for observation noise
v and use it to update the hidden states Z_out.

Terminology:
- V: Gaussian random variable describing the error variance sigma^2. N(0,
sqrt(V))
- V2: Square of the error (V^2)
- V2_bar: Gaussian random variable describing the expected value of V2
(mu_V2)
- V2_bar_tilde: Gaussian random variable describing V2 after passing through
an exponential activation function to restrict values to the positive domain


For more detail see https://www.jmlr.org/papers/volume22/20-1009/20-1009.pdf

Args:
    mu_a: mean of the hidden states, i.e., Z_out and V2_bar_tilde
    var_a: variance of the hidden states
    jcb: Jacobian of the hidden states
    obs: observed data
    start_chunk: start index of the hidden states
    end_chunk: end index of the hidden states
    delta_mu: delta mean of the hidden states
    delta_var: delta variance of the hidden states
*/
{
    const float zero_pad = 0.0f;

    for (int col = start_chunk; col < end_chunk; col += 2) {
        // mean of the Gaussian distribution for the output
        float var_a_col = var_a[col];
        float mu_a_col = mu_a[col];
        float jcb_col = jcb[col];

        // V2_bar_tilde
        float mu_V2_bar_tilde = mu_a[col + 1];
        float var_V2_bar_tilde = var_a[col + 1];
        float cov_V2_bar_tilde = jcb[col + 1];

        // Compute the prior predictive PDF for v2
        float mu_V2 = mu_V2_bar_tilde;
        float var_V2 =
            3.0f * var_V2_bar_tilde + 2.0f * mu_V2_bar_tilde * mu_V2_bar_tilde;
        float cov_y_V = mu_V2;

        // Variance of the output
        float var_sum = var_a_col + mu_V2;

        // Compute updating quantities for the mean of the output
        float tmp = jcb_col / var_sum;
        if (std::isinf(tmp) || std::isnan(tmp)) {
            delta_mu[col] = zero_pad;
            delta_var[col] = zero_pad;
        } else {
            float obs_diff = obs[col / 2] - mu_a_col;
            delta_mu[col] = tmp * obs_diff;
            delta_var[col] = -tmp * jcb_col;
        }

        // Compute the posterior mean and variance for V
        float mu_V_pos = cov_y_V / var_sum * (obs[col / 2] - mu_a_col);
        float var_V_pos = mu_V2 - cov_y_V / var_sum * cov_y_V;

        // Compute the posterior mean and variance for V2
        float mu_V2_pos = mu_V_pos * mu_V_pos + var_V_pos;
        float var_V2_pos = 2.0f * var_V_pos * var_V_pos +
                           4.0f * var_V_pos * mu_V_pos * mu_V_pos;

        // Compute the posterior mean and variance for V2_bar_tilde
        float k = var_V2_bar_tilde / var_V2;
        float mu_V2_bar_tilde_pos = mu_V2_bar_tilde + k * (mu_V2_pos - mu_V2);
        float var_V2_bar_tilde_pos =
            var_V2_bar_tilde + k * k * (var_V2_pos - var_V2);

        // Compute deltas for V2_bar
        float Jv = cov_V2_bar_tilde / var_V2_bar_tilde;
        delta_mu[col + 1] = Jv * (mu_V2_bar_tilde_pos - mu_V2_bar_tilde);
        delta_var[col + 1] =
            Jv * Jv * (var_V2_bar_tilde_pos - var_V2_bar_tilde);
    }
}

void compute_delta_z_heteros_mp(std::vector<float> &mu_a,
                                std::vector<float> &var_a,
                                std::vector<float> &jcb,
                                std::vector<float> &obs, int n,
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
        threads[i] = std::thread(compute_delta_z_heteros, std::ref(mu_a),
                                 std::ref(var_a), std::ref(jcb), std::ref(obs),
                                 start_chunk, end_chunk, std::ref(delta_mu),
                                 std::ref(delta_var));
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

void BaseOutputUpdater::update_output_delta_z_heteros(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = obs.size * 2;

    delta_states.reset_zeros();

    compute_delta_z_heteros(
        output_states.mu_a, output_states.var_a, output_states.jcb, obs.mu_obs,
        start_chunk, end_chunk, delta_states.delta_mu, delta_states.delta_var);
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

void OutputUpdater::update_heteros(BaseHiddenStates &output_states,
                                   std::vector<float> &mu_obs,
                                   BaseDeltaStates &delta_states)
/*
 */
{
    auto var_obs = std::vector<float>(mu_obs.size(), 0.0f);

    this->obs->set_obs(mu_obs, var_obs);
    this->obs->block_size = output_states.block_size;
    this->obs->size = mu_obs.size();
    this->obs->actual_size = mu_obs.size() / output_states.block_size;

    this->updater->update_output_delta_z_heteros(output_states, *this->obs,
                                                 delta_states);
}
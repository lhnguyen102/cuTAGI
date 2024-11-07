#include "../include/output_updater_cuda.cuh"

__global__ void update_delta_z_using_indices_cuda(
    float const *mu_a, float const *var_a, float const *jcb, float const *obs,
    float const *var_obs, int const *selected_idx, int n_obs, int n_enc,
    int size, float *delta_mu, float *delta_var)
/* Update output layer based on selected indices.
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0.0f;
    float tmp = 0.0f;
    int idx;
    if (col < size) {
        // minus 1 because the encoder index starts at 1
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
__global__ void update_delta_z_cuda(float const *mu_a, float const *var_a,
                                    float const *jcb, float const *obs,
                                    float const *var_obs, int size,
                                    float *delta_mu, float *delta_var) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0;
    float tmp = 0;
    if (col < size) {
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

__global__ void update_delta_z_cuda_heteros(float const *mu_a,
                                            float const *var_a,
                                            float const *jcb, float const *obs,
                                            int size, float *delta_mu,
                                            float *delta_var) {
    /*
    Compute delta hidden states for output layer with learned heteroscedastic
    noise. This function receives a vector of observations and the twice
    output hidden states. Using AGVI, we can infere the posterior for
    observation noise v and use it to update the hidden states Z_out.

    Terminology:
    - V: Gaussian random variable describing the error variance sigma^2. N(0,
    sqrt(V))
    - V2: Square of the error (V^2)
    - V2_bar: Gaussian random variable describing the expected value of V2
    (mu_V2)
    - V2_bar_tilde: Gaussian random variable describing V2 after passing through
    an exponential activation function to restrict values to the positive domain

    */
    const float zero_pad = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Output layer will have twice the size of the common one because one is
    // representing the mean and the other the variance
    int obs_col = col * 2;

    if (col < size) {
        // mean of the Gaussian distribution for the output
        float var_a_col = var_a[obs_col];
        float mu_a_col = mu_a[obs_col];
        float jcb_col = jcb[obs_col];

        // V2_bar_tilde
        float mu_v2_bar_tilde = mu_a[obs_col + 1];
        float var_v2_bar_tilde = var_a[obs_col + 1];
        float cov_v2_bar_tilde = jcb[obs_col + 1];

        // Compute the prior predictive PDF for v2
        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 =
            3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde;
        float cov_y_v = mu_v2;

        // Variance of the output
        float var_sum = var_a_col + mu_v2;

        // Compute updating quantities for the mean of the output
        float tmp = jcb_col / var_sum;
        if (std::isinf(tmp) || std::isnan(tmp)) {
            delta_mu[obs_col] = zero_pad;
            delta_var[obs_col] = zero_pad;
        } else {
            float obs_diff = obs[col] - mu_a_col;
            delta_mu[obs_col] = tmp * obs_diff;
            delta_var[obs_col] = -tmp * jcb_col;
        }

        // Compute the posterior mean and variance for V
        float mu_v_post = cov_y_v / var_sum * (obs[col] - mu_a_col);
        float var_v_post = mu_v2 - cov_y_v / var_sum * cov_y_v;

        // Compute the posterior mean and variance for V2
        float mu_v2_post = mu_v_post * mu_v_post + var_v_post;
        float var_v2_post = 2.0f * var_v_post * var_v_post +
                            4.0f * var_v_post * mu_v_post * mu_v_post;

        // Compute the posterior mean and variance for V2_bar_tilde
        float tmp_ratio = var_v2_bar_tilde / var_v2;
        float mu_v2_bar_tilde_post =
            mu_v2_bar_tilde + tmp_ratio * (mu_v2_post - mu_v2);
        float var_v2_bar_tilde_post =
            var_v2_bar_tilde + tmp_ratio * tmp_ratio * (var_v2_post - var_v2);

        // Compute update for V2_bar
        float jv = cov_v2_bar_tilde / var_v2_bar_tilde;
        delta_mu[obs_col + 1] = jv * (mu_v2_bar_tilde_post - mu_v2_bar_tilde);
        delta_var[obs_col + 1] =
            jv * jv * (var_v2_bar_tilde_post - var_v2_bar_tilde);
    }
}

OutputUpdaterCuda::OutputUpdaterCuda() {}

void OutputUpdaterCuda::set_num_cuda_threads(unsigned int num_threads) {
    this->num_cuda_threads = num_threads;
}

void OutputUpdaterCuda::update_output_delta_z(BaseHiddenStates &output_states,
                                              BaseObservation &obs,
                                              BaseDeltaStates &delta_states)
/*
 */
{
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->size;
    int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    update_delta_z_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
        num_states, cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);
}

void OutputUpdaterCuda::update_selected_output_delta_z(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states)
/*
 */
{
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->idx_size;
    int num_enc = cu_obs->idx_size / cu_obs->block_size;
    int num_outputs = cu_output_states->actual_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    update_delta_z_using_indices_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, cu_obs->d_var_obs,
        cu_obs->d_selected_idx, num_outputs, num_enc, num_states,
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);
}

void OutputUpdaterCuda::update_output_delta_z_heteros(
    BaseHiddenStates &output_states, BaseObservation &obs,
    BaseDeltaStates &delta_states)
/*
 */
{
    // Cast to cuda object
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    ObservationCuda *cu_obs = dynamic_cast<ObservationCuda *>(&obs);
    DeltaStateCuda *cu_delta_states =
        dynamic_cast<DeltaStateCuda *>(&delta_states);

    if (cu_obs->d_mu_obs == nullptr) {
        cu_obs->allocate_memory();
    }

    cu_obs->to_device();

    // Reset delta to zero
    cu_delta_states->reset_zeros();

    // Kernel
    int num_states = cu_obs->size;
    int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    update_delta_z_cuda_heteros<<<blocks, this->num_cuda_threads>>>(
        cu_output_states->d_mu_a, cu_output_states->d_var_a,
        cu_output_states->d_jcb, cu_obs->d_mu_obs, num_states,
        cu_delta_states->d_delta_mu, cu_delta_states->d_delta_var);
}
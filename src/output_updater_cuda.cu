///////////////////////////////////////////////////////////////////////////////
// File:         output_updater_cuda.cu
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 27, 2023
// Updated:      August 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
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
    const float zero_pad = 0.0f;

    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col % 2 == 0) {
        // Need to pass through exponential activantion function in the update
        // class in CUDA. To do: just use the exponential activation function
        // from the activation class.
        float mu_V2_bar_tilde = expf(mu_a[col + 1] + 0.5f * var_a[col + 1]);
        float var_V2_bar_tilde = expf(2.0f * mu_a[col + 1] + var_a[col + 1]) *
                                 (expf(var_a[col + 1]) - 1.0f);
        float cov_V2_bar_tilde = var_a[col + 1] * mu_V2_bar_tilde;

        // float mu_V2_bar_tilde = mu_a[col + 1];
        // float var_V2_bar_tilde = var_a[col + 1];
        // float cov_V2_bar_tilde = jcb[col + 1];

        float cov_y_V2 = mu_V2_bar_tilde;
        float mu_V2 = mu_V2_bar_tilde;
        float var_V2 =
            3.0f * var_V2_bar_tilde + 2.0f * mu_V2_bar_tilde * mu_V2_bar_tilde;

        float var_a_col = var_a[col];
        float mu_a_col = mu_a[col];
        float jcb_col = jcb[col];
        float var_sum = var_a_col + mu_V2;

        float tmp = jcb_col / var_sum;
        if (std::isinf(tmp) || std::isnan(tmp)) {
            delta_mu[col] = zero_pad;
            delta_var[col] = zero_pad;
        } else {
            float obs_diff = obs[col / 2] - mu_a_col;
            delta_mu[col] = tmp * obs_diff;
            delta_var[col] = -tmp * jcb_col;
        }

        float mu_V_pos = 0 + cov_y_V2 / var_sum * (obs[col / 2] - mu_a_col);
        float var_V_pos = mu_V2 - cov_y_V2 / var_sum * cov_y_V2;

        float mu_V2_pos = mu_V_pos * mu_V_pos + var_V_pos;
        float var_V2_pos = 2.0f * var_V_pos * var_V_pos +
                           4.0f * var_V_pos * mu_V_pos * mu_V_pos;

        float k = var_V2_bar_tilde / var_V2;
        float mu_V2_bar_tilde_pos = mu_V2_bar_tilde + k * (mu_V2_pos - mu_V2);
        float var_V2_bar_tilde_pos =
            var_V2_bar_tilde + k * k * (var_V2_pos - var_V2);

        float Jv = cov_V2_bar_tilde / var_V2_bar_tilde;
        delta_mu[col + 1] = Jv * (mu_V2_bar_tilde_pos - mu_V2_bar_tilde);
        delta_var[col + 1] =
            Jv * Jv * (var_V2_bar_tilde_pos - var_V2_bar_tilde);
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
///////////////////////////////////////////////////////////////////////////////
// File:         linear_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      August 06, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/slinear_layer.h"

#include "../include/common.h"
#include "../include/linear_layer.h"

////////////////////////////////////////////////////////////////////////////////
// SLinear: Linear layer with smoother for LSTM
////////////////////////////////////////////////////////////////////////////////

std::string SLinear::get_layer_info() const
/*
 */
{
    return "SLinear(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string SLinear::get_layer_name() const
/*
 */
{
    return "SLinear";
}

LayerType SLinear::get_layer_type() const
/*
 */
{
    return LayerType::SLinear;
}

void linear_update_hidden_states_worker(int time_step,
                                        std::vector<float> &mu_a_prior,
                                        std::vector<float> &var_a_prior,
                                        std::vector<float> &delta_mu,
                                        std::vector<float> &delta_var,
                                        std::vector<float> &mu_a_post,
                                        std::vector<float> &var_a_post)
/*
 */
{
    mu_a_post[time_step] =
        mu_a_prior[time_step] + delta_mu[0] * var_a_prior[time_step];
    var_a_post[time_step] =
        (1.0f + delta_var[0] * var_a_prior[time_step]) * var_a_prior[time_step];
}

void slinear_cov_zo_smoother(
    int ni, int no, int time_step, std::vector<float> &mu_w,
    std::vector<float> &var_w, std::vector<float> &var_b,
    std::vector<float> &mu_h_prev, std::vector<float> &mu_h_prior,
    std::vector<float> &cov_hh, std::vector<float> &cov_zo)
/*
 */
{
    float C_zo_zo = 0;
    int m;

    for (int t = 0; t < ni; t++) {
        for (int j = 0; j < ni; j++) {
            // C_zo_zo : cov(z^{O}_{t-1}, z^{O}_{t})
            m = t * ni + j;
            if (t != j) {
                C_zo_zo += cov_hh[m] * mu_w[t] * mu_w[j];
            } else {
                C_zo_zo +=
                    var_w[t] * (cov_hh[m] + mu_h_prev[t] * mu_h_prior[j]) +
                    cov_hh[m] * mu_w[t] * mu_w[j];
            }
        }
    }
    C_zo_zo = C_zo_zo + var_b[0];
    cov_zo[time_step] = C_zo_zo;
}

void slinear_smoother_z_ouput(int num_timestep, std::vector<float> &cov,
                              std::vector<float> &mu_priors,
                              std::vector<float> &var_priors,
                              std::vector<float> &mu_posts,
                              std::vector<float> &var_posts,
                              std::vector<float> &mu_smooths,
                              std::vector<float> &var_smooths)
/*
 */
{
    for (int i = num_timestep - 2; i >= 0; i--) {
        float tmp = cov[i + 1] / var_priors[i + 1];
        mu_smooths[i] =
            mu_posts[i] + tmp * (mu_smooths[i + 1] - mu_priors[i + 1]);
        var_smooths[i] =
            var_posts[i] + tmp * (var_smooths[i + 1] - var_priors[i + 1]) * tmp;
    }
}

void SLinear::storing_states_for_training_smooth(
    SmoothingHiddenStates &input_states, SmoothingHiddenStates &output_states)
/*
 */
{
    int act_size = input_states.actual_size * input_states.block_size;
    if (this->bwd_states->size != act_size) {
        this->allocate_bwd_vector(act_size);
    }

    // Activation's jacobian and mean from the previous layer
    this->fill_bwd_vector(input_states);

    // Send a copy of activation's mean and variance to the output buffer
    // for the current layer.
    // TODO: consider to have only mu_a and var_a in struct HiddenStates
    this->fill_output_states(output_states);
}

void SLinear::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    SmoothingHiddenStates *smooth_input_states =
        dynamic_cast<SmoothingHiddenStates *>(&input_states);
    SmoothingHiddenStates *smooth_output_states =
        dynamic_cast<SmoothingHiddenStates *>(&output_states);

    // Initialization
    int batch_size = smooth_input_states->block_size;
    this->set_cap_factor_udapte(batch_size);
    // Initialize smoothing hidden states for SLinear
    if (this->smoothing_states.num_timesteps !=
        smooth_input_states->num_timesteps) {
        smoothing_states.set_num_states(smooth_input_states->num_timesteps);
    }

    // Forward pass
    if (this->num_threads > 1) {
        linear_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               smooth_input_states->mu_a,
                               smooth_input_states->var_a, this->input_size,
                               this->output_size, batch_size, this->bias,
                               this->num_threads, smooth_output_states->mu_a,
                               smooth_output_states->var_a);
    } else {
        int start_chunk = 0;
        int end_chunk = this->output_size * batch_size;
        linear_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                            smooth_input_states->mu_a,
                            smooth_input_states->var_a, start_chunk, end_chunk,
                            this->input_size, this->output_size, batch_size,
                            this->bias, smooth_output_states->mu_a,
                            smooth_output_states->var_a);
    }
    // Update number of actual states.
    smooth_output_states->width = this->out_width;
    smooth_output_states->height = this->out_height;
    smooth_output_states->depth = this->out_channels;
    smooth_output_states->block_size = batch_size;
    smooth_output_states->actual_size = this->output_size;

    // save z_output prior for smoothing
    smoothing_states.mu_zo_priors[this->time_step] =
        smooth_output_states->mu_a[0];
    smoothing_states.var_zo_priors[this->time_step] =
        smooth_output_states->var_a[0];

    // save cov_zo for smoother
    slinear_cov_zo_smoother(
        this->input_size, this->output_size, this->time_step, this->mu_w,
        this->var_w, this->var_b, smooth_input_states->mu_h_prev,
        smooth_input_states->mu_a, smooth_input_states->cov_hh,
        smoothing_states.cov_zo);

    if (this->training) {
        this->storing_states_for_training_smooth(*smooth_input_states,
                                                 *smooth_output_states);
    }

    // // Increase index for time step
    ++this->time_step;
}

void SLinear::backward(BaseDeltaStates &input_delta_states,
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
        linear_update_hidden_states_worker(
            this->time_step, smoothing_states.mu_zo_priors,
            smoothing_states.var_zo_priors, input_delta_states.delta_mu,
            input_delta_states.delta_var, smoothing_states.mu_zo_posts,
            smoothing_states.var_zo_posts);
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

void SLinear::smoother()
/*
 */
{
    smoothing_states.mu_zo_smooths.back() = smoothing_states.mu_zo_posts.back();
    smoothing_states.var_zo_smooths.back() =
        smoothing_states.var_zo_posts.back();

    slinear_smoother_z_ouput(
        this->smoothing_states.num_timesteps, smoothing_states.cov_zo,
        smoothing_states.mu_zo_priors, smoothing_states.var_zo_priors,
        smoothing_states.mu_zo_posts, smoothing_states.var_zo_posts,
        smoothing_states.mu_zo_smooths, smoothing_states.var_zo_smooths);

    // Clear variables for next epoch
    this->time_step = 0;
    // this->smoothing_states.reset_zeros();
}

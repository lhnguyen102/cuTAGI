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

void linear_update_hidden_states_worker(std::vector<float> &mu_a_prior,
                                        std::vector<float> &var_a_prior,
                                        std::vector<float> &delta_mu,
                                        std::vector<float> &delta_var,
                                        std::vector<float> &mu_a_post,
                                        std::vector<float> &var_a_post)
/*
 */
{
    float mu_tmp = mu_a_prior.back();
    float var_tmp = var_a_prior.back();
    mu_tmp = mu_tmp + delta_mu[0] * var_tmp;
    var_tmp = (1.0f + delta_var[0] * var_tmp) * var_tmp;
    mu_a_post.push_back(mu_tmp);
    var_a_post.push_back(var_tmp);
}

void slinear_cov_zo_smoother(int ni, int no, std::vector<float> &mu_w,
                             std::vector<float> &var_w,
                             std::vector<float> &var_b,
                             std::vector<float> &mu_h_prev,
                             std::vector<float> &mu_h_prior,
                             std::vector<float> &cov_hh,
                             std::vector<float> &cov_zo)
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
    cov_zo.push_back(C_zo_zo);
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

void SLinear::forward(BaseHiddenStates &input_states,
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

    // save z_output prior for smoothing
    slinear_states.mu_zo_priors.push_back(output_states.mu_a[0]);
    slinear_states.var_zo_priors.push_back(output_states.var_a[0]);

    // save cov_zo for smoother
    slinear_cov_zo_smoother(this->input_size, this->output_size, this->mu_w,
                            this->var_w, this->var_b,
                            temp_states.slinear.mu_h_prev, input_states.mu_a,
                            temp_states.slinear.cov_hh, slinear_states.cov_zo);

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
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
            slinear_states.mu_zo_priors, slinear_states.var_zo_priors,
            input_delta_states.delta_mu, input_delta_states.delta_var,
            slinear_states.mu_zo_posts, slinear_states.var_zo_posts);
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

void SLinear::smoother(BaseTempStates &temp_states)
/*
 */
{
    int num_timestep = slinear_states.cov_zo.size();

    // Initialize the last time step for smoothing
    temp_states.slinear.mu_zo_smooths.resize(num_timestep);
    temp_states.slinear.var_zo_smooths.resize(num_timestep);
    temp_states.slinear.mu_zo_smooths.back() =
        slinear_states.mu_zo_posts.back();
    temp_states.slinear.var_zo_smooths.back() =
        slinear_states.var_zo_posts.back();

    slinear_smoother_z_ouput(
        num_timestep, slinear_states.cov_zo, slinear_states.mu_zo_priors,
        slinear_states.var_zo_priors, slinear_states.mu_zo_posts,
        slinear_states.var_zo_posts, temp_states.slinear.mu_zo_smooths,
        temp_states.slinear.var_zo_smooths);

    // Clear variables for next epoch
    this->slinear_states.cov_zo.clear();
    this->slinear_states.mu_zo_priors.clear();
    this->slinear_states.var_zo_priors.clear();
    this->slinear_states.mu_zo_posts.clear();
    this->slinear_states.var_zo_posts.clear();
}

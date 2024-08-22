///////////////////////////////////////////////////////////////////////////////
// File:         lstm_layer.cpp
// Description:  Header file for Long-Short Term Memory (LSTM) forward pass
//               in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 22, 2024
// Updated:      March 27, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "slstm_layer.h"

#include <cmath>
#include <thread>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/lstm_layer.h"
#include "../include/param_init.h"

////////////////////////////////////////////////////////////////////////////////
// SLSTM: LSTM layer with smoother
////////////////////////////////////////////////////////////////////////////////

std::string SLSTM::get_layer_info() const
/*
 */
{
    return "SLSTM(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string SLSTM::get_layer_name() const
/*
 */
{
    return "SLSTM";
}

LayerType SLSTM::get_layer_type() const
/*
 */
{
    return LayerType::SLSTM;
}

void lstm_cov_cell_states_smoother(std::vector<float> &var_c_prev,
                                   std::vector<float> &mu_f_ga,
                                   std::vector<std::vector<float>> &cov_cc)
/*
 */
{
    std::vector<float> Cc_c(var_c_prev.size());
    for (int i = 0; i < var_c_prev.size(); i++) {
        // cov(c_{t-1},c_{t})
        Cc_c[i] = var_c_prev[i] * mu_f_ga[i];
    }
    cov_cc.push_back(Cc_c);
}

void lstm_cov_hidden_cell_states_smoother(
    std::vector<float> &var_c_prior, std::vector<float> &mu_o_ga,
    std::vector<float> &jcb_ca, std::vector<std::vector<float>> &cov_hc)
/*
 */
{
    std::vector<float> Ch_c(var_c_prior.size());
    for (int i = 0; i < var_c_prior.size(); i++) {
        // cov(h_{t},c_{t})
        Ch_c[i] = var_c_prior[i] * jcb_ca[i] * mu_o_ga[i];
    }
    cov_hc.push_back(Ch_c);
}

void lstm_cov_hidden_states_smoother(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mu_h_prior, std::vector<float> &mu_h_prev,
    std::vector<float> &var_h_prev, std::vector<float> &mc_prev,
    std::vector<float> &mca, std::vector<float> &Jca, int w_pos_f, int w_pos_i,
    int w_pos_c, int w_pos_o, int no, int ni, int start_idx, int end_idx,
    std::vector<std::vector<float>> &cov_hh)
/*
 */
{
    std::vector<float> Ch_h(end_idx * end_idx, 0);
    float Czz_f, Czz_i, Czz_c, Czz_o;
    int m;

    for (int t = start_idx; t < end_idx; t++) {
        for (int j = 0; j < no; j++) {
            // Forget gate
            Czz_f = var_h_prev[t] * Jca[j] * mo_ga[j] * Jf_ga[j] *
                    mw[(ni + no) * j + t + ni + w_pos_f] * mc_prev[j];

            // Input gate
            Czz_i = var_h_prev[t] * Jca[j] * mo_ga[j] * Ji_ga[j] *
                    mw[(ni + no) * j + t + ni + w_pos_i] * mc_ga[j];

            // Cell state gate
            Czz_c = var_h_prev[t] * Jca[j] * mo_ga[j] * Jc_ga[j] *
                    mw[(ni + no) * j + t + ni + w_pos_c] * mi_ga[j];

            // Output gate
            Czz_o = var_h_prev[t] * Jo_ga[j] *
                    mw[(ni + no) * j + t + ni + w_pos_o] * mca[j];

            // Updating quantities
            m = t * no + j;
            Ch_h[m] = Czz_f + Czz_i + Czz_c + Czz_o;
        }
    }
    cov_hh.push_back(Ch_h);
}

void flatten2DVector(const std::vector<std::vector<float>> &vec2D,
                     std::vector<float> &vec1D) {
    // Reserve space in vec1D for efficiency
    vec1D.reserve(vec2D.size() * vec2D[0].size());

    for (const auto &row : vec2D) {
        vec1D.insert(vec1D.end(), row.begin(), row.end());
    }
}

void lstm_smoother_cell_states(int num_timestep,
                               std::vector<std::vector<float>> &cov_cc,
                               std::vector<std::vector<float>> &mu_c_priors,
                               std::vector<std::vector<float>> &var_c_priors,
                               std::vector<std::vector<float>> &mu_c_posts,
                               std::vector<std::vector<float>> &var_c_posts,
                               std::vector<std::vector<float>> &mu_c_smooths,
                               std::vector<std::vector<float>> &var_c_smooths)
/*
 */
{
    int num_states = mu_c_priors[0].size();
    for (int i = num_timestep - 2; i >= 0; --i) {
        for (int j = 0; j < num_states; j++) {
            float tmp = cov_cc[i + 1][j] / var_c_priors[i + 1][j];
            mu_c_smooths[i][j] =
                mu_c_posts[i][j] +
                tmp * (mu_c_smooths[i + 1][j] - mu_c_priors[i + 1][j]);
            var_c_smooths[i][j] =
                var_c_posts[i][j] +
                tmp * (var_c_smooths[i + 1][j] - var_c_priors[i + 1][j]) * tmp;
        }
    }

    // int n = mu_c_priors[0].size();
    // int current, next;
    // std::vector<float> cov_cc_1D, mu_c_priors_1D, var_c_priors_1D,
    //     mu_c_posts_1D, var_c_posts_1D, mu_c_smooths_1D, var_c_smooths_1D;
    // flatten2DVector(cov_cc, cov_cc_1D);
    // flatten2DVector(mu_c_priors, mu_c_priors_1D);
    // flatten2DVector(var_c_priors, var_c_priors_1D);
    // flatten2DVector(mu_c_posts, mu_c_posts_1D);
    // flatten2DVector(var_c_posts, var_c_posts_1D);
    // flatten2DVector(mu_c_smooths, mu_c_smooths_1D);
    // flatten2DVector(var_c_smooths, var_c_smooths_1D);

    // for (int i = num_timestep - 2; i >= 0; --i) {
    //     for (int j = n - 1; j >= 0; --j) {
    //         current = i * n + j;
    //         next = (i + 1) * n + j;
    //         float tmp = cov_cc_1D[current] / var_c_priors_1D[next];

    //         mu_c_smooths_1D[current] =
    //             mu_c_posts_1D[current] +
    //             tmp * (mu_c_smooths_1D[next] - mu_c_priors_1D[next]);

    //         var_c_smooths_1D[current] =
    //             var_c_posts_1D[current] +
    //             tmp * (var_c_smooths_1D[next] - var_c_priors_1D[next]) * tmp;
    //         std::cout << var_c_smooths_1D[current] << ". ";
    //     }
    // }
}

void lstm_smoother_hidden_states(int num_timestep,
                                 std::vector<std::vector<float>> &cov_hc,
                                 std::vector<std::vector<float>> &mu_c_priors,
                                 std::vector<std::vector<float>> &var_c_priors,
                                 std::vector<std::vector<float>> &mu_h_posts,
                                 std::vector<std::vector<float>> &var_h_posts,
                                 std::vector<std::vector<float>> &mu_c_smooths,
                                 std::vector<std::vector<float>> &var_c_smooths,
                                 std::vector<std::vector<float>> &mu_h_smooths,
                                 std::vector<std::vector<float>> &var_h_smooths)
/*
 */
{
    int num_states = mu_c_priors[0].size();
    for (int i = num_timestep - 2; i >= 0; i--) {
        for (int j = 0; j < num_states; j++) {
            float tmp = cov_hc[i + 1][j] / var_c_priors[i + 1][j];
            mu_h_smooths[i][j] =
                mu_h_posts[i][j] +
                tmp * (mu_c_smooths[i + 1][j] - mu_c_priors[i + 1][j]);
            var_h_smooths[i][j] =
                var_h_posts[i][j] +
                tmp * (var_c_smooths[i + 1][j] - var_c_priors[i + 1][j]) * tmp;
        }
    }
}

void SLSTM::forward(BaseHiddenStates &input_states,
                    BaseHiddenStates &output_states,
                    BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_states.set_num_states(
            batch_size * this->seq_len * this->output_size,
            batch_size * this->seq_len * this->input_size);
    }
    // Update number of actual states.
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size * this->seq_len;

    // TODO: This is not efficient for memory and performance. Update the
    // previous states
    if (this->seq_len == 1 && batch_size == 1) {
        lstm_to_prev_states(this->lstm_states.mu_h_prior,
                            this->lstm_states.mu_h_prior.size(),
                            this->lstm_states.mu_h_prev);
        lstm_to_prev_states(this->lstm_states.var_h_prior,
                            this->lstm_states.var_h_prior.size(),
                            this->lstm_states.var_h_prev);
        lstm_to_prev_states(this->lstm_states.mu_c_prior,
                            this->lstm_states.mu_c_prior.size(),
                            this->lstm_states.mu_c_prev);
        lstm_to_prev_states(this->lstm_states.var_c_prior,
                            this->lstm_states.var_c_prior.size(),
                            this->lstm_states.var_c_prev);
    }

    this->prepare_input(input_states);
    this->forget_gate(batch_size);
    this->input_gate(batch_size);
    this->cell_state_gate(batch_size);
    this->output_gate(batch_size);

    int end_chunk = this->output_size * batch_size * this->seq_len;

    if (this->num_threads > 1) {
        lstm_cov_input_cell_states_mp(
            lstm_states.var_ha, this->mu_w, lstm_states.jcb_i_ga,
            lstm_states.jcb_c_ga, this->w_pos_i, this->w_pos_c,
            this->input_size, this->output_size, this->seq_len, batch_size,
            this->num_threads, lstm_states.cov_i_c);

        lstm_cell_state_mean_var_mp(
            lstm_states.mu_f_ga, lstm_states.var_f_ga, lstm_states.mu_i_ga,
            lstm_states.var_i_ga, lstm_states.mu_c_ga, lstm_states.var_c_ga,
            lstm_states.mu_c_prev, lstm_states.var_c_prev, lstm_states.cov_i_c,
            this->output_size, this->seq_len, batch_size, this->num_threads,
            lstm_states.mu_c, lstm_states.var_c);

        tanh_mean_var_mp(lstm_states.mu_c, lstm_states.var_c, end_chunk,
                         this->num_threads, lstm_states.mu_ca,
                         lstm_states.jcb_ca, lstm_states.var_ca);

        lstm_cov_output_tanh_cell_states_mp(
            this->mu_w, lstm_states.var_ha, lstm_states.mu_c_prev,
            lstm_states.jcb_ca, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
            lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
            lstm_states.jcb_o_ga, this->w_pos_f, this->w_pos_i, this->w_pos_c,
            this->w_pos_o, this->input_size, this->output_size, this->seq_len,
            batch_size, this->num_threads, lstm_states.cov_o_tanh_c);

        lstm_hidden_state_mean_var_mp(
            lstm_states.mu_o_ga, lstm_states.var_o_ga, lstm_states.mu_ca,
            lstm_states.var_ca, lstm_states.cov_o_tanh_c, this->output_size,
            this->seq_len, batch_size, this->num_threads, output_states.mu_a,
            output_states.var_a);

    } else {
        lstm_cov_input_cell_states(
            lstm_states.var_ha, this->mu_w, lstm_states.jcb_i_ga,
            lstm_states.jcb_c_ga, this->w_pos_i, this->w_pos_c,
            this->input_size, this->output_size, this->seq_len, batch_size,
            lstm_states.cov_i_c);

        lstm_cell_state_mean_var(
            lstm_states.mu_f_ga, lstm_states.var_f_ga, lstm_states.mu_i_ga,
            lstm_states.var_i_ga, lstm_states.mu_c_ga, lstm_states.var_c_ga,
            lstm_states.mu_c_prev, lstm_states.var_c_prev, lstm_states.cov_i_c,
            this->output_size, this->seq_len, batch_size, lstm_states.mu_c,
            lstm_states.var_c);

        tanh_mean_var(lstm_states.mu_c, lstm_states.var_c, 0, end_chunk,
                      lstm_states.mu_ca, lstm_states.jcb_ca,
                      lstm_states.var_ca);

        lstm_cov_output_tanh_cell_states(
            this->mu_w, lstm_states.var_ha, lstm_states.mu_c_prev,
            lstm_states.jcb_ca, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
            lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
            lstm_states.jcb_o_ga, this->w_pos_f, this->w_pos_i, this->w_pos_c,
            this->w_pos_o, this->input_size, this->output_size, this->seq_len,
            batch_size, lstm_states.cov_o_tanh_c);

        lstm_hidden_state_mean_var(
            lstm_states.mu_o_ga, lstm_states.var_o_ga, lstm_states.mu_ca,
            lstm_states.var_ca, lstm_states.cov_o_tanh_c, this->output_size,
            this->seq_len, batch_size, output_states.mu_a, output_states.var_a);
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
    // Save the previous states
    if (this->seq_len == 1 && batch_size == 1) {
        lstm_to_prev_states(output_states.mu_a,
                            this->lstm_states.mu_h_prior.size(),
                            this->lstm_states.mu_h_prior);
        lstm_to_prev_states(output_states.var_a,
                            this->lstm_states.var_h_prior.size(),
                            this->lstm_states.var_h_prior);
        lstm_to_prev_states(this->lstm_states.mu_c,
                            this->lstm_states.mu_c_prior.size(),
                            this->lstm_states.mu_c_prior);
        lstm_to_prev_states(this->lstm_states.var_c,
                            this->lstm_states.var_c_prior.size(),
                            this->lstm_states.var_c_prior);

        // Save priors for smoothing
        this->lstm_states.mu_h_priors.push_back(this->lstm_states.mu_h_prior);
        this->lstm_states.var_h_priors.push_back(this->lstm_states.var_h_prior);
        this->lstm_states.mu_c_priors.push_back(this->lstm_states.mu_c_prior);
        this->lstm_states.var_c_priors.push_back(this->lstm_states.var_c_prior);

        // Save the cross-covariances for smoothing
        if (this->seq_len == 1 && batch_size == 1) {
            lstm_cov_cell_states_smoother(lstm_states.var_c_prev,
                                          lstm_states.mu_f_ga,
                                          lstm_states.cov_cc);
            lstm_cov_hidden_cell_states_smoother(
                lstm_states.var_c_prior, lstm_states.mu_o_ga,
                lstm_states.jcb_ca, lstm_states.cov_hc);

            int end_chunk_ = batch_size * this->seq_len * this->output_size;
            lstm_cov_hidden_states_smoother(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prior, lstm_states.mu_h_prev,
                lstm_states.var_h_prev, lstm_states.mu_c_prev,
                lstm_states.mu_ca, lstm_states.jcb_ca, this->w_pos_f,
                this->w_pos_i, this->w_pos_c, this->w_pos_o, this->output_size,
                this->input_size, 0, end_chunk_, lstm_states.cov_hh);

            // save temporary variables for smoothing z_output in slinear layer
            temp_states.slinear.mu_h_prev = lstm_states.mu_h_prev;
            temp_states.slinear.cov_hh = lstm_states.cov_hh.back();
        }
    }
}

void SLSTM::backward(BaseDeltaStates &input_delta_states,
                     BaseDeltaStates &output_delta_states,
                     BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    int batch_size = input_delta_states.block_size;
    if (state_udapte) {
        if (this->num_threads > 1) {
            lstm_delta_mean_var_z_mp(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, batch_size,
                this->num_threads, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        } else {
            int end_chunk = batch_size * this->seq_len * this->input_size;
            lstm_delta_mean_var_z_worker(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, 0,
                end_chunk, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        }
    }

    if (param_update) {
        if (this->num_threads > 1) {
            lstm_delta_mean_var_w_mp(
                this->var_w, lstm_states.mu_ha, lstm_states.jcb_f_ga,
                lstm_states.mu_i_ga, lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                lstm_states.jcb_c_ga, lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, batch_size,
                this->num_threads, this->delta_mu_w, this->delta_var_w);

            if (this->bias) {
                lstm_delta_mean_var_b_mp(
                    this->var_b, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                    lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                    lstm_states.jcb_c_ga, lstm_states.mu_o_ga,
                    lstm_states.jcb_o_ga, lstm_states.mu_c_prev,
                    lstm_states.mu_ca, lstm_states.jcb_ca,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->b_pos_f, this->b_pos_i, this->b_pos_c, this->b_pos_o,
                    this->output_size, this->seq_len, batch_size,
                    this->num_threads, this->delta_mu_b, this->delta_var_b);
            }
        } else {
            int end_chunk_w =
                (this->input_size + this->output_size) * this->output_size;
            lstm_delta_mean_var_w_worker(
                this->var_w, lstm_states.mu_ha, lstm_states.jcb_f_ga,
                lstm_states.mu_i_ga, lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                lstm_states.jcb_c_ga, lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, batch_size,
                0, end_chunk_w, this->delta_mu_w, this->delta_var_w);

            if (this->bias) {
                lstm_delta_mean_var_b_worker(
                    this->var_b, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                    lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                    lstm_states.jcb_c_ga, lstm_states.mu_o_ga,
                    lstm_states.jcb_o_ga, lstm_states.mu_c_prev,
                    lstm_states.mu_ca, lstm_states.jcb_ca,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->b_pos_f, this->b_pos_i, this->b_pos_c, this->b_pos_o,
                    this->output_size, this->seq_len, batch_size, 0,
                    this->output_size, this->delta_mu_b, this->delta_var_b);
            }
        }
    }
    if (this->seq_len == 1 && batch_size == 1) {
        if (this->num_threads > 1) {
            lstm_update_prev_hidden_states_mp(
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->lstm_states.num_states, this->num_threads,
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior);
            lstm_update_prev_cell_states_mp(
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior,
                this->lstm_states.jcb_ca, this->lstm_states.mu_o_ga,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->lstm_states.num_states, this->num_threads,
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior);
        } else {
            lstm_update_prev_hidden_states_worker(
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior,
                input_delta_states.delta_mu, input_delta_states.delta_var, 0,
                this->lstm_states.num_states, this->lstm_states.mu_h_prior,
                this->lstm_states.var_h_prior);
            lstm_update_prev_cell_states_worker(
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior,
                this->lstm_states.jcb_ca, this->lstm_states.mu_o_ga,
                input_delta_states.delta_mu, input_delta_states.delta_var, 0,
                this->lstm_states.num_states, this->lstm_states.mu_c_prior,
                this->lstm_states.var_c_prior);
        }

        // save posteriors for smoothing
        this->lstm_states.mu_h_posts.push_back(this->lstm_states.mu_h_prior);
        this->lstm_states.var_h_posts.push_back(this->lstm_states.var_h_prior);
        this->lstm_states.mu_c_posts.push_back(this->lstm_states.mu_c_prior);
        this->lstm_states.var_c_posts.push_back(this->lstm_states.var_c_prior);
    }
}

void SLSTM::smoother(BaseTempStates &temp_states)
/*
 */
{
    int num_timestep = lstm_states.cov_cc.size();
    int num_hidden_states = lstm_states.cov_cc[0].size();

    lstm_states.mu_c_smooths.resize(num_timestep);
    lstm_states.var_c_smooths.resize(num_timestep);
    lstm_states.mu_h_smooths.resize(num_timestep);
    lstm_states.var_h_smooths.resize(num_timestep);

    for (auto &val : lstm_states.mu_c_smooths) {
        val.resize(num_hidden_states, 0.0f);
    }
    for (auto &val : lstm_states.var_c_smooths) {
        val.resize(num_hidden_states, 0.0f);
    }
    for (auto &val : lstm_states.mu_h_smooths) {
        val.resize(num_hidden_states, 0.0f);
    }
    for (auto &val : lstm_states.var_h_smooths) {
        val.resize(num_hidden_states, 0.0f);
    }

    // Initialize the last time step for smoothing
    lstm_states.mu_c_smooths.back() = lstm_states.mu_c_posts.back();
    lstm_states.var_c_smooths.back() = lstm_states.var_c_posts.back();
    lstm_states.mu_h_smooths.back() = lstm_states.mu_h_posts.back();
    lstm_states.var_h_smooths.back() = lstm_states.var_h_posts.back();

    lstm_smoother_cell_states(num_timestep, lstm_states.cov_cc,
                              lstm_states.mu_c_priors, lstm_states.var_c_priors,
                              lstm_states.mu_c_posts, lstm_states.var_c_posts,
                              lstm_states.mu_c_smooths,
                              lstm_states.var_c_smooths);

    lstm_smoother_hidden_states(
        num_timestep, lstm_states.cov_hc, lstm_states.mu_c_priors,
        lstm_states.var_c_priors, lstm_states.mu_h_posts,
        lstm_states.var_h_posts, lstm_states.mu_c_smooths,
        lstm_states.var_c_smooths, lstm_states.mu_h_smooths,
        lstm_states.var_h_smooths);

    // transfer h and c to the first time step of the next epoch
    // this->lstm_states.mu_h_prior = this->lstm_states.mu_h_smooths[0];
    // this->lstm_states.var_h_prior =
    // this->lstm_states.var_h_smooths[0]; this->lstm_states.mu_c_prior
    // = this->lstm_states.mu_c_smooths[0];
    // this->lstm_states.var_c_prior =
    // this->lstm_states.var_c_smooths[0];

    // Clear variables for next epoch
    this->lstm_states.mu_h_priors.clear();
    this->lstm_states.var_h_priors.clear();
    this->lstm_states.mu_c_priors.clear();
    this->lstm_states.var_c_priors.clear();
    this->lstm_states.mu_h_posts.clear();
    this->lstm_states.var_h_posts.clear();
    this->lstm_states.mu_c_posts.clear();
    this->lstm_states.var_c_posts.clear();
    this->lstm_states.mu_h_smooths.clear();
    this->lstm_states.var_h_smooths.clear();
    this->lstm_states.mu_c_smooths.clear();
    this->lstm_states.var_c_smooths.clear();
    this->lstm_states.cov_cc.clear();
    this->lstm_states.cov_hc.clear();
    this->lstm_states.cov_hh.clear();
    this->lstm_states.reset_zeros();
}

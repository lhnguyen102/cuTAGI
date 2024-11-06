#include "slstm_layer.h"

#include <cmath>
#include <thread>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/custom_logger.h"
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

void save_cov_cell_states_smoother(int time_step, int num_states,
                                   std::vector<float> &var_c_prev,
                                   std::vector<float> &mu_f_ga,
                                   std::vector<float> &cov_cc)
/*
 */
{
    for (int i = 0; i < num_states; i++) {
        // cov(c_{t-1},c_{t})
        cov_cc[time_step * num_states + i] = var_c_prev[i] * mu_f_ga[i];
    }
}

void save_cov_hidden_cell_states_smoother(int time_step, int num_states,
                                          std::vector<float> &var_c_prior,
                                          std::vector<float> &mu_o_ga,
                                          std::vector<float> &jcb_ca,
                                          std::vector<float> &cov_hc)
/*
 */
{
    for (int i = 0; i < num_states; i++) {
        // cov(h_{t},c_{t})
        cov_hc[time_step * num_states + i] =
            var_c_prior[i] * jcb_ca[i] * mu_o_ga[i];
    }
}

void save_cov_hidden_states_smoother(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &var_h_prev, std::vector<float> &mc_prev,
    std::vector<float> &mca, std::vector<float> &Jca, int w_pos_f, int w_pos_i,
    int w_pos_c, int w_pos_o, int no, int ni, int start_idx, int end_idx,
    std::vector<float> &cov_hh)
/*
 */
{
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
            cov_hh[m] = Czz_f + Czz_i + Czz_c + Czz_o;
        }
    }
}

void smooth_cell_states(
    int num_timestep, int num_states, std::vector<float> &cov_cc,
    std::vector<float> &mu_c_priors, std::vector<float> &var_c_priors,
    std::vector<float> &mu_c_posts, std::vector<float> &var_c_posts,
    std::vector<float> &mu_c_smooths, std::vector<float> &var_c_smooths)
/*
 */
{
    int current, next;
    for (int i = num_timestep - 2; i >= 0; --i) {
        for (int j = num_states - 1; j >= 0; --j) {
            current = i * num_states + j;
            next = (i + 1) * num_states + j;
            float tmp = cov_cc[next] / var_c_priors[next];

            mu_c_smooths[current] =
                mu_c_posts[current] +
                tmp * (mu_c_smooths[next] - mu_c_priors[next]);

            var_c_smooths[current] =
                var_c_posts[current] +
                tmp * (var_c_smooths[next] - var_c_priors[next]) * tmp;
        }
    }
}

void smooth_hidden_states(
    int num_timestep, int num_states, std::vector<float> &cov_hc,
    std::vector<float> &mu_c_priors, std::vector<float> &var_c_priors,
    std::vector<float> &mu_c_smooths, std::vector<float> &var_c_smooths,
    std::vector<float> &mu_h_posts, std::vector<float> &var_h_posts,
    std::vector<float> &mu_h_smooths, std::vector<float> &var_h_smooths)
/*
 */
{
    int current, next;
    for (int i = num_timestep - 2; i >= 0; --i) {
        for (int j = num_states - 1; j >= 0; --j) {
            current = i * num_states + j;
            next = (i + 1) * num_states + j;
            float tmp = cov_hc[next] / var_c_priors[next];
            mu_h_smooths[current] =
                mu_h_posts[current] +
                tmp * (mu_c_smooths[next] - mu_c_priors[next]);
            var_h_smooths[current] =
                var_h_posts[current] +
                tmp * (var_c_smooths[next] - var_c_priors[next]) * tmp;
        }
    }
}

void save_priors_smoother(int time_step, int num_states,
                          BaseLSTMStates &lstm_states,
                          SmoothSLSTM &smooth_states) {
    // Save priors for smoothing
    for (int i = 0; i < num_states; i++) {
        smooth_states.mu_h_priors[time_step * num_states + i] =
            lstm_states.mu_h_prior[i];
        smooth_states.var_h_priors[time_step * num_states + i] =
            lstm_states.var_h_prior[i];
        smooth_states.mu_c_priors[time_step * num_states + i] =
            lstm_states.mu_c_prior[i];
        smooth_states.var_c_priors[time_step * num_states + i] =
            lstm_states.var_c_prior[i];
    }
}

void save_posteriors_smoother(int time_step, int num_states,
                              BaseLSTMStates &lstm_states,
                              SmoothSLSTM &smooth_states) {
    // Save priors for smoothing
    for (int i = 0; i < num_states; i++) {
        smooth_states.mu_h_posts[time_step * num_states + i] =
            lstm_states.mu_h_prior[i];
        smooth_states.var_h_posts[time_step * num_states + i] =
            lstm_states.var_h_prior[i];
        smooth_states.mu_c_posts[time_step * num_states + i] =
            lstm_states.mu_c_prior[i];
        smooth_states.var_c_posts[time_step * num_states + i] =
            lstm_states.var_c_prior[i];
    }
}

void SLSTM::prepare_input_smooth(SmoothingHiddenStates &input_state)
/*
 */
{
    int batch_size = input_state.block_size;
    if (this->num_threads > 1) {
        lstm_cat_activations_and_prev_states_mp(
            input_state.mu_a, lstm_states.mu_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, this->num_threads,
            lstm_states.mu_ha);
        lstm_cat_activations_and_prev_states_mp(
            input_state.var_a, lstm_states.var_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, this->num_threads,
            lstm_states.var_ha);
    } else {
        lstm_cat_activations_and_prev_states(
            input_state.mu_a, lstm_states.mu_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, lstm_states.mu_ha);
        lstm_cat_activations_and_prev_states(
            input_state.var_a, lstm_states.var_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, lstm_states.var_ha);
    }
}

void SLSTM::forward(BaseHiddenStates &input_states,
                    BaseHiddenStates &output_states,
                    BaseTempStates &temp_states)
/*
 */
{
    // Checkout input size
    if (this->input_size * this->seq_len != input_states.actual_size) {
        std::string message = "Input size mismatch: " +
                              std::to_string(this->input_size * this->seq_len) +
                              " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    // New poitner will point to the same memory location when casting
    SmoothingHiddenStates *smooth_input_states =
        dynamic_cast<SmoothingHiddenStates *>(&input_states);
    SmoothingHiddenStates *smooth_output_states =
        dynamic_cast<SmoothingHiddenStates *>(&output_states);

    int batch_size = smooth_input_states->block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_states.set_num_states(
            batch_size * this->seq_len * this->output_size,
            batch_size * this->seq_len * this->input_size);
    }

    // Initialize smoothing hidden states for SLSTM layer
    if (this->smooth_states.num_timesteps !=
        smooth_input_states->num_timesteps) {
        this->smooth_states.set_num_states(this->output_size,
                                           smooth_input_states->num_timesteps);
    }

    // Update number of actual states.
    smooth_output_states->width = this->out_width;
    smooth_output_states->height = this->out_height;
    smooth_output_states->depth = this->out_channels;
    smooth_output_states->block_size = batch_size;
    smooth_output_states->actual_size = this->output_size * this->seq_len;

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

    this->prepare_input_smooth(*smooth_input_states);
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
            this->seq_len, batch_size, this->num_threads,
            smooth_output_states->mu_a, smooth_output_states->var_a);

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
            this->seq_len, batch_size, smooth_output_states->mu_a,
            smooth_output_states->var_a);
    }

    if (this->training) {
        this->storing_states_for_training(*smooth_input_states,
                                          *smooth_output_states);
    }

    // Save the previous states
    lstm_to_prev_states(smooth_output_states->mu_a,
                        this->lstm_states.mu_h_prior.size(),
                        this->lstm_states.mu_h_prior);
    lstm_to_prev_states(smooth_output_states->var_a,
                        this->lstm_states.var_h_prior.size(),
                        this->lstm_states.var_h_prior);
    lstm_to_prev_states(this->lstm_states.mu_c,
                        this->lstm_states.mu_c_prior.size(),
                        this->lstm_states.mu_c_prior);
    lstm_to_prev_states(this->lstm_states.var_c,
                        this->lstm_states.var_c_prior.size(),
                        this->lstm_states.var_c_prior);

    // Save for smoothing
    save_priors_smoother(this->time_step, this->output_size, this->lstm_states,
                         this->smooth_states);

    save_cov_cell_states_smoother(
        this->time_step, this->output_size, this->lstm_states.var_c_prev,
        this->lstm_states.mu_f_ga, this->smooth_states.cov_cc);

    save_cov_hidden_cell_states_smoother(
        this->time_step, this->output_size, this->lstm_states.var_c_prior,
        this->lstm_states.mu_o_ga, this->lstm_states.jcb_ca,
        this->smooth_states.cov_hc);

    int end_chunk_ = batch_size * this->seq_len * this->output_size;
    save_cov_hidden_states_smoother(
        this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
        lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
        lstm_states.mu_o_ga, lstm_states.jcb_o_ga, lstm_states.var_h_prev,
        lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
        this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
        this->output_size, this->input_size, 0, end_chunk_,
        smooth_output_states->cov_hh);

    smooth_output_states->mu_h_prev = lstm_states.mu_h_prev;
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

    // Save for smoothing
    save_posteriors_smoother(this->time_step, this->output_size,
                             this->lstm_states, this->smooth_states);

    // TODO: Increase index for next time step
    ++this->time_step;
}

void SLSTM::smoother()
/*
 */
{
    // Initialize the last time step for smoothing
    this->smooth_states.mu_c_smooths.back() =
        this->smooth_states.mu_c_posts.back();
    this->smooth_states.var_c_smooths.back() =
        this->smooth_states.var_c_posts.back();
    this->smooth_states.mu_h_smooths.back() =
        this->smooth_states.mu_h_posts.back();
    this->smooth_states.var_h_smooths.back() =
        this->smooth_states.var_h_posts.back();

    smooth_cell_states(
        this->smooth_states.num_timesteps, this->smooth_states.num_states,
        this->smooth_states.cov_cc, this->smooth_states.mu_c_priors,
        this->smooth_states.var_c_priors, this->smooth_states.mu_c_posts,
        this->smooth_states.var_c_posts, this->smooth_states.mu_c_smooths,
        this->smooth_states.var_c_smooths);

    smooth_hidden_states(
        this->smooth_states.num_timesteps, this->smooth_states.num_states,
        this->smooth_states.cov_hc, this->smooth_states.mu_c_priors,
        this->smooth_states.var_c_priors, this->smooth_states.mu_c_smooths,
        this->smooth_states.var_c_smooths, this->smooth_states.mu_h_posts,
        this->smooth_states.var_h_posts, this->smooth_states.mu_h_smooths,
        this->smooth_states.var_h_smooths);

    // // TODO: Clear variables for next epoch
    this->time_step = 0;
    this->smooth_states.reset_zeros();
    this->lstm_states.reset_zeros();
}

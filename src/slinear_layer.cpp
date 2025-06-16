#include "../include/slinear_layer.h"

#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/linear_layer.h"

////////////////////////////////////////////////////////////////////////////////
// SLinear: Linear layer with smoother for LSTM
////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <sstream>

void SLinear::print_summary() const {
    std::ofstream summary_file("linear_state_summary.csv");

    if (!summary_file.is_open()) {
        LOG(LogLevel::ERROR, "Failed to open linear_state_summary.csv");
        return;
    }

    summary_file << "TimeStep,StateType,Variable,Values\n";

    size_t T = this->smooth_states.num_timesteps;

    auto write_vector = [&](auto const &state_type, auto const &variable,
                            std::vector<float> const &vec) {
        // Write one row: StateType,Variable,val0,val1,...,valT-1
        summary_file << state_type << "," << variable;
        for (size_t t = 0; t < T; ++t) {
            summary_file << "," << vec[t];
        }
        summary_file << "\n";
    };

    // Write Priors
    write_vector("Priors", "mu_zo_priors", this->smooth_states.mu_zo_priors);
    write_vector("Priors", "var_zo_priors", this->smooth_states.var_zo_priors);

    // Write Posteriors
    write_vector("Posteriors", "mu_zo_posts", this->smooth_states.mu_zo_posts);
    write_vector("Posteriors", "var_zo_posts",
                 this->smooth_states.var_zo_posts);

    // Write Smoothed
    write_vector("Smoothed", "mu_zo_smooths",
                 this->smooth_states.mu_zo_smooths);
    write_vector("Smoothed", "var_zo_smooths",
                 this->smooth_states.var_zo_smooths);
    write_vector("Smoothed", "cov_zo", this->smooth_states.cov_zo);

    summary_file.close();
}

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

void linear_update_hidden_states(int time_step, std::vector<float> &mu_a_prior,
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

void smooth_zo(int num_timestep, int input_size, int output_size,
               std::vector<float> &mu_w, std::vector<float> &var_w,
               std::vector<float> &mu_b, std::vector<float> &var_b,
               const std::vector<float> &prev_mu_h_smooths,
               const std::vector<float> &prev_var_h_smooths,
               std::vector<float> &mu_zo_smooths,
               std::vector<float> &var_zo_smooths)
/*
 */
{
    int idx_h, idx_w;
    bool print_clip_z = true;

    for (int i = num_timestep - 1; i >= 0; i--) {
        for (int k = 0; k <= output_size - 1; ++k) {
            float mu_zo = 0.0f;
            float var_zo = 0.0f;
            for (int j = 0; j <= input_size - 1; ++j) {
                idx_h = i * input_size + k * output_size + j;
                idx_w = k * output_size + j;
                mu_zo += prev_mu_h_smooths[idx_h] * mu_w[idx_w];
                var_zo +=
                    prev_var_h_smooths[idx_h] * var_w[idx_w] +
                    prev_var_h_smooths[idx_h] * mu_w[idx_w] * mu_w[idx_w] +
                    var_w[idx_w] * prev_mu_h_smooths[idx_h] *
                        prev_mu_h_smooths[idx_h];
            }
            mu_zo_smooths[i] = mu_zo + mu_b[k];
            var_zo_smooths[i] = var_zo + var_b[k];
            if (var_zo_smooths[i] < 0 && print_clip_z) {
                LOG(LogLevel::WARNING,
                    "Negative variance for z output at SLSTM smoother at time "
                    "step " +
                        std::to_string(i));
                print_clip_z = false;
            }
        }
    }
}

void SLinear::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    // Checkout input size
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    // New poitner will point to the same memory location when casting
    SmoothingHiddenStates *smooth_input_states =
        dynamic_cast<SmoothingHiddenStates *>(&input_states);
    SmoothingHiddenStates *smooth_output_states =
        dynamic_cast<SmoothingHiddenStates *>(&output_states);

    // Initialization
    int batch_size = smooth_input_states->block_size;
    this->set_cap_factor_udapte(batch_size);

    // Initialize smoothing hidden states for SLinear layer
    if (this->smooth_states.num_timesteps !=
        smooth_input_states->num_timesteps) {
        this->smooth_states.set_num_states(smooth_input_states->num_timesteps);
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
    if (this->training) {
        this->smooth_states.mu_zo_priors[this->time_step] =
            smooth_output_states->mu_a[0];
        this->smooth_states.var_zo_priors[this->time_step] =
            smooth_output_states->var_a[0];
    }

    if (this->training) {
        this->storing_states_for_training(*smooth_input_states,
                                          *smooth_output_states);
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

        linear_update_hidden_states(
            this->time_step, this->smooth_states.mu_zo_priors,
            this->smooth_states.var_zo_priors, input_delta_states.delta_mu,
            input_delta_states.delta_var, this->smooth_states.mu_zo_posts,
            this->smooth_states.var_zo_posts);
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
    // TODO: Increase index for next time step
    ++this->time_step;
}

void SLinear::smoother(const std::vector<float> &prev_mu_h_smooths,
                       const std::vector<float> &prev_var_h_smooths)
/*
 */
{
    smooth_zo(this->smooth_states.num_timesteps, this->input_size,
              this->output_size, this->mu_w, this->var_w, this->mu_b,
              this->var_b, prev_mu_h_smooths, prev_var_h_smooths,
              this->smooth_states.mu_zo_smooths,
              this->smooth_states.var_zo_smooths);

    // this->print_summary();
    this->time_step = 0;
}

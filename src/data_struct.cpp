#include "../include/data_struct.h"

////////////////////////////////////////////////////////////////////////////////
// Base Hidden States
////////////////////////////////////////////////////////////////////////////////
BaseHiddenStates::BaseHiddenStates(size_t n, size_t m)
    : mu_a(n, 0.0f),
      var_a(n, 0.0f),
      jcb(n, 1.0f),
      size(n),
      block_size(m)
/**/
{
    this->actual_size = n / m;
}

BaseHiddenStates::BaseHiddenStates() {}

void BaseHiddenStates::set_input_x(const std::vector<float>& mu_x,
                                   const std::vector<float>& var_x,
                                   const size_t block_size)
/*
 */
{
    int data_size = mu_x.size();
    this->actual_size = data_size / block_size;
    this->block_size = block_size;
    for (int i = 0; i < data_size; i++) {
        this->mu_a[i] = mu_x[i];
        this->jcb[i] = 1.0f;
    }
    if (var_x.size() == data_size) {
        for (int i = 0; i < data_size; i++) {
            this->var_a[i] = var_x[i];
        }
    } else {
        for (int i = 0; i < data_size; i++) {
            this->var_a[i] = 0.0f;
        }
    }
}

void BaseHiddenStates::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    if (new_size > this->size) {
        this->size = new_size;
        this->mu_a.resize(this->size, 0.0f);
        this->var_a.resize(this->size, 0.0f);
        this->jcb.resize(this->size, 1.0f);
    }

    this->block_size = new_block_size;

    // TODO check if we need to modify it
    this->actual_size = new_size / new_block_size;
}

void BaseHiddenStates::swap(BaseHiddenStates& other)
/*
 */
{
    std::swap(mu_a, other.mu_a);
    std::swap(var_a, other.var_a);
    std::swap(jcb, other.jcb);
    std::swap(size, other.size);
    std::swap(block_size, other.block_size);
    std::swap(actual_size, other.actual_size);
    std::swap(width, other.width);
    std::swap(height, other.height);
    std::swap(depth, other.depth);
}

void BaseHiddenStates::copy_from(const BaseHiddenStates& source, int num_data)
/*
 */
{
    // TODO: Revise the actual size copy
    if (num_data == -1) {
        num_data = std::min(source.size, this->size);
    }
    for (int i = 0; i < num_data; i++) {
        this->mu_a[i] = source.mu_a[i];
        this->var_a[i] = source.var_a[i];
        this->jcb[i] = source.jcb[i];
    }
    this->block_size = source.block_size;
    this->actual_size = source.actual_size;
    this->width = source.width;
    this->height = source.height;
    this->depth = source.depth;
}

////////////////////////////////////////////////////////////////////////////////
// Base Smoothing Hidden States
////////////////////////////////////////////////////////////////////////////////
void SmoothingHiddenStates::set_size(size_t new_size, size_t new_block_size) {
    // Set the size of BaseHiddenStates variables
    BaseHiddenStates::set_size(new_size, new_block_size);
    this->cov_hh.resize(this->size, 0.0f);
    this->mu_h_prev.resize(this->size * this->size, 0.0f);
}

//  Override the copy_from method
void SmoothingHiddenStates::copy_from(const BaseHiddenStates& source,
                                      int num_data) {
    BaseHiddenStates::copy_from(source, num_data);

    const SmoothingHiddenStates* source_other =
        dynamic_cast<const SmoothingHiddenStates*>(&source);

    this->cov_hh = source_other->cov_hh;
    this->mu_h_prev = source_other->mu_h_prev;
    this->num_timesteps = source_other->num_timesteps;
}

//  Override the swap method
void SmoothingHiddenStates::swap(BaseHiddenStates& other) {
    SmoothingHiddenStates* smooth_other =
        dynamic_cast<SmoothingHiddenStates*>(&other);
    std::swap(this->cov_hh, smooth_other->cov_hh);
    std::swap(this->mu_h_prev, smooth_other->mu_h_prev);
    std::swap(this->num_timesteps, smooth_other->num_timesteps);
}

////////////////////////////////////////////////////////////////////////////////
// Base Delta States
////////////////////////////////////////////////////////////////////////////////
BaseDeltaStates::BaseDeltaStates(size_t n, size_t m)
    : delta_mu(n, 0.0f), delta_var(n, 0.0f), size(n), block_size(m) {
    this->actual_size = this->size / this->block_size;
}

BaseDeltaStates::BaseDeltaStates() {}

void BaseDeltaStates::reset_zeros() {
    std::fill(this->delta_mu.begin(), this->delta_mu.end(), 0);
    std::fill(this->delta_var.begin(), this->delta_var.end(), 0);
}

void BaseDeltaStates::copy_from(const BaseDeltaStates& source, int num_data)
/*
 */
{
    if (num_data == -1) {
        num_data = std::min(source.size, this->size);
    }
    for (int i = 0; i < num_data; i++) {
        this->delta_mu[i] = source.delta_mu[i];
        this->delta_var[i] = source.delta_var[i];
    }

    this->block_size = source.block_size;
}

void BaseDeltaStates::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    if (new_size > this->size) {
        this->size = new_size;
        this->reset_zeros();
    }
    this->block_size = new_block_size;
    this->actual_size = new_size / new_block_size;
}

void BaseDeltaStates::swap(BaseDeltaStates& other)
/**/
{
    std::swap(delta_mu, other.delta_mu);
    std::swap(delta_var, other.delta_var);
    std::swap(size, other.size);
    std::swap(block_size, other.block_size);
    std::swap(actual_size, other.actual_size);
}

////////////////////////////////////////////////////////////////////////////////
// Base Temp States
////////////////////////////////////////////////////////////////////////////////

BaseTempStates::BaseTempStates(size_t n, size_t m)
    : tmp_1(n, 0.0f), tmp_2(n, 0.0f), size(n), block_size(m) {}

BaseTempStates::BaseTempStates() {}

void BaseTempStates::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    if (new_size > this->size) {
        this->size = new_size;
        this->tmp_1.resize(this->size, 0.0f);
        this->tmp_2.resize(this->size, 0.0f);
    }

    this->block_size = new_block_size;
    this->actual_size = new_size / new_block_size;
}

////////////////////////////////////////////////////////////////////////////////
// Base backward States
////////////////////////////////////////////////////////////////////////////////
BaseBackwardStates::BaseBackwardStates(int num) : size(num) {}
BaseBackwardStates::BaseBackwardStates() {}
void BaseBackwardStates::set_size(size_t new_size)
/*
 */
{
    if (new_size > this->size) {
        this->size = new_size;
        this->mu_a.resize(new_size, 0.0);
        this->jcb.resize(new_size, 1.0f);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Base Observation
////////////////////////////////////////////////////////////////////////////////
BaseObservation::BaseObservation(size_t n, size_t m, size_t k)
    : size(n), block_size(m), idx_size(k) {}

BaseObservation::BaseObservation() {}

void BaseObservation::set_obs(std::vector<float>& mu_obs,
                              std::vector<float>& var_obs)
/**/
{
    this->mu_obs = mu_obs;
    this->var_obs = var_obs;
}

void BaseObservation::set_selected_idx(std::vector<int>& selected_idx)
/*
 */
{
    this->selected_idx = selected_idx;
}

void BaseObservation::set_size(size_t new_size, size_t new_block_size)
/*
 */
{
    if (new_size > this->size) {
        this->size = new_size;
    }
    this->block_size = new_block_size;
    this->actual_size = new_size / new_block_size;
}

////////////////////////////////////////////////////////////////////////////////
// LSTM States
////////////////////////////////////////////////////////////////////////////////
BaseLSTMStates::BaseLSTMStates() {}
BaseLSTMStates::BaseLSTMStates(size_t num_states, size_t num_inputs)
    : num_states(num_states),
      num_inputs(num_inputs)
/*
 */
{
    this->reset_zeros();
}

void BaseLSTMStates::set_num_states(size_t num_states, size_t num_inputs)
/*
 */
{
    this->num_states = num_states;
    this->num_inputs = num_inputs;
    this->reset_zeros();
}

void BaseLSTMStates::reset_zeros()
/**/
{
    // Resize and reset mu_ha and var_ha
    if (mu_ha.size() != num_states + num_inputs)
        mu_ha.resize(num_states + num_inputs);
    if (var_ha.size() != num_states + num_inputs)
        var_ha.resize(num_states + num_inputs);
    for (auto& val : mu_ha) val = 0;
    for (auto& val : var_ha) val = 0;

    // Resize and reset mu_f_ga, var_f_ga, and jcb_f_ga
    if (mu_f_ga.size() != num_states) mu_f_ga.resize(num_states);
    if (var_f_ga.size() != num_states) var_f_ga.resize(num_states);
    if (jcb_f_ga.size() != num_states) jcb_f_ga.resize(num_states);
    for (auto& val : mu_f_ga) val = 0;
    for (auto& val : var_f_ga) val = 0;
    for (auto& val : jcb_f_ga) val = 1.0f;

    // Resize and reset mu_i_ga, var_i_ga, and jcb_i_ga
    if (mu_i_ga.size() != num_states) mu_i_ga.resize(num_states);
    if (var_i_ga.size() != num_states) var_i_ga.resize(num_states);
    if (jcb_i_ga.size() != num_states) jcb_i_ga.resize(num_states);
    for (auto& val : mu_i_ga) val = 0;
    for (auto& val : var_i_ga) val = 0;
    for (auto& val : jcb_i_ga) val = 1.0f;

    // Resize and reset mu_c_ga, var_c_ga, and jcb_c_ga
    if (mu_c_ga.size() != num_states) mu_c_ga.resize(num_states);
    if (var_c_ga.size() != num_states) var_c_ga.resize(num_states);
    if (jcb_c_ga.size() != num_states) jcb_c_ga.resize(num_states);
    for (auto& val : mu_c_ga) val = 0;
    for (auto& val : var_c_ga) val = 0;
    for (auto& val : jcb_c_ga) val = 1.0f;

    // Resize and reset mu_o_ga, var_o_ga, and jcb_o_ga
    if (mu_o_ga.size() != num_states) mu_o_ga.resize(num_states);
    if (var_o_ga.size() != num_states) var_o_ga.resize(num_states);
    if (jcb_o_ga.size() != num_states) jcb_o_ga.resize(num_states);
    for (auto& val : mu_o_ga) val = 0;
    for (auto& val : var_o_ga) val = 0;
    for (auto& val : jcb_o_ga) val = 1.0f;

    // Resize and reset mu_ca, var_ca, and jcb_ca
    if (mu_ca.size() != num_states) mu_ca.resize(num_states);
    if (var_ca.size() != num_states) var_ca.resize(num_states);
    if (jcb_ca.size() != num_states) jcb_ca.resize(num_states);
    for (auto& val : mu_ca) val = 0;
    for (auto& val : var_ca) val = 0;
    for (auto& val : jcb_ca) val = 1.0f;

    // Resize and reset mu_c, var_c, mu_c_prev, and var_c_prev
    if (mu_c.size() != num_states) mu_c.resize(num_states);
    if (var_c.size() != num_states) var_c.resize(num_states);
    if (mu_c_prev.size() != num_states) mu_c_prev.resize(num_states);
    if (var_c_prev.size() != num_states) var_c_prev.resize(num_states);
    for (auto& val : mu_c) val = 0;
    for (auto& val : var_c) val = 0;
    for (auto& val : mu_c_prev) val = 0;
    for (auto& val : var_c_prev) val = 0;

    // Resize and reset mu_h_prev and var_h_prev
    if (mu_h_prev.size() != num_states) mu_h_prev.resize(num_states);
    if (var_h_prev.size() != num_states) var_h_prev.resize(num_states);
    for (auto& val : mu_h_prev) val = 0;
    for (auto& val : var_h_prev) val = 0;

    // Resize and reset cov_i_c and cov_o_tanh_c
    if (cov_i_c.size() != num_states) cov_i_c.resize(num_states);
    if (cov_o_tanh_c.size() != num_states) cov_o_tanh_c.resize(num_states);
    for (auto& val : cov_i_c) val = 0;
    for (auto& val : cov_o_tanh_c) val = 0;

    // Resize and reset mu_c_prior, var_c_prior, mu_h_prior, and var_h_prior
    if (mu_c_prior.size() != num_states) mu_c_prior.resize(num_states);
    if (var_c_prior.size() != num_states) var_c_prior.resize(num_states);
    if (mu_h_prior.size() != num_states) mu_h_prior.resize(num_states);
    if (var_h_prior.size() != num_states) var_h_prior.resize(num_states);
    for (auto& val : mu_c_prior) val = 0;
    for (auto& val : var_c_prior) val = 0;
    for (auto& val : mu_h_prior) val = 0;
    for (auto& val : var_h_prior) val = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Smoother for Slinear layer
////////////////////////////////////////////////////////////////////////////////
SmoothSLinear::SmoothSLinear() {}
SmoothSLinear::SmoothSLinear(size_t num_timesteps)
    : num_timesteps(num_timesteps)
/*
 */
{
    this->reset_zeros();
}

void SmoothSLinear::set_num_states(size_t num_timesteps)
/*
 */
{
    this->num_timesteps = num_timesteps;
    this->reset_zeros();
}

void SmoothSLinear::reset_zeros()
/**/
{
    // Resize and reset cov_zo
    if (cov_zo.size() != num_timesteps) cov_zo.resize(num_timesteps);
    for (auto& val : cov_zo) val = 0;

    // Resize and reset mu_zo_priors
    if (mu_zo_priors.size() != num_timesteps)
        mu_zo_priors.resize(num_timesteps);
    for (auto& val : mu_zo_priors) val = 0;

    // Resize and reset var_zo_priors
    if (var_zo_priors.size() != num_timesteps)
        var_zo_priors.resize(num_timesteps);
    for (auto& val : var_zo_priors) val = 0;

    // Resize and reset mu_zo_posts
    if (mu_zo_posts.size() != num_timesteps) mu_zo_posts.resize(num_timesteps);
    for (auto& val : mu_zo_posts) val = 0;

    // Resize and reset var_zo_posts
    if (var_zo_posts.size() != num_timesteps)
        var_zo_posts.resize(num_timesteps);
    for (auto& val : var_zo_posts) val = 0;

    // Resize and reset mu_zo_smooths
    if (mu_zo_smooths.size() != num_timesteps)
        mu_zo_smooths.resize(num_timesteps);
    for (auto& val : mu_zo_smooths) val = 0;

    // Resize and reset var_zo_smooths
    if (var_zo_smooths.size() != num_timesteps)
        var_zo_smooths.resize(num_timesteps);
    for (auto& val : var_zo_smooths) val = 0;
}

////////////////////////////////////////////////////////////////////////////////
// Smoother for SLSTM layer
////////////////////////////////////////////////////////////////////////////////
SmoothSLSTM::SmoothSLSTM() {}
SmoothSLSTM::SmoothSLSTM(size_t num_states, size_t num_timesteps)
    : num_states(num_states),
      num_timesteps(num_timesteps)
/*
 */
{
    this->reset_zeros();
}

void SmoothSLSTM::set_num_states(size_t num_states, size_t num_timesteps)
/*
 */
{
    this->num_states = num_states;
    this->num_timesteps = num_timesteps;
    this->reset_zeros();
}

void SmoothSLSTM::reset_zeros()
/**/
{
    // Resize and reset mu_h_priors
    if (mu_h_priors.size() != num_states * num_timesteps)
        mu_h_priors.resize(num_states * num_timesteps);
    for (auto& val : mu_h_priors) val = 0;

    // Resize and reset var_h_priors
    if (var_h_priors.size() != num_states * num_timesteps)
        var_h_priors.resize(num_states * num_timesteps);
    for (auto& val : var_h_priors) val = 0;

    // Resize and reset mu_c_priors
    if (mu_c_priors.size() != num_states * num_timesteps)
        mu_c_priors.resize(num_states * num_timesteps);
    for (auto& val : mu_c_priors) val = 0;

    // Resize and reset var_c_priors
    if (var_c_priors.size() != num_states * num_timesteps)
        var_c_priors.resize(num_states * num_timesteps);
    for (auto& val : var_c_priors) val = 0;

    // Resize and reset mu_h_posts
    if (mu_h_posts.size() != num_states * num_timesteps)
        mu_h_posts.resize(num_states * num_timesteps);
    for (auto& val : mu_h_posts) val = 0;

    // Resize and reset var_h_posts
    if (var_h_posts.size() != num_states * num_timesteps)
        var_h_posts.resize(num_states * num_timesteps);
    for (auto& val : var_h_posts) val = 0;

    // Resize and reset mu_c_posts
    if (mu_c_posts.size() != num_states * num_timesteps)
        mu_c_posts.resize(num_states * num_timesteps);
    for (auto& val : mu_c_posts) val = 0;

    // Resize and reset var_c_posts
    if (var_c_posts.size() != num_states * num_timesteps)
        var_c_posts.resize(num_states * num_timesteps);
    for (auto& val : var_c_posts) val = 0;

    // Resize and reset mu_h_smooths
    if (mu_h_smooths.size() != num_states * num_timesteps)
        mu_h_smooths.resize(num_states * num_timesteps);
    for (auto& val : mu_h_smooths) val = 0;

    // Resize and reset var_h_smooths
    if (var_h_smooths.size() != num_states * num_timesteps)
        var_h_smooths.resize(num_states * num_timesteps);
    for (auto& val : var_h_smooths) val = 0;

    // Resize and reset mu_c_smooths
    if (mu_c_smooths.size() != num_states * num_timesteps)
        mu_c_smooths.resize(num_states * num_timesteps);
    for (auto& val : mu_c_smooths) val = 0;

    // Resize and reset var_c_smooths
    if (var_c_smooths.size() != num_states * num_timesteps)
        var_c_smooths.resize(num_states * num_timesteps);
    for (auto& val : var_c_smooths) val = 0;

    // Resize and reset cov_hc
    if (cov_hc.size() != num_states * num_timesteps)
        cov_hc.resize(num_states * num_timesteps);
    for (auto& val : cov_hc) val = 0;

    // Resize and reset cov_cc
    if (cov_cc.size() != num_states * num_timesteps)
        cov_cc.resize(num_states * num_timesteps);
    for (auto& val : cov_cc) val = 0;
}

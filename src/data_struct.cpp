///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 10, 2023
// Updated:      March 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/data_struct.h"

BaseHiddenStates::BaseHiddenStates(size_t n, size_t m)
    : mu_a(n, 0.0f), var_a(n, 0.0f), jcb(n, 1.0f), size(n), block_size(m) {}

BaseHiddenStates::BaseHiddenStates() {}

void BaseHiddenStates::set_input_x(const std::vector<float> &mu_x,
                                   const std::vector<float> &var_x,
                                   const size_t block_size)
/*
 */
{
    int data_size = mu_x.size();
    this->actual_size = data_size / block_size;
    this->block_size = block_size;
    for (int i = 0; i < data_size; i++) {
        this->mu_a[i] = mu_x[i];
    }
    if (var_x.size() == data_size) {
        for (int i = 0; i < data_size; i++) {
            this->var_a[i] = var_x[i];
        }
    }
}

BaseDeltaStates::BaseDeltaStates(size_t n, size_t m)
    : delta_mu(n, 0.0f), delta_var(n, 0.0f), size(n), block_size(m) {}

BaseDeltaStates::BaseDeltaStates() {}

void BaseDeltaStates::reset_zeros() {
    std::fill(this->delta_mu.begin(), this->delta_mu.end(), 0);
    std::fill(this->delta_var.begin(), this->delta_var.end(), 0);
}

void BaseDeltaStates::copy_from(const BaseDeltaStates &source, int num_data)
/*
 */
{
    if (num_data == -1) {
        num_data = this->size;
    }
    for (int i = 0; i < num_data; i++) {
        this->delta_mu[i] = source.delta_mu[i];
        this->delta_var[i] = source.delta_var[i];
    }
}

BaseTempStates::BaseTempStates(size_t n, size_t m)
    : tmp_1(n, 0.0f), tmp_2(n, 0.0f), size(n), block_size(m) {}

BaseTempStates::BaseTempStates() {}

BaseBackwardStates::BaseBackwardStates(int num) : size(num) {}
BaseBackwardStates::BaseBackwardStates() {}

BaseObservation::BaseObservation(size_t n, size_t m, size_t k)
    : size(n), block_size(m), idx_size(k) {}

BaseObservation::BaseObservation() {}

void BaseObservation::set_obs(std::vector<float> &mu_obs,
                              std::vector<float> &var_obs) {
    this->mu_obs = mu_obs;
    this->var_obs = var_obs;
}

void BaseObservation::set_selected_idx(std::vector<int> &selected_idx) {
    this->selected_idx = selected_idx;
}

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
    mu_ha.resize(num_states + num_inputs, 0);
    var_ha.resize(num_states + num_inputs, 0);
    mu_f_ga.resize(num_states, 0);
    var_f_ga.resize(num_states, 0);
    jcb_f_ga.resize(num_states, 0);
    mu_i_ga.resize(num_states, 0);
    var_i_ga.resize(num_states, 0);
    jcb_i_ga.resize(num_states, 0);
    mu_c_ga.resize(num_states, 0);
    var_c_ga.resize(num_states, 0);
    jcb_c_ga.resize(num_states, 0);
    mu_o_ga.resize(num_states, 0);
    var_o_ga.resize(num_states, 0);
    jcb_o_ga.resize(num_states, 0);
    mu_ca.resize(num_states, 0);
    var_ca.resize(num_states, 0);
    jcb_ca.resize(num_states, 0);
    mu_c.resize(num_states, 0);
    var_c.resize(num_states, 0);
    mu_c_prev.resize(num_states, 0);
    var_c_prev.resize(num_states, 0);
    mu_h_prev.resize(num_states, 0);
    var_h_prev.resize(num_states, 0);
    cov_i_c.resize(num_states, 0);
    cov_o_tanh_c.resize(num_states, 0);
}
///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.cpp
// Description:  TAGI network including feed forward & backward (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      January 27, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/tagi_network_cpu.h"

TagiNetworkCPU::TagiNetworkCPU(Network &net_prop) {
    this->prop = net_prop;
    init_net();
}

TagiNetworkCPU::~TagiNetworkCPU() {}

void TagiNetworkCPU::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                                  std::vector<float> &Sx_f) {
    // Set input data
    this->net_input.set_values(x, Sx, Sx_f);
    int input_size = this->prop.batch_size * this->prop.input_seq_len;

    // Initialize input
    initialize_states_cpu(this->net_input.x_batch, this->net_input.Sx_batch,
                          this->net_input.Sx_f_batch, this->prop.n_x,
                          input_size, this->state);

    // Feed forward
    feed_forward_cpu(this->prop, this->theta, this->idx, this->state);
}

void TagiNetworkCPU::connected_feed_forward(std::vector<float> &ma,
                                            std::vector<float> &Sa,
                                            std::vector<float> &mz,
                                            std::vector<float> &Sz,
                                            std::vector<float> &J) {
    // Initialize input
    initialize_full_states_cpu(mz, Sz, ma, Sa, J, this->state.mz,
                               this->state.Sz, this->state.ma, this->state.Sa,
                               this->state.J);

    // Feed forward
    feed_forward_cpu(this->prop, this->theta, this->idx, this->state);
}

void TagiNetworkCPU::state_feed_backward(std::vector<float> &y,
                                         std::vector<float> &Sy,
                                         std::vector<int> &idx_ud) {
    // Set output data
    this->obs.set_values(y, Sy, idx_ud);

    // Compute update quantities for hidden states
    state_backward_cpu(this->prop, this->theta, this->state, this->idx,
                       this->obs, this->d_state);
}

void TagiNetworkCPU::param_feed_backward() {
    // Feed backward for parameters
    param_backward_cpu(this->prop, this->theta, this->state, this->d_state,
                       this->idx, this->d_theta);

    // Update model parameters
    global_param_update_cpu(this->d_theta, this->prop.cap_factor,
                            this->num_weights, this->num_biases,
                            this->num_weights_sc, this->num_biases_sc,
                            this->theta);
}

void TagiNetworkCPU::init_net() {
    net_default(this->prop);
    get_net_props(this->prop);
    get_similar_layer(this->prop);

    // Check feature availability
    check_feature_availability(this->prop);

    // Indices
    tagi_idx(this->idx, this->prop);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(this->prop);
    this->state = initialize_net_states(this->prop);

    // Update quantities
    this->d_state.set_values(this->prop);
    this->d_theta.set_values(this->theta.mw.size(), this->theta.mb.size(),
                             this->theta.mw_sc.size(),
                             this->theta.mb_sc.size());

    this->num_weights = theta.mw.size();
    this->num_biases = theta.mb.size();
    this->num_weights_sc = theta.mw_sc.size();
    this->num_biases_sc = theta.mb_sc.size();

    // Input layer
    int input_size =
        this->prop.n_x * this->prop.batch_size * this->prop.input_seq_len;
    this->ma_init.resize(input_size, 0);
    this->Sa_init.resize(input_size, 0);
    this->mz_init.resize(input_size, 0);
    this->Sz_init.resize(input_size, 0);
    this->J_init.resize(input_size, 1);

    // Output layer
    this->ma.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->Sa.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->mz.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->Sz.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->J.resize(this->prop.nodes.back() * this->prop.batch_size, 1);
    this->m_pred.resize(this->prop.n_y * this->prop.batch_size, 0);
    this->v_pred.resize(this->prop.n_y * this->prop.batch_size, 0);
}

void TagiNetworkCPU::get_network_outputs() {
    // Last layer's hidden state
    int num_outputs = this->prop.nodes.back() * this->prop.batch_size;
    for (int i = 0; i < num_outputs; i++) {
        this->ma[i] = this->state.ma[this->prop.z_pos.back() + i];
        this->Sa[i] = this->state.Sa[this->prop.z_pos.back() + i];
    }
}
void TagiNetworkCPU::get_predictions()
/*Get prediction distributions. Note that the predictions might be different to
   output values of a network e.g., heteroscedastic noise inference where output
   layer includes the mean and std of the prediction of an outcome */
{
    int num_preds = this->prop.n_y * this->prop.batch_size;
    this->get_network_outputs();
    if (this->prop.noise_type.compare("heteros") != 0) {
        this->m_pred = this->ma;
        this->v_pred = this->Sa;
        if (this->prop.noise_type.compare("homosce") == 0) {
            for (int i = 0; i < num_preds; i++) {
                this->v_pred[i] += this->state.noise_state.ma_v2b_prior[i];
            }
        }
    } else {
        get_output_hidden_states_ni_cpu(this->ma, this->prop.nodes.back(), 0,
                                        this->m_pred);
        get_output_hidden_states_ni_cpu(this->Sa, this->prop.nodes.back(), 0,
                                        this->v_pred);
        for (int i = 0; i < num_preds; i++) {
            this->v_pred[i] += this->state.noise_state.ma_v2b_prior[i];
        }
    }
}

void TagiNetworkCPU::get_all_network_outputs() {
    // Last layer's hidden state
    int num_outputs = this->prop.nodes.back() * this->prop.batch_size;
    int z_pos = this->prop.z_pos.back();
    for (int i = 0; i < num_outputs; i++) {
        this->ma[i] = this->state.ma[z_pos + i];
        this->Sa[i] = this->state.Sa[z_pos + i];
        this->mz[i] = this->state.mz[z_pos + i];
        this->Sz[i] = this->state.Sz[z_pos + i];
        this->J[i] = this->state.J[z_pos + i];
    }
}

void TagiNetworkCPU::get_all_network_inputs() {
    // Last layer's hidden state
    int input_size =
        this->prop.n_x * this->prop.batch_size * this->prop.input_seq_len;
    for (int i = 0; i < input_size; i++) {
        this->ma_init[i] = this->state.ma[i];
        this->Sa_init[i] = this->state.Sa[i];
        this->mz_init[i] = this->state.mz[i];
        this->Sz_init[i] = this->state.Sz[i];
        this->J_init[i] = this->state.J[i];
    }
}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetworkCPU::get_derivatives(int layer)
/*Compute derivative of neural network using TAGI. NOTE: current version only
   support the fully-connected layer*/
{
    if (!this->prop.collect_derivative) {
        throw std::invalid_argument(
            "Set collect_derivative model in network properties to True");
    }
    int num_derv = this->prop.batch_size * this->prop.nodes[layer];
    std::vector<float> mdy_batch_in(num_derv, 0);
    std::vector<float> Sdy_batch_in(num_derv, 0);
    compute_network_derivatives_cpu(this->prop, this->theta, this->state,
                                    layer);
    get_input_derv_states(this->state.derv_state.md_layer,
                          this->state.derv_state.Sd_layer, mdy_batch_in,
                          Sdy_batch_in);

    return {mdy_batch_in, Sdy_batch_in};
}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetworkCPU::get_inovation_mean_var(int layer) {
    int num_data;
    if (layer == 0) {  // input layer
        num_data = this->prop.nodes[layer] * this->prop.batch_size *
                   this->prop.input_seq_len;

    } else {
        num_data = this->prop.nodes[layer] * this->prop.batch_size;
    }
    int z_pos = this->prop.z_pos[layer];
    std::vector<float> delta_m(num_data, 0);
    std::vector<float> delta_S(num_data, 0);
    for (int i = 0; i < num_data; i++) {
        delta_m[i] = this->d_state.delta_m[z_pos + i];
        delta_S[i] = this->d_state.delta_S[z_pos + i];
    }

    return {delta_m, delta_S};
}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetworkCPU::get_state_delta_mean_var()
/*Get updating quantites (delta_mz, delta_Sz) of the input layer for the hidden
states. NOTE: delta_mz and delta_Sz are overridable vector during the backward
pass in order to save memory allocation on the device.
*/
{
    int num_data =
        this->prop.nodes[0] * this->prop.batch_size * this->prop.input_seq_len;
    std::vector<float> delta_mz(num_data, 0);
    std::vector<float> delta_Sz(num_data, 0);
    for (int i = 0; i < num_data; i++) {
        delta_mz[i] = this->d_state.delta_mz[i];
        delta_Sz[i] = this->d_state.delta_Sz[i];
    }

    return {delta_mz, delta_Sz};
}

void TagiNetworkCPU::set_parameters(Param &init_theta)
/*Set parameters to network*/
{
    // Check if parameters are valids
    if ((init_theta.mw.size() != this->num_weights) ||
        (init_theta.Sw.size() != this->num_weights)) {
        throw std::invalid_argument("Length of weight parameters is invalid");
    }
    if ((init_theta.mb.size() != this->num_biases) ||
        (init_theta.Sb.size() != this->num_biases)) {
        throw std::invalid_argument("Length of bias parameters is invalid");
    }

    // Weights
    for (int i = 0; i < this->num_weights; i++) {
        this->theta.mw[i] = init_theta.mw[i];
        this->theta.Sw[i] = init_theta.Sw[i];
    }

    // Biases
    for (int j = 0; j < this->num_biases; j++) {
        this->theta.mb[j] = init_theta.mb[j];
        this->theta.Sb[j] = init_theta.Sb[j];
    }

    // Residual network
    if (this->num_weights_sc > 0) {
        // Weights
        for (int i = 0; i < this->num_weights_sc; i++) {
            this->theta.mw_sc[i] = init_theta.mw_sc[i];
            this->theta.Sw_sc[i] = init_theta.Sw_sc[i];
        }

        // Biases
        for (int j = 0; j < this->num_biases_sc; j++) {
            this->theta.mb_sc[j] = init_theta.mb_sc[j];
            this->theta.Sb_sc[j] = init_theta.Sb_sc[j];
        }
    }
}

Param TagiNetworkCPU::get_parameters() { return this->theta; }
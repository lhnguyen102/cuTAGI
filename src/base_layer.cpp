///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 11, 2023
// Updated:      March 11, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/base_layer.h"

InitArgs::InitArgs(size_t width, size_t height, size_t depth, int batch_size)
    : width(width), height(height), depth(depth), batch_size(batch_size) {}

BaseLayer::BaseLayer() {}

const char *BaseLayer::get_layer_type_name() const {
    return typeid(*this).name();
}

std::string BaseLayer::get_layer_info() const { return "Base()"; }

std::string BaseLayer::get_layer_name() const { return "Base"; }

LayerType BaseLayer::get_layer_type() const { return LayerType::Base; };

int BaseLayer::get_input_size() { return static_cast<int>(this->input_size); }

int BaseLayer::get_output_size() { return static_cast<int>(this->output_size); }

void BaseLayer::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states) {}

void BaseLayer::state_backward(BaseBackwardStates &next_bwd_states,
                               BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_hidden_states,
                               BaseTempStates &temp_states) {}

void BaseLayer::param_backward(BaseBackwardStates &next_bwd_states,
                               BaseDeltaStates &delta_states,
                               BaseTempStates &temp_states) {}

void BaseLayer::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);
    this->delta_mu_b.resize(this->num_biases, 0.0f);
    this->delta_var_b.resize(this->num_biases, 0.0f);
}

void BaseLayer::allocate_bwd_vector(int size)
/*
 */
{
    if (size <= 0) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    " - Invalid size: " + std::to_string(size));
    }

    this->bwd_states->mu_a.resize(size, 0.0f);
    this->bwd_states->jcb.resize(size, 0.0f);
}

void BaseLayer::fill_output_states(BaseHiddenStates &output_states)
/**/
{
    for (int j = 0; j < output_states.actual_size * output_states.block_size;
         j++) {
        output_states.jcb[j] = 1.0f;
    }
}

void BaseLayer::fill_bwd_vector(BaseHiddenStates &input_states)
/*
 */
{
    for (int i = 0; i < input_states.actual_size * input_states.block_size;
         i++) {
        this->bwd_states->mu_a[i] = input_states.mu_a[i];
        this->bwd_states->jcb[i] = input_states.jcb[i];
    }
}

void BaseLayer::update_weights()
/*
 */
{
    for (int i = 0; i < this->mu_w.size(); i++) {
        this->mu_w[i] += this->delta_mu_w[i];
        this->var_w[i] += this->delta_var_w[i];
    }
}

void BaseLayer::update_biases()
/*

 */
{
    if (this->bias) {
        for (int i = 0; i < this->mu_b.size(); i++) {
            this->mu_b[i] += this->delta_mu_b[i];
            this->var_b[i] += this->delta_var_b[i];
        }
    }
}

void BaseLayer::compute_input_output_size(const InitArgs &args)
/*
 */
{
    // TODO: find a way to add specialized paramters for each layer
    this->in_width = args.width;
    this->in_height = args.height;
    this->in_channels = args.depth;

    this->out_width = args.width;
    this->out_height = args.height;
    this->out_channels = args.depth;
}

void BaseLayer::storing_states_for_training(BaseHiddenStates &input_states,
                                            BaseHiddenStates &output_states)
/*
 */
{
    if (this->bwd_states->mu_a.size() == 0) {
        int act_size = input_states.actual_size * input_states.block_size;
        this->allocate_bwd_vector(act_size);
    }

    // Activation's jacobian and mean from the previous layer
    this->fill_bwd_vector(input_states);

    // Send a copy of activation's mean and variance to the output buffer
    // for the current layer.
    // TODO: consider to have only mu_a and var_a in struct HiddenStates
    this->fill_output_states(output_states);
}

void BaseLayer::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }

    // Save the name length and name
    auto layer_name = this->get_layer_name();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }
}

void BaseLayer::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_name();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }
}

std::tuple<std::vector<float>, std::vector<float>>
BaseLayer::get_running_mean_var()
/*
 */
{
    return {std::vector<float>(), std::vector<float>()};
}

void BaseLayer::preinit_layer()
/* Pre-initialize the layer property e.g., number of weights & biases
 */
{
    // We do nothing by default
}

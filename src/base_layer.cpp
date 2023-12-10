///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 11, 2023
// Updated:      December 10, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/base_layer.h"

BaseLayer::BaseLayer() {}

const char *BaseLayer::get_layer_type_name() const {
    return typeid(*this).name();
}

int BaseLayer::get_input_size() { return static_cast<int>(input_size); }

int BaseLayer::get_output_size() { return static_cast<int>(output_size); }

void BaseLayer::forward(HiddenStateBase &input_states,
                        HiddenStateBase &output_states,
                        TempStateBase &temp_states) {}

void BaseLayer::state_backward(std::vector<float> &jcb,
                               DeltaStateBase &input_delta_states,
                               DeltaStateBase &output_hidden_states,
                               TempStateBase &temp_states) {}

void BaseLayer::param_backward(std::vector<float> &mu_a,
                               DeltaStateBase &delta_states,
                               TempStateBase &temp_states) {}

void BaseLayer::allocate_bwd_vector(int size)
/*
 */
{
    if (size <= 0) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    " - Invalid size: " + std::to_string(size));
    }

    this->mu_a.resize(size, 0.0f);
    this->jcb.resize(size, 0.0f);
}

void BaseLayer::fill_output_states(HiddenStateBase &output_states)
/**/
{
    for (int j = 0; j < output_states.actual_size * output_states.block_size;
         j++) {
        output_states.mu_a[j] = output_states.mu_z[j];
        output_states.var_a[j] = output_states.var_z[j];
        output_states.jcb[j] = 1.0f;
    }
}

void BaseLayer::fill_bwd_vector(HiddenStateBase &input_states)
/*
 */
{
    for (int i = 0; i < input_states.actual_size * input_states.block_size;
         i++) {
        this->mu_a[i] = input_states.mu_a[i];
        this->jcb[i] = input_states.jcb[i];
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
    for (int i = 0; i < this->mu_b.size(); i++) {
        this->mu_b[i] += this->delta_mu_b[i];
        this->var_b[i] += this->delta_var_b[i];
    }
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
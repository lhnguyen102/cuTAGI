///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 11, 2023
// Updated:      October 19, 2023
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

void BaseLayer::forward(HiddenStates &input_states, HiddenStates &output_states,
                        TempStates &temp_states) {}

void BaseLayer::state_backward(std::vector<float> &jcb,
                               DeltaStates &input_delta_states,
                               DeltaStates &output_hidden_states,
                               TempStates &temp_states) {}

void BaseLayer::param_backward(std::vector<float> &mu_a,
                               DeltaStates &delta_states,
                               TempStates &temp_states) {}

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

void BaseLayer::fill_output_states(HiddenStates &output_states)
/**/
{
    for (int j = 0; j < output_states.actual_size * output_states.block_size;
         j++) {
        output_states.mu_a[j] = output_states.mu_z[j];
        output_states.var_a[j] = output_states.var_z[j];
    }
}

void BaseLayer::fill_bwd_vector(HiddenStates &input_states)
/*
 */
{
    for (int i = 0; i < input_states.actual_size * input_states.block_size;
         i++) {
        this->mu_a[i] = input_states.mu_a[i];
        this->jcb[i] = input_states.jcb[i];
    }
}
///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 11, 2023
// Updated:      October 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/base_layer.h"

int BaseLayer::get_input_size() { return static_cast<int>(input_size); }

int BaseLayer::get_output_size() { return static_cast<int>(output_size); }

void BaseLayer::forward(HiddenStates &input_states, HiddenStates &output_states,
                        TempStates &temp_states) {}

void BaseLayer::state_backward(std::vector<float> &jcb,
                               DeltaStates &input_delta_states,
                               DeltaStates &output_hidden_states,
                               TempStates &temp_states) {}

void BaseLayer::param_backward(DeltaStates &delta_states,
                               TempStates &temp_states) {}

void BaseLayer::allocate_activation_bwd_vector(int size)
/*
 */
{
    if (size <= 0) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    " - Invalid size: " + std::to_string(size));
    }

    this->input_size = size;
    this->mu_a.resize(size, 0.0f);
    this->jcb.resize(size, 0.0f);
}
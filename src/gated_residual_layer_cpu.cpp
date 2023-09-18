///////////////////////////////////////////////////////////////////////////////
// File:         gated_residual_layer_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 17, 2023
// Updated:      September 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/gated_residual_layer_cpu.h"

int get_number_of_params(GatedResidualProp &prop)
/*
 */
{
    // 1st FC layer
    prop.num_weights += prop.input_size * prop.hidden_size;
    prop.num_biases += prop.hidden_size;

    // Update weight and bias position vectors
    prop.w_pos.push_back(prop.num_weights);
    prop.b_pos.push_back(prop.num_biases);

    // Context layer
    if (prop.context_size > 0) {
        prop.num_weights += prop.context_size * prop.hidden_size;
        prop.num_biases += prop.context_size;
    }

    // 2nd FC Layer
    prop.w_pos.push_back(prop.num_weights);
    prop.b_pos.push_back(prop.num_biases);
    prop.num_weights += prop.hidden_size * prop.output_size;
    prop.num_biases += prop.output_size;
}

void forward_cpu(Network &net, Param &theta, IndexOut &idx, NetState &state)
/*
 */
{
    // First FC layer
    // Context FC layer
    // ELU layer
    // Second FC layer
    // SiLU layer
    // Add residual before normalization layer
    // Layer norm
}

void state_backward_cpu() {}

void param_backward_cpu() {}
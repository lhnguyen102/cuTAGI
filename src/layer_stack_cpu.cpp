///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/layer_stack_cpu.h"

LayerStack::LayerStack() {}
LayerStack::~LayerStack() {}

void LayerStack::add_layer(std::unique_ptr<BaseLayer> layer)
/*
 */
{
    // Get buffer size
    int output_size = layer->get_output_size();
    this->output_buffer_size = std::max(output_size, this->output_buffer_size);

    // Stack layer
    layers.push_back(std::move(layer));
}

void LayerStack::init_output_state_buffer()
/*
 */
{
    this->output_z_buffer =
        HiddenStates(this->output_buffer_size, this->output_buffer_block_size);
}

void LayerStack::init_delta_state_buffer()
/*
 */
{
    this->output_delta_z_buffer =
        DeltaStates(this->output_buffer_size, this->output_buffer_block_size);
    this->input_delta_z_buffer =
        DeltaStates(this->output_buffer_size, this->output_buffer_block_size);
}

void LayerStack::update_output_delta_z(HiddenStates &last_layer_states,
                                       std::vector<float> &obs,
                                       std::vector<float> &var_obs)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = obs.size();
    compute_delta_z_output(last_layer_states.mu_a, last_layer_states.var_a,
                           last_layer_states.var_z, last_layer_states.jcb, obs,
                           var_obs, start_chunk, end_chunk,
                           this->input_delta_z_buffer.delta_mu,
                           this->input_delta_z_buffer.delta_var);
}

HiddenStates LayerStack::forward(HiddenStates &input_states)
/*
 */
{
    // Resize the buffer for delta and output states
    if (input_states.block_size != this->output_buffer_block_size) {
        this->output_buffer_block_size = input_states.block_size;
        init_output_state_buffer();
    }

    // Forward pass for all layers
    for (const auto &layer : this->layers) {
        layer->forward(input_states, this->output_z_buffer, this->temp_states);
        input_states = this->output_z_buffer;
    }
    return this->output_z_buffer;
}

void LayerStack::backward()
/*
 */
{
    // Output layer
    int last_layer_idx = this->layers.size() - 1;

    // Hidden layers
    for (int i = last_layer_idx - 1; i >= 0; --i) {
        // Backward pass for hidden states
        this->layers[i]->state_backward(
            this->layers[i + 1]->jcb, this->input_delta_z_buffer,
            this->output_delta_z_buffer, this->temp_states);

        // Backward pass for parameters
        if (this->param_update) {
            this->layers[i]->param_backward(this->output_delta_z_buffer,
                                            this->temp_states);
        }

        // Pass new input data for next iteration
        this->input_delta_z_buffer = this->output_delta_z_buffer;
    }
}
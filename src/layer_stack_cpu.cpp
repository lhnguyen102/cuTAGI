///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      November 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/layer_stack_cpu.h"

LayerStack::LayerStack() {}
LayerStack::~LayerStack() {}

void LayerStack::add_layer(std::unique_ptr<BaseLayer> layer)
/*
NOTE: The output buffer size is determinated based on the output size for each
layer assuming that batch size = 1. If the batch size in the forward pass > 1,
it will be corrected at the first run in the forward pass.
 */
{
    // Get buffer size
    int output_size = layer->get_output_size();
    int input_size = layer->get_input_size();
    this->z_buffer_size = std::max(output_size, this->z_buffer_size);
    this->z_buffer_size = std::max(input_size, this->z_buffer_size);

    // Stack layer
    this->layers.push_back(std::move(layer));
}

void LayerStack::init_output_state_buffer()
/*
 */
{
    this->output_z_buffer =
        HiddenStates(this->z_buffer_size, this->z_buffer_block_size);
    this->input_z_buffer =
        HiddenStates(this->z_buffer_size, this->z_buffer_block_size);
}

void LayerStack::init_delta_state_buffer()
/*
 */
{
    this->output_delta_z_buffer =
        DeltaStates(this->z_buffer_size, this->z_buffer_block_size);
    this->input_delta_z_buffer =
        DeltaStates(this->z_buffer_size, this->z_buffer_block_size);
}

void LayerStack::to_z_buffer(const std::vector<float> &mu_x,
                             const std::vector<float> &var_x,
                             HiddenStates &hidden_states)
/*
 */
{
    int data_size = mu_x.size();
    for (int i = 0; i < data_size; i++) {
        hidden_states.mu_z[i] = mu_x[i];
        hidden_states.mu_a[i] = mu_x[i];
    }
    if (var_x.size() == data_size) {
        for (int i = 0; i < data_size; i++) {
            hidden_states.var_z[i] = var_x[i];
            hidden_states.var_a[i] = var_x[i];
        }
    }
    hidden_states.size = data_size;
    hidden_states.block_size = data_size / this->layers.front()->input_size;
    hidden_states.actual_size = this->layers.front()->input_size;
}

void LayerStack::forward(const std::vector<float> &mu_x,
                         const std::vector<float> &var_x)
/*
 */
{
    // Batch size
    int batch_size = mu_x.size() / this->layers.front()->input_size;

    // Only initialize if batch size changes
    if (batch_size != this->z_buffer_block_size) {
        this->z_buffer_block_size = batch_size;
        this->z_buffer_size = batch_size * this->z_buffer_size;
        init_output_state_buffer();
        if (this->training) {
            init_delta_state_buffer();
        }
    }

    // Merge input data to the input buffer
    this->to_z_buffer(mu_x, var_x, this->input_z_buffer);

    // Forward pass for all layers
    for (const auto &layer : this->layers) {
        layer->forward(this->input_z_buffer, this->output_z_buffer,
                       this->temp_states);
        this->input_z_buffer = this->output_z_buffer;
    }
}

void LayerStack::backward()
/*
 */
{
    // TODO: need to fix the case we we dont have activation function and need
    // to pass mu_a to parameters Output layer
    int last_layer_idx = this->layers.size() - 1;

    // Hidden layers
    for (int i = last_layer_idx; i >= 1; --i) {
        // TODO: need to perform parameter update for input layer and potential
        // update hidden state
        // Backward pass for parameters
        if (this->param_update) {
            this->layers[i]->param_backward(this->layers[i - 1]->mu_a,
                                            this->input_delta_z_buffer,
                                            this->temp_states);
        }

        // Backward pass for hidden states
        this->layers[i]->state_backward(
            this->layers[i - 1]->jcb, this->input_delta_z_buffer,
            this->output_delta_z_buffer, this->temp_states);

        // Pass new input data for next iteration
        this->input_delta_z_buffer = this->output_delta_z_buffer;
    }
}
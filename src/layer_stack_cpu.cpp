///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 19, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/layer_stack.h"

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
    this->output_state_buffer = HiddenStates(this->output_buffer_size);
}

void LayerStack::init_delta_state_buffer()
/*
 */
{
    this->delta_state_buffer = DeltaStates(this->output_buffer_size);
}

HiddenStates LayerStack::forward(HiddenStates &input_states)
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->forward(input_states, this->output_state_buffer);
        input_states = this->output_state_buffer;
    }
    return this->output_state_buffer;
}

void LayerStack::state_backward(std::vector<float> &jcb,
                                std::vector<float> &delta_mu,
                                std::vector<float> &delta_var)
/*
 */
{
    // Output layer
    int last_layer_idx = this->layers.size() - 1;
    this->layers[last_layer_idx]->state_backward(jcb, delta_mu, delta_var);

    // Hidden layers
    for (int i = last_layer_idx - 1; i >= 0; --i) {
        this->layers[i]->state_backward(this->layers[i + 1]->jcb,
                                        this->layers[i + 1]->delta_mu,
                                        this->layers[i + 1]->delta_var);
    }
}

void LayerStack::param_backward()
/*
 */
{
    for (int i = this->layers.size() - 1; i >= 0; --i) {
        this->layers[i]->param_backward();
    }
}

///////////////////////////////////////////////////////////////////////////
/// MODULE
///////////////////////////////////////////////////////////////////////////
void Module::add(std::shared_ptr<BaseLayer> layer) { layers.push_back(layer); }
///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 22, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <memory>

#include "base_layer.h"
#include "struct_var.h"

class LayerStack {
   public:
    HiddenStates output_state_buffer;
    DeltaStates output_delta_state_buffer;
    DeltaStates input_delta_state_buffer;
    TempStates temp_states;
    int output_buffer_size = 0;
    int output_buffer_block_size = 1;

    LayerStack();

    ~LayerStack();

    void add_layer(std::unique_ptr<BaseLayer> layer){};

    void init_output_state_buffer();

    void init_delta_state_buffer();

    HiddenStates forward(HiddenStates &input_states);

    void backward(std::vector<float> &obs, std::vector<float> &obs_var);

   private:
    std::vector<std::unique_ptr<BaseLayer>> layers;
};

class Module {
   public:
    void add(std::shared_ptr<BaseLayer> layer){};

   private:
    std::vector<std::shared_ptr<BaseLayer>> layers;
};
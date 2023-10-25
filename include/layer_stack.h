///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <memory>

#include "base_layer.h"
#include "output_layer_update_cpu.h"
#include "struct_var.h"

class LayerStack {
   public:
    HiddenStates output_z_buffer;
    DeltaStates output_delta_z_buffer;
    DeltaStates input_delta_z_buffer;
    TempStates temp_states;
    int output_buffer_size = 0;        // e.g., batch size x input size
    int output_buffer_block_size = 1;  // e.g., batch size
    bool param_update = true;

    LayerStack();

    ~LayerStack();

    void add_layer(std::unique_ptr<BaseLayer> layer){};

    void init_output_state_buffer();

    void init_delta_state_buffer();

    void update_output_delta_z(HiddenStates &output_states,
                               std::vector<float> &obs,
                               std::vector<float> &var_obs);

    HiddenStates forward(HiddenStates &input_states);

    void backward();

   private:
    std::vector<std::unique_ptr<BaseLayer>> layers;
};

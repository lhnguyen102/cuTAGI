///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      November 17, 2023
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
    HiddenStates input_z_buffer;
    DeltaStates output_delta_z_buffer;
    DeltaStates input_delta_z_buffer;
    TempStates temp_states;
    int z_buffer_size = 0;        // e.g., batch size x input size
    int z_buffer_block_size = 1;  // e.g., batch size
    int input_size = 0;
    bool training = true;
    bool param_update = true;

    LayerStack();

    ~LayerStack();

    void add_layer(std::unique_ptr<BaseLayer> layer);

    void init_output_state_buffer();

    void init_delta_state_buffer();

    void forward(const std::vector<float>& mu_a,
                 const std::vector<float>& var_a = std::vector<float>());

    void to_z_buffer(const std::vector<float>& mu_x,
                     const std::vector<float>& var_x,
                     HiddenStates& hidden_states);

    void backward();

   private:
    std::vector<std::unique_ptr<BaseLayer>> layers;
};

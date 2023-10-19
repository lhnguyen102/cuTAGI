///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 19, 2023
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
    DeltaStates delta_state_buffer;
    int output_buffer_size = 0;
    int input_buffer_size = 0;
    LayerStack();
    ~LayerStack();

    void add_layer(std::unique_ptr<BaseLayer> layer){};

    void init_output_state_buffer();

    void init_delta_state_buffer();

    HiddenStates forward(HiddenStates &input_states);

    void state_backward(std::vector<float> &jcb, std::vector<float> &delta_mu,
                        std::vector<float> &delta_var);

    void param_backward();

   private:
    std::vector<std::unique_ptr<BaseLayer>> layers;
};

class Module {
   public:
    void add(std::shared_ptr<BaseLayer> layer){};

   private:
    std::vector<std::shared_ptr<BaseLayer>> layers;
};
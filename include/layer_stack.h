///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 09, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <memory>

#include "base_layer.h"
#include "struct_var.h"

class LayerStack {
   public:
    LayerStack();
    ~LayerStack();
    void addLayer(std::unique_ptr<BaseLayer> layer){};

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
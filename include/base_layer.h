///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      November 11, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "struct_var.h"

class BaseLayer {
   public:
    size_t input_size, output_size;
    std::vector<float> mu_w;
    std::vector<float> var_w;
    std::vector<float> mu_b;
    std::vector<float> var_b;
    std::vector<float> mu_a;
    std::vector<float> jcb;
    std::vector<float> delta_mu_w;
    std::vector<float> delta_var_w;
    std::vector<float> delta_mu_b;
    std::vector<float> delta_var_b;

    virtual int get_input_size();

    virtual int get_output_size();

    virtual void forward(HiddenStates &input_states,
                         HiddenStates &output_states, TempStates &temp_states);
    virtual void state_backward(std::vector<float> &jcb,
                                DeltaStates &input_delta_states,
                                DeltaStates &output_hidden_states,
                                TempStates &temp_states);
    virtual void param_backward(DeltaStates &delta_states,
                                TempStates &temp_states);

   protected:
    virtual void allocate_activation_bwd_vector(int size);
};
///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      November 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <fstream>
#include <iostream>
#include <vector>

#include "struct_var.h"

class BaseLayer {
   public:
    size_t input_size = 0, output_size = 0;
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
    bool training = true;

    BaseLayer();
    ~BaseLayer() = default;

    virtual const char *get_layer_type_name() const;

    virtual std::string get_layer_info() const = 0;

    virtual std::string get_layer_name() const = 0;

    virtual int get_input_size();

    virtual int get_output_size();

    virtual void forward(HiddenStates &input_states,
                         HiddenStates &output_states, TempStates &temp_states);
    virtual void state_backward(std::vector<float> &jcb,
                                DeltaStates &input_delta_states,
                                DeltaStates &output_hidden_states,
                                TempStates &temp_states);
    virtual void param_backward(std::vector<float> &mu_a,
                                DeltaStates &delta_states,
                                TempStates &temp_states);

    virtual void update_weights();

    virtual void update_biases();

    virtual void save(std::ofstream &file);
    virtual void load(std::ifstream &file);

   protected:
    virtual void allocate_bwd_vector(int size);
    virtual void fill_output_states(HiddenStates &output_states);
    virtual void fill_bwd_vector(HiddenStates &input_states);
};
///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      December 09, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <fstream>
#include <iostream>
#include <vector>

#include "data_struct.h"
#ifdef USE_CUDA
#include "data_struct_cuda.cuh"
#endif

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
    unsigned int num_threads = 1;
    bool training = true;
    std::string device = "cpu";

    BaseLayer();
    ~BaseLayer() = default;

    virtual const char *get_layer_type_name() const;

    virtual std::string get_layer_info() const = 0;

    virtual std::string get_layer_name() const = 0;

    int get_input_size();

    int get_output_size();

    virtual void forward(HiddenStateBase &input_states,
                         HiddenStateBase &output_states,
                         TempStateBase &temp_states);

    virtual void state_backward(std::vector<float> &jcb,
                                DeltaStateBase &input_delta_states,
                                DeltaStateBase &output_hidden_states,
                                TempStateBase &temp_states);

    virtual void param_backward(std::vector<float> &mu_a,
                                DeltaStateBase &delta_states,
                                TempStateBase &temp_states);

    virtual void update_weights();

    virtual void update_biases();

    virtual void save(std::ofstream &file);
    virtual void load(std::ifstream &file);

   protected:
    void allocate_bwd_vector(int size);
    void fill_output_states(HiddenStateBase &output_states);
    void fill_bwd_vector(HiddenStateBase &input_states);
};
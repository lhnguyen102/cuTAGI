///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      December 30, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "data_struct.h"

enum class LayerType { Base, Linear, CNN, LSTM, Activation };

class BaseLayer {
   public:
    size_t input_size = 0, output_size = 0;
    size_t num_weights = 0, num_biases = 0;
    std::vector<float> mu_w;
    std::vector<float> var_w;
    std::vector<float> mu_b;
    std::vector<float> var_b;
    std::vector<float> delta_mu_w;
    std::vector<float> delta_var_w;
    std::vector<float> delta_mu_b;
    std::vector<float> delta_var_b;
    std::unique_ptr<BaseBackwardStates> bwd_states;

    unsigned int num_threads = 1;
    bool training = true;
    std::string device = "cpu";

    BaseLayer();
    virtual ~BaseLayer() = default;

    // Delete copy constructor and copy assignment meaning this object class
    // cannot be copy
    BaseLayer(const BaseLayer &) = delete;
    BaseLayer &operator=(const BaseLayer &) = delete;

    // Optionally implement move constructor and move assignment
    BaseLayer(BaseLayer &&) = default;
    BaseLayer &operator=(BaseLayer &&) = default;

    virtual const char *get_layer_type_name() const;

    virtual std::string get_layer_info() const;

    virtual std::string get_layer_name() const;

    virtual LayerType get_layer_type() const;

    int get_input_size();

    int get_output_size();

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states);

    virtual void state_backward(BaseBackwardStates &next_bwd_states,
                                BaseDeltaStates &input_delta_states,
                                BaseDeltaStates &output_hidden_states,
                                BaseTempStates &temp_states);

    virtual void param_backward(BaseBackwardStates &next_bwd_states,
                                BaseDeltaStates &delta_states,
                                BaseTempStates &temp_states);

    virtual void update_weights();

    virtual void update_biases();

    virtual void save(std::ofstream &file);
    virtual void load(std::ifstream &file);
    virtual std::unique_ptr<BaseLayer> to_cuda() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };

   protected:
    void allocate_bwd_vector(int size);
    void fill_output_states(BaseHiddenStates &output_states);
    void fill_bwd_vector(BaseHiddenStates &input_states);
};
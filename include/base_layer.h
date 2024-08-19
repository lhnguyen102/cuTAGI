///////////////////////////////////////////////////////////////////////////////
// File:         base_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      April 26, 2024
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

enum class LayerType {
    Base,
    Linear,
    SLinear,
    Conv2d,
    ConvTranspose2d,
    Pool2d,
    LSTM,
    SLSTM,
    Activation,
    Norm,
    LayerBlock,
    ResNetBlock
};

class InitArgs {
   public:
    size_t width = 0;
    size_t height = 0;
    size_t depth = 0;
    int batch_size = 1;
    InitArgs(size_t width = 0, size_t height = 0, size_t depth = 0,
             int batch_size = 1);
    virtual ~InitArgs() = default;
};

class BaseLayer {
   public:
    size_t input_size = 0, output_size = 0;
    size_t num_weights = 0, num_biases = 0;
    size_t in_width = 0, in_height = 0, in_channels = 0;
    size_t out_width = 0, out_height = 0, out_channels = 0;
    bool bias = true;
    bool param_update = true;
    float cap_factor_update = 1.0f;

    std::vector<float> mu_w;
    std::vector<float> var_w;
    std::vector<float> mu_b;
    std::vector<float> var_b;
    std::vector<float> delta_mu_w;
    std::vector<float> delta_var_w;
    std::vector<float> delta_mu_b;
    std::vector<float> delta_var_b;
    std::unique_ptr<BaseBackwardStates> bwd_states;

    unsigned int num_threads = 4;
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

    virtual int get_input_size();

    virtual int get_output_size();

    virtual int get_max_num_states();

    virtual std::string get_device();

    virtual void init_weight_bias();

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states);

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte = true);

    virtual void allocate_param_delta();

    virtual void update_weights();

    virtual void update_biases();

    virtual void raw_update_weights();

    virtual void raw_update_biases();

    virtual void set_cap_factor_udapte(int batch_size);

    virtual void set_threads(int num);

    virtual void compute_input_output_size(const InitArgs &args);

    virtual void train();
    virtual void eval();

    void storing_states_for_training(BaseHiddenStates &input_states,
                                     BaseHiddenStates &output_states);

    virtual void save(std::ofstream &file);
    virtual void load(std::ifstream &file);

    // NOTE: each layer has its own conversion to cuda layer. The idea is to
    // move the ownership of the layer to cuda layer, so unique_ptr is used.
    virtual std::unique_ptr<BaseLayer> to_cuda() {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Cuda device is not available");
    };

    // DEBUG
    virtual std::tuple<std::vector<float>, std::vector<float>>
    get_running_mean_var();

    virtual void preinit_layer();

   protected:
    void allocate_bwd_vector(int size);
    void fill_output_states(BaseHiddenStates &output_states);
    void fill_bwd_vector(BaseHiddenStates &input_states);
};
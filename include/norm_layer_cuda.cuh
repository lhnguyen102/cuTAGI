///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      April 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_layer_cuda.cuh"

class LayerNormCuda : public BaseLayerCuda {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> mu_ra, var_ra;
    float *d_mu_ra = nullptr, *d_var_ra = nullptr;

    float epsilon;
    bool bias;
    int _batch_size = 0;

    LayerNormCuda(const std::vector<int> &normalized_shape, float eps = 1e-5,
                  bool bias = true);
    ~LayerNormCuda();

    // Delete copy constructor and copy assignment
    LayerNormCuda(const LayerNormCuda &) = delete;
    LayerNormCuda &operator=(const LayerNormCuda &) = delete;

    // Optionally implement move constructor and move assignment
    LayerNormCuda(LayerNormCuda &&) = default;
    LayerNormCuda &operator=(LayerNormCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    // DEBUG
    std::tuple<std::vector<float>, std::vector<float>> get_running_mean_var()
        override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

   protected:
    void allocate_running_mean_var();
    void deallocate_running_mean_var();
    void running_mean_var_to_host();
    void running_mean_var_to_device();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};

class BatchNorm2dCuda : public BaseLayerCuda {
   public:
    int num_features;
    std::vector<float> mu_ra, var_ra, mu_norm_batch, var_norm_batch;
    float *d_mu_ra, *d_var_ra, *d_mu_norm_batch, *d_var_norm_batch;
    float epsilon;
    float momentum;
    bool bias;

    // momentum of running average of first batch is set to zero
    bool first_batch = true;

    BatchNorm2dCuda(int num_features, float eps = 1e-5, float mometum = 0.9,
                    bool bias = true);
    ~BatchNorm2dCuda();

    // Delete copy constructor and copy assignment
    BatchNorm2dCuda(const BatchNorm2dCuda &) = delete;
    BatchNorm2dCuda &operator=(const BatchNorm2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    BatchNorm2dCuda(BatchNorm2dCuda &&) = default;
    BatchNorm2dCuda &operator=(BatchNorm2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias();

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

   protected:
    void allocate_running_mean_var();
    void deallocate_running_mean_var();
    void running_mean_var_to_host();
    void running_mean_var_to_device();
    void lazy_init();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};
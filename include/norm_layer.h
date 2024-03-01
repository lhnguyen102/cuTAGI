///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      February 04, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_layer.h"

std::tuple<int, int> get_number_params_layer_norm(
    const std::vector<int> &normalized_shape);

class LayerNorm : public BaseLayer {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> mu_ra, var_ra;
    float epsilon;
    float momentum;
    bool bias;

    LayerNorm(const std::vector<int> &normalized_shape, float eps = 1e-4,
              float mometum = 0.9, bool bias = true);
    ~LayerNorm();

    // Delete copy constructor and copy assignment
    LayerNorm(const LayerNorm &) = delete;
    LayerNorm &operator=(const LayerNorm &) = delete;

    // Optionally implement move constructor and move assignment
    LayerNorm(LayerNorm &&) = default;
    LayerNorm &operator=(LayerNorm &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias();

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void state_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_delta_states,
                        BaseTempStates &temp_states) override;

    void param_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &delta_states,
                        BaseTempStates &temp_states) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    // DEBUG
    std::tuple<std::vector<float>, std::vector<float>> get_running_mean_var()
        override;

   protected:
    void allocate_param_delta();
    void allocate_running_mean_var(int batch_size);
};

class BatchNorm2d : public BaseLayer {
   public:
    int num_features;
    std::vector<float> mu_ra, var_ra;
    float epsilon;
    float momentum;
    bool bias;

    BatchNorm2d(int num_features, float eps = 1e-4, float mometum = 0.9,
                bool bias = true);
    ~BatchNorm2d();

    // Delete copy constructor and copy assignment
    BatchNorm2d(const BatchNorm2d &) = delete;
    BatchNorm2d &operator=(const BatchNorm2d &) = delete;

    // Optionally implement move constructor and move assignment
    BatchNorm2d(BatchNorm2d &&) = default;
    BatchNorm2d &operator=(BatchNorm2d &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias();

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void state_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_delta_states,
                        BaseTempStates &temp_states) override;

    void param_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &delta_states,
                        BaseTempStates &temp_states) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

   protected:
    void allocate_param_delta();
    void allocate_running_mean_var();
};
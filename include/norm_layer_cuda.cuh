///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 24, 2024
// Updated:      January 29, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "base_layer_cuda.cuh"

class LayerNormCuda : public : BaseLayerCuda {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> mu_ra, var_ra;
    float *d_mu_ra, *d_var_ra;
    float epsilon;
    float momentum;
    bool bias;

    LayerNormCuda(const std::vector<int> &normalized_shape, float eps = 1e-5,
                  float mometum = 0.9, bool bias = true);
    ~LayerNormCuda;

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

    std::unique_ptr<BaseLayer> to_host() override;

   protected:
    void allocate_param_delta();
    void allocate_running_mean_var(int batch_size);
    void running_mean_var_to_host();
    void running_mean_var_to_device();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};
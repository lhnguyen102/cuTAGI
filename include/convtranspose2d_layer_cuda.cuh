///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      March 10, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "base_layer_cuda.cuh"

class ConvTranspose2dCuda : public BaseLayerCuda {
   public:
    float gain_w;
    float gain_b;

    ConvTranspose2dCuda();
    ~ConvTranspose2dCuda();

    // Delete copy constructor and copy assignment
    ConvTranspose2dCuda(const ConvTranspose2dCuda &) = delete;
    ConvTranspose2dCuda &operator=(const ConvTranspose2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    ConvTranspose2dCuda(ConvTranspose2dCuda &&) = default;
    ConvTranspose2dCuda &operator=(ConvTranspose2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

    void get_number_param();

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

    void preinit_layer() override;

   protected:
    void allocate_param_delta();
    void allocate_conv_index();
    void conv_index_to_device();
    void lazy_index_init();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};
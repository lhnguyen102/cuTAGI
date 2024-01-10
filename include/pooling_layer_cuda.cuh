///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 08, 2024
// Updated:      January 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "base_layer_cuda.cuh"

class AvgPool2dCuda : public BaseLayerCuda {
   public:
    size_t kernel_size = 0;
    size_t in_channels = 0;
    size_t out_channels = 0;
    int stride = 0;
    int padding_type = 1;
    int padding = 0;

    int *d_pool_idx, *d_z_ud_idx;

    AvgPool2dCuda(size_t kernel_size, int stride = -1, int padding = 0,
                  int padding_type = 1);

    ~AvgPool2dCuda();

    // Delete copy constructor and copy assignment
    AvgPool2dCuda(const AvgPool2dCuda &) = delete;
    AvgPool2dCuda &operator=(const AvgPool2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    AvgPool2dCuda(AvgPool2dCuda &&) = default;
    AvgPool2dCuda &operator=(AvgPool2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void state_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_hidden_states,
                        BaseTempStates &temp_states) override;

    void param_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &delta_states,
                        BaseTempStates &temp_states) override;

   protected:
    void lazy_init(size_t width, size_t height, size_t depth, int batch_size);
};
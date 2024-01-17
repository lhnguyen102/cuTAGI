///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 08, 2024
// Updated:      January 14, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "base_layer.h"

struct Pool2dIndex {
    std::vector<int> pool_idx, z_ud_idx;
    int w, h;
};

Pool2dIndex get_pool_index(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out);

class AvgPool2d : public BaseLayer {
   public:
    size_t kernel_size = 0;
    int stride = 0;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> pool_idx, z_ud_idx;
    size_t row_zw = 0, col_z_ud = 0;
    bool overlap = true;

    AvgPool2d(size_t kernel_size, int stride = -1, int padding = 0,
              int padding_type = 0);

    virtual ~AvgPool2d();

    // Delete copy constructor and copy assignment
    AvgPool2d(const AvgPool2d &) = delete;
    AvgPool2d &operator=(const AvgPool2d &) = delete;

    // Optionally implement move constructor and move assignment
    AvgPool2d(AvgPool2d &&) = default;
    AvgPool2d &operator=(AvgPool2d &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

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

    void update_weights() override{};

    void update_biases() override{};

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
   protected:
    void lazy_init(size_t width, size_t height, int batch_size);
};
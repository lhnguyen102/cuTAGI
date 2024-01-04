///////////////////////////////////////////////////////////////////////////////
// File:         conv2d_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 03, 2024
// Updated:      January 04, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "base_layer.h"

struct Conv2dIndex {
    std::vector<int> Fmwa_2_idx, FCzwa_1_idx, FCzwa_2_idx, Szz_ud_idx;
    int w, h;
};
std::tuple<int, int> compute_downsample_img_size_v2(int kernel, int stride,
                                                    int wi, int hi, int pad,
                                                    int pad_type);

std::tuple<int, int> get_number_param_conv_v2(int kernel, int fi, int fo,
                                              bool use_bias);

Conv2dIndex get_conv2d_idx(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out, int param_pad_idx);

class Conv2d : public BaseLayer {
   public:
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t in_channels = 0;
    size_t out_channels = 0;
    size_t kernel_size = 0;
    int stride = 1;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> idx_mwa_2;
    std::vector<int> idx_cov_zwa_1;
    std::vector<int> idx_var_z_ud;
    int row_zw = 0, col_z_ud = 0;

    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           int stride = 1, int padding = 0, int padding_type = 1,
           float gain_w = 1.0f, float gain_b = 1.0f,
           std::string init_method = "He", bool bias = true);
    ~Conv2d();

    // Delete copy constructor and copy assignment
    Conv2d(const Conv2d &) = delete;
    Conv2d &operator=(const Conv2d &) = delete;

    // Optionally implement move constructor and move assignment
    Conv2d(Conv2d &&) = default;
    Conv2d &operator=(Conv2d &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void get_number_param_conv2d(int kernel, int fi, int fo, bool use_bias);

    void init_weight_bias();

    void allocate_param_delta();

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

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};
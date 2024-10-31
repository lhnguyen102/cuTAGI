///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      April 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "conv2d_layer.h"

struct ConvTranspose2dIndex {
    std::vector<int> FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx;
    int w_wz, h_wz, w_zz, h_zz;
};

ConvTranspose2dIndex get_tconv_idx(int pad_idx_in, int pad_idx_out,
                                   int param_pad_idx, Conv2dIndex &convIndex);

std::tuple<int, int> compute_upsample_img_size_v2(int kernel, int stride,
                                                  int wi, int hi, int pad,
                                                  int pad_type);

class ConvTranspose2d : public BaseLayer {
   public:
    std::string init_method;
    size_t kernel_size = 0;
    int stride = 1;
    int padding_type = 1;
    int padding = 0;
    float gain_w;
    float gain_b;
    std::vector<int> idx_mwa_1;
    std::vector<int> idx_mwa_2;
    std::vector<int> idx_cov_wz_2;
    std::vector<int> idx_var_wz_ud;
    std::vector<int> idx_cov_z_wa_1;
    std::vector<int> idx_var_z_ud;
    int row_zw = 0, col_z_ud = 0;
    int col_cov_mwa_1 = 0;

    ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_size,
                    bool bias = true, int stride = 1, int padding = 0,
                    int padding_type = 1, size_t in_width = 0,
                    size_t in_height = 0, float gain_w = 1.0f,
                    float gain_b = 1.0f, std::string init_method = "He");
    ~ConvTranspose2d();

    // Delete copy constructor and copy assignment
    ConvTranspose2d(const ConvTranspose2d &) = delete;
    ConvTranspose2d &operator=(const ConvTranspose2d &) = delete;

    // Optionally implement move constructor and move assignment
    ConvTranspose2d(ConvTranspose2d &&) = default;
    ConvTranspose2d &operator=(ConvTranspose2d &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

    void get_number_param();

    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    using BaseLayer::storing_states_for_training;
    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void preinit_layer() override;

   protected:
    void lazy_index_init();
};
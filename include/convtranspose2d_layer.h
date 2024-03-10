///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      March 10, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector.h>

#include "conv2d_layer.h"

struct ConvTranspose2dIndex {
    std::vector<int> FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx;
    int w_wz, h_wz, w_zz, h_zz;
};

ConvTranspose2dIndex get_tconv_idx(int kernel, int wi, int hi, int wo, int ho,
                                   int pad_idx_in, int pad_idx_out,
                                   int param_pad_idx, Conv2dIndex &convIndex);

class ConvTranspose2d : public BaseLayer {
   public:
    float gain_w;
    float gain_b;

    ConvTranspose2d();
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

    using BaseLayer::storing_states_for_training;
    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void preinit_layer() override;

   protected:
    void allocate_param_delta();
    void lazy_index_init();
}
///////////////////////////////////////////////////////////////////////////////
// File:         conv2d_layer.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 03, 2024
// Updated:      April 18, 2024
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
    size_t kernel_size = 0;
    int stride = 1;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> idx_mwa_2;
    std::vector<int> idx_cov_zwa_1;
    std::vector<int> idx_var_z_ud;
    int row_zw = 0, col_z_ud = 0;

    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           bool bias = true, int stride = 1, int padding = 0,
           int padding_type = 1, size_t in_width = 0, size_t in_height = 0,
           float gain_w = 1.0f, float gain_b = 1.0f,
           std::string init_method = "He");
    virtual ~Conv2d();

    // Delete copy constructor and copy assignment
    Conv2d(const Conv2d &) = delete;
    Conv2d &operator=(const Conv2d &) = delete;

    // Optionally implement move constructor and move assignment
    Conv2d(Conv2d &&) = default;
    Conv2d &operator=(Conv2d &&) = default;

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

////////////////////////////////////////////////////////////////////////////////
// Conv2d Backward and Forward
////////////////////////////////////////////////////////////////////////////////
void conv2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &aidx, int woho, int fo, int wihi, int fi, int ki,
    int batch_size, int pad_idx, bool bias, int start_chunk, int end_chunk,
    std::vector<float> &mu_z, std::vector<float> &var_z);

void conv2d_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &zw_idx,
    const std::vector<int> &zud_idx, int woho, int fo, int wihi, int fi, int ki,
    int nr, int n, int batch_size, int pad_idx, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu, std::vector<float> &delta_var);

void permute_jacobian(std::vector<float> &jcb_0, int wihi, int fi,
                      int batch_size, std::vector<float> &jcb);

void conv2d_bwd_delta_w(const std::vector<float> &var_w,
                        const std::vector<float> &mu_a,
                        const std::vector<float> &delta_mu_out,
                        const std::vector<float> &delta_var_out,
                        const std::vector<int> &aidx, int batch_size, int k,
                        int woho, int wihi, int fi, int ki, int pad_idx,
                        int start_chunk, int end_chunk,
                        std::vector<float> &delta_mu_w,
                        std::vector<float> &delta_var_w);

void conv2d_bwd_delta_b(const std::vector<float> &var_b,
                        const std::vector<float> &delta_mu_out,
                        const std::vector<float> &delta_var_out, int n, int k,
                        std::vector<float> &delta_mu_b,
                        std::vector<float> &delta_var_b);

void permute_delta(const std::vector<float> &delta_mu_0,
                   const std::vector<float> &delta_var_0, int woho, int kp,
                   int batch_size, std::vector<float> &delta_mu,
                   std::vector<float> &delta_var);

void conv2d_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<int> &aidx, int woho, int fo, int wihi, int fi, int ki,
    int batch_size, int pad_idx, bool bias, unsigned int num_threads,
    std::vector<float> &mu_z, std::vector<float> &var_z);

void conv2d_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, const std::vector<int> &zw_idx,
    const std::vector<int> &zud_idx, int woho, int fo, int wihi, int fi, int ki,
    int nr, int n, int batch_size, int pad_idx, unsigned int num_threads,
    std::vector<float> &delta_mu, std::vector<float> &delta_var);

void conv2d_bwd_delta_w_mp(const std::vector<float> &var_w,
                           const std::vector<float> &mu_a,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           const std::vector<int> &aidx, int batch_size, int k,
                           int woho, int wihi, int fi, int ki, int pad_idx,
                           unsigned int num_threads,
                           std::vector<float> &delta_mu_w,
                           std::vector<float> &delta_var_w);
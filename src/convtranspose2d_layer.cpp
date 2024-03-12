///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      March 10, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/convtranspose2d_layer.h"

#include "../include/base_layer.h"
#include "../include/common.h"
#include "../include/indices.h"
#include "../include/param_init.h"

ConvTranspose2dIndex get_tconv_idx(int pad_idx_in, int pad_idx_out,
                                   int param_pad_idx, Conv2dIndex &convIndex)
/**/
{
    // Initialize pointers
    std::vector<int> FCwz_2_idx, Swz_ud_idx;
    std::vector<int> FCzwa_1_idx, Szz_ud_idx;
    std::vector<int> FCzwa_1_idx_t, FCzwa_2_idx_t, Szz_ud_idx_t, tmp_1, tmp_2,
        tmp_3, tmp_4;
    int pad = 1;

    // Transpose convolutional index matrix
    FCzwa_1_idx_t =
        transpose_matrix(convIndex.FCzwa_1_idx, convIndex.w, convIndex.h);
    FCzwa_2_idx_t =
        transpose_matrix(convIndex.FCzwa_2_idx, convIndex.w, convIndex.h);
    Szz_ud_idx_t =
        transpose_matrix(convIndex.Szz_ud_idx, convIndex.w, convIndex.h);

    ///////////////////////////////////////////
    /* Indices for FCwz 2 and Swz ud */
    // Get unique indices and its frequency of the receptive field
    auto FCwz_1 = get_ref_idx(convIndex.FCzwa_1_idx, pad, param_pad_idx);

    // Get indices for FCwz_2
    tmp_1 = reorganize_idx_from_ref(Szz_ud_idx_t, pad, FCwz_1.pad_pos,
                                    pad_idx_out, FCwz_1.ref, FCwz_1.base_idx,
                                    FCwz_1.w, FCwz_1.h);
    FCwz_2_idx = transpose_matrix(tmp_1, FCwz_1.w, FCwz_1.h);

    // Get indices for Swz ud
    tmp_2 = reorganize_idx_from_ref(FCzwa_2_idx_t, pad, FCwz_1.pad_pos,
                                    pad_idx_in, FCwz_1.ref, FCwz_1.base_idx,
                                    FCwz_1.w, FCwz_1.h);

    Swz_ud_idx = transpose_matrix(tmp_2, FCwz_1.w, FCwz_1.h);

    //////////////////////////////////////////
    /* Indices for FCzz 2 and Szz ud */
    // Get unique indices and its frequency of the receptive field
    auto Szz_ud = get_ref_idx(Szz_ud_idx_t, pad, pad_idx_out);

    // Get indices for FCwz_2
    tmp_3 = reorganize_idx_from_ref(convIndex.FCzwa_1_idx, pad, Szz_ud.pad_pos,
                                    param_pad_idx, Szz_ud.ref, Szz_ud.base_idx,
                                    Szz_ud.w, Szz_ud.h);

    FCzwa_1_idx = transpose_matrix(tmp_3, Szz_ud.w, Szz_ud.h);

    // Get indices for Szz ud
    tmp_4 = reorganize_idx_from_ref(FCzwa_2_idx_t, pad, Szz_ud.pad_pos,
                                    pad_idx_in, Szz_ud.ref, Szz_ud.base_idx,
                                    Szz_ud.w, Szz_ud.h);

    Szz_ud_idx = transpose_matrix(tmp_4, Szz_ud.w, Szz_ud.h);

    return {FCwz_2_idx, Swz_ud_idx, FCzwa_1_idx, Szz_ud_idx,
            FCwz_1.w,   FCwz_1.h,   Szz_ud.w,    Szz_ud.h};
}

std::tuple<int, int> compute_upsample_img_size_v2(int kernel, int stride,
                                                  int wi, int hi, int pad,
                                                  int pad_type)
/* Compute the size of upsampling images i.e. increase of image size.
 *
 * Args:
 *    Kernel: size of the receptive field
 *    stride: Stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    pad: Number of paddings
 *    pad_type: Padding type
 *
 * Returns:
 *    wo: Width of the output image
 *    ho: Height of the output image
 *    */
{
    int wo, ho, nom_w, nom_h;
    // Compute nominator of tconv. formulation given a padding type
    if (pad_type == 1) {
        wo = stride * (wi - 1) + kernel - 2 * pad;
        ho = stride * (hi - 1) + kernel - 2 * pad;
        nom_w = wo - kernel + 2 * pad;
        nom_h = ho - kernel + 2 * pad;
    }

    // Check validity of the conv. hyper-parameters such as wi, hi, kernel,
    // stride
    else if (pad_type == 2) {
        wo = stride * (wi - 1) + kernel - pad;
        ho = stride * (hi - 1) + kernel - pad;
        nom_w = wo - kernel + pad;
        nom_h = ho - kernel + pad;
    }

    if (nom_w % stride != 0 || nom_h % stride != 0) {
        throw std::invalid_argument(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Invalid hyperparameters for ConvTranspose2d layer");
    }

    return {wo, ho};
}

ConvTranspose2d::ConvTranspose2d(size_t in_channels, size_t out_channels,
                                 size_t kernel_size, bool bias, int stride,
                                 int padding, int padding_type, size_t in_width,
                                 size_t in_height, float gain_w, float gain_b,
                                 std::string init_method)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/**/
{
    this->in_width = in_width;
    this->in_height = in_height;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->bias = bias;

    if (this->training) {
        this->bwd_states = std::make_unique<BaseBackwardStates>();
    }
}

ConvTranspose2d::~ConvTranspose2d() {}

std::string ConvTranspose2d::get_layer_name() const {
    return "ConvTranspose2d()";
}

std::string ConvTranspose2d::get_layer_info() const {
    return "ConvTranspose2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->out_width) + "," +
           std::to_string(this->out_height) + "," +
           std::to_string(this->kernel_size) + ")";
}

LayerType ConvTranspose2d::get_layer_type() const {
    return LayerType::ConvTranspose2d;
};

void ConvTranspose2d::compute_input_output_size(const InitArgs &args)
/*
 */
{
    this->in_width = args.width;
    this->in_height = args.height;
    std::tie(this->out_width, this->out_height) = compute_upsample_img_size_v2(
        this->kernel_size, this->stride, this->in_width, this->in_height,
        this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void ConvTranspose2d::get_number_param()
/*
 */
{
    this->num_weights =
        this->kernel_size * this->in_channels * this->out_channels;
    this->num_bias = 0;
    if (this->bias) {
        this->num_bias = this->out_channels;
    }
}

ConvTranspose2d::init_weight_bias()
/**/
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
}

void ConvTranspose2d::lazy_index_init()
/*
 */
{
    int ki2 = this->kernel_size * this->kernel_size;
    int param_pad_idx = ki2 * this->in_channels * this->out_channels + 1;

    auto conv_idx = get_conv2d_idx(
        this->kernel_size, this->stride, this->out_width, this->out_height,
        this->in_width, this->in_height, this->padding, this->padding_type, -1,
        -1, param_pad_idx);

    auto conv_transpose_idx = get_tconv_idx(-1, -1, param_pad_idx, conv_idx);

    this->idx_mwa_1 = conv_idx.FCzwa_1_idx;
    this->idx_mwa_2 =
        transpose_matrix(conv_index.Szz_ud_idx, conv_index.w, conv_index.h);
    this->idx_cov_wz_2 = conv_transpose_idx.FCwz_2_idx;
    this->idx_var_wz_ud = conv_transpose_idx.Swz_ud_idx;
    this->idx_cov_z_wa_1 = conv_transpose_idx.FCzwa_1_idx;
    this->idx_var_z_ud = conv_transpose_idx.Szz_ud_idx;

    // Dimension
    this->row_zw = conv_transpose_idx.w_wz;
    this->col_z_ud = conv_transpose_idx.w_zz;
    this->col_cov_mwa_1 = conv_transpose_idx.h;
}

void ConvTranspose2d::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{}

void ConvTranpose2d::state_backward(BaseBackwardStates &next_bwd_states,
                                    BaseDeltaStates &input_delta_states,
                                    BaseDeltaStates &output_delta_states,
                                    BaseTempStates &temp_states)
/*
 */
{}

void ConvTranpose2d::param_backward(BaseBackwardStates &next_bwd_states,
                                    BaseDeltaStates &delta_states,
                                    BaseTempStates &temp_states)
/**/
{}

void ConvTranpose2d::preinit_layer() {
    this->get_number_param_conv2d();
    this->init_weight_bias();
    this->lazy_index_init();
}

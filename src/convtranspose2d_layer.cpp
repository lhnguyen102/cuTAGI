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

ConvTranspose2dIndex get_tconv_idx(int kernel, int wi, int hi, int wo, int ho,
                                   int pad_idx_in, int pad_idx_out,
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

ConvTranspose2d::ConvTranspose2d() {}
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
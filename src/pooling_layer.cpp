///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 08, 2024
// Updated:      January 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/pooling_layer.h"

#include "../include/indices.h"
#ifdef USE_CUDA
#include "../include/pooling_layer_cuda.cuh"
#endif

AvgPool2d::AvgPool2d(size_t kernel_size, int stride, int padding,
                     int padding_type)
    : kernel_size(kernel_size),
      stride(stride),
      padding_type(padding_type),
      padding(padding) {}

AvgPool2d::~AvgPool2d() {}

std::string AvgPool2d::get_layer_info() const {
    return "AvgPool2d(" + std::to_string(this->kernel_size) + ")";
    ;
}

std::string AvgPool2d::get_layer_name() const { return "AvgPool2d"; }

LayerType AvgPool2d::get_layer_type() const { return LayerType::AvgPool2d; }

void AvgPool2d::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/*
 */
{
    throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                " at line: " + std::to_string(__LINE__) +
                                ". AvgPool2d forward is unavaialble on CPU");
}

void AvgPool2d::state_backward(BaseBackwardStates &next_bwd_states,
                               BaseDeltaStates &input_delta_states,
                               BaseDeltaStates &output_hidden_states,
                               BaseTempStates &temp_states)
/*
 */
{
    throw std::invalid_argument(
        "Error in file: " + std::string(__FILE__) +
        " at line: " + std::to_string(__LINE__) +
        ". AvgPool2d state backward is unavaialble on CPU");
}

void AvgPool2d::param_backward(BaseBackwardStates &next_bwd_states,
                               BaseDeltaStates &delta_states,
                               BaseTempStates &temp_states)
/*
 */
{
    throw std::invalid_argument(
        "Error in file: " + std::string(__FILE__) +
        " at line: " + std::to_string(__LINE__) +
        ". AvgPool2d param backward is unavaialble on CPU");
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> AvgPool2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<AvgPool2dCuda>(this->kernel_size, this->stride,
                                           this->padding, this->padding_type);
}
#endif

void AvgPool2d::lazy_init(size_t width, size_t height, int batch_size) {}

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

Pool2dIndex get_pool_index(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out) {
    // Initialize pointers
    std::vector<int> raw_img, img, padded_img, Fmwa_2_idx, tmp;
    std::vector<int> Szz_ud_idx;
    RefIndexOut Fmwa_2;
    int w_p, h_p;

    // Generate image indices
    std::tie(raw_img, img, padded_img, w_p, h_p) =
        image_construction(wi, hi, pad, pad_idx_in, pad_type);

    // Get indices for receptive field
    tmp =
        get_receptive_field(img, padded_img, kernel, stride, wo, ho, w_p, h_p);
    if (!(kernel == stride || (kernel == wi && stride == 1))) {
        // Get unique indices and its frequency of the receptive field
        Fmwa_2 = get_ref_idx(tmp, pad, pad_idx_in);

        // Get indices for Szz ud
        Szz_ud_idx =
            get_Szz_ud_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                           Fmwa_2.base_idx, pad_idx_out, Fmwa_2.w, Fmwa_2.h);
    }

    // NOTE THAT DOUBLE CHECK WHY WE NEED THE TRANSPOSE HEAR AND SIZE OF MATRIX
    Fmwa_2_idx = transpose_matrix(tmp, kernel * kernel, wo * ho);

    return {Fmwa_2_idx, Szz_ud_idx, Fmwa_2.w, Fmwa_2.h};
}
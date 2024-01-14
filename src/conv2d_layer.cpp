///////////////////////////////////////////////////////////////////////////////
// File:         conv2d_layer.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 04, 2024
// Updated:      January 14, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/conv2d_layer.h"

#include "../include/indices.h"
#include "../include/param_init.h"
#ifdef USE_CUDA
#include "../include/conv2d_layer_cuda.cuh"
#endif

////////////////////////////////////////////////////////////////////////////////
/// Conv2d
////////////////////////////////////////////////////////////////////////////////
Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
               int stride, int padding, int padding_type, size_t in_width,
               size_t in_height, float gain_w, float gain_b,
               std::string init_method, bool bias)
    : kernel_size(kernel_size),
      stride(stride),
      padding(padding),
      padding_type(padding_type),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/*
 */
{
    this->in_width = in_width;
    this->in_height = in_height;
    this->in_channels = in_channels;
    this->out_channels = out_channels;
    this->bias = bias;

    // if (in_width != 0 && in_height != 0) {
    //     InitArgs args(in_width, in_height);
    //     this->compute_input_output_size(args);
    // }
}

Conv2d::~Conv2d() {}

std::string Conv2d::get_layer_info() const { return "Conv2d()"; }

std::string Conv2d::get_layer_name() const {
    return "Conv2d(" + std::to_string(this->in_channels) + "," +
           std::to_string(this->out_channels) + "," +
           std::to_string(this->kernel_size) + ")";
}

LayerType Conv2d::get_layer_type() const { return LayerType::Conv2d; };

void Conv2d::compute_input_output_size(const InitArgs &args)
/*
 */
{
    this->in_width = args.width;
    this->in_height = args.height;
    std::tie(this->out_width, this->out_height) =
        compute_downsample_img_size_v2(this->kernel_size, this->stride,
                                       this->in_width, this->in_height,
                                       this->padding, this->padding_type);

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
}

void Conv2d::get_number_param_conv2d(int kernel, int fi, int fo, bool use_bias)

/* Get the number of parameters for conv. and tconv. layer.
 *
 * Args:
 *    kernel: Size of the receptive field
 *    fi: Number of filters for input image
 *    fo: Number of filters for output image
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = kernel * kernel * fi * fo;
    if (use_bias) {
        n_b = fo;
    } else {
        n_b = 0;
    }
    this->num_weights = n_w;
    this->bias = n_b;
}

void Conv2d::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_conv2d(this->kernel_size, this->in_channels,
                                this->out_channels, this->init_method,
                                this->gain_w, this->gain_b, this->num_weights,
                                this->num_biases);
}

void Conv2d::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights);
    this->delta_var_w.resize(this->num_weights);
    this->delta_mu_b.resize(this->num_biases);
    this->delta_var_b.resize(this->num_biases);
}

void Conv2d::forward(BaseHiddenStates &input_states,
                     BaseHiddenStates &output_states,
                     BaseTempStates &temp_states)
/*
 */
{
    throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                " at line: " + std::to_string(__LINE__) +
                                ". Conv2d forward is unavaialble on CPU");
}

void Conv2d::state_backward(BaseBackwardStates &next_bwd_states,
                            BaseDeltaStates &input_delta_states,
                            BaseDeltaStates &output_hidden_states,
                            BaseTempStates &temp_states)
/*
 */
{
    throw std::invalid_argument(
        "Error in file: " + std::string(__FILE__) +
        " at line: " + std::to_string(__LINE__) +
        ". Conv2d state backward is unavaialble on CPU");
}

void Conv2d::param_backward(BaseBackwardStates &next_bwd_states,
                            BaseDeltaStates &delta_states,
                            BaseTempStates &temp_states)
/*
 */
{
    throw std::invalid_argument(
        "Error in file: " + std::string(__FILE__) +
        " at line: " + std::to_string(__LINE__) +
        ". Conv2d param backward is unavaialble on CPU");
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Conv2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<Conv2dCuda>(
        this->in_channels, this->out_channels, this->kernel_size, this->stride,
        this->padding, this->padding_type, this->in_width, this->in_height,
        this->gain_w, this->gain_b, this->init_method, this->bias);
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Utility functions
////////////////////////////////////////////////////////////////////////////////

std::tuple<int, int> compute_downsample_img_size_v2(int kernel, int stride,
                                                    int wi, int hi, int pad,
                                                    int pad_type)
/* compute the size of downsampling images i.e. reduction of image size
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: width of the input image
 *    hi: height of the input image
 *    pad: number of paddings
 *    pad_type: padding type
 *
 * returns:
 *    wo: width of the output image
 *    ho: height of the output image
 *    */
{
    int wo, ho, nom_w, nom_h;

    // Compute nominator of conv. formulation given a padding type
    if (pad_type == 1) {
        nom_w = wi - kernel + 2 * pad;
        nom_h = hi - kernel + 2 * pad;
    } else if (pad_type == 2) {
        nom_w = wi - kernel + pad;
        nom_h = hi - kernel + pad;
    } else {
        nom_w = wi - kernel;
        nom_h = hi - kernel;
    }

    // Check validity of the conv. hyper-parameters such as wi, hi, kernel,
    // stride

    if (nom_w % stride == 0 && nom_h % stride == 0) {
        wo = nom_w / stride + 1;
        ho = nom_h / stride + 1;
    } else {
        throw std::invalid_argument(
            "Error in file: " + std::string(__FILE__) +
            " at line: " + std::to_string(__LINE__) +
            ". Invalid hyperparameters for conv2d layer");
    }

    return {wo, ho};
}

std::tuple<int, int> get_number_param_conv_v2(int kernel, int fi, int fo,
                                              bool use_bias)
/* Get the number of parameters for conv. and tconv. layer.
 *
 * Args:
 *    kernel: Size of the receptive field
 *    fi: Number of filters for input image
 *    fo: Number of filters for output image
 *    use_bias: Whether to include the bias parameters.
 *
 * Returns:
 *    n_w: Number of weight paramerers
 *    n_b: Number of bias parameters
 *    */
{
    int n_w, n_b;
    n_w = kernel * kernel * fi * fo;
    if (use_bias) {
        n_b = fo;
    } else {
        n_b = 0;
    }

    return {n_w, n_b};
}

Conv2dIndex get_conv2d_idx(int kernel, int stride, int wi, int hi, int wo,
                           int ho, int pad, int pad_type, int pad_idx_in,
                           int pad_idx_out, int param_pad_idx)
/*
 * Get index matrices for convolutional layer.
 *
 * Args:
 *    kernel: size of the receptive field
 *    stride: stride for the receptive field
 *    wi: Width of the input image
 *    hi: Height of the input image
 *    wo: width of the output image
 *    ho: height of the output image
 *    pad: Number of padding
 *    pad_type: Type of paddings
 *    pad_idx_in: Padding index for the input image
 *    pad_idx_out: Index for the padding of the output image
 *    param_pad_idx: Index for the padding of the parameters
 *
 * Returns:
 *    FCzwa_1_idx: Index for the parameters sorted as the input hidden state
 *      ordering
 *    FCzwa_2_idx: Index for the receptive field indices sorted as the input
 *      hidden state ordering
 *    Szz_ud_idx: Index for the output hidden states sorted as the input
 *      hidden state ordering
 *    w: Width of three above-mentioned index matrix
 *    h: Height of three above_mentioned idex matrix
 * */
{
    // Initialize pointers
    std::vector<int> raw_img, img, padded_img, Fmwa_2_idx;
    std::vector<int> FCzwa_1_idx, FCzwa_2_idx, Szz_ud_idx, tmp;
    int w_p, h_p, num_elements;

    // Generate image indices
    std::tie(raw_img, img, padded_img, w_p, h_p) =
        image_construction(wi, hi, pad, pad_idx_in, pad_type);

    // Get indices for receptive field
    Fmwa_2_idx =
        get_receptive_field(img, padded_img, kernel, stride, wo, ho, w_p, h_p);

    // Get unique indices and its frequency of the receptive field
    auto Fmwa_2 = get_ref_idx(Fmwa_2_idx, pad, pad_idx_in);

    // Get indices for FCzwa 1
    tmp = get_FCzwa_1_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                          Fmwa_2.base_idx, param_pad_idx, Fmwa_2.w, Fmwa_2.h);

    FCzwa_1_idx = transpose_matrix(tmp, Fmwa_2.w, Fmwa_2.h);

    // Get indices for FCzwa 2
    FCzwa_2_idx = get_FCzwa_2_idx(Fmwa_2_idx, pad, pad_idx_in, Fmwa_2.ref,
                                  Fmwa_2.base_idx, Fmwa_2.w, Fmwa_2.h);

    // Get indices for Szz ud
    Szz_ud_idx =
        get_Szz_ud_idx(kernel, wo, ho, pad, Fmwa_2.pad_pos, Fmwa_2.ref,
                       Fmwa_2.base_idx, pad_idx_out, Fmwa_2.w, Fmwa_2.h);

    return {Fmwa_2_idx, FCzwa_1_idx, FCzwa_2_idx,
            Szz_ud_idx, Fmwa_2.w,    Fmwa_2.h};
}

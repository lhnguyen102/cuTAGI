
///////////////////////////////////////////////////////////////////////////////
// File:         main_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      August 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/main_bindings.h"

PYBIND11_MODULE(cutagi, modo) {
    modo.doc() =
        "Tractable Approximate Gaussian Inference - Backend C++/CUDA  ";

    bind_base_hidden_states(modo);
    bind_base_delta_states(modo);
    bind_hrcsoftmax(modo);
    bind_base_layer(modo);
    bind_relu(modo);
    bind_sigmoid(modo);
    bind_tanh(modo);
    bind_mixture_relu(modo);
    bind_mixture_sigmoid(modo);
    bind_mixture_tanh(modo);
    bind_softplus(modo);
    bind_leakyrelu(modo);
    bind_softmax(modo);
    bind_even_exp(modo);
    bind_linear_layer(modo);
    bind_slinear_layer(modo);
    bind_conv2d_layer(modo);
    bind_convtranspose2d_layer(modo);
    bind_avgpool2d_layer(modo);
    bind_layernorm_layer(modo);
    bind_batchnorm_layer(modo);
    bind_lstm_layer(modo);
    bind_slstm_layer(modo);
    bind_layer_block(modo);
    bind_resnet_block(modo);
    bind_sequential(modo);
    bind_output_updater(modo);
    bind_utils(modo);
}

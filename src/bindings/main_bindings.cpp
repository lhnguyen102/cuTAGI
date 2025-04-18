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
    bind_maxpool2d_layer(modo);
    bind_layernorm_layer(modo);
    bind_batchnorm_layer(modo);
    bind_lstm_layer(modo);
    bind_slstm_layer(modo);
    bind_layer_block(modo);
    bind_resnet_block(modo);
    bind_sequential(modo);
    bind_distributed(modo);
    bind_output_updater(modo);
    bind_utils(modo);
    bind_manual_seed(modo);
    bind_is_cuda_available(modo);
    bind_cuda_device_memory(modo);
    bind_cuda_device_properties(modo);
    bind_cuda_device_count(modo);
    bind_cuda_current_device(modo);
    bind_cuda_set_device(modo);
    bind_cuda_device_available(modo);
    bind_nccl_available(modo);
}

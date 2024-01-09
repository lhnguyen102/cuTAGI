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
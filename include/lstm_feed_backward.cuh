///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_backward.cuh
// Description:  Header file for Long-Short Term Memory (LSTM) state backward
//               pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      September 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "lstm_feed_forward.cuh"
#include "net_prop.h"
#include "struct_var.h"

void lstm_state_update(Network &net, StateGPU &state, ParamGPU &theta,
                       DeltaStateGPU &d_state, int l);

void lstm_parameter_update(Network &net, StateGPU &state, ParamGPU &theta,
                           DeltaStateGPU &d_state, DeltaParamGPU &d_theta,
                           int l);
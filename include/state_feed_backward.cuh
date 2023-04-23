///////////////////////////////////////////////////////////////////////////////
// File:         state_feed_backward.cuh
// Description:  Header file for state feed backward  in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 22, 2022
// Updated:      February 02, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>

#include <cmath>

#include "activation_fun.cuh"
#include "data_transfer.cuh"
#include "lstm_feed_backward.cuh"
#include "net_prop.h"
#include "struct_var.h"

__global__ void getInputDeltaState(float const *delta_m, float const *delta_S,
                                   int niB, float *delta_m_0, float *delta_S_0);
///////////////////////////////////////////////////////////////////////////
/// STATE BACKWARD PASS
///////////////////////////////////////////////////////////////////////////
void stateBackward(Network &net, ParamGPU &theta, StateGPU &state,
                   IndexGPU &idx, ObsGPU &obs, DeltaStateGPU &d_state);

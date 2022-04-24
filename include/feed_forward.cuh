///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cuh
// Description:  Header file for feed forward in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 22, 2022
// Updated:      March 20, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "net_prop.h"
#include "struct_var.h"

__global__ void initializeStates(float const *x, float const *Sx, float *mz,
                                 float *Sz, float *ma, float *Sa, float *J,
                                 int niB);

__global__ void initializeFullStates(float const *mz_0, float const *Sz_0,
                                     float const *ma_0, float const *Sa_0,
                                     float const *J_0, int niB, int zposIn,
                                     float *mz, float *Sz, float *ma, float *Sa,
                                     float *J);

////////////////////////////////////////////////////////////////////////////////
/// TAGI-FEEDFORWARD PASS
////////////////////////////////////////////////////////////////////////////////
void feedForward(Network &net, ParamGPU &theta, IndexGPU &idx, StateGPU &state);

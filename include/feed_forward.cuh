///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cuh
// Description:  Header file for feed forward in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 22, 2022
// Updated:      June 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "net_prop.h"
#include "struct_var.h"

__global__ void initializeFullStates(float const *mz_0, float const *Sz_0,
                                     float const *ma_0, float const *Sa_0,
                                     float const *J_0, int niB, int zposIn,
                                     float *mz, float *Sz, float *ma, float *Sa,
                                     float *J);
__global__ void updateMraSra(float const *mra, float const *Sra, int N,
                             float *mra_prev, float *Sra_prev);

void initializeStates(StateGPU &state, InputGPU &ip, Network &net);

void feedForward(Network &net, ParamGPU &theta, IndexGPU &idx, StateGPU &state);

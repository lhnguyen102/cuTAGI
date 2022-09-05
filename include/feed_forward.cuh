///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cuh
// Description:  Header file for feed forward in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 22, 2022
// Updated:      September 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "derivative_calcul.cuh"
#include "net_prop.h"
#include "struct_var.h"

__global__ void fcMean(float const *mw, float const *mb, float const *ma,
                       float *mz, int wpos, int bpos, int zposIn, int zposOut,
                       int m, int n, int k);

__global__ void fcVar(float const *mw, float const *Sw, float const *Sb,
                      float const *ma, float const *Sa, float *Sz, int wpos,
                      int bpos, int zposIn, int zposOut, int m, int n, int k);

__global__ void initializeFullStates(float const *mz_0, float const *Sz_0,
                                     float const *ma_0, float const *Sa_0,
                                     float const *J_0, int niB, int zposIn,
                                     float *mz, float *Sz, float *ma, float *Sa,
                                     float *J);
__global__ void updateMraSra(float const *mra, float const *Sra, int N,
                             float *mra_prev, float *Sra_prev);

void initializeStates(StateGPU &state, InputGPU &ip, Network &net);

void feedForward(Network &net, ParamGPU &theta, IndexGPU &idx, StateGPU &state);

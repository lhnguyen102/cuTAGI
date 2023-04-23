///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cuh
// Description:  Header file for feed forward in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 22, 2022
// Updated:      October 08, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "activation_fun.cuh"
#include "data_transfer.cuh"
#include "derivative_calcul.cuh"
#include "lstm_feed_forward.cuh"
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

__global__ void get_output_hidden_states(float const *z, int z_pos, int n,
                                         float *z_mu);

void initializeStates(StateGPU &state, InputGPU &ip, Network &net);

void feedForward(Network &net, ParamGPU &theta, IndexGPU &idx, StateGPU &state);

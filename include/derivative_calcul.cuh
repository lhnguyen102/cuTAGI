///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calcul.cuh
// Description:  Header file for derivative calculations of neural networks in
//               cuda.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 17, 2022
// Updated:      July 27, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "net_prop.h"
#include "struct_var.h"

void compute_network_derivatives(Network &net, ParamGPU &theta, StateGPU &state,
                                 int l);

void compute_activation_derivatives(Network &net, StateGPU &state, int j);
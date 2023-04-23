///////////////////////////////////////////////////////////////////////////////
// File:         global_param_update.cuh
// Description:  Header file for global parameter update in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      March 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>

#include "data_transfer.cuh"

void global_param_update(DeltaParamGPU &d_theta, float cap_factor, int wN,
                         int bN, int wN_sc, int bN_sc, int THREADS,
                         ParamGPU &theta);

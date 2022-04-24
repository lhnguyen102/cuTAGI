///////////////////////////////////////////////////////////////////////////////
// File:         global_param_update.cuh
// Description:  Header file for global parameter update in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      March 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include "data_transfer.cuh"

void globalParamUpdate(DeltaParamGPU &d_theta, int wN, int bN, int wN_sc, 
    int bN_sc, int THREADS, ParamGPU &theta);


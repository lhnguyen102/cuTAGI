///////////////////////////////////////////////////////////////////////////////
// File:         gpu_debug_utils.h
// Description:  Header file for Debug utils for GPU
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 22, 2022
// Updated:      May 22, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include <vector>

#include "common.h"
#include "data_transfer.cuh"
#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

void save_inference_results(std::string &res_path, DeltaStateGPU &d_state_gpu,
                            Param &theta);

void save_delta_param(std::string &res_path, DeltaParamGPU &d_param);
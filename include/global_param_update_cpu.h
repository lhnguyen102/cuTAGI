///////////////////////////////////////////////////////////////////////////////
// File:         global_param_update_cpu.h
// Description:  Header file for global parameter update in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 20, 2022
// Updated:      March 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "data_transfer_cpu.h"
#include "struct_var.h"

void global_param_update_cpu(DeltaParam &d_theta, float cap_factor, int wN,
                             int bN, int wN_sc, int bN_sc, Param &theta);
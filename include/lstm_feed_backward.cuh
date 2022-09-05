///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_backward.cuh
// Description:  Header file for Long-Short Term Memory (LSTM) state backward
//               pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      September 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "lstm_feed_forward.cuh"
#include "net_prop.h"
#include "struct_var.h"
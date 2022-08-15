///////////////////////////////////////////////////////////////////////////////
// File:         lstm_state_feed_backward_cpu.h
// Description:  Hearder file for Long-Short Term Memory (LSTM) backward pass
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      August 15, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "common.h"
#include "data_transfer_cpu.h"
#include "feed_forward_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

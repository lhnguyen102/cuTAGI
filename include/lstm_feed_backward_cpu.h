///////////////////////////////////////////////////////////////////////////////
// File:         lstm_state_feed_backward_cpu.h
// Description:  Hearder file for Long-Short Term Memory (LSTM) backward pass
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      September 01, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "common.h"
#include "data_transfer_cpu.h"
#include "feed_forward_cpu.h"
#include "lstm_feed_forward_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void lstm_state_update_cpu(Network &net, NetState &state, Param &theta,
                           DeltaState &d_state, int l);

void lstm_parameter_update_cpu(Network &net, NetState &state, Param &theta,
                               DeltaState &d_state, DeltaParam &d_theta, int l);

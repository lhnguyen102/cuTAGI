///////////////////////////////////////////////////////////////////////////
// File:         state_feed_backward_cpu.h
// Description:  Header file for CPU backward pass for hidden state
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      June 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////
#pragma once
#include <math.h>

#include <thread>

#include "data_transfer_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void state_backward_cpu(Network &net, Param &theta, NetState &state,
                        IndexOut &idx, Obs &obs, DeltaState &d_state);

///////////////////////////////////////////////////////////////////////////
// File:         param_feed_backward_cpu.h
// Description:  Header file for CPU backward pass for parametes
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      August 17, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////
#pragma once

#include <thread>

#include "data_transfer_cpu.h"
#include "lstm_feed_backward_cpu.h"
#include "net_prop.h"
#include "struct_var.h"

void param_backward_cpu(Network &net, Param &theta, NetState &state,
                        DeltaState &d_state, IndexOut &idx,
                        DeltaParam &d_theta);

///////////////////////////////////////////////////////////////////////////////
// File:         test_lstm.h
// Description:  Header of independent script to perform lstm
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      March 16, 2023
// Updated:      March 16, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "../include/common.h"
#include "../include/cost.h"
#include "../include/data_transfer_cpu.h"
#include "../include/dataloader.h"
#include "../include/derivative_calcul_cpu.h"
#include "../include/feed_forward_cpu.h"
#include "../include/global_param_update_cpu.h"
#include "../include/indices.h"
#include "../include/net_init.h"
#include "../include/net_prop.h"
#include "../include/param_feed_backward_cpu.h"
#include "../include/state_feed_backward_cpu.h"
#include "../include/struct_var.h"
#include "../include/tagi_network_cpu.h"
#include "../include/task_cpu.h"
#include "../include/user_input.h"
#include "../include/utils.h"
#include "test_utils.h"

/**
 * @brief Train the network with the time series task
 *
 * @param net TagiNetworkCPU object
 * @param db Dataloader object
 * @param epochs number of epochs
 */
void train_time_series(TagiNetworkCPU &net, Dataloader &db, int n_epochs);
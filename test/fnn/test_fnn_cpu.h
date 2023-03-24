///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu.h
// Description:  Header file for fnn test
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com, luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../../include/common.h"
#include "../../include/cost.h"
#include "../../include/data_transfer_cpu.h"
#include "../../include/dataloader.h"
#include "../../include/derivative_calcul_cpu.h"
#include "../../include/feed_forward_cpu.h"
#include "../../include/global_param_update_cpu.h"
#include "../../include/indices.h"
#include "../../include/net_init.h"
#include "../../include/net_prop.h"
#include "../../include/param_feed_backward_cpu.h"
#include "../../include/state_feed_backward_cpu.h"
#include "../../include/struct_var.h"
#include "../../include/tagi_network_cpu.h"
#include "../../include/task_cpu.h"
#include "../../include/user_input.h"
#include "../../include/utils.h"
#include "../test_dataloader.h"
#include "../test_regression.h"
#include "../test_utils.h"

/**
 * @brief Test the FNN network
 *
 * @param recompute_outputs indicates if the outputs should be recomputed
 * @param date date of the test
 * @param arch architecture of the network
 * @param data dataset used
 *
 * @return true if the test was successful
 */
bool test_fnn_cpu(bool recompute_outputs, std::string date, std::string arch,
                  std::string data);
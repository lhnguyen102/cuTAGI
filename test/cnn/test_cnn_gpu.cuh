///////////////////////////////////////////////////////////////////////////////
// File:         test_cnn_gpu.cuh
// Description:  Header file for cnn test
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      March 21, 2023
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
#include "../../include/dataloader.h"
#include "../../include/derivative_calcul.cuh"
#include "../../include/feed_forward.cuh"
#include "../../include/global_param_update.cuh"
#include "../../include/indices.h"
#include "../../include/net_init.h"
#include "../../include/net_prop.h"
#include "../../include/param_feed_backward.cuh"
#include "../../include/state_feed_backward.cuh"
#include "../../include/struct_var.h"
#include "../../include/tagi_network.cuh"
#include "../../include/task.cuh"
#include "../../include/utils.h"
#include "../test_classification.cuh"
#include "../test_dataloader.h"
#include "../test_utils.h"


/**
 * @brief Test the CNN network
 *
 * @param recompute_outputs indicates if the outputs should be recomputed
 * @param date date of the test
 * @param arch architecture of the network
 * @param data dataset used
 *
 * @return true if the test was successful
 */
bool test_cnn_gpu(bool recompute_outputs, std::string date, std::string arch,
                  std::string data);
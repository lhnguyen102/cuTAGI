///////////////////////////////////////////////////////////////////////////////
// File:         test_autoencoder_gpu.cuh
// Description:  Header file for autoencoder test
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      April 11, 2023
// Updated:      April 12, 2023
// Contact:      miquelflorensa11@gmail.com, luongha.nguyen@gmail.com &
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
#include "../test_autoencoder.cuh"
#include "../test_dataloader.h"
#include "../test_utils.h"

/**
 * @brief Test the autoencoder network
 *
 * @param recompute_outputs indicates if the outputs should be recomputed
 * @param date date of the test
 * @param arch architecture of the network
 * @param data dataset used
 *
 * @return true if the test was successful
 */
bool test_autoencoder_gpu(bool recompute_outputs, std::string date,
                          std::string arch, std::string data);

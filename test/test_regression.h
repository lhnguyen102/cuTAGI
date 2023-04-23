///////////////////////////////////////////////////////////////////////////////
// File:         test_regression.h
// Description:  Auxiliar independent script to perform regression
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
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
 * @brief Perform a linear regression on the train data.
 *
 * @param[out] net the network to train with specified architecture and
 * parameters
 * @param[in] db the database to train the network
 * @param[in] epochs the number of epochs to train the network
 */
void regression_train(TagiNetworkCPU &net, Dataloader &db, int epochs);

/**
 * @brief Perform a forward pass on the test data.
 *
 * @param[out] net the network to train with specified architecture and
 * parameters
 * @param[in] db the database to train the network
 */
void forward_pass(TagiNetworkCPU &net, Dataloader &db);

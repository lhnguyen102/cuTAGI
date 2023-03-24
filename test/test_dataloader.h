///////////////////////////////////////////////////////////////////////////////
// File:         test_dataloader.cpp
// Description:  Header of data loader file for testing
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      March 22, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
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
 * @brief Train the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 *
 * @return the training data
 */
Dataloader train_data(std::string problem, TagiNetworkCPU &net,
                      std::string data_path, bool normalize);

/**
 * @brief Test the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 * @param train_db contains the training data
 *
 * @return the testing data
 */
Dataloader test_data(std::string problem, TagiNetworkCPU &net,
                     std::string data_path, Dataloader &train_db,
                     bool normalize);

/**
 * @brief Test the LSTM data
 *
 * @param net contains the network
 * @param mode contains the mode: train or test
 * @param num_features contains the number of features
 * @param data_path contains the path to the data
 * @param output_col contains the output column
 * @param data_norm true if we want to normalize the data
 */
Dataloader test_time_series_datloader(Network &net, std::string mode,
                                      int num_features, std::string data_path,
                                      std::vector<int> output_col,
                                      bool data_norm);


ImageData image_dataloader(std::string data_name, std::string data_path, 
                           std::string mode, std::vector<float> mu, 
                           std::vector<float> sigma, 
                           int num_classes, Network &net_prop);
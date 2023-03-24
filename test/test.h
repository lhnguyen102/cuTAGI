///////////////////////////////////////////////////////////////////////////////
// File:         test.h
// Description:  Header file for main script to test the CPU & GPU
// implementation
//               of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      March 21, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "act_func/test_act_func_cpu.h"
#include "cnn/test_cnn_gpu.cuh"
#include "fnn/test_fnn_cpu.h"
#include "fnn_derivatives/test_fnn_derivatives_cpu.h"
#include "fnn_full_cov/test_fnn_full_cov_cpu.h"
#include "fnn_heteros/test_fnn_heteros_cpu.h"
#include "lstm/test_lstm_cpu.h"

/**
 * @brief Read the last dates of the tests
 *
 * @param user_input_options vector with the user input options
 */
void test(std::vector<std::string> &user_input_options);

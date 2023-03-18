///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.h
// Description:  Header file for main script to test the CPU implementation of
//               cuTAGI
// Authors:      Florensa Miquel , Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Florensa Miquel, Luong-Ha Nguyen & James-A. Goulet.
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
void test_cpu(std::vector<std::string> &user_input_options);
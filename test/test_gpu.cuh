///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.h
// Description:  Header file for main script to test the CPU implementation
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

#include "cnn/test_cnn_gpu.cuh"
#include "test_cpu.h"

/**
 * @brief Read the last dates of the tests
 *
 * @param user_input_options vector with the user input options
 * @param num_tests_passed_cpu number of cpu passed
 * @return Returns number of passed test or -1 for any error
 */
int test_gpu(std::vector<std::string> &user_input_options,
             int num_tests_passed_cpu);

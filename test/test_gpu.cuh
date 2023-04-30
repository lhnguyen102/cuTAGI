///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.h
// Description:  Header file for main script to test the CPU implementation
//               of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      April 13, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "autoencoder/test_autoencoder_gpu.cuh"
#include "cnn/test_cnn_gpu.cuh"
#include "cnn_batch_norm/test_cnn_batch_norm_gpu.cuh"
#include "test_cpu.h"

/**
 * @brief Read the last dates of the tests
 *
 * @param user_input_options vector with the user input options
 * @param num_tests_passed_cpu number of cpu passed
 * @param test_start time point of the start of the test
 * @return Returns number of passed test or -1 for any error
 */
int test_gpu(std::vector<std::string> &user_input_options,
             int num_tests_passed_cpu,
             std::chrono::steady_clock::time_point test_start,
             const int NUM_TESTS);

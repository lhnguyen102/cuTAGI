///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.h
// Description:  Header file for main script to test the CPU implementation
//               of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      April 4, 2023
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

#include "act_func/test_act_func_cpu.h"
#include "fnn/test_fnn_cpu.h"
#include "fnn_derivatives/test_fnn_derivatives_cpu.h"
#include "fnn_full_cov/test_fnn_full_cov_cpu.h"
#include "fnn_heteros/test_fnn_heteros_cpu.h"
#include "lstm/test_lstm_cpu.h"

/**
 * @brief Read the last dates of the tests
 *
 * @return std::vector<std::string> vector with the last dates of the tests
 */
std::vector<std::string> read_dates();

/**
 * @brief Write the last dates of the tests
 *
 * @param dates vector with the last dates of the tests
 * @param column column to change
 * @param date new current date
 */
void write_dates(std::vector<std::string> dates, int column, std::string date);

/**
 * @brief Check if the user input architecture is valid
 *
 * @param test_architecture architecture to test
 */
void check_valid_input_architecture(std::string test_architecture);

/**
 * @brief Read the last dates of the tests
 *
 * @param user_input_options vector with the user input options
 * @param compute_gpu_tests true if the gpu tests will be also computed
 * @param test_start time when the test started
 * @return Returns number of passed test or -1 for any error
 */
int test_cpu(std::vector<std::string> &user_input_options,
             bool compute_gpu_tests,
             std::chrono::steady_clock::time_point test_start);

/**
 * @brief Print the test results
 * @param single_test true if there is only one test being executed
 * @param test_passed true if the test passed
 * @param num_tests number of tests
 * @param test_num number of the test
 * @param arch_name name of the architecture
 * @param run_time time taken to run the test
 */
void print_test_results(bool single_test, bool test_passed, int num_tests,
                        int test_num, std::string arch_name,
                        std::chrono::milliseconds run_time);

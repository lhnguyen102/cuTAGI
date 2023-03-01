///////////////////////////////////////////////////////////////////////////////
// File:         test_cpu.h
// Description:  Header file for main script to test the CPU implementation of
// cuTAGI
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "fnn/test_fnn_cpu.h"
#include "fnn_heteros/test_fnn_heteros_cpu.h"

/**
 * @brief Read the last dates of the tests
 *
 * @param user_input_options vector with the user input options
 */
void test_cpu(std::vector<std::string> &user_input_options);
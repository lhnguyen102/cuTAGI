///////////////////////////////////////////////////////////////////////////////
// File:         user_input.h
// Description:  Load user input
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 05, 2022
// Updated:      May 29, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"

struct UserInput {
    std::string model_name, net_name, task_name, data_name, encoder_net_name,
        decoder_net_name;
    std::string device = "cuda";
    int num_classes, num_epochs, num_train_data, num_test_data;
    bool load_param = false, debug = false;
    std::vector<float> mu, sigma;
    std::vector<std::string> x_train_dir, y_train_dir, x_test_dir, y_test_dir;
};

UserInput load_userinput(std::string &user_input_file);
///////////////////////////////////////////////////////////////////////////////
// File:         main.cpp
// Description:  API for c++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      April 24, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>

#include "test/autoencoder/test_autoencoder_v2.h"
#include "test/fnn/test_fnn_mnist_cpu.h"
#include "test/heteros/test_fnn_heteros_cpu_v2.h"
#include "test/load_state_dict/test_load_state_dict.h"
#include "test/lstm/test_lstm_v2.h"
#include "test/resnet/test_resnet_1d_toy.h"
#include "test/resnet/test_resnet_cifar10.h"
#include "test/smoother/test_smoother.h"

int main(int argc, char *argv[]) {
    // User input file
    std::string user_input_file;
    std::vector<std::string> user_input_options;
    if (argc == 0) {
        throw std::invalid_argument(
            "User need to provide user input file -> see README");
    } else {
        user_input_file = argv[1];
        for (int i = 2; i < argc; i++) {
            user_input_options.push_back(argv[i]);
        }
    }

    // Run task
    if (user_input_file.compare("test_fc_mnist") == 0) {
        auto is_passed = test_fnn_mnist();
    } else if (user_input_file.compare("test_fc_heteros") == 0) {
        auto is_passed = test_fnn_heteros_cpu_v2();
    } else if (user_input_file.compare("autoencoder_mnist") == 0) {
        auto is_passed = test_autoecoder_v2();
    } else if (user_input_file.compare("lstm_toy") == 0) {
        auto is_passed = test_lstm_v2();
    } else if (user_input_file.compare("smoother_toy") == 0) {
        auto is_passed = test_smoother();
    } else if (user_input_file.compare("resnet_toy") == 0) {
        auto is_passed = test_resnet_1d_toy();
    } else if (user_input_file.compare("resnet_cifar10") == 0) {
        auto is_passed = test_resnet_cifar10();
    } else if (user_input_file.compare("load_state_dict") == 0) {
        auto is_passed = test_load_state_dict();
    }
    return 0;
}

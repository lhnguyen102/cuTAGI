///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_heteroscedastic_cpu.cpp
// Description:  CPU version for testing the FNN
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2023
// Updated:      January 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_derivatives_cpu.h"

// Specific constant for the network
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES_1D = {1, 15, 20, 1};
const std::vector<int> NODES_BH = {13, 15, 20, 1};
const std::vector<int> ACTIVATIONS = {0, 4, 4, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 50;
const bool COLLECT_DERIVATIVE = true;

const bool NORMALIZE = true;

/**
 * @brief Test the FNN Heteroscedastic network
 *
 */
bool test_fnn_derivatives_cpu(bool recompute_outputs, std::string date,
                              std::string arch, std::string data) {
    // Create TAGI network
    Network net;

    net.layers = LAYERS;
    if (data == "Boston_housing") {
        net.nodes = NODES_BH;
    } else {
        net.nodes = NODES_1D;
        net.activations = ACTIVATIONS;
    }
    net.activations = ACTIVATIONS;
    net.batch_size = BATCH_SIZE;
    net.collect_derivative = COLLECT_DERIVATIVE;
    net.multithreading = true;

    TagiNetworkCPU tagi_net(net);

    // Put it in the main test file
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data/" + data;
    std::string init_param_path = path.curr_path + "/test/" + arch + "/data/" +
                                  date + "_init_param_" + arch + "_" + data +
                                  ".csv";
    std::string opt_param_path = path.curr_path + "/test/" + arch + "/data/" +
                                 date + "_opt_param_" + arch + "_" + data +
                                 ".csv";
    std::string opt_param_path_2 = path.curr_path + "/test/" + arch + "/data/" +
                                   date + "_opt_param_2_" + arch + "_" + data +
                                   ".csv";
    std::string forward_states_path =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_forward_hidden_states_" + arch + "_" + data + ".csv";
    std::string forward_states_path_2 =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_forward_hidden_states_2_" + arch + "_" + data + ".csv";
    std::string backward_states_path =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_backward_hidden_states_" + arch + "_" + data + ".csv";
    std::string backward_states_path_2 =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_backward_hidden_states_2_" + arch + "_" + data + ".csv";
    std::string input_derivatives_path =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_input_derivative_" + arch + "_" + data + ".csv";
    std::string input_derivatives_path_2 =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_input_derivative_2_" + arch + "_" + data + ".csv";

    // Train data
    Dataloader train_db = train_data(data, tagi_net, data_path, NORMALIZE);
    // Test data
    Dataloader test_db =
        test_data(data, tagi_net, data_path, train_db, NORMALIZE);

    if (recompute_outputs) {
        write_params(init_param_path, tagi_net.theta);
    }

    // Read the initial parameters (see tes_utils.cpp for more details)
    read_params(init_param_path, tagi_net.theta);

    // Train the network
    regression_train(tagi_net, train_db, EPOCHS);

    // Test the network
    regression_test(tagi_net, test_db);

    if (recompute_outputs) {
        // Write the parameters and hidden states
        write_params(opt_param_path, tagi_net.theta);

        write_forward_hidden_states(forward_states_path, tagi_net.state);

        write_backward_hidden_states(backward_states_path, tagi_net,
                                     net.layers.size() - 2);

        write_input_derivatives(input_derivatives_path, tagi_net);
    } else {
        // Write new parameters and compare
        write_params(opt_param_path_2, tagi_net.theta);

        // Compare optimal values with the ones we got
        if (!compare_csv_files(opt_param_path, opt_param_path_2)) {
            std::cout << "\033[1;31mTest for FNN PARAMS has FAILED in " + data +
                             " data\033[0m\n"
                      << std::endl;
            return false;
        }
        // Delete the new parameters
        if (remove(opt_param_path_2.c_str()) != 0) {
            std::cout << "Error deleting " << opt_param_path_2 << std::endl;
            return false;
        }

        // Write new backward hidden states and compare
        write_forward_hidden_states(forward_states_path_2, tagi_net.state);

        if (!compare_csv_files(forward_states_path, forward_states_path_2)) {
            std::cout << "\033[1;31mTest for FNN FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Delete the new forward hidden states
        if (remove(forward_states_path_2.c_str()) != 0) {
            std::cout << "Error deleting " << forward_states_path_2
                      << std::endl;
            return false;
        }

        // Write new backward hidden states and compare
        write_backward_hidden_states(backward_states_path_2, tagi_net,
                                     net.layers.size() - 2);

        if (!compare_csv_files(backward_states_path, backward_states_path_2)) {
            std::cout << "\033[1;32mTest for FNN BACKWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Delete the new backward hidden states
        if (remove(backward_states_path_2.c_str()) != 0) {
            std::cout << "Error deleting " << backward_states_path_2
                      << std::endl;
            return false;
        }

        // Write new input derivatives and compare
        write_input_derivatives(input_derivatives_path_2, tagi_net);

        if (!compare_csv_files(input_derivatives_path,
                               input_derivatives_path_2)) {
            std::cout << "\033[1;32mTest for FNN INPUT DERIVATIVES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        if (remove(input_derivatives_path_2.c_str()) != 0) {
            std::cout << "Error deleting " << input_derivatives_path_2
                      << std::endl;
            return false;
        }
    }
    return true;
}
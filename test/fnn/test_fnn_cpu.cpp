///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu.cpp
// Description:  Script to test the FNN CPU implementation of cuTAGI
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_fnn_cpu.h"

// Specific constant for the network
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES_1D = {1, 10, 15, 1};
const std::vector<int> NODES_BH = {13, 10, 15, 1};
const std::vector<int> ACTIVATIONS = {0, 4, 4, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 50;
const bool NORMALIZE = true;

/**
 * @brief Test the FNN network
 *
 */
bool test_fnn_cpu(bool recompute_outputs, std::string date, std::string arch,
                  std::string data) {
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
    net.sigma_v = 0.06;
    net.sigma_v_min = 0.06;

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
    }
    return true;
}
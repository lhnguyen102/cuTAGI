///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu.cpp
// Description:  CPU version for testing the FNN
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2023
// Updated:      January 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "test_cnn.cuh"

// Specific constant for the network
const std::vector<int> LAYERS = {2, 2, 4, 2, 4, 1, 1};
const std::vector<int> NODES = {784, 0, 0, 0, 0, 20, 11};
const std::vector<int> KERNELS = {4, 3, 5, 3, 1, 1, 1};
const std::vector<int> STRIDES = {1, 2, 1, 2, 0, 0, 0};
const std::vector<int> WIDTHS = {28, 0, 0, 0, 0, 0, 0};
const std::vector<int> HEIGHTS = {28, 0, 0, 0, 0, 0, 0};
const std::vector<int> FILTERS = {1, 16, 16, 32, 32, 1, 1};
const std::vector<int> PADS = {1, 0, 0, 0, 0, 0, 0};
const std::vector<int> PAD_TYPES = {1, 0, 0, 0, 0, 0, 0};
const std::vector<int> ACTIVATIONS = {0, 4, 0, 4, 0, 4, 0};
const int BATCH_SIZE = 10;
const int SIGMA_V = 1;
const int EPOCHS = 2;
const int NUM_CLASSES = 10;
const std::vector<float> MU = {0.1309};
const std::vector<float> SIGMA = {1.0};

/**
 * @brief Test the FNN network
 *
 */
bool test_cnn(bool recompute_outputs, std::string date, std::string arch,
              std::string data) {
    // Create TAGI network
    Network net;

    net.layers = LAYERS;
    net.nodes = NODES;
    net.kernels = KERNELS;
    net.strides = STRIDES;
    net.widths = WIDTHS;
    net.heights = HEIGHTS;
    net.filters = FILTERS;
    net.pads = PADS;
    net.pad_types = PAD_TYPES;
    net.activations = ACTIVATIONS;
    net.batch_size = BATCH_SIZE;
    net.sigma_v = SIGMA_V;

    std::string device = "cuda";
    net.device = device;
    auto hrs = class_to_obs(NUM_CLASSES);
    net.nye = hrs.n_obs;

    TagiNetwork tagi_net(net);

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

    // Data
    auto imdb = image_train_data(data_path, MU, SIGMA, WIDTHS[0], HEIGHTS[0],
                                 tagi_net.prop.filters[0], hrs);

    auto test_imdb = image_test_data(data_path, MU, SIGMA, WIDTHS[0],
                                     HEIGHTS[0], tagi_net.prop.filters[0], hrs);

    if (recompute_outputs) {
        write_params(init_param_path, tagi_net.theta);
    }

    // Read the initial parameters (see tes_utils.cpp for more details)
    read_params(init_param_path, tagi_net.theta);

    // Classify
    test_classification(tagi_net, imdb, test_imdb, EPOCHS, NUM_CLASSES);

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
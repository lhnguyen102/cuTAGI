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

// Specify network properties
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES_1D = {1, 10, 15, 1};
const std::vector<int> NODES_BH = {13, 10, 15, 1};
const std::vector<int> ACTIVATIONS = {0, 4, 4, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 50;
const bool NORMALIZE = true;


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
    std::string data_dir = path.curr_path + "/test/" + arch + "/data";
    std::string init_param_path_w = path.curr_path + "/test/" + arch +
                                    "/data/" + date + "_init_param_weights_" +
                                    arch + "_" + data + ".csv";
    std::string init_param_path_b = path.curr_path + "/test/" + arch +
                                    "/data/" + date + "_init_param_bias_" +
                                    arch + "_" + data + ".csv";
    std::string opt_param_path_w = path.curr_path + "/test/" + arch + "/data/" +
                                   date + "_opt_param_weights_" + arch + "_" +
                                   data + ".csv";
    std::string opt_param_path_b = path.curr_path + "/test/" + arch + "/data/" +
                                   date + "_opt_param_bias_" + arch + "_" +
                                   data + ".csv";
    std::string forward_states_path =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_forward_hidden_states_" + arch + "_" + data + ".csv";
    std::string backward_states_path =
        path.curr_path + "/test/" + arch + "/data/" + date +
        "_backward_hidden_states_" + arch + "_" + data + ".csv";

    // Cheks if the data directory exists
    if (!create_directory_if_not_exists(data_dir)) {
        std::cout << "Error: could not create data directory" << std::endl;
        return false;
    }


    // Train data
    Dataloader train_db = train_data(data, tagi_net, data_path, NORMALIZE);
    // Test data
    Dataloader test_db =
        test_data(data, tagi_net, data_path, train_db, NORMALIZE);

    std::vector<std::vector<float> *> weights;
    weights.push_back(&tagi_net.theta.mw);
    weights.push_back(&tagi_net.theta.Sw);
    weights.push_back(&tagi_net.theta.mb);
    weights.push_back(&tagi_net.theta.Sb);

    std::vector<std::vector<float> *> bias;
    bias.push_back(&tagi_net.theta.mw_sc);
    bias.push_back(&tagi_net.theta.Sw_sc);
    bias.push_back(&tagi_net.theta.mb_sc);
    bias.push_back(&tagi_net.theta.Sb_sc);

    if (recompute_outputs) {
        write_vector_to_csv(init_param_path_w, "mw,Sw,mb,Sb", weights);
        write_vector_to_csv(init_param_path_b, "mw_sc,Sw_sc,mb_sc,Sb_sc", bias);
    }

    // Read the initial parameters (see tes_utils.cpp for more details)
    read_vector_from_csv(init_param_path_w, weights);
    read_vector_from_csv(init_param_path_b, bias);

    // Train the network
    regression_train(tagi_net, train_db, EPOCHS);

    // Test the network
    regression_test(tagi_net, test_db);

    std::vector<std::vector<float> *> forward_states;
    forward_states.push_back(&tagi_net.state.mz);
    forward_states.push_back(&tagi_net.state.Sz);
    forward_states.push_back(&tagi_net.state.ma);
    forward_states.push_back(&tagi_net.state.Sa);
    forward_states.push_back(&tagi_net.state.J);

    std::vector<std::vector<float>> backward_states;
    std::string backward_states_header = "mean_i,sigma_i";

    for (int i = 0; i < net.layers.size() - 2; i++) {
        backward_states.push_back(
            std::get<0>(tagi_net.get_inovation_mean_var(i)));
        backward_states.push_back(
            std::get<1>(tagi_net.get_inovation_mean_var(i)));
    }

    std::vector<std::vector<float> *> backward_states_ptr;
    for (int i = 0; i < backward_states.size(); i++)
        backward_states_ptr.push_back(&backward_states[i]);

    if (recompute_outputs) {
        // RESET OUPUTS

        // Write the parameters and hidden states
        write_vector_to_csv(opt_param_path_w, "mw,Sw,mb,Sb", weights);
        write_vector_to_csv(opt_param_path_b, "mw_sc,Sw_sc,mb_sc,Sb_sc", bias);

        // Write the forward hidden states
        write_vector_to_csv(forward_states_path, "mz,Sz,ma,Sa,J",
                            forward_states);

        // Write the backward hidden states
        write_vector_to_csv(backward_states_path, backward_states_header,
                            backward_states_ptr);

    } else {
        // PERFORM TESTS

        // Read the saved parameters
        std::vector<std::vector<float> *> updated_weights;
        for (int i = 0; i < 4; i++)
            updated_weights.push_back(new std::vector<float>());
        read_vector_from_csv(opt_param_path_w, updated_weights);

        std::vector<std::vector<float> *> updated_bias;
        for (int i = 0; i < 4; i++)
            updated_bias.push_back(new std::vector<float>());
        read_vector_from_csv(opt_param_path_b, updated_bias);

        // Compare optimal values with the ones we got
        if (!compare_vectors(updated_weights, weights) ||
            !compare_vectors(updated_bias, bias)) {
            std::cout << "\033[1;31mTest for FNN PARAMS has FAILED in " + data +
                             " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved forward hidden states
        std::vector<std::vector<float> *> updated_forward_states;
        for (int i = 0; i < 5; i++)
            updated_forward_states.push_back(new std::vector<float>());

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(updated_forward_states, forward_states)) {
            std::cout << "\033[1;31mTest for FNN FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved backward hidden states
        std::vector<std::vector<float> *> updated_backward_states;
        for (int i = 0; i < 2 * (net.layers.size() - 2); i++)
            updated_backward_states.push_back(new std::vector<float>());

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(updated_backward_states, backward_states_ptr)) {
            std::cout << "\033[1;32mTest for FNN BACKWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }
    }
    return true;
}
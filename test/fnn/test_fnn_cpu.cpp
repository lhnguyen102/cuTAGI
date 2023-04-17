///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu.cpp
// Description:  Script to test the FNN CPU implementation of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      March 18, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_fnn_cpu.h"

// Specify network properties
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES_1D = {1, 10, 15, 1};
const std::vector<int> NODES_BH = {13, 10, 15, 1};
const std::vector<int> ACTIVATIONS = {0, 4, 4, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 1;
const float SIGMA_V = 0.06;
const float SIGMA_V_MIN = 0.06;
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
    net.sigma_v = SIGMA_V;
    net.sigma_v_min = SIGMA_V_MIN;

    TagiNetworkCPU tagi_net(net);

    // Put it in the main test file
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data/" + data;
    std::string data_dir = path.curr_path + "/test/" + arch + "/data/";

    TestSavingPaths test_saving_paths(path.curr_path, arch, data, date);

    // Train data
    Dataloader train_db = train_data(data, tagi_net, data_path, NORMALIZE);
    // Test data
    Dataloader test_db =
        test_data(data, tagi_net, data_path, train_db, NORMALIZE);

    TestParamAndStates params_and_states(tagi_net);

    // If we want to test but no data is available, we throw an error
    if (!recompute_outputs && !directory_exists(data_dir)) {
        throw std::runtime_error(
            "Tested data are not available. Please regenerate or provide the "
            "tested data");
    } else if (recompute_outputs) {
        // Cheks if the data directory exists and if not, creates it
        if (!create_directory_if_not_exists(data_dir)) {
            std::cout << "Error: could not create data directory" << std::endl;
            return false;
        }
        params_and_states.write_params(test_saving_paths, true);
    }

    // Read the initial parameters (see tes_utils.cpp for more details)
    params_and_states.read_params(test_saving_paths, true);

    // Train the network
    regression_train(tagi_net, train_db, EPOCHS);

    add_forward_states(params_and_states.forward_states, tagi_net);

    std::vector<std::vector<float>> backward_states;
    std::string backward_states_header = "";

    add_backward_states(backward_states, backward_states_header, tagi_net,
                        net.layers.size());

    for (int i = 0; i < backward_states.size(); i++)
        params_and_states.backward_states.push_back(&backward_states[i]);

    if (recompute_outputs) {
        // RESET OUPUTS

        // Write the parameters and hidden states
        params_and_states.write_params(test_saving_paths, false);

        // Write the forward hidden states
        write_vector_to_csv(test_saving_paths.forward_states_path,
                            "mz,Sz,ma,Sa,J", params_and_states.forward_states);

        // Write the backward hidden states
        write_vector_to_csv(test_saving_paths.backward_states_path,
                            backward_states_header,
                            params_and_states.backward_states);

    } else {
        // PERFORM TESTS

        // Read the saved reference parameters
        TestParamAndStates params_and_states_reference(tagi_net);

        params_and_states_reference.read_params(test_saving_paths, false);

        // Compare optimal values with the ones we got
        if (!compare_vectors(params_and_states_reference.weights,
                             params_and_states.weights, data, "fnn weights") ||
            !compare_vectors(params_and_states_reference.weights_sc,
                             params_and_states.weights_sc, data,
                             "fnn weights for residual network") ||
            !compare_vectors(params_and_states_reference.bias,
                             params_and_states.bias, data, "fnn bias") ||
            !compare_vectors(params_and_states_reference.bias_sc,
                             params_and_states.bias_sc, data,
                             "fnn bias for residual network")) {
            std::cout << "\033[1;31mTest for FNN PARAMS has FAILED in " + data +
                             " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved forward hidden states reference
        for (int i = 0; i < 5; i++)
            params_and_states_reference.forward_states.push_back(
                new std::vector<float>());

        read_vector_from_csv(test_saving_paths.forward_states_path,
                             params_and_states_reference.forward_states);

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(params_and_states_reference.forward_states,
                             params_and_states.forward_states, data,
                             "fnn forward hidden states")) {
            std::cout << "\033[1;31mTest for FNN FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved backward hidden states reference
        for (int i = 0; i < 2 * (net.layers.size() - 2); i++)
            params_and_states_reference.backward_states.push_back(
                new std::vector<float>());

        read_vector_from_csv(test_saving_paths.backward_states_path,
                             params_and_states_reference.backward_states);

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(params_and_states_reference.backward_states,
                             params_and_states.backward_states, data,
                             "fnn backward hidden states")) {
            std::cout << "\033[1;31mTest for FNN BACKWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }
    }
    return true;
}

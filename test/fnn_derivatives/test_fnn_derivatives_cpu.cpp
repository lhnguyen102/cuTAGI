///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_derivatives_cpu.cpp
// Description:  Script to test the derivatives of input layer of FNN network
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      March 13, 2023
// Updated:      March 18, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_fnn_derivatives_cpu.h"

// Specify network properties
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES = {1, 10, 15, 1};
const std::vector<int> ACTIVATIONS = {0, 1, 4, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 1;

const float SIGMA_V = 0.3;
const float SIGMA_V_MIN = 0.1;
const std::string INIT_METHOD = "He";
const float DECAY_FACTOR_SIGMA_V = 0.99;
const bool NORMALIZE = false;
const bool MULTITHREADING = false;
const bool COLLECT_DERIVATIVE = true;

bool test_fnn_derivatives_cpu(bool recompute_outputs, std::string date,
                              std::string arch, std::string data) {
    // Create TAGI network
    Network net;

    net.layers = LAYERS;
    net.nodes = NODES;
    net.activations = ACTIVATIONS;
    net.batch_size = BATCH_SIZE;
    net.sigma_v = SIGMA_V;
    net.sigma_v_min = SIGMA_V_MIN;
    net.decay_factor_sigma_v = DECAY_FACTOR_SIGMA_V;
    net.init_method = INIT_METHOD;
    net.multithreading = MULTITHREADING;
    net.collect_derivative = COLLECT_DERIVATIVE;

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

    // Take pointers from all forward states
    add_forward_states(params_and_states.forward_states, tagi_net);

    std::vector<std::vector<float>> backward_states;
    std::string backward_states_header = "";

    add_backward_states(backward_states, backward_states_header, tagi_net,
                        net.layers.size());

    for (int i = 0; i < backward_states.size(); i++)
        params_and_states.backward_states.push_back(&backward_states[i]);

    // Take pointers from all input derivatives
    std::vector<std::vector<float>> input_derivatives;
    input_derivatives.push_back(std::get<0>(tagi_net.get_derivatives(0)));
    input_derivatives.push_back(std::get<1>(tagi_net.get_derivatives(0)));

    for (int i = 0; i < input_derivatives.size(); i++)
        params_and_states.input_derivatives.push_back(&input_derivatives[i]);

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

        // Write estimated derivatives of the input layer
        write_vector_to_csv(test_saving_paths.input_derivative_path, "md,Sd",
                            params_and_states.input_derivatives);

    } else {
        // PERFORM TESTS

        // Read the saved reference parameters
        TestParamAndStates params_and_states_reference(tagi_net);

        params_and_states_reference.read_params(test_saving_paths, false);

        // Compare optimal values with the ones we got
        if (!compare_vectors(params_and_states_reference.weights,
                             params_and_states.weights, data,
                             "fnn input derivatives weights") ||
            !compare_vectors(
                params_and_states_reference.weights_sc,
                params_and_states.weights_sc, data,
                "fnn input derivatives weights for residual network") ||
            !compare_vectors(params_and_states_reference.bias,
                             params_and_states.bias, data,
                             "fnn input derivatives bias") ||
            !compare_vectors(
                params_and_states_reference.bias_sc, params_and_states.bias_sc,
                data, "fnn input derivatives bias for residual network")) {
            std::cout << "\033[1;31mTest for FNN INPUT DERIVATIVES PARAMS has "
                         "FAILED in " +
                             data + " data\033[0m\n"
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
                             "fnn input derivatives forward hidden states")) {
            std::cout << "\033[1;31mTest for FNN INPUT DERIVATIVES FORWARD "
                         "HIDDEN STATES has "
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
                             "fnn input derivatives backward hidden states")) {
            std::cout << "\033[1;31mTest for FNN INPUT DERIVATIVES BACKWARD "
                         "HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved input derivative reference
        for (int i = 0; i < 2; i++)
            params_and_states_reference.input_derivatives.push_back(
                new std::vector<float>());

        read_vector_from_csv(test_saving_paths.input_derivative_path,
                             params_and_states_reference.input_derivatives);

        // Compare the saved input derivatives with the ones we got
        if (!compare_vectors(params_and_states_reference.input_derivatives,
                             params_and_states.input_derivatives, data,
                             "fnn input derivatives")) {
            std::cout << "\033[1;31mTest for FNN INPUT DERIVATIVES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }
    }
    return true;
}

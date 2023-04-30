///////////////////////////////////////////////////////////////////////////////
// File:         test_cnn_batch_norm_gpu.cu
// Description:  Script to test the CNN batch norm GPU implementation of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      April 5, 2023
// Updated:      April 13, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_cnn_batch_norm_gpu.cuh"

// Specify network properties
const std::vector<int> LAYERS = {2, 2, 6, 4, 2, 6, 4, 1, 1};
const std::vector<int> NODES = {784, 0, 0, 0, 0, 0, 0, 150, 11};
const std::vector<int> KERNELS = {4, 1, 3, 5, 1, 3, 1, 1, 1};
const std::vector<int> STRIDES = {1, 0, 2, 1, 0, 2, 0, 0, 0};
const std::vector<int> WIDTHS = {28, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> HEIGHTS = {28, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> FILTERS = {1, 4, 4, 4, 8, 8, 8, 1, 1};
const std::vector<int> PADS = {1, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> PAD_TYPES = {1, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> ACTIVATIONS = {0, 4, 0, 0, 4, 0, 0, 4, 12};
const int BATCH_SIZE = 2;
const int SIGMA_V = 1;
const int NUM_CLASSES = 10;
const std::vector<float> MU = {0.1309};
const std::vector<float> SIGMA = {1.0};

bool test_cnn_batch_norm_gpu(bool recompute_outputs, std::string date,
                             std::string arch, std::string data) {
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

    if (net.activations.back() == net.act_names.hr_softmax) {
        net.is_idx_ud = true;
        auto hrs = class_to_obs(NUM_CLASSES);
        net.nye = hrs.n_obs;
    }

    TagiNetwork tagi_net(net);

    // Put it in the main test file
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data/" + data;
    std::string data_dir = path.curr_path + "/test/" + arch + "/data/";

    TestSavingPaths test_saving_paths(path.curr_path, arch, data, date);

    // Train data
    ImageData imdb = image_dataloader(data, data_path, MU, SIGMA, NUM_CLASSES,
                                      tagi_net.prop);

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

    // Read the initial parameters (see test_utils.cpp for more details)
    params_and_states.read_params(test_saving_paths, true);

    tagi_net.theta_gpu.copy_host_to_device();

    // Classify
    train_classification(tagi_net, imdb, NUM_CLASSES);

    tagi_net.theta_gpu.copy_device_to_host();
    tagi_net.d_state_gpu.copy_device_to_host();

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

        TagiNetworkCPU tagi_net_ref(net);
        // Read the saved reference parameters
        TestParamAndStates params_and_states_reference(tagi_net_ref);

        params_and_states_reference.read_params(test_saving_paths, false);

        tagi_net.theta_gpu.copy_host_to_device();

        // Compare optimal values with the ones we got
        if (!compare_vectors(params_and_states_reference.weights,
                             params_and_states.weights, data,
                             "cnn batch norm. weights") ||
            !compare_vectors(params_and_states_reference.weights_sc,
                             params_and_states.weights_sc, data,
                             "cnn batch norm. weights for residual network") ||
            !compare_vectors(params_and_states_reference.bias,
                             params_and_states.bias, data,
                             "cnn batch norm. bias") ||
            !compare_vectors(params_and_states_reference.bias_sc,
                             params_and_states.bias_sc, data,
                             "cnn batch norm. bias for residual network")) {
            std::cout
                << "\033[1;31mTest for CNN batch norm. PARAMS has FAILED in " +
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
                             "cnn batch norm. forward hidden states")) {
            std::cout << "\033[1;31mTest for CNN batch norm. FORWARD HIDDEN "
                         "STATES has "
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
                             "cnn batch norm. backward hidden states")) {
            std::cout << "\033[1;31mTest for CNN batch norm. BACKWARD HIDDEN "
                         "STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }
    }
    return true;
}

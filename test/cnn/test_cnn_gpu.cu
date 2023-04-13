///////////////////////////////////////////////////////////////////////////////
// File:         test_cnn_gpu.cu
// Description:  Script to test the CNN GPU implementation of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      April 4, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_cnn_gpu.cuh"

// Specify network properties
const std::vector<int> LAYERS = {2, 2, 4, 2, 4, 1, 1};
const std::vector<int> NODES = {784, 0, 0, 0, 0, 20, 11};
const std::vector<int> KERNELS = {4, 3, 5, 3, 1, 1, 1};
const std::vector<int> STRIDES = {1, 2, 1, 2, 0, 0, 0};
const std::vector<int> WIDTHS = {28, 0, 0, 0, 0, 0, 0};
const std::vector<int> HEIGHTS = {28, 0, 0, 0, 0, 0, 0};
const std::vector<int> FILTERS = {1, 4, 4, 8, 8, 1, 1};
const std::vector<int> PADS = {1, 0, 0, 0, 0, 0, 0};
const std::vector<int> PAD_TYPES = {1, 0, 0, 0, 0, 0, 0};
const std::vector<int> ACTIVATIONS = {0, 7, 0, 7, 0, 7, 12};
const int BATCH_SIZE = 2;
const int SIGMA_V = 4;
const int NUM_CLASSES = 10;
const std::vector<float> MU = {0.1309};
const std::vector<float> SIGMA = {1.0};

bool test_cnn_gpu(bool recompute_outputs, std::string date, std::string arch,
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

    std::vector<std::vector<float> *> weights;
    weights.push_back(&tagi_net.theta.mw);
    weights.push_back(&tagi_net.theta.Sw);
    std::vector<std::vector<float> *> weights_sc;
    weights_sc.push_back(&tagi_net.theta.mw_sc);
    weights_sc.push_back(&tagi_net.theta.Sw_sc);

    std::vector<std::vector<float> *> bias;
    bias.push_back(&tagi_net.theta.mb);
    bias.push_back(&tagi_net.theta.Sb);
    std::vector<std::vector<float> *> bias_sc;
    bias_sc.push_back(&tagi_net.theta.mb_sc);
    bias_sc.push_back(&tagi_net.theta.Sb_sc);

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

        write_vector_to_csv(test_saving_paths.init_param_path_w, "mw,Sw",
                            weights);
        write_vector_to_csv(test_saving_paths.init_param_path_w_sc,
                            "mw_sc,Sw_sc", weights_sc);
        write_vector_to_csv(test_saving_paths.init_param_path_b, "mb,Sb", bias);
        write_vector_to_csv(test_saving_paths.init_param_path_b_sc,
                            "mb_sc,Sb_sc", bias_sc);
    }

    // Read the initial parameters (see test_utils.cpp for more details)
    read_vector_from_csv(test_saving_paths.init_param_path_w, weights);
    read_vector_from_csv(test_saving_paths.init_param_path_w_sc, weights_sc);
    read_vector_from_csv(test_saving_paths.init_param_path_b, bias);
    read_vector_from_csv(test_saving_paths.init_param_path_b_sc, bias_sc);

    tagi_net.theta_gpu.copy_host_to_device();

    // Classify
    train_classification(tagi_net, imdb, NUM_CLASSES);

    tagi_net.theta_gpu.copy_device_to_host();
    tagi_net.d_state_gpu.copy_device_to_host();

    std::vector<std::vector<float> *> forward_states;
    forward_states.push_back(&tagi_net.state.mz);
    forward_states.push_back(&tagi_net.state.Sz);
    forward_states.push_back(&tagi_net.state.ma);
    forward_states.push_back(&tagi_net.state.Sa);
    forward_states.push_back(&tagi_net.state.J);

    std::vector<std::vector<float>> backward_states;
    std::string backward_states_header = "";

    for (int i = 0; i < net.layers.size() - 2; i++) {
        backward_states_header +=
            "mean_" + std::to_string(i) + ",sigma_" + std::to_string(i) + ",";
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
        write_vector_to_csv(test_saving_paths.opt_param_path_w, "mw,Sw",
                            weights);
        write_vector_to_csv(test_saving_paths.opt_param_path_w_sc,
                            "mw_sc,Sw_sc", weights_sc);
        write_vector_to_csv(test_saving_paths.opt_param_path_b, "mb,Sb", bias);
        write_vector_to_csv(test_saving_paths.opt_param_path_b_sc,
                            "mb_sc,Sb_sc", bias_sc);

        // Write the forward hidden states
        write_vector_to_csv(test_saving_paths.forward_states_path,
                            "mz,Sz,ma,Sa,J", forward_states);

        // Write the backward hidden states
        write_vector_to_csv(test_saving_paths.backward_states_path,
                            backward_states_header, backward_states_ptr);

    } else {
        // PERFORM TESTS

        // Read the saved reference parameters
        std::vector<std::vector<float> *> ref_weights;
        std::vector<std::vector<float> *> ref_weights_sc;
        std::vector<std::vector<float> *> ref_bias;
        std::vector<std::vector<float> *> ref_bias_sc;

        for (int i = 0; i < 2; i++) {
            ref_weights.push_back(new std::vector<float>());
            ref_weights_sc.push_back(new std::vector<float>());
            ref_bias.push_back(new std::vector<float>());
            ref_bias_sc.push_back(new std::vector<float>());
        }

        read_vector_from_csv(test_saving_paths.opt_param_path_w, ref_weights);
        read_vector_from_csv(test_saving_paths.opt_param_path_w_sc,
                             ref_weights_sc);
        read_vector_from_csv(test_saving_paths.opt_param_path_b, ref_bias);
        read_vector_from_csv(test_saving_paths.opt_param_path_b_sc,
                             ref_bias_sc);

        tagi_net.theta_gpu.copy_host_to_device();

        // Compare optimal values with the ones we got
        if (!compare_vectors(ref_weights, weights, data, "cnn weights") ||
            !compare_vectors(ref_weights_sc, weights_sc, data,
                             "cnn weights for residual network") ||
            !compare_vectors(ref_bias, bias, data, "cnn bias") ||
            !compare_vectors(ref_bias_sc, bias_sc, data,
                             "cnn bias for residual network")) {
            std::cout << "\033[1;31mTest for CNN PARAMS has FAILED in " + data +
                             " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved forward hidden states reference
        std::vector<std::vector<float> *> ref_forward_states;
        for (int i = 0; i < 5; i++)
            ref_forward_states.push_back(new std::vector<float>());

        read_vector_from_csv(test_saving_paths.forward_states_path,
                             ref_forward_states);

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(ref_forward_states, forward_states, data,
                             "cnn forward hidden states")) {
            std::cout << "\033[1;31mTest for CNN FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved backward hidden states reference
        std::vector<std::vector<float> *> ref_backward_states;
        for (int i = 0; i < 2 * (net.layers.size() - 2); i++)
            ref_backward_states.push_back(new std::vector<float>());

        read_vector_from_csv(test_saving_paths.backward_states_path,
                             ref_backward_states);

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(ref_backward_states, backward_states_ptr, data,
                             "cnn backward hidden states")) {
            std::cout << "\033[1;31mTest for CNN BACKWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_full_cov_cpu.cpp
// Description:  Script to test FNN with full covariance
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      March 07, 2023
// Updated:      March 07, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_fnn_full_cov_cpu.h"

// Specify network properties
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES = {1, 10, 15, 1};
const std::vector<int> ACTIVATIONS = {0, 7, 7, 0};
const int BATCH_SIZE = 5;
const int EPOCHS = 1;

const float SIGMA_V = 0.5;
const float SIGMA_V_MIN = 0.065;
const std::string INIT_METHOD = "He";
const float DECAY_FACTOR_SIGMA_V = 0.95;
const bool NORMALIZE = true;
const float SIGMA_X = 0.3485;
const bool IS_FULL_COV = true;

bool test_fnn_full_cov_cpu(bool recompute_outputs, std::string date,
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
    net.sigma_x = SIGMA_X;
    net.init_method = INIT_METHOD;
    net.is_full_cov = IS_FULL_COV;

    TagiNetworkCPU tagi_net(net);

    // Put it in the main test file
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data/" + data;
    std::string data_dir = path.curr_path + "/test/" + arch + "/data/";
    std::string init_param_path_w =
        data_dir + date + "_" + data + "_init_param_weights_w.csv";
    std::string init_param_path_w_sc =
        data_dir + date + "_" + data + "_init_param_weights_w_sc.csv";
    std::string init_param_path_b =
        data_dir + date + "_" + data + "_init_param_bias_b.csv";
    std::string init_param_path_b_sc =
        data_dir + date + "_" + data + "_init_param_bias_b_sc.csv";
    std::string opt_param_path_w =
        data_dir + date + "_" + data + "_opt_param_weights_w.csv";
    std::string opt_param_path_w_sc =
        data_dir + date + "_" + data + "_opt_param_weights_w_sc.csv";
    std::string opt_param_path_b =
        data_dir + date + "_" + data + "_opt_param_bias_b.csv";
    std::string opt_param_path_b_sc =
        data_dir + date + "_" + data + "_opt_param_bias_b_sc.csv";
    std::string forward_states_path = data_dir + date +
                                      "_forward_hidden_states_" + arch + "_" +
                                      data + ".csv";
    std::string backward_states_path = data_dir + date +
                                       "_backward_hidden_states_" + arch + "_" +
                                       data + ".csv";

    // Train data
    Dataloader train_db = train_data(data, tagi_net, data_path, NORMALIZE);
    // Test data
    Dataloader test_db =
        test_data(data, tagi_net, data_path, train_db, NORMALIZE);

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
        write_vector_to_csv(init_param_path_w, "mw,Sw", weights);
        write_vector_to_csv(init_param_path_w_sc, "mw_sc,Sw_sc", weights_sc);

        write_vector_to_csv(init_param_path_b, "mb,Sb", bias);
        write_vector_to_csv(init_param_path_b_sc, "mb_sc,Sb_sc", bias_sc);
    }

    // Read the initial parameters (see tes_utils.cpp for more details)
    read_vector_from_csv(init_param_path_w, weights);
    read_vector_from_csv(init_param_path_w_sc, weights_sc);
    read_vector_from_csv(init_param_path_b, bias);
    read_vector_from_csv(init_param_path_b_sc, bias_sc);

    // Train the network
    regression_train(tagi_net, train_db, EPOCHS);

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
        write_vector_to_csv(opt_param_path_w, "mw,Sw", weights);
        write_vector_to_csv(opt_param_path_w_sc, "mw_sc,Sw_sc", weights_sc);
        write_vector_to_csv(opt_param_path_b, "mb,Sb", bias);
        write_vector_to_csv(opt_param_path_b_sc, "mb_sc,Sb_sc", bias_sc);

        // Write the forward hidden states
        write_vector_to_csv(forward_states_path, "mz,Sz,ma,Sa,J",
                            forward_states);

        // Write the backward hidden states
        write_vector_to_csv(backward_states_path, backward_states_header,
                            backward_states_ptr);

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

        read_vector_from_csv(opt_param_path_w, ref_weights);
        read_vector_from_csv(opt_param_path_w_sc, ref_weights_sc);
        read_vector_from_csv(opt_param_path_b, ref_bias);
        read_vector_from_csv(opt_param_path_b_sc, ref_bias_sc);

        // Compare optimal values with the ones we got
        if (!compare_vectors(ref_weights, weights) ||
            !compare_vectors(ref_weights_sc, weights_sc) ||
            !compare_vectors(ref_bias, bias) ||
            !compare_vectors(ref_bias_sc, bias_sc)) {
            std::cout << "\033[1;31mTest for FNN PARAMS has FAILED in " + data +
                             " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved forward hidden states reference
        std::vector<std::vector<float> *> ref_forward_states;
        for (int i = 0; i < 5; i++)
            ref_forward_states.push_back(new std::vector<float>());

        read_vector_from_csv(forward_states_path, ref_forward_states);

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(ref_forward_states, forward_states)) {
            std::cout << "\033[1;31mTest for FNN FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved backward hidden states reference
        std::vector<std::vector<float> *> ref_backward_states;
        for (int i = 0; i < 2 * (net.layers.size() - 2); i++)
            ref_backward_states.push_back(new std::vector<float>());

        read_vector_from_csv(backward_states_path, ref_backward_states);

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(ref_backward_states, backward_states_ptr)) {
            std::cout << "\033[1;31mTest for FNN BACKWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }
    }
    return true;
}
///////////////////////////////////////////////////////////////////////////////
// File:         test_act_func_cpu.cpp
// Description:  Script to test all activation functions available in cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      March 18, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_act_func_cpu.h"

// Specify network properties
const std::vector<int> LAYERS = {1, 1, 1, 1};
const std::vector<int> NODES = {13, 10, 15, 1};
ActLabel act;
const std::vector<int> ACTIVATIONS = {act.tanh,  act.sigmoid, act.relu,
                                      act.mrelu, act.mtanh,   act.msigmoid};
const std::vector<std::string> ACTIVATIONS_NAMES = {
    "tanh", "sigmoid", "relu", "mrelu", "mtanh", "msigmoid"};
const int BATCH_SIZE = 5;
const float SIGMA_V = 0.06;
const float SIGMA_V_MIN = 0.06;
const bool NORMALIZE = true;

bool test_act_func_cpu(bool recompute_outputs, std::string date,
                       std::string arch, std::string data) {
    // Define paths
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data/" + data;
    std::string data_dir = path.curr_path + "/test/" + arch + "/data/";

    TestSavingPaths test_saving_paths(path.curr_path, arch, data, date);

    // Iterate over all activation functions
    for (int i = 0; i < ACTIVATIONS.size(); i++) {
        // Define path for forward hidden states
        std::string forward_states_path =
            data_dir + date + "_forward_hidden_states_" + ACTIVATIONS_NAMES[i] +
            "_" + arch + "_" + data + ".csv";
        // Create TAGI network
        Network net;

        net.layers = LAYERS;
        net.nodes = NODES;
        net.activations = {act.no_act, ACTIVATIONS[i], ACTIVATIONS[i],
                           act.no_act};
        net.batch_size = BATCH_SIZE;
        net.sigma_v = SIGMA_V;
        net.sigma_v_min = SIGMA_V_MIN;

        TagiNetworkCPU tagi_net(net);

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
                "Tested data are not available. Please regenerate or provide "
                "the tested data");
        } else if (recompute_outputs && i == 0) {
            // Cheks if the data directory exists and if not, creates it
            if (!create_directory_if_not_exists(data_dir)) {
                std::cout << "Error: could not create data directory"
                          << std::endl;
                return false;
            }
            write_vector_to_csv(test_saving_paths.init_param_path_w, "mw,Sw",
                                weights);
            write_vector_to_csv(test_saving_paths.init_param_path_w_sc,
                                "mw_sc,Sw_sc", weights_sc);

            write_vector_to_csv(test_saving_paths.init_param_path_b, "mb,Sb",
                                bias);
            write_vector_to_csv(test_saving_paths.init_param_path_b_sc,
                                "mb_sc,Sb_sc", bias_sc);
        }

        // Read the initial parameters (see tes_utils.cpp for more details)
        read_vector_from_csv(test_saving_paths.init_param_path_w, weights);
        read_vector_from_csv(test_saving_paths.init_param_path_w_sc,
                             weights_sc);
        read_vector_from_csv(test_saving_paths.init_param_path_b, bias);
        read_vector_from_csv(test_saving_paths.init_param_path_b_sc, bias_sc);

        // Train the network
        forward_pass(tagi_net, train_db);

        std::vector<std::vector<float> *> forward_states;
        forward_states.push_back(&tagi_net.state.mz);
        forward_states.push_back(&tagi_net.state.Sz);
        forward_states.push_back(&tagi_net.state.ma);
        forward_states.push_back(&tagi_net.state.Sa);
        forward_states.push_back(&tagi_net.state.J);

        if (recompute_outputs) {
            // RESET OUPUTS

            // Write the forward hidden states
            write_vector_to_csv(forward_states_path, "mz,Sz,ma,Sa,J",
                                forward_states);

        } else {
            // PERFORM TESTS

            // Read the saved forward hidden states reference
            std::vector<std::vector<float> *> ref_forward_states;
            for (int i = 0; i < 5; i++)
                ref_forward_states.push_back(new std::vector<float>());

            read_vector_from_csv(forward_states_path, ref_forward_states);

            // Compare the saved forward hidden states with the ones we got
            if (!compare_vectors(ref_forward_states, forward_states, data,
                                 "fnn forward hidden states")) {
                std::cout << "\033[1;31mTest for " + ACTIVATIONS_NAMES[i] +
                                 " " + "activation function has FAILED in " +
                                 data + " data\033[0m\n"
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// File:         test_autoencoder_gpu.cu
// Description:  Script to test the autoencoder GPU implementation of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      April 11, 2023
// Updated:      April 12, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_autoencoder_gpu.cuh"

// Specify network properties for the encoder
const std::vector<int> LAYERS_E = {2, 2, 6, 4, 2, 6, 4, 1, 1};
const std::vector<int> NODES_E = {784, 0, 0, 0, 0, 0, 0, 100, 10};
const std::vector<int> KERNELS_E = {3, 1, 3, 3, 1, 3, 1, 1, 1};
const std::vector<int> STRIDES_E = {1, 0, 2, 1, 0, 2, 0, 0, 0};
const std::vector<int> WIDTHS_E = {28, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> HEIGHTS_E = {28, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> FILTERS_E = {1, 4, 4, 4, 8, 8, 8, 1, 1};
const std::vector<int> PADS_E = {1, 0, 1, 1, 0, 1, 0, 0, 0};
const std::vector<int> PAD_TYPES_E = {1, 0, 2, 1, 0, 2, 0, 0, 0};
const std::vector<int> ACTIVATIONS_E = {0, 4, 0, 0, 4, 0, 0, 4, 0};

// Specify network properties for the decoder
const std::vector<int> LAYERS_D = {1, 1, 21, 21, 21};
const std::vector<int> NODES_D = {10, 1568, 0, 0, 784};
const std::vector<int> KERNELS_D = {1, 3, 3, 3, 1};
const std::vector<int> STRIDES_D = {0, 2, 2, 1, 0};
const std::vector<int> WIDTHS_D = {0, 7, 0, 0, 0};
const std::vector<int> HEIGHTS_D = {0, 7, 0, 0, 0};
const std::vector<int> FILTERS_D = {1, 8, 8, 4, 1};
const std::vector<int> PADS_D = {0, 1, 1, 1, 0};
const std::vector<int> PAD_TYPES_D = {0, 2, 2, 1, 0};
const std::vector<int> ACTIVATIONS_D = {0, 4, 4, 4, 0};

const int BATCH_SIZE = 2;
const int SIGMA_V = 8;
const int SIGMA_V_MIN = 2;
const int DECAT_FACTOR_SIGMA_V = 0.95;
const int NUM_CLASSES = 10;
const std::vector<float> MU = {0.1309};
const std::vector<float> SIGMA = {1.0};
const std::string INIT_METHOD = "He";

bool test_autoencoder_gpu(bool recompute_outputs, std::string date,
                          std::string arch, std::string data) {
    // Encoder
    Network net_prop_e;

    net_prop_e.layers = LAYERS_E;
    net_prop_e.nodes = NODES_E;
    net_prop_e.kernels = KERNELS_E;
    net_prop_e.strides = STRIDES_E;
    net_prop_e.widths = WIDTHS_E;
    net_prop_e.heights = HEIGHTS_E;
    net_prop_e.filters = FILTERS_E;
    net_prop_e.pads = PADS_E;
    net_prop_e.pad_types = PAD_TYPES_E;
    net_prop_e.activations = ACTIVATIONS_E;
    net_prop_e.batch_size = BATCH_SIZE;
    net_prop_e.init_method = INIT_METHOD;

    std::string device = "cuda";
    net_prop_e.device = device;

    TagiNetwork net_e(net_prop_e);
    net_e.prop.is_output_ud = false;

    // Decoder
    Network net_prop_d;
    net_prop_d.layers = LAYERS_D;
    net_prop_d.nodes = NODES_D;
    net_prop_d.kernels = KERNELS_D;
    net_prop_d.strides = STRIDES_D;
    net_prop_d.widths = WIDTHS_D;
    net_prop_d.heights = HEIGHTS_D;
    net_prop_d.filters = FILTERS_D;
    net_prop_d.pads = PADS_D;
    net_prop_d.pad_types = PAD_TYPES_D;
    net_prop_d.activations = ACTIVATIONS_D;
    net_prop_d.batch_size = BATCH_SIZE;
    net_prop_d.sigma_v = SIGMA_V;
    net_prop_d.sigma_v_min = SIGMA_V_MIN;
    net_prop_d.decay_factor_sigma_v = DECAT_FACTOR_SIGMA_V;
    net_prop_d.init_method = INIT_METHOD;

    net_prop_d.device = device;

    TagiNetwork net_d(net_prop_d);
    net_d.prop.last_backward_layer = 0;

    SavePath path;
    path.curr_path = get_current_dir();

    TestSavingPaths test_saving_paths_encoder(path.curr_path, arch, data, date,
                                              true);
    TestSavingPaths test_saving_paths_decoder(path.curr_path, arch, data, date,
                                              false, true);

    std::string data_path = path.curr_path + "/test/data/" + data;
    std::string data_dir = path.curr_path + "/test/" + arch + "/data/";

    // Train data
    ImageData imdb =
        image_dataloader(data, data_path, MU, SIGMA, NUM_CLASSES, net_e.prop);

    std::vector<std::vector<float> *> weights_e;
    weights_e.push_back(&net_e.theta.mw);
    weights_e.push_back(&net_e.theta.Sw);
    std::vector<std::vector<float> *> weights_d;
    weights_d.push_back(&net_d.theta.mw);
    weights_d.push_back(&net_d.theta.Sw);

    std::vector<std::vector<float> *> weights_sc_e;
    weights_sc_e.push_back(&net_e.theta.mw_sc);
    weights_sc_e.push_back(&net_e.theta.Sw_sc);
    std::vector<std::vector<float> *> weights_sc_d;
    weights_sc_d.push_back(&net_d.theta.mw_sc);
    weights_sc_d.push_back(&net_d.theta.Sw_sc);

    std::vector<std::vector<float> *> bias_e;
    bias_e.push_back(&net_e.theta.mb);
    bias_e.push_back(&net_e.theta.Sb);
    std::vector<std::vector<float> *> bias_d;
    bias_d.push_back(&net_d.theta.mb);
    bias_d.push_back(&net_d.theta.Sb);

    std::vector<std::vector<float> *> bias_sc_e;
    bias_sc_e.push_back(&net_e.theta.mb_sc);
    bias_sc_e.push_back(&net_e.theta.Sb_sc);
    std::vector<std::vector<float> *> bias_sc_d;
    bias_sc_d.push_back(&net_d.theta.mb_sc);
    bias_sc_d.push_back(&net_d.theta.Sb_sc);

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

        write_vector_to_csv(test_saving_paths_encoder.init_param_path_w,
                            "mw,Sw", weights_e);
        write_vector_to_csv(test_saving_paths_encoder.init_param_path_w_sc,
                            "mw_sc,Sw_sc", weights_sc_e);
        write_vector_to_csv(test_saving_paths_encoder.init_param_path_b,
                            "mb,Sb", bias_e);
        write_vector_to_csv(test_saving_paths_encoder.init_param_path_b_sc,
                            "mb_sc,Sb_sc", bias_sc_e);

        write_vector_to_csv(test_saving_paths_decoder.init_param_path_w,
                            "mw,Sw", weights_d);
        write_vector_to_csv(test_saving_paths_decoder.init_param_path_w_sc,
                            "mw_sc,Sw_sc", weights_sc_d);
        write_vector_to_csv(test_saving_paths_decoder.init_param_path_b,
                            "mb,Sb", bias_d);
        write_vector_to_csv(test_saving_paths_decoder.init_param_path_b_sc,
                            "mb_sc,Sb_sc", bias_sc_d);
    }

    // Read the initial parameters (see test_utils.cpp for more details)
    read_vector_from_csv(test_saving_paths_encoder.init_param_path_w,
                         weights_e);
    read_vector_from_csv(test_saving_paths_encoder.init_param_path_w_sc,
                         weights_sc_e);
    read_vector_from_csv(test_saving_paths_encoder.init_param_path_b, bias_e);
    read_vector_from_csv(test_saving_paths_encoder.init_param_path_b_sc,
                         bias_sc_e);

    read_vector_from_csv(test_saving_paths_decoder.init_param_path_w,
                         weights_d);
    read_vector_from_csv(test_saving_paths_decoder.init_param_path_w_sc,
                         weights_sc_d);
    read_vector_from_csv(test_saving_paths_decoder.init_param_path_b, bias_d);
    read_vector_from_csv(test_saving_paths_decoder.init_param_path_b_sc,
                         bias_sc_d);

    net_e.theta_gpu.copy_host_to_device();
    net_d.theta_gpu.copy_host_to_device();

    // Autoencoder
    train_autoencoder(net_e, net_d, imdb, NUM_CLASSES);

    net_e.theta_gpu.copy_device_to_host();
    net_d.theta_gpu.copy_device_to_host();
    net_e.d_state_gpu.copy_device_to_host();
    net_d.d_state_gpu.copy_device_to_host();

    std::vector<std::vector<float> *> forward_states_e;
    forward_states_e.push_back(&net_e.state.mz);
    forward_states_e.push_back(&net_e.state.Sz);
    forward_states_e.push_back(&net_e.state.ma);
    forward_states_e.push_back(&net_e.state.Sa);
    forward_states_e.push_back(&net_e.state.J);
    std::vector<std::vector<float> *> forward_states_d;
    forward_states_d.push_back(&net_d.state.mz);
    forward_states_d.push_back(&net_d.state.Sz);
    forward_states_d.push_back(&net_d.state.ma);
    forward_states_d.push_back(&net_d.state.Sa);
    forward_states_d.push_back(&net_d.state.J);

    std::vector<std::vector<float>> backward_states_e;
    std::string backward_states_header_e = "";

    for (int i = 0; i < net_prop_e.layers.size() - 2; i++) {
        backward_states_header_e +=
            "mean_" + std::to_string(i) + ",sigma_" + std::to_string(i) + ",";
        backward_states_e.push_back(
            std::get<0>(net_e.get_inovation_mean_var(i)));
        backward_states_e.push_back(
            std::get<1>(net_e.get_inovation_mean_var(i)));
    }

    std::vector<std::vector<float>> backward_states_d;
    std::string backward_states_header_d = "";

    for (int i = 0; i < net_prop_d.layers.size() - 2; i++) {
        backward_states_header_d +=
            "mean_" + std::to_string(i) + ",sigma_" + std::to_string(i) + ",";
        backward_states_d.push_back(
            std::get<0>(net_d.get_inovation_mean_var(i)));
        backward_states_d.push_back(
            std::get<1>(net_d.get_inovation_mean_var(i)));
    }

    std::vector<std::vector<float> *> backward_states_e_ptr;
    for (int i = 0; i < backward_states_e.size(); i++)
        backward_states_e_ptr.push_back(&backward_states_e[i]);

    std::vector<std::vector<float> *> backward_states_d_ptr;
    for (int i = 0; i < backward_states_d.size(); i++)
        backward_states_d_ptr.push_back(&backward_states_d[i]);

    if (recompute_outputs) {
        // RESET OUPUTS

        // Write the parameters and hidden states
        write_vector_to_csv(test_saving_paths_encoder.opt_param_path_w, "mw,Sw",
                            weights_e);
        write_vector_to_csv(test_saving_paths_encoder.opt_param_path_w_sc,
                            "mw_sc,Sw_sc", weights_sc_e);
        write_vector_to_csv(test_saving_paths_encoder.opt_param_path_b, "mb,Sb",
                            bias_e);
        write_vector_to_csv(test_saving_paths_encoder.opt_param_path_b_sc,
                            "mb_sc,Sb_sc", bias_sc_e);

        write_vector_to_csv(test_saving_paths_decoder.opt_param_path_w, "mw,Sw",
                            weights_d);
        write_vector_to_csv(test_saving_paths_decoder.opt_param_path_w_sc,
                            "mw_sc,Sw_sc", weights_sc_d);
        write_vector_to_csv(test_saving_paths_decoder.opt_param_path_b, "mb,Sb",
                            bias_d);
        write_vector_to_csv(test_saving_paths_decoder.opt_param_path_b_sc,
                            "mb_sc,Sb_sc", bias_sc_d);

        // Write the forward hidden states
        write_vector_to_csv(test_saving_paths_encoder.forward_states_path,
                            "mz,Sz,ma,Sa,J", forward_states_e);
        write_vector_to_csv(test_saving_paths_decoder.forward_states_path,
                            "mz,Sz,ma,Sa,J", forward_states_d);

        // Write the backward hidden states
        write_vector_to_csv(test_saving_paths_encoder.backward_states_path,
                            backward_states_header_e, backward_states_e_ptr);
        write_vector_to_csv(test_saving_paths_decoder.backward_states_path,
                            backward_states_header_d, backward_states_d_ptr);

    } else {
        // PERFORM TESTS

        // Read the saved reference parameters
        std::vector<std::vector<float> *> ref_weights_e;
        std::vector<std::vector<float> *> ref_weights_sc_e;
        std::vector<std::vector<float> *> ref_bias_e;
        std::vector<std::vector<float> *> ref_bias_sc_e;
        std::vector<std::vector<float> *> ref_weights_d;
        std::vector<std::vector<float> *> ref_weights_sc_d;
        std::vector<std::vector<float> *> ref_bias_d;
        std::vector<std::vector<float> *> ref_bias_sc_d;

        for (int i = 0; i < 2; i++) {
            ref_weights_e.push_back(new std::vector<float>());
            ref_weights_sc_e.push_back(new std::vector<float>());
            ref_bias_e.push_back(new std::vector<float>());
            ref_bias_sc_e.push_back(new std::vector<float>());
        }
        for (int i = 0; i < 2; i++) {
            ref_weights_d.push_back(new std::vector<float>());
            ref_weights_sc_d.push_back(new std::vector<float>());
            ref_bias_d.push_back(new std::vector<float>());
            ref_bias_sc_d.push_back(new std::vector<float>());
        }

        read_vector_from_csv(test_saving_paths_encoder.opt_param_path_w,
                             ref_weights_e);
        read_vector_from_csv(test_saving_paths_encoder.opt_param_path_w_sc,
                             ref_weights_sc_e);
        read_vector_from_csv(test_saving_paths_encoder.opt_param_path_b,
                             ref_bias_e);
        read_vector_from_csv(test_saving_paths_encoder.opt_param_path_b_sc,
                             ref_bias_sc_e);

        read_vector_from_csv(test_saving_paths_decoder.opt_param_path_w,
                             ref_weights_d);
        read_vector_from_csv(test_saving_paths_decoder.opt_param_path_w_sc,
                             ref_weights_sc_d);
        read_vector_from_csv(test_saving_paths_decoder.opt_param_path_b,
                             ref_bias_d);
        read_vector_from_csv(test_saving_paths_decoder.opt_param_path_b_sc,
                             ref_bias_sc_d);

        net_e.theta_gpu.copy_host_to_device();
        net_d.theta_gpu.copy_host_to_device();

        // Compare optimal values with the ones we got
        if (!compare_vectors(ref_weights_e, weights_e, data,
                             "encoder weights") ||
            !compare_vectors(ref_weights_sc_e, weights_sc_e, data,
                             "encoder weights for residual network") ||
            !compare_vectors(ref_bias_e, bias_e, data, "encoder bias") ||
            !compare_vectors(ref_bias_sc_e, bias_sc_e, data,
                             "encoder bias for residual network")) {
            std::cout << "\033[1;31mTest for encoder PARAMS has FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Compare optimal values with the ones we got
        if (!compare_vectors(ref_weights_d, weights_d, data,
                             "decoder weights") ||
            !compare_vectors(ref_weights_sc_d, weights_sc_d, data,
                             "decoder weights for residual network") ||
            !compare_vectors(ref_bias_d, bias_d, data, "decoder bias") ||
            !compare_vectors(ref_bias_sc_d, bias_sc_d, data,
                             "decoder bias for residual network")) {
            std::cout << "\033[1;31mTest for decoder PARAMS has FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved forward hidden states reference
        std::vector<std::vector<float> *> ref_forward_states_e;
        for (int i = 0; i < 5; i++)
            ref_forward_states_e.push_back(new std::vector<float>());
        std::vector<std::vector<float> *> ref_forward_states_d;
        for (int i = 0; i < 5; i++)
            ref_forward_states_d.push_back(new std::vector<float>());

        read_vector_from_csv(test_saving_paths_encoder.forward_states_path,
                             ref_forward_states_e);
        read_vector_from_csv(test_saving_paths_decoder.forward_states_path,
                             ref_forward_states_d);

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(ref_forward_states_e, forward_states_e, data,
                             "encoder forward hidden states")) {
            std::cout << "\033[1;31mTest for encoder FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(ref_forward_states_d, forward_states_d, data,
                             "decoder forward hidden states")) {
            std::cout << "\033[1;31mTest for decoder FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved backward hidden states reference
        std::vector<std::vector<float> *> ref_backward_states_e;
        for (int i = 0; i < 2 * (net_prop_e.layers.size() - 2); i++)
            ref_backward_states_e.push_back(new std::vector<float>());
        std::vector<std::vector<float> *> ref_backward_states_d;
        for (int i = 0; i < 2 * (net_prop_d.layers.size() - 2); i++)
            ref_backward_states_d.push_back(new std::vector<float>());

        read_vector_from_csv(test_saving_paths_encoder.backward_states_path,
                             ref_backward_states_e);
        read_vector_from_csv(test_saving_paths_decoder.backward_states_path,
                             ref_backward_states_d);

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(ref_backward_states_e, backward_states_e_ptr, data,
                             "endoer backward hidden states")) {
            std::cout
                << "\033[1;31mTest for encoder BACKWARD HIDDEN STATES has "
                   "FAILED in " +
                       data + " data\033[0m\n"
                << std::endl;
            return false;
        }

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(ref_backward_states_d, backward_states_d_ptr, data,
                             "decoder backward hidden states")) {
            std::cout
                << "\033[1;31mTest for decoder BACKWARD HIDDEN STATES has "
                   "FAILED in " +
                       data + " data\033[0m\n"
                << std::endl;
            return false;
        }
    }
    return true;
}

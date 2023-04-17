///////////////////////////////////////////////////////////////////////////////
// File:         test_autoencoder_gpu.cu
// Description:  Script to test the autoencoder GPU implementation of cuTAGI
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      April 11, 2023
// Updated:      April 16, 2023
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

    TestParamAndStates params_and_states_encoder(net_e);
    TestParamAndStates params_and_states_decoder(net_d);

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

        params_and_states_encoder.write_params(test_saving_paths_encoder, true);
        params_and_states_decoder.write_params(test_saving_paths_decoder, true);
    }

    // Read the initial parameters (see test_utils.cpp for more details)
    params_and_states_encoder.read_params(test_saving_paths_encoder, true);
    params_and_states_decoder.read_params(test_saving_paths_decoder, true);

    net_e.theta_gpu.copy_host_to_device();
    net_d.theta_gpu.copy_host_to_device();

    // Autoencoder
    train_autoencoder(net_e, net_d, imdb, NUM_CLASSES);

    // Transfer data to CPU
    net_e.theta_gpu.copy_device_to_host();
    net_d.theta_gpu.copy_device_to_host();
    net_e.d_state_gpu.copy_device_to_host();
    net_d.d_state_gpu.copy_device_to_host();

    // Recover forward and backward hidden states after running the autoencoder
    add_forward_states(params_and_states_encoder.forward_states, net_e);
    add_forward_states(params_and_states_decoder.forward_states, net_d);

    // Recover forward and backward hidden states after running the autoencoder

    std::vector<std::vector<float>> backward_states_e;
    std::vector<std::vector<float>> backward_states_d;
    std::string backward_states_header_e = "";
    std::string backward_states_header_d = "";

    add_backward_states(backward_states_e, backward_states_header_e, net_e,
                        net_prop_e.layers.size());
    add_backward_states(backward_states_d, backward_states_header_d, net_d,
                        net_prop_d.layers.size());

    for (int i = 0; i < backward_states_e.size(); i++)
        params_and_states_encoder.backward_states.push_back(
            &backward_states_e[i]);

    for (int i = 0; i < backward_states_d.size(); i++)
        params_and_states_decoder.backward_states.push_back(
            &backward_states_d[i]);

    if (recompute_outputs) {
        // RESET OUPUTS

        // Write the parameters and hidden states
        params_and_states_encoder.write_params(test_saving_paths_encoder,
                                               false);
        params_and_states_decoder.write_params(test_saving_paths_decoder,
                                               false);

        // Write the forward hidden states
        write_vector_to_csv(test_saving_paths_encoder.forward_states_path,
                            "mz,Sz,ma,Sa,J",
                            params_and_states_encoder.forward_states);
        write_vector_to_csv(test_saving_paths_decoder.forward_states_path,
                            "mz,Sz,ma,Sa,J",
                            params_and_states_decoder.forward_states);

        // Write the backward hidden states
        write_vector_to_csv(test_saving_paths_encoder.backward_states_path,
                            backward_states_header_e,
                            params_and_states_encoder.backward_states);
        write_vector_to_csv(test_saving_paths_decoder.backward_states_path,
                            backward_states_header_d,
                            params_and_states_decoder.backward_states);

    } else {
        // PERFORM TESTS

        // Read the saved reference parameters
        TestParamAndStates params_and_states_encoder_reference(net_e);
        TestParamAndStates params_and_states_decoder_reference(net_d);

        params_and_states_encoder_reference.read_params(
            test_saving_paths_encoder, false);
        params_and_states_decoder_reference.read_params(
            test_saving_paths_decoder, false);

        net_e.theta_gpu.copy_host_to_device();
        net_d.theta_gpu.copy_host_to_device();

        // Compare optimal values with the ones we got
        if (!compare_vectors(params_and_states_encoder_reference.weights,
                             params_and_states_encoder.weights, data,
                             "encoder weights") ||
            !compare_vectors(params_and_states_encoder_reference.weights_sc,
                             params_and_states_encoder.weights_sc, data,
                             "encoder weights for residual network") ||
            !compare_vectors(params_and_states_encoder_reference.bias,
                             params_and_states_encoder.bias, data,
                             "encoder bias") ||
            !compare_vectors(params_and_states_encoder_reference.bias_sc,
                             params_and_states_encoder.bias_sc, data,
                             "encoder bias for residual network")) {
            std::cout << "\033[1;31mTest for encoder PARAMS has FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        if (!compare_vectors(params_and_states_decoder_reference.weights,
                             params_and_states_decoder.weights, data,
                             "decoder weights") ||
            !compare_vectors(params_and_states_decoder_reference.weights_sc,
                             params_and_states_decoder.weights_sc, data,
                             "decoder weights for residual network") ||
            !compare_vectors(params_and_states_decoder_reference.bias,
                             params_and_states_decoder.bias, data,
                             "decoder bias") ||
            !compare_vectors(params_and_states_decoder_reference.bias_sc,
                             params_and_states_decoder.bias_sc, data,
                             "decoder bias for residual network")) {
            std::cout << "\033[1;31mTest for decoder PARAMS has FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved forward hidden states reference
        for (int i = 0; i < 5; i++)
            params_and_states_encoder_reference.forward_states.push_back(
                new std::vector<float>());
        for (int i = 0; i < 5; i++)
            params_and_states_decoder_reference.forward_states.push_back(
                new std::vector<float>());

        read_vector_from_csv(
            test_saving_paths_encoder.forward_states_path,
            params_and_states_encoder_reference.forward_states);
        read_vector_from_csv(
            test_saving_paths_decoder.forward_states_path,
            params_and_states_decoder_reference.forward_states);

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(params_and_states_encoder_reference.forward_states,
                             params_and_states_encoder.forward_states, data,
                             "encoder forward hidden states")) {
            std::cout << "\033[1;31mTest for encoder FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Compare the saved forward hidden states with the ones we got
        if (!compare_vectors(params_and_states_decoder_reference.forward_states,
                             params_and_states_decoder.forward_states, data,
                             "decoder forward hidden states")) {
            std::cout << "\033[1;31mTest for decoder FORWARD HIDDEN STATES has "
                         "FAILED in " +
                             data + " data\033[0m\n"
                      << std::endl;
            return false;
        }

        // Read the saved backward hidden states reference
        for (int i = 0; i < 2 * (net_prop_e.layers.size() - 2); i++)
            params_and_states_encoder_reference.backward_states.push_back(
                new std::vector<float>());
        for (int i = 0; i < 2 * (net_prop_d.layers.size() - 2); i++)
            params_and_states_decoder_reference.backward_states.push_back(
                new std::vector<float>());

        read_vector_from_csv(
            test_saving_paths_encoder.backward_states_path,
            params_and_states_encoder_reference.backward_states);
        read_vector_from_csv(
            test_saving_paths_decoder.backward_states_path,
            params_and_states_decoder_reference.backward_states);

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(
                params_and_states_encoder_reference.backward_states,
                params_and_states_encoder.backward_states, data,
                "encoder backward hidden states")) {
            std::cout
                << "\033[1;31mTest for encoder BACKWARD HIDDEN STATES has "
                   "FAILED in " +
                       data + " data\033[0m\n"
                << std::endl;
            return false;
        }

        // Compare the saved backward hidden states with the ones we got
        if (!compare_vectors(
                params_and_states_decoder_reference.backward_states,
                params_and_states_decoder.backward_states, data,
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

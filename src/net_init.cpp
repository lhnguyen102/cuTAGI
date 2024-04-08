///////////////////////////////////////////////////////////////////////////////
// File:         net_init.cpp
// Description:  Network initialization
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 07, 2021
// Updated:      October 03, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/net_init.h"

void net_init(std::string &net_file, std::string &device, Network &net,
              Param &theta, NetState &state, IndexOut &idx)
/* Initalize the network
   Args:
    net_file: Filename of the network
    device: cuda or cpu
    net: Network
    theta: Parameters of the network
    state: Hidden states of the network
    idx: Indices of the network
 */
{
    // Add extestion to file name
    std::string net_file_ext = net_file + ".txt";

    // Initialize network
    load_cfg(net_file_ext, net);
    net_default(net);
    get_net_props(net);
    get_similar_layer(net);
    net.device = device;

    // Check feature availability
    check_feature_availability(net);

    // Indices
    tagi_idx(idx, net);
    index_default(idx);
    theta = initialize_param(net);
    state = initialize_net_states(net);
}

void map_config_to_prop(NetConfig &config, Network &net) {
    net.layers = config.layers;
    net.nodes = config.nodes;
    net.kernels = config.kernels;
    net.strides = config.strides;
    net.widths = config.widths;
    net.heights = config.heights;
    net.filters = config.filters;
    net.pads = config.pads;
    net.pad_types = config.pad_types;
    net.shortcuts = config.shortcuts;
    net.activations = config.activations;
    net.sigma_v = config.sigma_v;
    net.sigma_v_min = config.sigma_v_min;
    net.sigma_x = config.sigma_x;
    net.decay_factor_sigma_v = config.decay_factor_sigma_v;
    net.mu_v2b = config.mu_v2b;
    net.sigma_v2b = config.sigma_v2b;
    net.noise_gain = config.noise_gain;
    net.noise_gain = config.out_gain;
    net.batch_size = config.batch_size;
    net.input_seq_len = config.input_seq_len;
    net.output_seq_len = config.output_seq_len;
    net.seq_stride = config.seq_stride;
    net.multithreading = config.multithreading;
    net.collect_derivative = config.collect_derivative;
    net.is_full_cov = config.is_full_cov;
    net.init_method = config.init_method;
    net.noise_type = config.noise_type;
    net.device = config.device;
}

void reset_net_batchsize(std::string &net_file, std::string &device,
                         Network &net, NetState &state, IndexOut &idx,
                         int batch_size)
/* Reset network's batchsize.

Args:
    net_file: Filename of the network
    device: cuda or cpu
    net: Network
    state: Hidden states of the network
    idx: Indices of the network

*NOTE: It is commonly used for initializing the test network where the batch
size of train network is incompatible with the test set.
*/
{
    // Add extestion to file name
    std::string net_file_ext = net_file + ".txt";

    // Initialize network
    load_cfg(net_file_ext, net);
    net_default(net);
    net.batch_size = batch_size;
    get_net_props(net);
    get_similar_layer(net);
    net.device = device;

    // Check feature availability
    check_feature_availability(net);

    // Indices
    tagi_idx(idx, net);
    index_default(idx);
    state = initialize_net_states(net);
}

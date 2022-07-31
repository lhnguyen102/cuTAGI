///////////////////////////////////////////////////////////////////////////////
// File:         net_init.cpp
// Description:  Network initialization
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 07, 2021
// Updated:      July 03, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
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

///////////////////////////////////////////////////////////////////////////////
// File:         net_init.cpp
// Description:  Network initialization
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 07, 2021
// Updated:      April 10, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/net_init.h"

void net_init(std::string &net_file, Network &net, Param &theta,
              NetState &state, IndexOut &idx)
/* Initalize the network
   Args:
    net_file: Filename of the network
    net: Network
    theta: Parameters of the network
    state: Hidden states of the network
    idx: Indices of the network
 */
{
    // Add extestion to file name
    std::string net_file_ext = net_file + ".txt";

    // Initialize network
    net = load_cfg(net_file_ext);
    net_default(net);
    get_net_props(net);
    get_similar_layer(net);
    tagi_idx(idx, net);
    index_default(idx);
    theta = initialize_param(net);
    state = initialize_net_states(net);
}

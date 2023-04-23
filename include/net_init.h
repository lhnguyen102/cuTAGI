///////////////////////////////////////////////////////////////////////////////
// File:         net_init.h
// Description:  Header for Network initialization
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 07, 2021
// Updated:      October 03, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>

#include "feature_availability.h"
#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"
#include "user_input.h"

void net_init(std::string &net_input, std::string &device, Network &net,
              Param &theta, NetState &state, IndexOut &idx);
void map_config_to_prop(NetConfig &config, Network &net);

void reset_net_batchsize(std::string &net_file, std::string &device,
                         Network &net, NetState &state, IndexOut &idx,
                         int batch_size);
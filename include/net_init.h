///////////////////////////////////////////////////////////////////////////////
// File:         net_init.h
// Description:  Header for Network initialization
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 07, 2021
// Updated:      April 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>

#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

void net_init(std::string &net_file, Network &net, Param &theta,
              NetState &state, IndexOut &idx);

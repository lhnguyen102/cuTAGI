///////////////////////////////////////////////////////////////////////////////
// File:         derivative_calculation_cpu.h
// Description:  Header file for derivative calculation of neural networks
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      July 17, 2022
// Updated:      July 19, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <algorithm>
#include <iostream>
#include <thread>
// #include <vector>

#include "common.h"
#include "net_prop.h"
#include "struct_var.h"

void compute_network_derivatives(Network &net, Param &theta, NetState &state,
                                 int l);

void compute_activation_derivatives(Network &net, NetState &state, int j);
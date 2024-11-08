#pragma once
#include <algorithm>
#include <iostream>
#include <thread>
// #include <vector>

#include "common.h"
#include "net_prop.h"
#include "struct_var.h"

void compute_network_derivatives_cpu(Network &net, Param &theta,
                                     NetState &state, int l);

void compute_activation_derivatives_cpu(Network &net, NetState &state, int j);
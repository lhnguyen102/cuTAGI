#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "net_prop.h"
#include "struct_var.h"

void compute_network_derivatives(Network &net, ParamGPU &theta, StateGPU &state,
                                 int l);

void compute_activation_derivatives(Network &net, StateGPU &state, int j);
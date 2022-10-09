///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network.cuh
// Description:  Header file for tagi network including feed forward & backward
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      October 08, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "feed_forward.cuh"
#include "global_param_update.cuh"
#include "param_feed_backward.cuh"
#include "state_feed_backward.cuh"
#include "tagi_network_base.h"

class TagiNetwork : public TagiNetworkBase {
   public:
    IndexGPU idx_gpu;
    StateGPU state_gpu;
    ParamGPU theta_gpu;
    DeltaStateGPU d_state_gpu;
    DeltaParamGPU d_theta_gpu;
    InputGPU net_input_gpu;
    ObsGPU obs_gpu;
    float *d_ma, *d_Sa;
    std::vector<float> ma, Sa;
    size_t num_output_bytes;

    TagiNetwork(Network &net_prop);
    ~TagiNetwork();
    void feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                      std::vector<float> &Sx_f);
    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud);
    void param_feed_backward();
    void get_network_outputs();

   private:
    void init_net();
    void allocate_output_memory();
    void output_to_device();
    void output_to_host();
};

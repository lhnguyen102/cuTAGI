///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network.cuh
// Description:  Header file for tagi network including feed forward & backward
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      November 06, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "data_transfer.cuh"
#include "derivative_calcul.cuh"
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
    ConnectorInputGPU connected_input_gpu;
    ObsGPU obs_gpu;
    float *d_ma = nullptr, *d_Sa = nullptr, *d_mz = nullptr, *d_Sz = nullptr,
          *d_J = nullptr;
    float *d_ma_init = nullptr, *d_Sa_init = nullptr, *d_mz_init = nullptr,
          *d_Sz_init = nullptr, *d_J_init = nullptr;
    size_t num_output_bytes;
    size_t num_input_bytes;

    TagiNetwork(Network &net_prop);
    TagiNetwork();

    ~TagiNetwork();

    void feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                      std::vector<float> &Sx_f);

    void connected_feed_forward(std::vector<float> &ma, std::vector<float> &Sa,
                                std::vector<float> &mz, std::vector<float> &Sz,
                                std::vector<float> &J);

    void state_feed_backward(std::vector<float> &y, std::vector<float> &Sy,
                             std::vector<int> &idx_ud);

    void param_feed_backward();

    void get_network_outputs();

    void get_predictions();

    void get_all_network_outputs();

    void get_all_network_inputs();

    std::tuple<std::vector<float>, std::vector<float>> get_derivatives(
        int layer);

    std::tuple<std::vector<float>, std::vector<float>> get_inovation_mean_var(
        int layer);

    std::tuple<std::vector<float>, std::vector<float>>
    get_state_delta_mean_var();

    void set_parameters(Param &init_theta);

    Param get_parameters();

   private:
    void init_net();
    void allocate_output_memory();
    void output_to_device();
    void output_to_host();
    void all_outputs_to_host();
    void allocate_input_memory();
    void input_to_device();
    void all_inputs_to_host();
};

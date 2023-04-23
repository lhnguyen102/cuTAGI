///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.h
// Description:  Header file for tagi network including feed forward & backward
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 03, 2022
// Updated:      November 06, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "tagi_network_base.h"

class TagiNetworkCPU : public TagiNetworkBase {
   public:
    TagiNetworkCPU(Network &net_prop);

    ~TagiNetworkCPU();

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
};
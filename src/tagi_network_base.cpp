///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_base.cpp
// Description:  tagi network base
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 08, 2022
// Updated:      November 06, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/tagi_network_base.h"

TagiNetworkBase::TagiNetworkBase() {}

TagiNetworkBase::~TagiNetworkBase() {}

void TagiNetworkBase::feed_forward(std::vector<float> &x,
                                   std::vector<float> &Sx,
                                   std::vector<float> &Sx_f) {}

void TagiNetworkBase::connected_feed_forward(std::vector<float> &ma,
                                             std::vector<float> &Sa,
                                             std::vector<float> &mz,
                                             std::vector<float> &Sz,
                                             std::vector<float> &J) {}

void TagiNetworkBase::state_feed_backward(std::vector<float> &y,
                                          std::vector<float> &Sy,
                                          std::vector<int> &idx_ud) {}

void TagiNetworkBase::param_feed_backward() {}

void TagiNetworkBase::get_network_outputs() {}

void TagiNetworkBase::get_predictions(){};

std::tuple<std::vector<float>, std::vector<float>>
TagiNetworkBase::get_derivatives(int layer) {
    return {};
};

void TagiNetworkBase::get_all_network_outputs() {}

void TagiNetworkBase::get_all_network_inputs() {}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetworkBase::get_inovation_mean_var(int layer) {
    std::vector<float> delta_m(1, 0);
    std::vector<float> delta_S(1, 0);
    return {delta_m, delta_S};
}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetworkBase::get_state_delta_mean_var() {
    std::vector<float> delta_mz(1, 0);
    std::vector<float> delta_Sz(1, 0);
    return {delta_mz, delta_Sz};
}

void TagiNetworkBase::set_parameters(Param &init_theta){};

Param TagiNetworkBase::get_parameters() { return this->theta; }
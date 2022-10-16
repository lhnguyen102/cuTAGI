///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_base.cpp
// Description:  tagi network base
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 08, 2022
// Updated:      October 16, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/tagi_network_base.h"

TagiNetworkBase::TagiNetworkBase() {}
TagiNetworkBase::~TagiNetworkBase() {}
void TagiNetworkBase::feed_forward(std::vector<float> &x,
                                   std::vector<float> &Sx,
                                   std::vector<float> &Sx_f) {}

void TagiNetworkBase::state_feed_backward(std::vector<float> &y,
                                          std::vector<float> &Sy,
                                          std::vector<int> &idx_ud) {}

void TagiNetworkBase::param_feed_backward() {}
void TagiNetworkBase::get_network_outputs(){};
void TagiNetworkBase::set_parameters(Param &init_theta){};
Param TagiNetworkBase::get_parameters() { return this->theta; }
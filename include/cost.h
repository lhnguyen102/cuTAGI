///////////////////////////////////////////////////////////////////////////////
// File:         cost.h
// Description:  Header file for cost function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 19, 2022
// Updated:      May 15, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserve.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "common.h"
#include "struct_var.h"

HrSoftmax class_to_obs(int n_classes);

std::tuple<std::vector<int>, std::vector<float>> get_error(
    std::vector<float> &mz, std::vector<float> &Sz, std::vector<int> &labels,
    HrSoftmax &hs, int n_classes, int B);

float mean_squared_error(std::vector<float> &pred, std::vector<float> &obs);
float avg_univar_log_lik(std::vector<float> &x, std::vector<float> &mu,
                         std::vector<float> &sigma);
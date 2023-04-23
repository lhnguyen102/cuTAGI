///////////////////////////////////////////////////////////////////////////////
// File:         feature_support.h
// Description:  Header file for the feature-support check
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      June 05, 2022
// Updated:      December 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "net_prop.h"
#include "struct_var.h"

void check_feature_availability(Network &net);
bool is_cuda_available();
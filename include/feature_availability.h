///////////////////////////////////////////////////////////////////////////////
// File:         feature_support.h
// Description:  Header file for the feature-support check
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      June 05, 2022
// Updated:      June 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "net_prop.h"
#include "struct_var.h"

void cpu_feature_availability(Network &net, LayerLabel &layer_names);
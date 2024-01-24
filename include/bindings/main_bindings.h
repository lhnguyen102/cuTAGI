///////////////////////////////////////////////////////////////////////////////
// File:         main_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      January 21, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "activation_bindings.h"
#include "base_layer_bindings.h"
#include "base_output_updater_bindings.h"
#include "conv2d_layer_bindings.h"
#include "data_struct_bindings.h"
#include "linear_layer_bindings.h"
#include "pooling_layer_bindings.h"
#include "sequential_bindings.h"

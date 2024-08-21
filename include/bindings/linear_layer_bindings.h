///////////////////////////////////////////////////////////////////////////////
// File:         linear_layer_binding.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 29, 2023
// Updated:      December 29, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../linear_layer.h"

void bind_linear_layer(pybind11::module_& modo);
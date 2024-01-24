///////////////////////////////////////////////////////////////////////////////
// File:         base_layer_binding.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      December 30, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../base_layer.h"

void bind_base_layer(pybind11::module_& modo);
///////////////////////////////////////////////////////////////////////////////
// File:         norm_layer_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 08, 2024
// Updated:      February 08, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/pybind11.h>

void bind_layernorm_layer(pybind11::module_& modo);
void bind_batchnorm_layer(pybind11::module_& modo);
///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 20, 2024
// Updated:      January 20, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/pybind11.h>

void bind_avgpool2d_layer(pybind11::module_& modo);
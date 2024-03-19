///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 19, 2024
// Updated:      March 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

void bind_convtranspose2d_layer(pybind11::module_& modo);
///////////////////////////////////////////////////////////////////////////////
// File:         slstm_layer_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 21, 2024
// Updated:      August 21, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <pybind11/pybind11.h>

#include "../slstm_layer.h"

void bind_slstm_layer(pybind11::module_& modo);
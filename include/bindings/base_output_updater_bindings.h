///////////////////////////////////////////////////////////////////////////////
// File:         base_output_updater_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 31, 2023
// Updated:      August 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
namespace pybind11 {
class module_;
}

void bind_output_updater(pybind11::module_& modo);
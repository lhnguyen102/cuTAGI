///////////////////////////////////////////////////////////////////////////////
// File:         data_struct_bindings.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 31, 2023
// Updated:      December 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
namespace pybind11 {
class module_;
}

void bind_base_hidden_states(pybind11::module_& modo);

void bind_base_delta_state_states(pybind11::module_& modo);

///////////////////////////////////////////////////////////////////////////////
// File:         base_output_updater_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 31, 2023
// Updated:      December 31, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/base_output_updater_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/base_output_updater.h"

void bind_output_updater(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<OutputUpdater>(modo, "OutputUpdater")
        .def(pybind11::init<const std::string&>())
        .def("update", &OutputUpdater::update)
        .def("update_using_indices", &OutputUpdater::update_using_indices)
        .def_readwrite("device", &OutputUpdater::device);
}
#include "../include/bindings/base_output_updater_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/base_output_updater.h"

void bind_output_updater(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<OutputUpdater, std::shared_ptr<OutputUpdater>>(
        modo, "OutputUpdater")
        .def(pybind11::init<const std::string>())
        .def("update", &OutputUpdater::update)
        .def("update_using_indices", &OutputUpdater::update_using_indices,
             pybind11::arg("output_states"), pybind11::arg("mu_obs"),
             pybind11::arg("var_obs"), pybind11::arg("selected_idx"),
             pybind11::arg("delta_states"),
             "Updates the output states using specified indices.")
        .def("update_heteros", &OutputUpdater::update_heteros,
             pybind11::arg("output_states"), pybind11::arg("mu_obs"),
             pybind11::arg("delta_states"),
             "Updates the output given heteroscedastic noise.")
        .def_readwrite("device", &OutputUpdater::device);
}

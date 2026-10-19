#include "../include/bindings/base_output_updater_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/base_output_updater.h"

template <typename T>
using ContiguousArray =
    pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;

void bind_output_updater(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<OutputUpdater, std::shared_ptr<OutputUpdater>>(
        modo, "OutputUpdater")
        .def(pybind11::init<const std::string>())
        .def(
            "update",
            [](OutputUpdater& self, BaseHiddenStates& output_states,
               ContiguousArray<float> mu_obs, ContiguousArray<float> var_obs,
               BaseDeltaStates& delta_states) {
                std::vector<float> mu_vec(mu_obs.data(),
                                          mu_obs.data() + mu_obs.size());
                std::vector<float> var_vec(var_obs.data(),
                                           var_obs.data() + var_obs.size());
                self.update(output_states, mu_vec, var_vec, delta_states);
            },
            pybind11::arg("output_states"), pybind11::arg("mu_obs"),
            pybind11::arg("var_obs"), pybind11::arg("delta_states"))
        .def(
            "update_using_indices",
            [](OutputUpdater& self, BaseHiddenStates& output_states,
               ContiguousArray<float> mu_obs, ContiguousArray<float> var_obs,
               ContiguousArray<int> selected_idx,
               BaseDeltaStates& delta_states) {
                std::vector<float> mu_vec(mu_obs.data(),
                                          mu_obs.data() + mu_obs.size());
                std::vector<float> var_vec(var_obs.data(),
                                           var_obs.data() + var_obs.size());
                std::vector<int> idx_vec(
                    selected_idx.data(),
                    selected_idx.data() + selected_idx.size());
                self.update_using_indices(output_states, mu_vec, var_vec,
                                          idx_vec, delta_states);
            },
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

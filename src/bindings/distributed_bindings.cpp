#include "../include/bindings/distributed_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "../include/data_struct.h"
#include "../include/ddp.h"
#include "../include/sequential.h"

void bind_distributed(pybind11::module_& m) {
    pybind11::class_<DDPConfig>(m, "DDPConfig")
        .def(pybind11::init<const std::vector<int>&, const std::string&, int,
                            int>(),
             pybind11::arg("device_ids"), pybind11::arg("backend") = "nccl",
             pybind11::arg("rank") = 0, pybind11::arg("world_size") = 1)
        .def_readwrite("device_ids", &DDPConfig::device_ids)
        .def_readwrite("backend", &DDPConfig::backend)
        .def_readwrite("rank", &DDPConfig::rank)
        .def_readwrite("world_size", &DDPConfig::world_size);

#if defined(DISTRIBUTED_AVAILABLE)
    pybind11::class_<NCCLCommunicator>(m, "NCCLCommunicator")
        .def(pybind11::init<int, const std::vector<int>&>())
        .def("all_reduce", &NCCLCommunicator::all_reduce)
        .def("barrier", &NCCLCommunicator::barrier)
        .def("check_async_error", &NCCLCommunicator::check_async_error)
        .def("get_world_size", &NCCLCommunicator::get_world_size)
        .def("get_rank", &NCCLCommunicator::get_rank);
#endif

    pybind11::class_<DDPSequential, std::shared_ptr<DDPSequential>>(
        m, "DDPSequential")
        .def(pybind11::init<std::shared_ptr<Sequential>, const DDPConfig&,
                            bool>(),
             pybind11::arg("model"), pybind11::arg("config"),
             pybind11::arg("average") = true)
        .def("forward",
             [](DDPSequential& self, pybind11::object arg1,
                pybind11::object arg2 = pybind11::none()) {
                 if (pybind11::isinstance<pybind11::array_t<float>>(arg1)) {
                     pybind11::array_t<float> mu_a_np =
                         arg1.cast<pybind11::array_t<float>>();
                     pybind11::array_t<float> var_a_np =
                         arg2.is_none() ? pybind11::array_t<float>()
                                        : arg2.cast<pybind11::array_t<float>>();
                     self.forward_py(mu_a_np, var_a_np);
                 } else {
                     // Handle the case for BaseHiddenStates
                     BaseHiddenStates& input_states =
                         arg1.cast<BaseHiddenStates&>();
                     self.forward(input_states);
                 }
             })
        .def("backward", &DDPSequential::backward)
        .def("step", &DDPSequential::step)
        .def("train", &DDPSequential::train)
        .def("eval", &DDPSequential::eval)
        .def("barrier", &DDPSequential::barrier)
        .def("get_outputs", &DDPSequential::get_outputs)
        .def("output_to_host", &DDPSequential::output_to_host)
        .def("get_device_with_index", &DDPSequential::get_device_with_index);
}

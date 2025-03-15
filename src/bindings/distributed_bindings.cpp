#include "../include/bindings/distributed_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "../include/distributed.h"
#include "../include/sequential.h"

void bind_distributed(pybind11::module_& m) {
    pybind11::class_<DistributedConfig>(m, "DistributedConfig")
        .def(pybind11::init<const std::vector<int>&, const std::string&, int,
                            int>(),
             pybind11::arg("device_ids"), pybind11::arg("backend") = "nccl",
             pybind11::arg("rank") = 0, pybind11::arg("world_size") = 1)
        .def_readwrite("device_ids", &DistributedConfig::device_ids)
        .def_readwrite("backend", &DistributedConfig::backend)
        .def_readwrite("rank", &DistributedConfig::rank)
        .def_readwrite("world_size", &DistributedConfig::world_size);

#if defined(DISTRIBUTED_AVAILABLE)
    pybind11::class_<NCCLCommunicator>(m, "NCCLCommunicator")
        .def(pybind11::init<int, const std::vector<int>&>())
        .def("all_reduce", &NCCLCommunicator::all_reduce)
        .def("barrier", &NCCLCommunicator::barrier)
        .def("check_async_error", &NCCLCommunicator::check_async_error)
        .def("get_world_size", &NCCLCommunicator::get_world_size)
        .def("get_rank", &NCCLCommunicator::get_rank);
#endif

    pybind11::class_<DistributedSequential,
                     std::shared_ptr<DistributedSequential>>(
        m, "DistributedSequential")
        .def(pybind11::init<std::shared_ptr<Sequential>,
                            const DistributedConfig&, bool>(),
             pybind11::arg("model"), pybind11::arg("config"),
             pybind11::arg("average") = true)
        .def("forward", &DistributedSequential::forward)
        .def("backward", &DistributedSequential::backward)
        .def("step", &DistributedSequential::step)
        .def("train", &DistributedSequential::train)
        .def("eval", &DistributedSequential::eval);
}
#include "../include/bindings/pooling_layer_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/max_pooling_layer.h"
#include "../include/pooling_layer.h"

void bind_avgpool2d_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<AvgPool2d, std::shared_ptr<AvgPool2d>, BaseLayer>(
        modo, "AvgPool2d")
        .def(pybind11::init<size_t, int, int, int>(),
             pybind11::arg("kernel_size"), pybind11::arg("stride") = -1,
             pybind11::arg("padding") = 0, pybind11::arg("padding_type") = 0)
        .def("get_layer_info", &AvgPool2d::get_layer_info)
        .def("get_layer_name", &AvgPool2d::get_layer_name)
        .def("forward", &AvgPool2d::forward)
        .def("backward", &AvgPool2d::backward);
    ;
}

void bind_maxpool2d_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MaxPool2d, std::shared_ptr<MaxPool2d>, BaseLayer>(
        modo, "MaxPool2d")
        .def(pybind11::init<size_t, int, int, int>(),
             pybind11::arg("kernel_size"), pybind11::arg("stride") = -1,
             pybind11::arg("padding") = 0, pybind11::arg("padding_type") = 0)
        .def("get_layer_info", &MaxPool2d::get_layer_info)
        .def("get_layer_name", &MaxPool2d::get_layer_name)
        .def("forward", &MaxPool2d::forward)
        .def("backward", &MaxPool2d::backward);
    ;
}
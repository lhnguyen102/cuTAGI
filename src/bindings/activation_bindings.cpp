///////////////////////////////////////////////////////////////////////////////
// File:         activation_bindings.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 30, 2023
// Updated:      December 30, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/activation_bindings.h"

void bind_relu(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<ReLU, std::shared_ptr<ReLU>, BaseLayer>(modo, "ReLU")
        .def(pybind11::init<>())
        .def("get_layer_info", &ReLU::get_layer_info)
        .def("get_layer_name", &ReLU::get_layer_name)
        .def("forward", &ReLU::forward)
        .def("update_weights", &ReLU::update_weights)
        .def("update_biases", &ReLU::update_biases)
        .def("load", &ReLU::load)
        .def("save", &ReLU::save)
        .def("to_cuda", &ReLU::to_cuda);
}

void bind_sigmoid(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Sigmoid, std::shared_ptr<Sigmoid>, BaseLayer>(modo,
                                                                   "Sigmoid")
        .def(pybind11::init<>())
        .def("get_layer_info", &Sigmoid::get_layer_info)
        .def("get_layer_name", &Sigmoid::get_layer_name)
        .def("forward", &Sigmoid::forward)
        .def("update_weights", &Sigmoid::update_weights)
        .def("update_biases", &Sigmoid::update_biases)
        .def("load", &Sigmoid::load)
        .def("save", &Sigmoid::save)
        .def("to_cuda", &Sigmoid::to_cuda);
}

void bind_tanh(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Tanh, std::shared_ptr<Tanh>, BaseLayer>(modo, "Tanh")
        .def(pybind11::init<>())
        .def("get_layer_info", &Tanh::get_layer_info)
        .def("get_layer_name", &Tanh::get_layer_name)
        .def("forward", &Tanh::forward)
        .def("update_weights", &Tanh::update_weights)
        .def("update_biases", &Tanh::update_biases)
        .def("load", &Tanh::load)
        .def("save", &Tanh::save)
        .def("to_cuda", &Tanh::to_cuda);
}

void bind_mixture_relu(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MixtureRelu, std::shared_ptr<MixtureRelu>, BaseLayer>(
        modo, "MixtureRelu")
        .def(pybind11::init<>())
        .def("get_layer_info", &MixtureRelu::get_layer_info)
        .def("get_layer_name", &MixtureRelu::get_layer_name)
        .def("forward", &MixtureRelu::forward)
        .def("update_weights", &MixtureRelu::update_weights)
        .def("update_biases", &MixtureRelu::update_biases)
        .def("load", &MixtureRelu::load)
        .def("save", &MixtureRelu::save)
        .def("to_cuda", &MixtureRelu::to_cuda);
}

void bind_mixture_sigmoid(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MixtureSigmoid, std::shared_ptr<MixtureSigmoid>,
                     BaseLayer>(modo, "MixtureSigmoid")
        .def(pybind11::init<>())
        .def("get_layer_info", &MixtureSigmoid::get_layer_info)
        .def("get_layer_name", &MixtureSigmoid::get_layer_name)
        .def("forward", &MixtureSigmoid::forward)
        .def("update_weights", &MixtureSigmoid::update_weights)
        .def("update_biases", &MixtureSigmoid::update_biases)
        .def("load", &MixtureSigmoid::load)
        .def("save", &MixtureSigmoid::save)
        .def("to_cuda", &MixtureSigmoid::to_cuda);
}

void bind_mixture_tanh(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MixtureTanh, std::shared_ptr<MixtureTanh>, BaseLayer>(
        modo, "MixtureTanh")
        .def(pybind11::init<>())
        .def("get_layer_info", &MixtureTanh::get_layer_info)
        .def("get_layer_name", &MixtureTanh::get_layer_name)
        .def("forward", &MixtureTanh::forward)
        .def("update_weights", &MixtureTanh::update_weights)
        .def("update_biases", &MixtureTanh::update_biases)
        .def("load", &MixtureTanh::load)
        .def("save", &MixtureTanh::save)
        .def("to_cuda", &MixtureTanh::to_cuda);
}

void bind_softplus(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Softplus, std::shared_ptr<Softplus>, BaseLayer>(
        modo, "MixtureSoftplus")
        .def(pybind11::init<>())
        .def("get_layer_info", &Softplus::get_layer_info)
        .def("get_layer_name", &Softplus::get_layer_name)
        .def("forward", &Softplus::forward)
        .def("update_weights", &Softplus::update_weights)
        .def("update_biases", &Softplus::update_biases)
        .def("load", &Softplus::load)
        .def("save", &Softplus::save)
        .def("to_cuda", &Softplus::to_cuda);
}

void bind_leakyrelu(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<LeakyRelu, std::shared_ptr<LeakyRelu>, BaseLayer>(
        modo, "LeakyRelu")
        .def(pybind11::init<>())
        .def("get_layer_info", &LeakyRelu::get_layer_info)
        .def("get_layer_name", &LeakyRelu::get_layer_name)
        .def("forward", &LeakyRelu::forward)
        .def("update_weights", &LeakyRelu::update_weights)
        .def("update_biases", &LeakyRelu::update_biases)
        .def("load", &LeakyRelu::load)
        .def("save", &LeakyRelu::save)
        .def("to_cuda", &LeakyRelu::to_cuda);
}

void bind_softmax(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Softmax, std::shared_ptr<Softmax>, BaseLayer>(modo,
                                                                   "Softmax")
        .def(pybind11::init<>())
        .def("get_layer_info", &Softmax::get_layer_info)
        .def("get_layer_name", &Softmax::get_layer_name)
        .def("forward", &Softmax::forward)
        .def("update_weights", &Softmax::update_weights)
        .def("update_biases", &Softmax::update_biases)
        .def("load", &Softmax::load)
        .def("save", &Softmax::save)
        .def("to_cuda", &Softmax::to_cuda);
}

#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../activation.h"

void bind_relu(pybind11::module_& modo);

void bind_sigmoid(pybind11::module_& modo);

void bind_tanh(pybind11::module_& modo);

void bind_mixture_relu(pybind11::module_& modo);

void bind_mixture_sigmoid(pybind11::module_& modo);

void bind_mixture_tanh(pybind11::module_& modo);

void bind_softplus(pybind11::module_& modo);

void bind_leakyrelu(pybind11::module_& modo);

void bind_softmax(pybind11::module_& modo);

void bind_even_exp(pybind11::module_& modo);

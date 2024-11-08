#pragma once
#include <pybind11/pybind11.h>

void bind_layernorm_layer(pybind11::module_& modo);
void bind_batchnorm_layer(pybind11::module_& modo);
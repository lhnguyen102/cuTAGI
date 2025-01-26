#pragma once
#include <pybind11/pybind11.h>

void bind_avgpool2d_layer(pybind11::module_& modo);

void bind_maxpool2d_layer(pybind11::module_& modo);
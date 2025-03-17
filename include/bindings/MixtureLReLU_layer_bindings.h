#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../MixtureLReLU_layer.h"

void bind_MixtureLReLU_layer(pybind11::module_ &modo);
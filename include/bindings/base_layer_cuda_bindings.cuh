#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../base_layer_cuda.h"

void base_layer_cuda_binding(pybind11::module_& modo)
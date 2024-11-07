#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../base_layer.h"

void bind_base_layer(pybind11::module_& modo);
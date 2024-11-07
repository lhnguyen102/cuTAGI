#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../slinear_layer.h"

void bind_slinear_layer(pybind11::module_& modo);
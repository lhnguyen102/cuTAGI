#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../embedding_cpu.h"

void bind_embedding(pybind11::module_& modo);

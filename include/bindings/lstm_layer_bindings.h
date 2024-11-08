#pragma once
#include <pybind11/pybind11.h>

#include "../lstm_layer.h"

void bind_lstm_layer(pybind11::module_& modo);
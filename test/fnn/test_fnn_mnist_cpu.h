///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_mnist_cpu.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 25, 2023
// Updated:      November 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>

#include "../../include/activation_layer_cpu.h"
#include "../../include/base_output_updater.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/layer_stack_cpu.h"

int test_fnn_mnist();
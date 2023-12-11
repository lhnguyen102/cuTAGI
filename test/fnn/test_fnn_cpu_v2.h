///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu_v2.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 25, 2023
// Updated:      October 25, 2023
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
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/fc_cpu_v2.h"
#include "../../include/layer_stack_cpu.h"

int test_fnn_cpu_v2();
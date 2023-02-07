///////////////////////////////////////////////////////////////////////////////
// File:         test_lstm_cpu.h
// Description:  Header file for lstm unitest
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 04, 2022
// Updated:      September 04, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "../../include/common.h"
#include "../../include/cost.h"
#include "../../include/dataloader.h"
#include "../../include/derivative_calcul.cuh"
#include "../../include/feed_forward.cuh"
#include "../../include/global_param_update.cuh"
#include "../../include/indices.h"
#include "../../include/net_init.h"
#include "../../include/net_prop.h"
#include "../../include/param_feed_backward.cuh"
#include "../../include/state_feed_backward.cuh"
#include "../../include/struct_var.h"
#include "../../include/tagi_network.cuh"
#include "../../include/task.cuh"
#include "../../include/utils.h"
#include "../test_utils.h"

void test_fnn_derivatives();
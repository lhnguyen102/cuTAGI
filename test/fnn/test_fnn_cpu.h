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
#include "../../include/data_transfer_cpu.h"
#include "../../include/dataloader.h"
#include "../../include/derivative_calcul_cpu.h"
#include "../../include/feed_forward_cpu.h"
#include "../../include/global_param_update_cpu.h"
#include "../../include/indices.h"
#include "../../include/net_init.h"
#include "../../include/net_prop.h"
#include "../../include/param_feed_backward_cpu.h"
#include "../../include/state_feed_backward_cpu.h"
#include "../../include/struct_var.h"
#include "../../include/tagi_network_cpu.h"
#include "../../include/task_cpu.h"
#include "../../include/test_utils.h"
#include "../../include/user_input.h"
#include "../../include/utils.h"

bool test_fnn_cpu(bool recompute_outputs, std::string date, std::string arch,
                  std::string data);
///////////////////////////////////////////////////////////////////////////////
// File:         task.cuh
// Description:  Header for task command
//               that uses TAGI approach.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      October 16, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "common.h"
#include "cost.h"
#include "data_transfer.cuh"
#include "dataloader.h"
#include "feed_forward.cuh"
#include "global_param_update.cuh"
#include "gpu_debug_utils.h"
#include "indices.h"
#include "lstm_feed_forward.cuh"
#include "net_init.h"
#include "net_prop.h"
#include "param_feed_backward.cuh"
#include "state_feed_backward.cuh"
#include "struct_var.h"
#include "tagi_network.cuh"
#include "user_input.h"
#include "utils.h"

void task_command(UserInput &user_input, SavePath &path);

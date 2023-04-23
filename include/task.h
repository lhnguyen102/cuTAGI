///////////////////////////////////////////////////////////////////////////////
// File:         task.cu
// Description:  providing different tasks such as regression, classification
//               that uses TAGI approach.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      April 18, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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
#include "indices.h"
#include "net_init.h"
#include "net_prop.h"
#include "param_feed_backward.cuh"
#include "state_feed_backward.cuh"
#include "struct_var.h"
#include "user_input.h"
#include "utils.h"

void set_task(std::string &user_input_file, SavePath &path);

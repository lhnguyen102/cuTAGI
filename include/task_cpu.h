///////////////////////////////////////////////////////////////////////////////
// File:         task_cpu.h
// Description:  Header for task command
//               that uses TAGI approach.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Updated:      May 20, 2022
// Updated:      October 30, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <math.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "common.h"
#include "cost.h"
#include "data_transfer_cpu.h"
#include "dataloader.h"
#include "derivative_calcul_cpu.h"
#include "feed_forward_cpu.h"
#include "global_param_update_cpu.h"
#include "indices.h"
#include "net_init.h"
#include "net_prop.h"
#include "param_feed_backward_cpu.h"
#include "state_feed_backward_cpu.h"
#include "struct_var.h"
#include "tagi_network_cpu.h"
#include "user_input.h"
#include "utils.h"

void task_command_cpu(UserInput &user_input, SavePath &path);
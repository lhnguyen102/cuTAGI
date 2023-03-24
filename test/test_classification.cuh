///////////////////////////////////////////////////////////////////////////////
// File:         test_classification.cuh
// Description:  Header of script to perform independent classification
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      March 21, 2023
// Updated:      March 21, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2023 Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#include "../include/common.h"
#include "../include/cost.h"
#include "../include/data_transfer.cuh"
#include "../include/dataloader.h"
#include "../include/feed_forward.cuh"
#include "../include/global_param_update.cuh"
#include "../include/gpu_debug_utils.h"
#include "../include/indices.h"
#include "../include/lstm_feed_forward.cuh"
#include "../include/net_init.h"
#include "../include/net_prop.h"
#include "../include/param_feed_backward.cuh"
#include "../include/state_feed_backward.cuh"
#include "../include/struct_var.h"
#include "../include/tagi_network.cuh"
#include "../include/user_input.h"

void train_classification(TagiNetwork &net, ImageData &imdb, int n_classes);
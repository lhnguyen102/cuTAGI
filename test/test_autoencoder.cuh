///////////////////////////////////////////////////////////////////////////////
// File:         test_autoencoder.cuh
// Description:  Header of script to perform independent image generation
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      April 12, 2023
// Updated:      April 13, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#pragma once
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

/**
 * @brief Auxiliar autoencoder function for unit tests
 *
 * @param net_e  The encoder network
 * @param net_d  The decoder network
 * @param imdb The image train data
 * @param n_classes The number of classes
 */
void train_autoencoder(TagiNetwork &net_e, TagiNetwork &net_d, ImageData &imdb,
                       int n_classes);

///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network_cpu.h
// Description:  Header file for embeddings layer
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 22, 2023
// Updated:      August 22, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector.h>

#include "data_transfer_cpu.h"
#include "struct_var.h"

std::vector<float> initalize_weight(int num_classes, int num_weights);
void forward(NetState &state, int z_pos_out, int num_states);
void param_backward(DeltaState &state, int z_pos_out, int num_states);
int calculate_embedding_size(int num_categories);

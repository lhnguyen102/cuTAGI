///////////////////////////////////////////////////////////////////////////////
// File:         fc_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 20, 2023
// Updated:      September 20, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/fc_cpu_v2.h"

FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size,
                                         size_t batch_size)
    : input_size(input_size),
      output_size(output_size),
      batch_size(batch_size),
      mu_z(output_size * batch_size),
      var_z(output_size * batch_size),
      mu_a(output_size * batch_size),
      var_a(output_size * batch_size),
      jcb(output_size * batch_size) {}
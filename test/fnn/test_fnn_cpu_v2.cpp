///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 25, 2023
// Updated:      October 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_cpu_v2.h"

void forward_fnn_v2()
/*
 */
{
    auto fc_1 = FullyConnectedLayer(1, 10);
    auto fc_2 = FullyConnectedLayer(10, 13);
}

int test_fnn_cpu_v2() {}
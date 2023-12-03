///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 01, 2023
// Updated:      December 01, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <vector>

struct HiddenStates {
    std::vector<float> mu_z;
    std::vector<float> var_z;
    std::vector<float> mu_a;
    std::vector<float> var_a;
    std::vector<float> jcb;
    int size = 0;         // size of data including buffer
    int block_size = 1;   // batch size
    int actual_size = 0;  // actual size of data

    // Default constructor
    HiddenStates() = default;

    // Constructor to initialize all vectors with a specific size
    HiddenStates(size_t n, size_t m)
        : mu_z(n, 0.0f),
          var_z(n, 0.0f),
          mu_a(n, 0.0f),
          var_a(n, 0.0f),
          jcb(n, 1.0f),
          size(n),
          block_size(m) {}
};

struct DeltaStates {
    std::vector<float> delta_mu;
    std::vector<float> delta_var;
    int size = 0, block_size = 1, actual_size = 0;

    // Default constructor
    DeltaStates() = default;

    // Constructor to initialize all vectors with a specific size
    DeltaStates(size_t n, size_t m)
        : delta_mu(n, 0.0f), delta_var(n, 0.0f), size(n), block_size(m) {}
};

struct TempStates {
    std::vector<float> tmp_1;
    std::vector<float> tmp_2;
    int size = 0, block_size = 1;

    // Default constructor
    TempStates() = default;

    // Constructor to initialize all vectors with a specific size
    TempStates(size_t n, size_t m)
        : tmp_1(n, 0.0f), tmp_2(n, 0.0f), size(n), block_size(m) {}
};
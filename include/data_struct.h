///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 01, 2023
// Updated:      December 11, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <vector>

class HiddenStateBase {
   public:
    std::vector<float> mu_z;
    std::vector<float> var_z;
    std::vector<float> mu_a;
    std::vector<float> var_a;
    std::vector<float> jcb;
    int size = 0;         // size of data including buffer
    int block_size = 1;   // batch size
    int actual_size = 0;  // actual size of data

    HiddenStateBase(size_t n, size_t m);
    HiddenStateBase();
    ~HiddenStateBase() = default;
};

class DeltaStateBase {
   public:
    std::vector<float> delta_mu;
    std::vector<float> delta_var;
    int size = 0, block_size = 1, actual_size = 0;

    DeltaStateBase(size_t n, size_t m);
    DeltaStateBase();
    ~DeltaStateBase() = default;
};

class TempStateBase {
   public:
    std::vector<float> tmp_1;
    std::vector<float> tmp_2;
    int size = 0, block_size = 1;

    TempStateBase(size_t n, size_t m);
    TempStateBase();
    ~TempStateBase() = default;
};

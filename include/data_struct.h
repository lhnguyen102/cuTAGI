///////////////////////////////////////////////////////////////////////////////
// File:         data_struct.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 01, 2023
// Updated:      December 19, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iostream>
#include <string>
#include <vector>

class BaseHiddenStates {
   public:
    std::vector<float> mu_z;
    std::vector<float> var_z;
    std::vector<float> mu_a;
    std::vector<float> var_a;
    std::vector<float> jcb;
    int size = 0;         // size of data including buffer
    int block_size = 1;   // batch size
    int actual_size = 0;  // actual size of data

    BaseHiddenStates(size_t n, size_t m);
    BaseHiddenStates();
    ~BaseHiddenStates() = default;
    virtual std::string get_name() const { return "BaseHiddenStates"; };
};

class BaseDeltaStates {
   public:
    std::vector<float> delta_mu;
    std::vector<float> delta_var;
    int size = 0, block_size = 1, actual_size = 0;

    BaseDeltaStates(size_t n, size_t m);
    BaseDeltaStates();
    ~BaseDeltaStates() = default;
    virtual std::string get_name() const { return "BaseDeltaStates"; };
};

class BaseTempStates {
   public:
    std::vector<float> tmp_1;
    std::vector<float> tmp_2;
    int size = 0, block_size = 1;

    BaseTempStates(size_t n, size_t m);
    BaseTempStates();
    ~BaseTempStates() = default;
    virtual std::string get_name() const { return "BaseTempStates"; };
};

class BaseBackwardStates {
   public:
    std::vector<float> mu_a;
    std::vector<float> jcb;

    BaseBackwardStates();
    ~BaseBackwardStates() = default;
    virtual std::string get_name() const { return "BaseBackwardStates"; };
};

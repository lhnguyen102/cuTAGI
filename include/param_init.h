
///////////////////////////////////////////////////////////////////////////////
// File:         param_init.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 15, 2023
// Updated:      March 23, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// #include "net_prop.h"

class SeedManager {
   public:
    static SeedManager& get_instance() {
        static SeedManager instance;
        return instance;
    }

    void set_seed(int seed) {
        current_seed = seed;
        global_random_engine.seed(seed);
    }

    void reset_seed() { global_random_engine.seed(current_seed); }

    std::mt19937& get_engine() { return global_random_engine; }

   private:
    SeedManager() : global_random_engine(std::random_device{}()) {}
    // Delete copy constructor and assignment operator
    SeedManager(const SeedManager&) = delete;
    SeedManager& operator=(const SeedManager&) = delete;
    std::mt19937 global_random_engine;
    int current_seed;
};

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_linear(const std::string& init_method, const float gain_w,
                        const float gain_b, const int input_size,
                        const int output_size, int num_weights, int num_biases);

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_conv2d(const size_t kernel_size, const size_t in_channels,
                        const size_t out_channels,
                        const std::string& init_method, const float gain_w,
                        const float gain_b, int num_weights, int num_biases);

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_lstm(const std::string& init_method, const float gain_w,
                      const float gain_b, const int input_size,
                      const int output_size, int num_weights, int num_biases);

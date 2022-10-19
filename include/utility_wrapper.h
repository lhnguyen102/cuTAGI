///////////////////////////////////////////////////////////////////////////////
// File:         utility_wrapper.h
// Description:  Python wrapper for utility functions in C++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 17, 2022
// Updated:      October 19, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "cost.h"
#include "dataloader.h"

class UtilityWrapper {
   public:
    UtilityWrapper();
    ~UtilityWrapper();
    std::tuple<std::vector<float>, std::vector<int>, int> hierarchical_softmax(
        std::vector<int> &labels, int num_classes);

    std::tuple<std::vector<float>, std::vector<int>> load_mnist_dataset(
        std::string &image_file, std::string &label_file, int num);

    std::tuple<std::vector<float>, std::vector<int>> load_cifar_dataset(
        std::string &image_file, int num);
};
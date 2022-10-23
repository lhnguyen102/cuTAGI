///////////////////////////////////////////////////////////////////////////////
// File:         utility_wrapper.h
// Description:  Python wrapper for utility functions in C++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 17, 2022
// Updated:      October 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "cost.h"
#include "dataloader.h"
#include "struct_var.h"

class UtilityWrapper {
   public:
    UtilityWrapper();
    ~UtilityWrapper();
    std::tuple<std::vector<float>, std::vector<int>, int> label_to_obs_wrapper(
        std::vector<int> &labels, int num_classes);

    std::tuple<std::vector<float>, std::vector<int>> load_mnist_dataset_wrapper(
        std::string &image_file, std::string &label_file, int num);

    std::tuple<std::vector<float>, std::vector<int>> load_cifar_dataset_wrapper(
        std::string &image_file, int num);

    std::tuple<std::vector<int>, std::vector<float>> get_labels_wrapper(
        std::vector<float> &mz, std::vector<float> &Sz, HrSoftmax &hs,
        int num_classes, int B);

    HrSoftmax hierarchical_softmax_wrapper(int num_classes);

    std::vector<float> obs_to_label_prob_wrapper(std::vector<float> &mz,
                                                 std::vector<float> &Sz,
                                                 HrSoftmax &hs,
                                                 int num_classes);
    std::tuple<std::vector<int>, std::vector<float>> get_error_wrapper(
        std::vector<float> &mz, std::vector<float> &Sz,
        std::vector<int> &labels, HrSoftmax &hs, int n_classes, int B);
};
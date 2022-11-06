///////////////////////////////////////////////////////////////////////////////
// File:         utility_wrapper.h
// Description:  Python wrapper for utility functions in C++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 17, 2022
// Updated:      November 06, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <vector>

#include "common.h"
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

    std::vector<float> load_mnist_images_wrapper(std::string &image_file,
                                                 int num);

    std::vector<int> load_mnist_labels_wrapper(std::string &image_file,
                                               int num);

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

    std::tuple<std::vector<float>, std::vector<float>>
    create_rolling_window_wrapper(std::vector<float> &data,
                                  std::vector<int> &output_col,
                                  int input_seq_len, int output_seq_len,
                                  int num_features, int stride);

    std::vector<float> get_upper_triu_cov_wrapper(int batch_size, int num_data,
                                                  float &sigma);
};

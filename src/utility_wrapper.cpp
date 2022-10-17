///////////////////////////////////////////////////////////////////////////////
// File:         utility_wrapper.cpp
// Description:  Python wrapper for utility functions in C++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 17, 2022
// Updated:      October 17, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/utility_wrapper"

UtilityWrapper::UtilityWrapper(){};
UtilityWrapper::~UtilityWrapper(){};
std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::hierarchical_softmax(std::vector<int> &labels,
                                     int num_classes) {
    // Create tree
    int num = labels.size();
    auto hrs = class_to_obs(user_input.num_classes);

    // Convert to observation and get observation indices
    std::vector<float> obs(hrs.n_obs * num);
    std::vector<int> obs_idx(hrs.n_obs * num);
    labels_to_hrs(labels, hrs, obs, obs_idx);

    return { obs, obs_idx }
}

std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::load_mnist_dataset(std::string &image_file,
                                   std::string &label_file, int num) {
    auto images = load_mnist_images(image_file, num);
    auto labels = load_mnist_labels(label_file, num);

    return { images, labels }
}

std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::load_cifar_dataset(std::string &image_file, int num) {
    std::vector<float> images;
    std::vector<int> labels;
    std::tie(images, labels) = load_cifar_images(image_file, num);

    return { images, labels }
}
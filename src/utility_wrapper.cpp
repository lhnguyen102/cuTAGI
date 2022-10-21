///////////////////////////////////////////////////////////////////////////////
// File:         utility_wrapper.cpp
// Description:  Python wrapper for utility functions in C++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 17, 2022
// Updated:      October 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/utility_wrapper"

UtilityWrapper::UtilityWrapper(){};
UtilityWrapper::~UtilityWrapper(){};
std::tuple<std::vector<float>, std::vector<int>, int>
UtilityWrapper::hierarchical_softmax_wrapper(std::vector<int> &labels,
                                             int num_classes) {
    // Create tree
    int num = labels.size();
    auto hrs = class_to_obs(user_input.num_classes);

    // Convert to observation and get observation indices
    std::vector<float> obs(hrs.n_obs * num);
    std::vector<int> obs_idx(hrs.n_obs * num);
    labels_to_hrs(labels, hrs, obs, obs_idx);

    return { obs, obs_idx, hrs.n_obs }
}

std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::load_mnist_dataset_wrapper(std::string &image_file,
                                           std::string &label_file, int num) {
    auto images = load_mnist_images(image_file, num);
    auto labels = load_mnist_labels(label_file, num);

    return { images, labels }
}

std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::load_cifar_dataset_wrapper(std::string &image_file, int num) {
    std::vector<float> images;
    std::vector<int> labels;
    std::tie(images, labels) = load_cifar_images(image_file, num);

    return { images, labels }
}

std::tuple<std::vector<int>, std::vector<float>> get_labels_wrapper(
    std::vector<float> &mz, std::vector<float> &Sz, HrSoftmax &hs,
    int num_classes, int B) {
    // Initialization
    std::vector<float> prob(B * n_classes);
    std::vector<float> mz_tmp(hs.len);
    std::vector<float> Sz_tmp(hs.len);

    // Compute probability for each class
    for (int r = 0; r < B; r++) {
        // Get sample
        for (int i = 0; i < hs.len; i++) {
            mz_tmp[i] = mz[r * hs.len + i];
            Sz_tmp[i] = Sz[r * hs.len + i];
        }

        // Compute probability
        std::vector<float> tmp(n_classes, 0);
        tmp = obs_to_class(mz_tmp, Sz_tmp, hs, n_classes);

        // Store in P matrix
        for (int c = 0; c < n_classes; c++) {
            prob[r * n_classes + c] = tmp[c];
        }

        // Prediction
        int pred = std::distance(tmp.begin(),
                                 std::max_element(tmp.begin(), tmp.end()));
    }

    return {pred, prob};
}

HrSoftmax label_to_obs_wrapper(int num_classes) {
    auto hs = class_to_obs(num_classes);

    return hs;
}
std::vector<float> obs_to_label_prob_wrapper(std::vector<float> &mz,
                                             std::vector<float> &Sz,
                                             HrSoftmax &hs, int num_classes) {
    auto prob = obs_to_class(std::vector<float> & mz, std::vector<float> & Sz,
                             HrSoftmax & hs, int num_classes);
    return prob;
}

///////////////////////////////////////////////////////////////////////////////
// File:         utility_wrapper.cpp
// Description:  Python wrapper for utility functions in C++
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 17, 2022
// Updated:      December 02, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/utility_wrapper.h"

UtilityWrapper::UtilityWrapper(){};
UtilityWrapper::~UtilityWrapper(){};
std::tuple<std::vector<float>, std::vector<int>, int>
UtilityWrapper::label_to_obs_wrapper(std::vector<int> &labels,
                                     int num_classes) {
    // Create tree
    int num = labels.size();
    auto hrs = class_to_obs(num_classes);

    // Convert to observation and get observation indices
    std::vector<float> obs(hrs.n_obs * num);
    std::vector<int> obs_idx(hrs.n_obs * num);
    labels_to_hrs(labels, hrs, obs, obs_idx);

    return {obs, obs_idx, hrs.n_obs};
}

std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::load_mnist_dataset_wrapper(std::string &image_file,
                                           std::string &label_file, int num) {
    auto images = load_mnist_images(image_file, num);
    auto labels = load_mnist_labels(label_file, num);

    return {images, labels};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
UtilityWrapper::load_mnist_dataset_wrapper_v2(std::string &image_file,
                                              std::string &label_file,
                                              int num) {
    auto images = load_mnist_images(image_file, num);
    auto labels = load_mnist_labels(label_file, num);
    py::array_t<float> py_images = py::array_t<float>(images.size());
    py::array_t<float> py_labels = py::array_t<float>(labels.size());
    auto img_data = images.data();
    auto label_data = labels.data();
    auto py_img_out = py_images.mutable_data();
    auto py_label_out = py_labels.mutable_data();
    for (int i = 0; i < images.size(); i++) {
        py_image_out[i] = img_data[i];
    }
    for (int j = 0; j < labels.size(); j++) {
        py_label_out[i] = label_data[i];
    }

    return {py_images, py_labels};
}

std::vector<float> UtilityWrapper::load_mnist_images_wrapper(
    std::string &image_file, int num) {
    // auto images = load_mnist_images(image_file, num);
    std::vector<float> images(60000 * 784, 0);

    return images;
}

std::vector<int> UtilityWrapper::load_mnist_labels_wrapper(
    std::string &label_file, int num) {
    auto labels = load_mnist_labels(label_file, num);

    return labels;
}

std::tuple<std::vector<float>, std::vector<int>>
UtilityWrapper::load_cifar_dataset_wrapper(std::string &image_file, int num) {
    std::vector<float> images;
    std::vector<int> labels;
    std::tie(images, labels) = load_cifar_images(image_file, num);

    return {images, labels};
}

std::tuple<std::vector<int>, std::vector<float>>
UtilityWrapper::get_labels_wrapper(std::vector<float> &mz,
                                   std::vector<float> &Sz, HrSoftmax &hs,
                                   int num_classes, int B) {
    // Initialization
    std::vector<float> prob(B * num_classes);
    std::vector<int> pred(B);
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
        auto tmp = obs_to_class(mz_tmp, Sz_tmp, hs, num_classes);

        // Store in P matrix
        for (int c = 0; c < num_classes; c++) {
            prob[r * num_classes + c] = tmp[c];
        }

        // Prediction
        pred[r] = std::distance(tmp.begin(),
                                std::max_element(tmp.begin(), tmp.end()));
    }

    return {pred, prob};
}

HrSoftmax UtilityWrapper::hierarchical_softmax_wrapper(int num_classes) {
    auto hs = class_to_obs(num_classes);

    return hs;
}
std::vector<float> UtilityWrapper::obs_to_label_prob_wrapper(
    std::vector<float> &mz, std::vector<float> &Sz, HrSoftmax &hs,
    int num_classes) {
    auto prob = obs_to_class(mz, Sz, hs, num_classes);
    return prob;
}
std::tuple<std::vector<int>, std::vector<float>>
UtilityWrapper::get_error_wrapper(std::vector<float> &mz,
                                  std::vector<float> &Sz,
                                  std::vector<int> &labels, HrSoftmax &hs,
                                  int n_classes, int B) {
    std::vector<int> er;
    std::vector<float> prob;
    std::tie(er, prob) = get_error(mz, Sz, labels, hs, n_classes, B);

    return {er, prob};
}

std::tuple<std::vector<float>, std::vector<float>>
UtilityWrapper::create_rolling_window_wrapper(std::vector<float> &data,
                                              std::vector<int> &output_col,
                                              int input_seq_len,
                                              int output_seq_len,
                                              int num_features, int stride) {
    int num_samples =
        (data.size() / num_features - input_seq_len - output_seq_len) / stride +
        1;
    int num_outputs = output_col.size();
    std::vector<float> input_data(input_seq_len * num_features * num_samples,
                                  0);
    std::vector<float> output_data(
        output_seq_len * output_col.size() * num_samples, 0);

    create_rolling_windows(data, output_col, input_seq_len, output_seq_len,
                           num_features, stride, input_data, output_data);

    return {input_data, output_data};
}

std::vector<float> UtilityWrapper::get_upper_triu_cov_wrapper(int batch_size,
                                                              int num_data,
                                                              float &sigma) {
    float var_x = powf(sigma, 2);
    auto Sx_f = initialize_upper_triu(var_x, num_data);
    auto Sx_f_batch = repmat_vector(Sx_f, batch_size);

    return Sx_f_batch;
}
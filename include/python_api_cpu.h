///////////////////////////////////////////////////////////////////////////////
// File:         python_api_cpu.h
// Description:  API for Python bindings of C++/CUDA
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 19, 2022
// Updated:      December 04, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.h"
#include "cost.h"
#include "dataloader.h"
#include "struct_var.h"
#include "tagi_network_base.h"
#include "tagi_network_cpu.h"

class UtilityWrapper {
   public:
    UtilityWrapper();
    ~UtilityWrapper();
    std::tuple<std::vector<float>, std::vector<int>, int> label_to_obs_wrapper(
        std::vector<int> &labels, int num_classes);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
    load_mnist_dataset_wrapper(std::string &image_file, std::string &label_file,
                               int num);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
    load_cifar_dataset_wrapper(std::string &image_file, int num);

    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
    get_labels_wrapper(std::vector<float> &mz, std::vector<float> &Sz,
                       HrSoftmax &hs, int num_classes, int B);

    HrSoftmax hierarchical_softmax_wrapper(int num_classes);

    std::vector<float> obs_to_label_prob_wrapper(std::vector<float> &mz,
                                                 std::vector<float> &Sz,
                                                 HrSoftmax &hs,
                                                 int num_classes);
    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
    get_error_wrapper(std::vector<float> &mz, std::vector<float> &Sz,
                      std::vector<int> &labels, HrSoftmax &hs, int n_classes,
                      int B);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    create_rolling_window_wrapper(std::vector<float> &data,
                                  std::vector<int> &output_col,
                                  int input_seq_len, int output_seq_len,
                                  int num_features, int stride);

    std::vector<float> get_upper_triu_cov_wrapper(int batch_size, int num_data,
                                                  float &sigma);
};

class NetworkWrapper {
   public:
    std::unique_ptr<TagiNetworkBase> tagi_net;
    NetworkWrapper(Network &net);
    ~NetworkWrapper();
    void feed_forward_wrapper(std::vector<float> &x, std::vector<float> &Sx,
                              std::vector<float> &Sx_f);
    void connected_feed_forward_wrapper(std::vector<float> &ma,
                                        std::vector<float> &Sa,
                                        std::vector<float> &mz,
                                        std::vector<float> &Sz,
                                        std::vector<float> &J);

    void state_feed_backward_wrapper(std::vector<float> &y,
                                     std::vector<float> &Sy,
                                     std::vector<int> &idx_ud);
    void param_feed_backward_wrapper();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_network_outputs_wrapper();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_network_prediction_wrapper();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
               pybind11::array_t<float>, pybind11::array_t<float>,
               pybind11::array_t<float>>
    get_all_network_outputs_wrapper();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
               pybind11::array_t<float>, pybind11::array_t<float>,
               pybind11::array_t<float>>
    get_all_network_inputs_wrapper();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_derivative_wrapper(int layer);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_inovation_mean_var_wrapper(int layer);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_state_delta_mean_var_wrapper();

    void set_parameters_wrapper(Param &init_theta);

    Param get_parameters_wrapper();
};

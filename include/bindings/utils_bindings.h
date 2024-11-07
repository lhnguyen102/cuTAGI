#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "common.h"
#include "cost.h"
#include "data_struct.h"
#include "dataloader.h"

class Utils {
   public:
    Utils();
    ~Utils();
    std::string get_name() { return "Utils"; }
    std::tuple<std::vector<float>, std::vector<int>, int> label_to_obs_wrapper(
        std::vector<int> &labels, int num_classes);

    pybind11::array_t<float> label_to_one_hot_wrapper(std::vector<int> &labels,
                                                      int n_classes);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
    load_mnist_dataset_wrapper(const std::string &image_file,
                               const std::string &label_file, int num);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
    load_cifar_dataset_wrapper(std::string &image_file, int num);

    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
    get_labels_wrapper(std::vector<float> &mz, std::vector<float> &Sz,
                       HRCSoftmax &hs, int num_classes, int B);

    HRCSoftmax hierarchical_softmax_wrapper(int num_classes);

    std::vector<float> obs_to_label_prob_wrapper(std::vector<float> &mz,
                                                 std::vector<float> &Sz,
                                                 HRCSoftmax &hs,
                                                 int num_classes);
    std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
    get_error_wrapper(std::vector<float> &mz, std::vector<float> &Sz,
                      std::vector<int> &labels, int n_classes, int B);

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    create_rolling_window_wrapper(std::vector<float> &data,
                                  std::vector<int> &output_col,
                                  int input_seq_len, int output_seq_len,
                                  int num_features, int stride);

    std::vector<float> get_upper_triu_cov_wrapper(int batch_size, int num_data,
                                                  float &sigma);
};

void bind_utils(pybind11::module_ &modo);
void bind_manual_seed(pybind11::module_ &modo);
void bind_is_cuda_available(pybind11::module_ &modo);
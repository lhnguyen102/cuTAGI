
///////////////////////////////////////////////////////////////////////////////
// File:         utils_bindings.cpp
// Description:  API for Python bindings of C++/CUDA
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 31, 2024
// Updated:      April 04, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "../include/bindings/utils_bindings.h"

Utils::Utils() {};
Utils::~Utils() {};
std::tuple<std::vector<float>, std::vector<int>, int>
Utils::label_to_obs_wrapper(std::vector<int> &labels, int num_classes) {
    // Create tree
    int num = labels.size();
    auto hrs = class_to_obs(num_classes);

    // Convert to observation and get observation indices
    std::vector<float> obs(hrs.n_obs * num);
    std::vector<int> obs_idx(hrs.n_obs * num);
    labels_to_hrs(labels, hrs, obs, obs_idx);

    return {obs, obs_idx, hrs.n_obs};
}

pybind11::array_t<float> Utils::label_to_one_hot_wrapper(
    std::vector<int> &labels, int n_classes) {
    auto obs = label_to_one_hot(labels, n_classes);
    auto py_obs = pybind11::array_t<float>(obs.size(), obs.data());

    return py_obs;
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
Utils::load_mnist_dataset_wrapper(const std::string &image_file,
                                  const std::string &label_file, int num) {
    auto images = load_mnist_images(image_file, num);
    auto labels = load_mnist_labels(label_file, num);
    auto py_images = pybind11::array_t<float>(images.size(), images.data());
    auto py_labels = pybind11::array_t<int>(labels.size(), labels.data());

    return {py_images, py_labels};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
Utils::load_cifar_dataset_wrapper(std::string &image_file, int num) {
    std::vector<float> images;
    std::vector<int> labels;
    std::tie(images, labels) = load_cifar_images(image_file, num);
    auto py_images = pybind11::array_t<float>(images.size(), images.data());
    auto py_labels = pybind11::array_t<int>(labels.size(), labels.data());

    return {py_images, py_labels};
}

std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
Utils::get_labels_wrapper(std::vector<float> &mz, std::vector<float> &Sz,
                          HRCSoftmax &hs, int num_classes, int B) {
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
    auto py_pred = pybind11::array_t<int>(pred.size(), pred.data());
    auto py_prob = pybind11::array_t<float>(prob.size(), prob.data());

    return {py_pred, py_prob};
}

HRCSoftmax Utils::hierarchical_softmax_wrapper(int num_classes) {
    auto hs = class_to_obs(num_classes);

    return hs;
}
std::vector<float> Utils::obs_to_label_prob_wrapper(std::vector<float> &mz,
                                                    std::vector<float> &Sz,
                                                    HRCSoftmax &hs,
                                                    int num_classes) {
    auto prob = obs_to_class(mz, Sz, hs, num_classes);
    return prob;
}
std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
Utils::get_error_wrapper(std::vector<float> &mz, std::vector<float> &Sz,
                         std::vector<int> &labels, int n_classes, int B) {
    std::vector<int> er;
    std::vector<float> prob;
    std::tie(er, prob) = get_error(mz, Sz, labels, n_classes, B);
    auto py_er = pybind11::array_t<int>(er.size(), er.data());
    auto py_prob = pybind11::array_t<float>(prob.size(), prob.data());

    return {py_er, py_prob};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
Utils::create_rolling_window_wrapper(std::vector<float> &data,
                                     std::vector<int> &output_col,
                                     int input_seq_len, int output_seq_len,
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

    auto py_input_data =
        pybind11::array_t<float>(input_data.size(), input_data.data());
    auto py_output_data =
        pybind11::array_t<float>(output_data.size(), output_data.data());

    return {py_input_data, py_output_data};
}

std::vector<float> Utils::get_upper_triu_cov_wrapper(int batch_size,
                                                     int num_data,
                                                     float &sigma) {
    float var_x = powf(sigma, 2);
    auto Sx_f = initialize_upper_triu(var_x, num_data);
    auto Sx_f_batch = repmat_vector(Sx_f, batch_size);

    return Sx_f_batch;
}

void bind_utils(pybind11::module_ &m) {
    pybind11::class_<Utils>(m, "Utils")
        .def(pybind11::init<>())
        .def("label_to_obs_wrapper", &Utils::label_to_obs_wrapper)
        .def("label_to_one_hot_wrapper", &Utils::label_to_one_hot_wrapper)
        .def("hierarchical_softmax_wrapper",
             &Utils::hierarchical_softmax_wrapper)
        .def("load_mnist_dataset_wrapper", &Utils::load_mnist_dataset_wrapper,
             pybind11::arg("image_file"), pybind11::arg("label_file"),
             pybind11::arg("num"))
        .def("load_cifar_dataset_wrapper", &Utils::load_cifar_dataset_wrapper)
        .def("get_labels_wrapper", &Utils::get_labels_wrapper)
        .def("obs_to_label_prob_wrapper", &Utils::obs_to_label_prob_wrapper)
        .def("get_error_wrapper", &Utils::get_error_wrapper)
        .def("create_rolling_window_wrapper",
             &Utils::create_rolling_window_wrapper)
        .def("get_name", &Utils::get_name)
        .def("get_upper_triu_cov_wrapper", &Utils::get_upper_triu_cov_wrapper);
}
///////////////////////////////////////////////////////////////////////////////
// File:         python_api_cpu.cpp
// Description:  API for Python bindings of C++/CUDA
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 19, 2022
// Updated:      January 27, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/python_api_cpu.h"

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

pybind11::array_t<float> UtilityWrapper::label_to_one_hot_wrapper(
    std::vector<int> &labels, int n_classes) {
    auto obs = label_to_one_hot(labels, n_classes);
    auto py_obs = pybind11::array_t<float>(obs.size(), obs.data());

    return py_obs;
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
UtilityWrapper::load_mnist_dataset_wrapper(std::string &image_file,
                                           std::string &label_file, int num) {
    auto images = load_mnist_images(image_file, num);
    auto labels = load_mnist_labels(label_file, num);
    auto py_images = pybind11::array_t<float>(images.size(), images.data());
    auto py_labels = pybind11::array_t<int>(labels.size(), labels.data());

    return {py_images, py_labels};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
UtilityWrapper::load_cifar_dataset_wrapper(std::string &image_file, int num) {
    std::vector<float> images;
    std::vector<int> labels;
    std::tie(images, labels) = load_cifar_images(image_file, num);
    auto py_images = pybind11::array_t<float>(images.size(), images.data());
    auto py_labels = pybind11::array_t<int>(labels.size(), labels.data());

    return {py_images, py_labels};
}

std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
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
    auto py_pred = pybind11::array_t<int>(pred.size(), pred.data());
    auto py_prob = pybind11::array_t<float>(prob.size(), prob.data());

    return {py_pred, py_prob};
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
std::tuple<pybind11::array_t<int>, pybind11::array_t<float>>
UtilityWrapper::get_error_wrapper(std::vector<float> &mz,
                                  std::vector<float> &Sz,
                                  std::vector<int> &labels, int n_classes,
                                  int B) {
    std::vector<int> er;
    std::vector<float> prob;
    std::tie(er, prob) = get_error(mz, Sz, labels, n_classes, B);
    auto py_er = pybind11::array_t<int>(er.size(), er.data());
    auto py_prob = pybind11::array_t<float>(prob.size(), prob.data());

    return {py_er, py_prob};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
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

    auto py_input_data =
        pybind11::array_t<float>(input_data.size(), input_data.data());
    auto py_output_data =
        pybind11::array_t<float>(output_data.size(), output_data.data());

    return {py_input_data, py_output_data};
}

std::vector<float> UtilityWrapper::get_upper_triu_cov_wrapper(int batch_size,
                                                              int num_data,
                                                              float &sigma) {
    float var_x = powf(sigma, 2);
    auto Sx_f = initialize_upper_triu(var_x, num_data);
    auto Sx_f_batch = repmat_vector(Sx_f, batch_size);

    return Sx_f_batch;
}

///////////////////////////////////////////////////////////////////////////////
// NETWORK WRAPPER
///////////////////////////////////////////////////////////////////////////////
NetworkWrapper::NetworkWrapper(Network &net) {
    this->tagi_net = std::make_unique<TagiNetworkCPU>(net);
}
NetworkWrapper::~NetworkWrapper(){};

void NetworkWrapper::feed_forward_wrapper(std::vector<float> &x,
                                          std::vector<float> &Sx,
                                          std::vector<float> &Sx_f) {
    this->tagi_net->feed_forward(x, Sx, Sx_f);
}
void NetworkWrapper::connected_feed_forward_wrapper(std::vector<float> &ma,
                                                    std::vector<float> &Sa,
                                                    std::vector<float> &mz,
                                                    std::vector<float> &Sz,
                                                    std::vector<float> &J) {
    this->tagi_net->connected_feed_forward(ma, Sa, mz, Sz, J);
}
void NetworkWrapper::state_feed_backward_wrapper(std::vector<float> &y,
                                                 std::vector<float> &Sy,
                                                 std::vector<int> &idx_ud) {
    this->tagi_net->state_feed_backward(y, Sy, idx_ud);
}
void NetworkWrapper::param_feed_backward_wrapper() {
    this->tagi_net->param_feed_backward();
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
NetworkWrapper::get_network_outputs_wrapper() {
    this->tagi_net->get_network_outputs();
    auto py_ma = pybind11::array_t<float>(this->tagi_net->ma.size(),
                                          this->tagi_net->ma.data());
    auto py_Sa = pybind11::array_t<float>(this->tagi_net->Sa.size(),
                                          this->tagi_net->Sa.data());

    return {py_ma, py_Sa};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
NetworkWrapper::get_network_prediction_wrapper() {
    this->tagi_net->get_predictions();
    auto py_m_pred = pybind11::array_t<float>(this->tagi_net->m_pred.size(),
                                              this->tagi_net->m_pred.data());
    auto py_v_pred = pybind11::array_t<float>(this->tagi_net->v_pred.size(),
                                              this->tagi_net->v_pred.data());

    return {py_m_pred, py_v_pred};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
           pybind11::array_t<float>, pybind11::array_t<float>,
           pybind11::array_t<float>>
NetworkWrapper::get_all_network_outputs_wrapper() {
    this->tagi_net->get_all_network_outputs();
    auto py_ma = pybind11::array_t<float>(this->tagi_net->ma.size(),
                                          this->tagi_net->ma.data());
    auto py_Sa = pybind11::array_t<float>(this->tagi_net->Sa.size(),
                                          this->tagi_net->Sa.data());
    auto py_mz = pybind11::array_t<float>(this->tagi_net->mz.size(),
                                          this->tagi_net->mz.data());
    auto py_Sz = pybind11::array_t<float>(this->tagi_net->Sz.size(),
                                          this->tagi_net->Sz.data());
    auto py_J = pybind11::array_t<float>(this->tagi_net->J.size(),
                                         this->tagi_net->J.data());

    return {py_ma, py_Sa, py_mz, py_Sz, py_J};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>,
           pybind11::array_t<float>, pybind11::array_t<float>,
           pybind11::array_t<float>>
NetworkWrapper::get_all_network_inputs_wrapper() {
    this->tagi_net->get_all_network_inputs();
    auto py_ma_init = pybind11::array_t<float>(this->tagi_net->ma_init.size(),
                                               this->tagi_net->ma_init.data());
    auto py_Sa_init = pybind11::array_t<float>(this->tagi_net->Sa_init.size(),
                                               this->tagi_net->Sa_init.data());
    auto py_mz_init = pybind11::array_t<float>(this->tagi_net->mz_init.size(),
                                               this->tagi_net->mz_init.data());
    auto py_Sz_init = pybind11::array_t<float>(this->tagi_net->Sz_init.size(),
                                               this->tagi_net->Sz_init.data());
    auto py_J_init = pybind11::array_t<float>(this->tagi_net->J_init.size(),
                                              this->tagi_net->J_init.data());

    return {py_ma_init, py_Sa_init, py_mz_init, py_Sz_init, py_J_init};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
NetworkWrapper::get_derivative_wrapper(int layer) {
    std::vector<float> mdy, Sdy;
    std::tie(mdy, Sdy) = this->tagi_net->get_derivatives(layer);
    auto py_mdy = pybind11::array_t<float>(mdy.size(), mdy.data());
    auto py_Sdy = pybind11::array_t<float>(Sdy.size(), Sdy.data());

    return {py_mdy, py_Sdy};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
NetworkWrapper::get_inovation_mean_var_wrapper(int layer) {
    std::vector<float> delta_m, delta_S;
    std::tie(delta_m, delta_S) = this->tagi_net->get_inovation_mean_var(layer);
    auto py_delta_m = pybind11::array_t<float>(delta_m.size(), delta_m.data());
    auto py_delta_S = pybind11::array_t<float>(delta_S.size(), delta_S.data());
    return {py_delta_m, py_delta_S};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
NetworkWrapper::get_state_delta_mean_var_wrapper() {
    std::vector<float> delta_mz, delta_Sz;
    std::tie(delta_mz, delta_Sz) = this->tagi_net->get_state_delta_mean_var();
    auto py_delta_mz =
        pybind11::array_t<float>(delta_mz.size(), delta_mz.data());
    auto py_delta_Sz =
        pybind11::array_t<float>(delta_Sz.size(), delta_Sz.data());
    return {py_delta_mz, py_delta_Sz};
}

void NetworkWrapper::set_parameters_wrapper(Param &init_theta) {
    this->tagi_net->set_parameters(init_theta);
}

Param NetworkWrapper::get_parameters_wrapper() { return this->tagi_net->theta; }

PYBIND11_MODULE(cutagi, m) {
    m.doc() = "Tractable Approximate Gaussian Inference - Backend C++/CUDA";

    pybind11::class_<Param>(m, "Param")
        .def(pybind11::init<>())
        .def_readwrite("mw", &Param::mw)
        .def_readwrite("Sw", &Param::Sw)
        .def_readwrite("mb", &Param::mb)
        .def_readwrite("Sb", &Param::Sb)
        .def_readwrite("mw_sc", &Param::mw_sc)
        .def_readwrite("Sw_sc", &Param::Sw_sc)
        .def_readwrite("mb_sc", &Param::mb_sc)
        .def_readwrite("Sb_sc", &Param::Sb_sc);

    pybind11::class_<Network>(m, "Network")
        .def(pybind11::init<>())
        .def_readwrite("layers", &Network::layers)
        .def_readwrite("nodes", &Network::nodes)
        .def_readwrite("kernels", &Network::kernels)
        .def_readwrite("strides", &Network::strides)
        .def_readwrite("widths", &Network::widths)
        .def_readwrite("heights", &Network::heights)
        .def_readwrite("filters", &Network::filters)
        .def_readwrite("pads", &Network::pads)
        .def_readwrite("pad_types", &Network::pad_types)
        .def_readwrite("shortcuts", &Network::shortcuts)
        .def_readwrite("activations", &Network::activations)
        .def_readwrite("mu_v2b", &Network::mu_v2b)
        .def_readwrite("sigma_v2b", &Network::sigma_v2b)
        .def_readwrite("sigma_v", &Network::sigma_v)
        .def_readwrite("sigma_v_min", &Network::sigma_v_min)
        .def_readwrite("sigma_x", &Network::sigma_x)
        .def_readwrite("is_idx_ud", &Network::is_idx_ud)
        .def_readwrite("is_output_ud", &Network::is_output_ud)
        .def_readwrite("last_backward_layer", &Network::last_backward_layer)
        .def_readwrite("nye", &Network::nye)
        .def_readwrite("decay_factor_sigma_v", &Network::decay_factor_sigma_v)
        .def_readwrite("noise_gain", &Network::noise_gain)
        .def_readwrite("batch_size", &Network::batch_size)
        .def_readwrite("input_seq_len", &Network::input_seq_len)
        .def_readwrite("output_seq_len", &Network::output_seq_len)
        .def_readwrite("seq_stride", &Network::seq_stride)
        .def_readwrite("multithreading", &Network::multithreading)
        .def_readwrite("collect_derivative", &Network::collect_derivative)
        .def_readwrite("is_full_cov", &Network::is_full_cov)
        .def_readwrite("init_method", &Network::init_method)
        .def_readwrite("noise_type", &Network::noise_type)
        .def_readwrite("device", &Network::device)
        .def_readwrite("ra_mt", &Network::ra_mt);

    pybind11::class_<HrSoftmax>(m, "HrSoftmax")
        .def(pybind11::init<>())
        .def_readwrite("obs", &HrSoftmax::obs)
        .def_readwrite("idx", &HrSoftmax::idx)
        .def_readwrite("num_obs", &HrSoftmax::n_obs)
        .def_readwrite("length", &HrSoftmax::len);

    pybind11::class_<UtilityWrapper>(m, "UtilityWrapper")
        .def(pybind11::init<>())
        .def("label_to_obs_wrapper", &UtilityWrapper::label_to_obs_wrapper)
        .def("label_to_one_hot_wrapper",
             &UtilityWrapper::label_to_one_hot_wrapper)
        .def("hierarchical_softmax_wrapper",
             &UtilityWrapper::hierarchical_softmax_wrapper)
        .def("load_mnist_dataset_wrapper",
             &UtilityWrapper::load_mnist_dataset_wrapper)
        .def("load_cifar_dataset_wrapper",
             &UtilityWrapper::load_cifar_dataset_wrapper)
        .def("get_labels_wrapper", &UtilityWrapper::get_labels_wrapper)
        .def("obs_to_label_prob_wrapper",
             &UtilityWrapper::obs_to_label_prob_wrapper)
        .def("get_error_wrapper", &UtilityWrapper::get_error_wrapper)
        .def("get_upper_triu_cov_wrapper",
             &UtilityWrapper::get_upper_triu_cov_wrapper);

    pybind11::class_<NetworkWrapper>(m, "NetworkWrapper")
        .def(pybind11::init<Network &>())
        .def("feed_forward_wrapper", &NetworkWrapper::feed_forward_wrapper)
        .def("connected_feed_forward_wrapper",
             &NetworkWrapper::connected_feed_forward_wrapper)
        .def("state_feed_backward_wrapper",
             &NetworkWrapper::state_feed_backward_wrapper)
        .def("param_feed_backward_wrapper",
             &NetworkWrapper::param_feed_backward_wrapper)
        .def("get_network_outputs_wrapper",
             &NetworkWrapper::get_network_outputs_wrapper)
        .def("get_network_prediction_wrapper",
             &NetworkWrapper::get_network_prediction_wrapper)
        .def("get_all_network_outputs_wrapper",
             &NetworkWrapper::get_all_network_outputs_wrapper)
        .def("get_all_network_inputs_wrapper",
             &NetworkWrapper::get_all_network_inputs_wrapper)
        .def("get_derivative_wrapper", &NetworkWrapper::get_derivative_wrapper)
        .def("get_inovation_mean_var_wrapper",
             &NetworkWrapper::get_inovation_mean_var_wrapper)
        .def("get_state_delta_mean_var_wrapper",
             &NetworkWrapper::get_state_delta_mean_var_wrapper)
        .def("set_parameters_wrapper", &NetworkWrapper::set_parameters_wrapper)
        .def("get_parameters_wrapper", &NetworkWrapper::get_parameters_wrapper);
}
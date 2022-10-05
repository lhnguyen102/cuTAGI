///////////////////////////////////////////////////////////////////////////////
// File:         network_wrapper.cpp
// Description:  Python binding of cutagi code
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      October 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/network_wrapper.cuh"

NetworkWrapper::NetworkWrapper(Network &net) {
    this->net = net;
    net_default(this->net);
    get_net_props(this->net);
    get_similar_layer(this->net);

    // Check feature availability
    check_feature_availability(net);

    // Indices
    tagi_idx(this->idx, net);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(net);
    this->state = initialize_net_states(net);

    // Update quantities
    if (net.device.compare("cpu") == 0) {
        this->d_state.set_values(this->net.n_state, this->state.msc.size(),
                                 this > state.mdsc.size(),
                                 this->net.n_max_state);
        this->d_theta.set_values(this->theta.mw.size(), this->theta.mb.size(),
                                 this->theta.mw_sc.size(),
                                 this->theta.mb_sc.size());
    } else if (net.device.compare("cuda") == 0) {
        // Data transfer for indices
        idx_gpu.set_values(idx);
        idx_gpu.allocate_cuda_memory();
        idx_gpu.copy_host_to_device(idx);

        // Data transfer for states
        state_gpu.set_values(state, net);
        state_gpu.allocate_cuda_memory();
        state_gpu.copy_host_to_device();

        // Data transfer for parameters
        theta_gpu.set_values(theta.mw.size(), theta.mb.size(),
                             theta.mw_sc.size(), theta.mb_sc.size());
        theta_gpu.allocate_cuda_memory();
        theta_gpu.copy_host_to_device(theta);

        // Data transfer for delta state
        d_state_gpu.set_values(net.n_state, state.msc.size(), state.mdsc.size(),
                               net.n_max_state);
        d_state_gpu.allocate_cuda_memory();
        d_state_gpu.copy_host_to_device();

        // Data transfer for delta parameters
        d_theta_gpu.set_values(theta.mw.size(), theta.mb.size(),
                               theta.mw_sc.size(), theta.mb_sc.size());
        d_theta_gpu.allocate_cuda_memory();
        d_theta_gpu.copy_host_to_device();
    } else {
        throw std::invalid_argument(
            "Device is invalid. Device can be either cpu or cuda")
    }
    this->num_weights = theta.mw.size();
    this->num_biases = theta.mb.size();
    this->num_weights_sc = theta.mw_sc.size();
    this->num_biases_sc = theta.mb_sc.size();
}

NetworkWrapper::~NetworkWrapper() {}

void NetworkWrapper::feed_forward_cpu(std::vector<float> &x,
                                      std::vector<float> &Sx,
                                      std::vector<float> &Sx_f) {
    // Set input data
    this->ip.set_values(x, Sx, Sx_f);

    // Initialize input
    initialize_states_cpu(this->ip.x_batch, this->ip.Sx_batch,
                          this->> ip.Sx_f_batch, this->net.n_x,
                          this->net.batch_size, this->state);

    // Feed forward
    feed_forward_cpu(this->net, this->theta, this->idx, this->state);
}

NetworkWrapper::state_feed_backward_cpu(std::vector<float> &y,
                                        std::vector<float> &Sy,
                                        std::vector<int> &idx_ud_batch) {
    // Set output data
    this->op.set_values(y, Sy, idx_ud);

    // Compute update quantities for hidden states
    state_feed_backward(this->net, this->theta, this->state, this->idx,
                        this->op, this->d_state);
}

NetworkWrapper::param_feed_backward_cpu() {
    // Feed backward for parameters
    param_backward_cpu(this->net, this->theta, this->state, this->d_state,
                       this->idx, this->d_theta);

    // Update model parameters
    global_param_update_cpu(this->d_theta, this->num_weights, this->num_biases,
                            this->num_weights_sc, this->num_biases_sc,
                            this->theta);
}

void NetworkWrapper::feed_forward_cuda(std::vector<float> &x,
                                       std::vector<float> &Sx,
                                       std::vector<float> &Sx_f) {
    this->ip_gpu.copy_host_to_device(x, Sx, Sx_f);

    // Initialize input
    initializeStates(this->state_gpu, this->ip_gpu, net);

    // Feed forward
    feedForward(this->net, this->theta_gpu, this->idx_gpu, this->state_gpu);
}

NetworkWrapper::state_feed_backward_cuda(std::vector<float> &y,
                                         std::vector<float> &Sy,
                                         std::vector<int> &idx_ud) {
    // Set output data
    this->op_gpu.copy_host_to_device(y, idx_ud, Sy);

    // Feed backward for hidden states
    stateBackward(this->net, this->theta_gpu, this->state_gpu, this->idx_gpu,
                  this->op_gpu, this->d_state_gpu);
}

NetworkWrapper::param_feed_backward_cuda() {
    // Feed backward for parameters
    paramBackward(this->net, this->theta_gpu, this->state_gpu,
                  this->d_state_gpu, this->idx_gpu, this->d_theta_gpu);

    // Update model parameters.
    globalParamUpdate(this->d_theta_gpu, this->num_weights, this->num_biases,
                      this->num_weights_sc, this->num_biases_sc,
                      this->net.num_gpu_threads, this->theta_gpu);
}

PYBIND11_MODULE(cutagi, m) {
    m.doc() = "Tractable Approximate Gaussian Inference";
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
        .def_read_write("shortcuts", &Network::shortcuts)
        .def_readwrite("activations", &Network::activations)
        .def_readwrite("mu_v2b", &Network::mu_v2b)
        .def_readwrite("sigma_v2b", &Network::sigma_v2b)
        .def_readwrite("sigma_v", &Network::sigma_v)
        .def_readwrite("sigma_v_min", &Network::sigma_v_min)
        .def_readwrite("sigma_x", &Network::sigma_x)
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
        .def_readwrite("device", &Network::device);

    pybind11::class_<NetworkWrapperCPU>(m, "NetworkWrapperCPU")
        .def(pybind11::init<>(Network &))
        .def("feed_forward", &NetworkWrapperCPU::feed_forward)
        .def("state_feed_backward", &NetworkWrapperCPU::state_feed_backward)
        .def("param_feed_backward", &NetworkWrapperCPU::param_feed_backward);
}
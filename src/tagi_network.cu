///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network.cu
// Description:  TAGI network including feed forward & backward
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      October 08, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/tagi_network.cuh"

TagiNetwork::TagiNetwork(Network &net) {
    this->net = net;
    this->init_net();
}

TagiNetwork::~TagiNetwork() {}

void TagiNetwork::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                               std::vector<float> &Sx_f) {
    this->net_input_gpu.copy_host_to_device(x, Sx, Sx_f);

    // Initialize input
    initializeStates(this->state_gpu, this->net_input_gpu, net);

    // Feed forward
    feedForward(this->net, this->theta_gpu, this->idx_gpu, this->state_gpu);
}

void TagiNetwork::state_feed_backward(std::vector<float> &y,
                                      std::vector<float> &Sy,
                                      std::vector<int> &idx_ud) {
    // Set output data
    this->obs_gpu.copy_host_to_device(y, idx_ud, Sy);

    // Feed backward for hidden states
    stateBackward(this->net, this->theta_gpu, this->state_gpu, this->idx_gpu,
                  this->obs_gpu, this->d_state_gpu);
}

void TagiNetwork::param_feed_backward() {
    // Feed backward for parameters
    paramBackward(this->net, this->theta_gpu, this->state_gpu,
                  this->d_state_gpu, this->idx_gpu, this->d_theta_gpu);

    // Update model parameters.
    globalParamUpdate(this->d_theta_gpu, this->num_weights, this->num_biases,
                      this->num_weights_sc, this->num_biases_sc,
                      this->net.num_gpu_threads, this->theta_gpu);
}

void TagiNetwork::init_net() {
    net_default(this->net);
    get_net_props(this->net);
    get_similar_layer(this->net);

    // Check feature availability
    check_feature_availability(this->net);

    tagi_idx(this->idx, this->net);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(this->net);
    this->state = initialize_net_states(this->net);

    this->num_weights = this->theta.mw.size();
    this->num_biases = this->theta.mb.size();
    this->num_weights_sc = this->theta.mw_sc.size();
    this->num_biases_sc = this->theta.mb_sc.size();

    // Data transfer for indices
    this->idx_gpu.set_values(this->idx);
    this->idx_gpu.allocate_cuda_memory();
    this->idx_gpu.copy_host_to_device(this->idx);

    // Data transfer for states
    this->state_gpu.set_values(this->state, this->net);
    this->state_gpu.allocate_cuda_memory();
    this->state_gpu.copy_host_to_device();

    // Data transfer for parameters
    this->theta_gpu.set_values(this->num_weights, this->num_biases,
                               this->num_weights_sc, this->num_biases_sc);
    this->theta_gpu.allocate_cuda_memory();
    this->theta_gpu.copy_host_to_device(this->theta);

    // Data transfer for delta state
    this->d_state_gpu.set_values(this->net.n_state, this->state.msc.size(),
                                 this->state.mdsc.size(),
                                 this->net.n_max_state);
    this->d_state_gpu.allocate_cuda_memory();
    this->d_state_gpu.copy_host_to_device();

    // Data transfer for delta parameters
    this->d_theta_gpu.set_values(this->num_weights, this->num_biases,
                                 this->num_weights_sc, this->num_biases_sc);
    this->d_theta_gpu.allocate_cuda_memory();
    this->d_theta_gpu.copy_host_to_device();

    // Output layer
    this->num_output_bytes =
        this->net.batch_size * this->net.nodes.back() * sizeof(float);
    this->allocate_output_memory();
    this->output_to_device();
}
void TagiNetwork::get_network_outputs() {
    int n = this->net.batch_size * this->net.nodes.back();
    std::vector<float> ma(n, 0);
    std::vector<float> Sa(n, 0);
    unsigned int THREADS = this->net.num_gpu_threads;
    unsigned int BLOCKS = (n + THREADS - 1) / THREADS;

    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_ma, this->net.z_pos.back(), n, this->d_ma);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_Sa, this->net.z_pos.back(), n, this->d_Sa);
    this->output_to_host();
}

void TagiNetwork::allocate_output_memory() {
    cudaMalloc(&d_ma, this->num_output_bytes);
    cudaMalloc(&d_Sa, this->num_output_bytes);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for output's hidden states - "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}
void TagiNetwork::output_to_device() {
    cudaMemcpy(d_ma, this->ma.data(), this->num_output_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma, this->Sa.data(), this->num_output_bytes,
               cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to device for output's hidden states "
            "- "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}
void TagiNetwork::output_to_host() {
    cudaMemcpy(this->ma.data(), d_ma, num_output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Sa.data(), d_Sa, num_output_bytes, cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to host for output's hidden states "
            "- "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}

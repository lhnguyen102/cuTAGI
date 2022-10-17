///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network.cu
// Description:  TAGI network including feed forward & backward
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      October 16, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "tagi_network.cuh"

TagiNetwork::TagiNetwork(Network &net_prop) {
    this->prop = net_prop;
    this->init_net();
}

TagiNetwork::~TagiNetwork() {}

void TagiNetwork::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                               std::vector<float> &Sx_f) {
    this->net_input_gpu.copy_host_to_device(x, Sx, Sx_f);

    // Initialize input
    initializeStates(this->state_gpu, this->net_input_gpu, this->prop);

    // Feed forward
    feedForward(this->prop, this->theta_gpu, this->idx_gpu, this->state_gpu);
}

void TagiNetwork::state_feed_backward(std::vector<float> &y,
                                      std::vector<float> &Sy,
                                      std::vector<int> &idx_ud) {
    // Set output data
    this->obs_gpu.copy_host_to_device(y, idx_ud, Sy);

    // Feed backward for hidden states
    stateBackward(this->prop, this->theta_gpu, this->state_gpu, this->idx_gpu,
                  this->obs_gpu, this->d_state_gpu);
}

void TagiNetwork::param_feed_backward() {
    // Feed backward for parameters
    paramBackward(this->prop, this->theta_gpu, this->state_gpu,
                  this->d_state_gpu, this->idx_gpu, this->d_theta_gpu);

    // Update model parameters.
    globalParamUpdate(this->d_theta_gpu, this->num_weights, this->num_biases,
                      this->num_weights_sc, this->num_biases_sc,
                      this->prop.num_gpu_threads, this->theta_gpu);
}

void TagiNetwork::init_net() {
    net_default(this->prop);
    get_net_props(this->prop);
    get_similar_layer(this->prop);

    // Check feature availability
    check_feature_availability(this->prop);

    tagi_idx(this->idx, this->prop);
    index_default(this->idx);  // TODO: To be removed
    this->theta = initialize_param(this->prop);
    this->state = initialize_net_states(this->prop);

    this->num_weights = this->theta.mw.size();
    this->num_biases = this->theta.mb.size();
    this->num_weights_sc = this->theta.mw_sc.size();
    this->num_biases_sc = this->theta.mb_sc.size();

    // Send indices to device
    this->idx_gpu.set_values(this->idx);
    this->idx_gpu.allocate_cuda_memory();
    this->idx_gpu.copy_host_to_device(this->idx);

    // Send states to device
    this->state_gpu.set_values(this->state, this->prop);
    this->state_gpu.allocate_cuda_memory();
    this->state_gpu.copy_host_to_device();

    // Send parameters to device
    this->theta_gpu.set_values(this->theta);
    this->theta_gpu.allocate_cuda_memory();
    this->theta_gpu.copy_host_to_device();

    // Send delta state to device
    this->d_state_gpu.set_values(this->prop.n_state, this->state.msc.size(),
                                 this->state.mdsc.size(),
                                 this->prop.n_max_state);
    this->d_state_gpu.allocate_cuda_memory();
    this->d_state_gpu.copy_host_to_device();

    // Send delta parameters to device
    this->d_theta_gpu.set_values(this->num_weights, this->num_biases,
                                 this->num_weights_sc, this->num_biases_sc);
    this->d_theta_gpu.allocate_cuda_memory();
    this->d_theta_gpu.copy_host_to_device();

    // Output layer
    this->ma.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->Sa.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->num_output_bytes =
        this->prop.batch_size * this->prop.nodes.back() * sizeof(float);
    this->allocate_output_memory();
    this->output_to_device();

    // IO data
    this->net_input_gpu.set_values(this->prop);
    this->obs_gpu.set_values(this->prop.n_y, this->prop.nye,
                             this->prop.batch_size);
    this->net_input_gpu.allocate_cuda_memory();
    this->obs_gpu.allocate_cuda_memory();
}
void TagiNetwork::get_network_outputs() {
    int n = this->prop.batch_size * this->prop.nodes.back();
    int THREADS = this->prop.num_gpu_threads;
    unsigned int BLOCKS = (n + THREADS - 1) / THREADS;

    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_ma, this->prop.z_pos.back(), n, this->d_ma);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_Sa, this->prop.z_pos.back(), n, this->d_Sa);
    this->output_to_host();
}

void TagiNetwork::set_parameters(Param &init_theta)
/*Set parameters to network*/
{
    // Weights
    for (int i = 0; i < this->num_weights; i++) {
        this->theta.mw[i] = init_theta.mw[i];
        this->theta.Sw[i] = init_theta.Sw[i];
    }

    // Biases
    for (int j = 0; j < this->num_biases; j++) {
        this->theta.mb[j] = init_theta.mb[j];
        this->theta.Sb[j] = init_theta.Sb[j];
    }

    // Residual network
    if (this->num_weights_sc > 0) {
        // Weights
        for (int i = 0; i < this->num_weights_sc; i++) {
            this->theta.mw_sc[i] = init_theta.mw_sc[i];
            this->theta.Sw_sc[i] = init_theta.Sw_sc[i];
        }

        // Biases
        for (int j = 0; j < this->num_biases_sc; j++) {
            this->theta.mb_sc[j] = init_theta.mb_sc[j];
            this->theta.Sb_sc[j] = init_theta.Sb_sc[j];
        }
    }
    this->theta_gpu.copy_host_to_device();
}

Param TagiNetwork::get_parameters() {
    this->theta_gpu.copy_device_to_host();
    return this->theta;
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
    cudaMemcpy(d_Sa, this->Sa.data(), this->num_output_bytes,
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
    cudaMemcpy(this->ma.data(), this->d_ma, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Sa.data(), this->d_Sa, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to host for output's hidden states "
            "- "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}

///////////////////////////////////////////////////////////////////////////////
// File:         tagi_network.cu
// Description:  TAGI network including feed forward & backward
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 05, 2022
// Updated:      November 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#include "tagi_network.cuh"

TagiNetwork::TagiNetwork(Network &net_prop) {
    this->prop = net_prop;
    this->init_net();
}

TagiNetwork::TagiNetwork() {}

TagiNetwork::~TagiNetwork() {
    cudaFree(this->d_ma);
    cudaFree(this->d_Sa);
    cudaFree(this->d_mz);
    cudaFree(this->d_Sz);
    cudaFree(this->d_J);
    cudaFree(this->d_ma_init);
    cudaFree(this->d_Sa_init);
    cudaFree(this->d_mz_init);
    cudaFree(this->d_Sz_init);
    cudaFree(this->d_J_init);
}

void TagiNetwork::feed_forward(std::vector<float> &x, std::vector<float> &Sx,
                               std::vector<float> &Sx_f) {
    this->net_input_gpu.copy_host_to_device(x, Sx, Sx_f);

    // Initialize input
    initializeStates(this->state_gpu, this->net_input_gpu, this->prop);

    // Feed forward
    feedForward(this->prop, this->theta_gpu, this->idx_gpu, this->state_gpu);
}

void TagiNetwork::connected_feed_forward(std::vector<float> &ma,
                                         std::vector<float> &Sa,
                                         std::vector<float> &mz,
                                         std::vector<float> &Sz,
                                         std::vector<float> &J) {
    this->connected_input_gpu.copy_host_to_device(ma, Sa, mz, Sz, J);

    // Initialize input. TODO: Find a proper way (see taks.cu)
    int input_size =
        this->prop.n_x * this->prop.batch_size * this->prop.input_seq_len;
    int THREADS = this->prop.num_gpu_threads;
    unsigned int BLOCKS = (input_size + THREADS - 1) / THREADS;

    initializeFullStates<<<BLOCKS, THREADS>>>(
        connected_input_gpu.d_mz, connected_input_gpu.d_Sz,
        connected_input_gpu.d_ma, connected_input_gpu.d_Sa,
        connected_input_gpu.d_J, input_size, 0, this->state_gpu.d_mz,
        this->state_gpu.d_Sz, this->state_gpu.d_ma, this->state_gpu.d_Sa,
        this->state_gpu.d_J);

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
    global_param_update(this->d_theta_gpu, this->prop.cap_factor,
                        this->num_weights, this->num_biases,
                        this->num_weights_sc, this->num_biases_sc,
                        this->prop.num_gpu_threads, this->theta_gpu);
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
void TagiNetwork::get_predictions()
/*Get prediction distributions. Note that the predictions might be different to
   output values of a network e.g., heteroscedastic noise inference where output
   layer includes the mean and std of the prediction of an outcome */
{
    int num_preds = this->prop.n_y * this->prop.batch_size;
    this->get_network_outputs();
    if (this->prop.noise_type.compare("heteros") != 0) {
        this->m_pred = this->ma;
        this->v_pred = this->Sa;
        if (this->prop.noise_type.compare("homosce") == 0) {
            this->state_gpu.noise_state.copy_device_to_host(
                this->state.noise_state);
            for (int i = 0; i < num_preds; i++) {
                this->v_pred[i] += this->state.noise_state.ma_v2b_prior[i];
            }
        }
    } else {
        // TODO: We don't need to copy all noise states to CPU. This will be a
        // subject of speed optimization.
        this->state_gpu.noise_state.copy_device_to_host(
            this->state.noise_state);
        get_output_hidden_states_ni_cpu(this->ma, this->prop.nodes.back(), 0,
                                        this->m_pred);
        get_output_hidden_states_ni_cpu(this->Sa, this->prop.nodes.back(), 0,
                                        this->v_pred);
        for (int i = 0; i < num_preds; i++) {
            this->v_pred[i] += this->state.noise_state.ma_v2b_prior[i];
        }
    }
}

void TagiNetwork::get_all_network_outputs() {
    int n = this->prop.batch_size * this->prop.nodes.back();
    int THREADS = this->prop.num_gpu_threads;
    unsigned int BLOCKS = (n + THREADS - 1) / THREADS;

    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_ma, this->prop.z_pos.back(), n, this->d_ma);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_Sa, this->prop.z_pos.back(), n, this->d_Sa);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_mz, this->prop.z_pos.back(), n, this->d_mz);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_Sz, this->prop.z_pos.back(), n, this->d_Sz);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(
        this->state_gpu.d_J, this->prop.z_pos.back(), n, this->d_J);
    this->all_outputs_to_host();
}

void TagiNetwork::get_all_network_inputs() {
    int n = this->prop.n_x * this->prop.batch_size * this->prop.input_seq_len;
    int THREADS = this->prop.num_gpu_threads;
    unsigned int BLOCKS = (n + THREADS - 1) / THREADS;

    get_output_hidden_states<<<BLOCKS, THREADS>>>(this->state_gpu.d_ma, 0, n,
                                                  this->d_ma);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(this->state_gpu.d_Sa, 0, n,
                                                  this->d_Sa);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(this->state_gpu.d_mz, 0, n,
                                                  this->d_mz);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(this->state_gpu.d_Sz, 0, n,
                                                  this->d_Sz);
    get_output_hidden_states<<<BLOCKS, THREADS>>>(this->state_gpu.d_J, 0, n,
                                                  this->d_J);
    this->all_inputs_to_host();
}

std::tuple<std::vector<float>, std::vector<float>> TagiNetwork::get_derivatives(
    int layer)
/*Compute derivative of neural network using TAGI. NOTE: current version only
   support the fully-connected layer*/
{
    if (!this->prop.collect_derivative) {
        throw std::invalid_argument(
            "Set collect_derivative model in network properties to True");
    }
    int num_derv = this->prop.batch_size * this->prop.nodes[layer];
    std::vector<float> mdy_batch_in(num_derv, 0);
    std::vector<float> Sdy_batch_in(num_derv, 0);
    compute_network_derivatives(this->prop, this->theta_gpu, this->state_gpu,
                                layer);
    // TODO: We don't need to copy all derivatives to CPU
    this->state_gpu.derv_state.copy_host_to_device(this->state.derv_state);

    get_input_derv_states(this->state.derv_state.md_layer,
                          this->state.derv_state.Sd_layer, mdy_batch_in,
                          Sdy_batch_in);

    return {mdy_batch_in, Sdy_batch_in};
}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetwork::get_inovation_mean_var(int layer) {
    // Transfer data from device to host. TODO: find a way to only the portion
    // of data requested
    this->d_state_gpu.copy_device_to_host();

    // Get data for requested layer
    int num_data;
    if (layer == 0) {  // input layer
        num_data = this->prop.nodes[layer] * this->prop.batch_size *
                   this->prop.input_seq_len;

    } else {
        num_data = this->prop.nodes[layer] * this->prop.batch_size;
    }
    int z_pos = this->prop.z_pos[layer];
    std::vector<float> delta_m(num_data, 0);
    std::vector<float> delta_S(num_data, 0);
    for (int i = 0; i < num_data; i++) {
        delta_m[i] = this->d_state_gpu.delta_m[z_pos + i];
        delta_S[i] = this->d_state_gpu.delta_S[z_pos + i];
    }

    return {delta_m, delta_S};
}

std::tuple<std::vector<float>, std::vector<float>>
TagiNetwork::get_state_delta_mean_var()
/*Get updating quantites (delta_mz, delta_Sz) of the input layer for the hidden
states. NOTE: delta_mz and delta_Sz are overridable vector during the backward
pass in order to save memory allocation on the device.
*/
{
    // Transfer data from device to host. TODO: find a way to only the portion
    // of data requested
    this->d_state_gpu.copy_device_to_host();

    // Get data for input layer
    int num_data =
        this->prop.nodes[0] * this->prop.batch_size * this->prop.input_seq_len;
    std::vector<float> delta_mz(num_data, 0);
    std::vector<float> delta_Sz(num_data, 0);
    for (int i = 0; i < num_data; i++) {
        delta_mz[i] = this->d_state_gpu.delta_mz[i];
        delta_Sz[i] = this->d_state_gpu.delta_Sz[i];
    }

    return {delta_mz, delta_Sz};
}

void TagiNetwork::set_parameters(Param &init_theta)
/*Set parameters to network*/
{
    // Check if parameters are valids
    if ((init_theta.mw.size() != this->num_weights) ||
        (init_theta.Sw.size() != this->num_weights)) {
        throw std::invalid_argument("Length of weight parameters is invalid");
    }
    if ((init_theta.mb.size() != this->num_biases) ||
        (init_theta.Sb.size() != this->num_biases)) {
        throw std::invalid_argument("Length of biases parameters is invalid");
    }

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
    this->d_state_gpu.set_values(this->prop);
    this->d_state_gpu.allocate_cuda_memory();
    this->d_state_gpu.copy_host_to_device();

    // Send delta parameters to device
    this->d_theta_gpu.set_values(this->num_weights, this->num_biases,
                                 this->num_weights_sc, this->num_biases_sc);
    this->d_theta_gpu.allocate_cuda_memory();
    this->d_theta_gpu.copy_host_to_device();

    // Input layers
    int input_size =
        this->prop.n_x * this->prop.batch_size * this->prop.input_seq_len;
    this->ma_init.resize(input_size, 0);
    this->Sa_init.resize(input_size, 0);
    this->mz_init.resize(input_size, 0);
    this->Sz_init.resize(input_size, 0);
    this->J_init.resize(input_size, 1);
    this->num_input_bytes = input_size * sizeof(float);

    // Output layer
    this->ma.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->Sa.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->mz.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->Sz.resize(this->prop.nodes.back() * this->prop.batch_size, 0);
    this->J.resize(this->prop.nodes.back() * this->prop.batch_size, 1);
    this->m_pred.resize(this->prop.n_y * this->prop.batch_size, 0);
    this->v_pred.resize(this->prop.n_y * this->prop.batch_size, 0);
    this->num_output_bytes =
        this->prop.batch_size * this->prop.nodes.back() * sizeof(float);
    this->allocate_output_memory();
    this->output_to_device();

    // IO data
    this->net_input_gpu.set_values(this->prop);
    this->net_input_gpu.allocate_cuda_memory();
    // TODO: This should be an option
    this->connected_input_gpu.set_values(input_size);
    this->connected_input_gpu.allocate_cuda_memory();

    this->obs_gpu.set_values(this->prop.n_y, this->prop.nye,
                             this->prop.batch_size);
    this->obs_gpu.allocate_cuda_memory();
}

void TagiNetwork::allocate_output_memory() {
    cudaMalloc(&d_ma, this->num_output_bytes);
    cudaMalloc(&d_Sa, this->num_output_bytes);
    cudaMalloc(&d_mz, this->num_output_bytes);
    cudaMalloc(&d_Sz, this->num_output_bytes);
    cudaMalloc(&d_J, this->num_output_bytes);
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
    cudaMemcpy(d_mz, this->mz.data(), this->num_output_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sz, this->Sz.data(), this->num_output_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_J, this->J.data(), this->num_output_bytes,
               cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to device for all output's hidden "
            "states "
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
void TagiNetwork::all_outputs_to_host() {
    cudaMemcpy(this->ma.data(), this->d_ma, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Sa.data(), this->d_Sa, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mz.data(), this->d_mz, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Sz.data(), this->d_Sz, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->J.data(), this->d_J, this->num_output_bytes,
               cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to host for ALL output's hidden "
            "states "
            "- "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}

void TagiNetwork::allocate_input_memory() {
    cudaMalloc(&d_ma_init, this->num_input_bytes);
    cudaMalloc(&d_Sa_init, this->num_input_bytes);
    cudaMalloc(&d_mz_init, this->num_input_bytes);
    cudaMalloc(&d_Sz_init, this->num_input_bytes);
    cudaMalloc(&d_J_init, this->num_input_bytes);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to allocate CUDA memory for input's hidden states - "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}

void TagiNetwork::input_to_device() {
    cudaMemcpy(d_ma_init, this->ma_init.data(), this->num_input_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sa_init, this->Sa_init.data(), this->num_input_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mz_init, this->mz_init.data(), this->num_input_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_Sz_init, this->Sz_init.data(), this->num_input_bytes,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_J_init, this->J_init.data(), this->num_input_bytes,
               cudaMemcpyHostToDevice);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to device for ALL input's hidden "
            "states "
            "- "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}

void TagiNetwork::all_inputs_to_host() {
    cudaMemcpy(this->ma_init.data(), this->d_ma_init, this->num_input_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Sa_init.data(), this->d_Sa_init, this->num_input_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->mz_init.data(), this->d_mz_init, this->num_input_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->Sz_init.data(), this->d_Sz_init, this->num_input_bytes,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(this->J_init.data(), this->d_J_init, this->num_input_bytes,
               cudaMemcpyDeviceToHost);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::string err_msg =
            "Failed to make data tranfer to host for input's hidden states "
            "- "
            "tagi_network.cu";
        std::cerr << error << ": " << err_msg << "\n";
    }
}

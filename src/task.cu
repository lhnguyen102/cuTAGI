///////////////////////////////////////////////////////////////////////////////
// File:         task.cu
// Description:  providing different tasks such as regression, classification
//               that uses TAGI approach.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      May 16, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/task.cuh"

///////////////////////////////////////////////////////////////////////
// MISC FUNCTIONS
///////////////////////////////////////////////////////////////////////
void compute_net_memory(Network &net, size_t &id_bytes, size_t &od_bytes,
                        size_t &ode_bytes, size_t &max_n_s_bytes)
/*TODO: Might be removed
 */
{
    id_bytes = net.batch_size * net.nodes[0] * sizeof(float);
    od_bytes = net.batch_size * net.nodes[net.nodes.size() - 1] * sizeof(float);
    ode_bytes = net.batch_size * net.nye * sizeof(int);
    max_n_s_bytes = net.n_max_state * sizeof(float);
}

void get_output_states(std::vector<float> &ma, std::vector<float> Sa,
                       std::vector<float> &ma_output,
                       std::vector<float> &Sa_output, int idx)
/*Get output's distrinution
Args:
    ma: Mean of activation units of the entire network
    ma: Variance of activation units of the entire network
    ma_output: mean of activation units of the output layer
    Sa_output: Variance of activation units of the output layer
    idx: Starting index of the output layer
*/
{
    for (int i = 0; i < ma_output.size(); i++) {
        ma_output[i] = ma[idx + i];
        Sa_output[i] = Sa[idx + i];
    }
}

template <typename T>
void update_vector(std::vector<T> &v, std::vector<T> &new_values, int idx,
                   int w)
/*Save new value to vector.
Args:
    v: Vector of interest
    new_values: Values to be stored in the vector v
    idx: Indices of new value in vector v
*/
{
    int N = new_values.size() / w;
    if (v.size() - idx < new_values.size()) {
        throw std::invalid_argument(
            "Vector capacity is insufficient - task.cu");
    }
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < w; c++) {
            v[idx + i * w + c] = new_values[w * i + c];
        }
    }
}

float compute_average_error_rate(std::vector<int> &error_rate, int curr_idx,
                                 int n_past_data)
/*Compute running error rate.
  Args:
    error_rate: Vector of error rate
    curr_idx: Index of the current error rate
    n_past_data: Number of past data from the current index
*/
{
    int end_idx = curr_idx - n_past_data;
    if (end_idx < 0) {
        end_idx = 0;
        n_past_data = curr_idx;
    }

    float tmp = 0;
    for (int i = 0; i < n_past_data; i++) {
        tmp += error_rate[end_idx + i];
    }

    float avg_error = tmp / n_past_data;

    return avg_error;
}

void initialize_network_to_device(Network &net, IndexOut &idx, NetState &state,
                                  Param &theta, IndexGPU &idx_gpu,
                                  StateGPU &state_gpu, ParamGPU &theta_gpu,
                                  DeltaStateGPU &d_state_gpu,
                                  DeltaParamGPU &d_theta_gpu)
/*Send network's data to device
  Args:
    net: Network properties on CPU
    idx: Indices of network on CPU
    state: Hidden states of network on CPU
    theta: Parameters of network on CPU
    idx_gpu: Indices of network on GPU
    state_gpu: Hidden states of network on GPU
    theta_gpu: Parameters of network on GPU
    d_state_gpu: Updated quantities for hidden states on GPU
    d_theta_gpu: Updated quantites for parameters on GPU
*/
{
    // Data transfer for indices
    idx_gpu.set_values(idx);
    idx_gpu.allocate_cuda_memory();
    idx_gpu.copy_host_to_device(idx);

    // Data transfer for states
    state_gpu.set_values(state);
    state_gpu.allocate_cuda_memory();
    state_gpu.copy_host_to_device(state);

    // Data transfer for parameters
    theta_gpu.set_values(theta.mw.size(), theta.mb.size(), theta.mw_sc.size(),
                         theta.mb_sc.size());
    theta_gpu.allocate_cuda_memory();
    theta_gpu.copy_host_to_device(theta);

    // Data transfer for delta state
    d_state_gpu.set_values(net.n_state, state.msc.size(), state.mdsc.size(),
                           net.n_max_state);
    d_state_gpu.allocate_cuda_memory();
    d_state_gpu.copy_host_to_device();

    // Data transfer for delta parameters
    d_theta_gpu.set_values(theta.mw.size(), theta.mb.size(), theta.mw_sc.size(),
                           theta.mb_sc.size());
    d_theta_gpu.allocate_cuda_memory();
    d_theta_gpu.copy_host_to_device();
}

///////////////////////////////////////////////////////////////////////
// AUTOENCODER
///////////////////////////////////////////////////////////////////////
void autoencoder(Network &net_e, IndexOut &idx_e, NetState &state_e,
                 Param &theta_e, Network &net_d, IndexOut &idx_d,
                 NetState &state_d, Param &theta_d, ImageData &imdb,
                 int n_epochs, int n_classes, SavePath &path, bool train_mode,
                 bool debug)
/* Autoencoder network for generating images
   Args:
    net_e: Network properties for encoder
    idx_e: Indices of network for encoder
    state_e: Hidden states of network for encoder
    theta_e: Parameters of network for encoder
    net_d: Network properties for decoder
    idx_d: Indices of network for decoder
    state_d: Hidden states of network for decoder
    theta_d: Parameters of network for decoder
    imdb: Image database
    n_iter: Number of iteration for each epoch
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
 */
{
    // Batch size check
    if (net_e.batch_size != net_d.batch_size) {
        throw std::invalid_argument(
            " Batch size is not equal - Task - Autoencoder");
    }

    // Compute number of data
    int n_iter = imdb.num_data / net_d.batch_size;

    // Input and output layer
    std::vector<float> x_batch, Sx_batch, y_batch, y_batch_e, V_batch_e;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(imdb.num_data);
    std::vector<int> batch_idx(net_d.batch_size);
    std::vector<int> idx_ud_batch(net_d.nye * net_d.batch_size, 0);
    std::vector<int> idx_ud_batch_e(net_e.nye * net_e.batch_size, 0);
    std::vector<int> label_batch(net_d.batch_size, 0);

    x_batch.resize(net_e.batch_size * net_e.nodes[0], 0);
    Sx_batch.resize(net_e.batch_size * net_e.nodes[0], 0);
    y_batch.resize(net_d.batch_size * net_d.nodes.back(), 0);
    y_batch_e.resize(net_e.batch_size * net_e.nodes.back(), 0);
    V_batch_e.resize(net_e.batch_size * net_e.nodes.back(), 0);

    // Transfer data for states of encoder
    IndexGPU idx_e_gpu;
    StateGPU state_e_gpu;
    ParamGPU theta_e_gpu;
    DeltaStateGPU d_state_e_gpu;
    DeltaParamGPU d_theta_e_gpu;
    initialize_network_to_device(net_e, idx_e, state_e, theta_e, idx_e_gpu,
                                 state_e_gpu, theta_e_gpu, d_state_e_gpu,
                                 d_theta_e_gpu);

    // Transfer data for states of decoder
    IndexGPU idx_d_gpu;
    StateGPU state_d_gpu;
    ParamGPU theta_d_gpu;
    DeltaStateGPU d_state_d_gpu;
    DeltaParamGPU d_theta_d_gpu;
    initialize_network_to_device(net_d, idx_d, state_d, theta_d, idx_d_gpu,
                                 state_d_gpu, theta_d_gpu, d_state_d_gpu,
                                 d_theta_d_gpu);

    // Transfer data for input and output
    InputGPU ip_gpu(net_e.nodes[0], net_d.batch_size);
    ip_gpu.allocate_cuda_memory();

    ObsGPU op_e_gpu(net_e.nodes.back(), net_e.nye, net_e.batch_size);
    op_e_gpu.allocate_cuda_memory();

    ObsGPU op_d_gpu(net_d.nodes.back(), net_d.nye, net_d.batch_size);
    op_d_gpu.allocate_cuda_memory();

    // Loop initialization
    int THREADS = 16;
    unsigned int BLOCKS =
        (net_e.batch_size * net_e.nodes[0] + THREADS - 1) / THREADS;
    unsigned int BLOCKS_D =
        (net_d.batch_size * net_d.nodes[0] + THREADS - 1) / THREADS;

    // Compute kernel block for normalization layer
    unsigned int BLOCKS_N_E = (state_e.mra.size() + THREADS - 1) / THREADS;
    unsigned int BLOCKS_N_D = (state_d.mra.size() + THREADS - 1) / THREADS;

    if (train_mode) {
        std::cout << "Training...\n";
        for (int e = 0; e < n_epochs; e++) {
            std::cout << "################\n";
            std::cout << "Epoch #" << e + 1 << "\n";

            // Decay observation noise
            if (e > 0) {
                decay_obs_noise(net_d.sigma_v, net_d.decay_factor_sigma_v,
                                net_d.sigma_v_min);
            }
            std::vector<float> V_batch(net_d.batch_size * net_d.nodes.back(),
                                       pow(net_d.sigma_v, 2));
            std::cout << "sigma v: " << V_batch[0] << "\n";

            // Shufle data
            std::random_shuffle(data_idx.begin(), data_idx.end());

            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // TODO: Make a cleaner way to handle both cases
                if (i == 0 && e == 0) {
                    net_e.ra_mt = 0.0f;
                    net_d.ra_mt = 0.0f;
                } else {
                    net_e.ra_mt = 0.9f;
                    net_d.ra_mt = 0.9f;
                }

                // Load input data for encoder and output data for decoder
                get_batch_idx(data_idx, i * net_d.batch_size, net_e.batch_size,
                              batch_idx);
                get_batch_data(imdb.images, batch_idx, net_e.nodes[0], x_batch);
                get_batch_data(imdb.labels, batch_idx, 1, label_batch);
                ip_gpu.copy_host_to_device(x_batch, Sx_batch);
                op_d_gpu.copy_host_to_device(x_batch, idx_ud_batch, V_batch);

                // Initialize input of encoder
                initializeStates<<<BLOCKS, THREADS>>>(
                    ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_e_gpu.d_mz,
                    state_e_gpu.d_Sz, state_e_gpu.d_ma, state_e_gpu.d_Sa,
                    state_e_gpu.d_J, net_e.batch_size * net_e.nodes[0]);

                // Feed forward for encoder
                feedForward(net_e, theta_e_gpu, idx_e_gpu, state_e_gpu);

                // Initialize the decoder's input. TODO double the position of
                // hidden state for encoder net.
                initializeFullStates<<<BLOCKS_D, THREADS>>>(
                    state_e_gpu.d_mz, state_e_gpu.d_Sz, state_e_gpu.d_ma,
                    state_e_gpu.d_Sa, state_e_gpu.d_J,
                    net_d.nodes[0] * net_d.batch_size, net_e.z_pos.back(),
                    state_d_gpu.d_mz, state_d_gpu.d_Sz, state_d_gpu.d_ma,
                    state_d_gpu.d_Sa, state_d_gpu.d_J);

                // Feed forward for decoder
                feedForward(net_d, theta_d_gpu, idx_d_gpu, state_d_gpu);

                // Feed backward for hidden state and parameters of decoder
                stateBackward(net_d, theta_d_gpu, state_d_gpu, idx_d_gpu,
                              op_d_gpu, d_state_d_gpu);
                paramBackward(net_d, theta_d_gpu, state_d_gpu, d_state_d_gpu,
                              idx_d_gpu, d_theta_d_gpu);

                // Update parameter for decoder
                globalParamUpdate(d_theta_d_gpu, theta_d.mw.size(),
                                  theta_d.mb.size(), theta_d.mw_sc.size(),
                                  theta_d.mb_sc.size(), THREADS, theta_d_gpu);

                // Load output data for encoder
                getInputDeltaState<<<BLOCKS_D, THREADS>>>(
                    d_state_d_gpu.d_delta_mz, d_state_d_gpu.d_delta_Sz,
                    net_d.nodes[0] * net_d.batch_size, op_e_gpu.d_y_batch,
                    op_e_gpu.d_V_batch);

                // op_e_gpu.copy_device_to_host(y_batch_e, idx_ud_batch_e,
                //                              V_batch_e);

                // Feed backward for hidden state and parameters of encoder
                stateBackward(net_e, theta_e_gpu, state_e_gpu, idx_e_gpu,
                              op_e_gpu, d_state_e_gpu);
                paramBackward(net_e, theta_e_gpu, state_e_gpu, d_state_e_gpu,
                              idx_e_gpu, d_theta_e_gpu);

                // Update parameter for encoder
                globalParamUpdate(d_theta_e_gpu, theta_e.mw.size(),
                                  theta_e.mb.size(), theta_e.mw_sc.size(),
                                  theta_e.mb_sc.size(), THREADS, theta_e_gpu);

                ///////////////////////////
                // DEBUG ONLY
                if (debug) {
                    // Transfer data from device to host
                    state_e_gpu.copy_device_to_host(state_e);
                    state_d_gpu.copy_device_to_host(state_d);
                    d_theta_e_gpu.copy_device_to_host();
                    d_theta_d_gpu.copy_device_to_host();

                    // Save results
                    std::string hs_path_d =
                        path.debug_path + "/saved_hidden_state_dec/";
                    std::string hs_path_e =
                        path.debug_path + "/saved_hidden_state_enc/";

                    std::string dp_path_d =
                        path.debug_path + "/saved_delta_param_dec/";
                    std::string dp_path_e =
                        path.debug_path + "/saved_delta_param_enc/";

                    save_hidden_states(hs_path_d, state_d);
                    save_hidden_states(hs_path_e, state_e);
                    save_delta_param(dp_path_e, d_theta_e_gpu);
                    save_delta_param(dp_path_d, d_theta_d_gpu);
                }
            }
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::seconds>(end - start)
                    .count();
            std::cout << " Time per epoch: " << run_time << " sec\n";
            std::cout << " Time left     : "
                      << run_time * (n_epochs - e - 1) / 60 << " mins\n";
        }

        d_state_e_gpu.copy_device_to_host();
        theta_e_gpu.copy_device_to_host(theta_e);
        d_state_d_gpu.copy_device_to_host();
        theta_d_gpu.copy_device_to_host(theta_d);
        state_e_gpu.copy_device_to_host(state_e);
        state_d_gpu.copy_device_to_host(state_d);

        // Save results
        if (debug) {
            std::string res_path_e = path.debug_path + "/saved_result_enc/";
            save_inference_results(res_path_e, d_state_e_gpu, theta_e);

            std::string res_path_d = path.debug_path + "/saved_result_dec/";
            save_inference_results(res_path_d, d_state_d_gpu, theta_d);
        }
    } else {
        std::cout << "Testing...\n";
        std::vector<float> ma_d_batch_out(net_d.batch_size * net_d.nodes.back(),
                                          0);
        std::vector<float> Sa_d_batch_out(net_d.batch_size * net_d.nodes.back(),
                                          0);
        std::vector<float> ma_d_out(imdb.num_data * net_d.nodes.back(), 0);
        std::vector<float> Sa_d_out(imdb.num_data * net_d.nodes.back(), 0);
        std::vector<float> V_batch(net_d.batch_size * net_d.nodes.back(),
                                   pow(net_d.sigma_v, 2));
        int mt_idx = 0;

        // Generate image from test set
        for (int i = 0; i < n_iter; i++) {
            // TODO: set momentum for normalization layer when i > i
            net_e.ra_mt = 1.0f;
            net_d.ra_mt = 1.0f;

            // Load input data for encoder and output data for decoder
            get_batch_idx(data_idx, i, net_e.batch_size, batch_idx);
            get_batch_data(imdb.images, batch_idx, net_e.nodes[0], x_batch);
            get_batch_data(imdb.labels, batch_idx, 1, label_batch);
            ip_gpu.copy_host_to_device(x_batch, Sx_batch);
            op_d_gpu.copy_host_to_device(x_batch, idx_ud_batch, V_batch);

            // Initialize input of encoder
            initializeStates<<<BLOCKS, THREADS>>>(
                ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_e_gpu.d_mz,
                state_e_gpu.d_Sz, state_e_gpu.d_ma, state_e_gpu.d_Sa,
                state_e_gpu.d_J, net_e.batch_size * net_e.nodes[0]);

            // Feed forward for encoder
            feedForward(net_e, theta_e_gpu, idx_e_gpu, state_e_gpu);

            // Initialize the decoder's input.
            initializeFullStates<<<BLOCKS_D, THREADS>>>(
                state_e_gpu.d_mz, state_e_gpu.d_Sz, state_e_gpu.d_ma,
                state_e_gpu.d_Sa, state_e_gpu.d_J,
                net_d.nodes[0] * net_d.batch_size, net_e.z_pos.back(),
                state_d_gpu.d_mz, state_d_gpu.d_Sz, state_d_gpu.d_ma,
                state_d_gpu.d_Sa, state_d_gpu.d_J);

            // Feed forward for decoder
            feedForward(net_d, theta_d_gpu, idx_d_gpu, state_d_gpu);

            // Get hidden states for output layers
            state_d_gpu.copy_device_to_host(state_d);
            get_output_states(state_d.ma, state_d.Sa, ma_d_batch_out,
                              Sa_d_batch_out, net_d.z_pos.back());

            // Update the final hidden state vector for last layer
            mt_idx = i * net_d.batch_size * net_d.nodes.back();
            update_vector(ma_d_out, ma_d_batch_out, mt_idx, net_d.nodes.back());
        }
        std::cout << std::endl;

        // Save generated images
        std::string suffix = "test";
        save_generated_images(path.saved_inference_path, ma_d_out, suffix);
    }
}

///////////////////////////////////////////////////////////////////////
// CLASSIFICATION
///////////////////////////////////////////////////////////////////////
void classification(Network &net, IndexOut &idx, NetState &state, Param &theta,
                    ImageData &imdb, ImageData &test_imdb, int n_epochs,
                    int n_classes, SavePath &path, bool train_mode, bool debug)
/*Classification task

  Args:
    Net: Network architecture
    idx: Indices of network
    theta: Weights & biases of network
    imdb: Image database
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
 */
{
    // Seed
    std::srand(unsigned(std::time(0)));

    // Number of bytes
    size_t id_bytes, od_bytes, ode_bytes, max_n_s_bytes;
    compute_net_memory(net, id_bytes, od_bytes, ode_bytes, max_n_s_bytes);

    // Input and output layer
    auto hrs = class_to_obs(n_classes);
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(test_imdb.num_data);
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);
    std::vector<int> label_batch(net.batch_size, 0);

    x_batch.resize(net.batch_size * net.nodes[0], 0);
    Sx_batch.resize(net.batch_size * net.nodes[0], 0);
    y_batch.resize(net.batch_size * hrs.n_obs, 0);
    V_batch.resize(net.batch_size * hrs.n_obs, pow(net.sigma_v, 2));

    IndexGPU idx_gpu;
    StateGPU state_gpu;
    ParamGPU theta_gpu;
    DeltaStateGPU d_state_gpu;
    DeltaParamGPU d_theta_gpu;

    initialize_network_to_device(net, idx, state, theta, idx_gpu, state_gpu,
                                 theta_gpu, d_state_gpu, d_theta_gpu);

    // Data transfer for input and output data
    InputGPU ip_gpu(net.nodes[0], net.batch_size);
    ip_gpu.allocate_cuda_memory();

    ObsGPU op_gpu(net.nodes[net.nodes.size() - 1], net.nye, net.batch_size);
    op_gpu.allocate_cuda_memory();

    // Initialization
    int wN = theta.mw.size();
    int bN = theta.mb.size();
    int wN_sc = theta.mw_sc.size();
    int bN_sc = theta.mb_sc.size();

    int THREADS = 16;
    unsigned int BLOCKS =
        (net.batch_size * net.nodes[0] + THREADS - 1) / THREADS;
    int mt_idx = 0;
    int n_iter = imdb.num_data / net.batch_size;
    int test_n_iter = test_imdb.num_data / net.batch_size;

    // Error rate for training
    std::vector<int> error_rate(imdb.num_data, 0);
    std::vector<float> prob_class(imdb.num_data * n_classes);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;

    // Error rate for testing
    std::vector<float> test_epoch_error_rate(n_epochs, 0);
    std::vector<int> test_error_rate(test_imdb.num_data, 0);
    std::vector<float> prob_class_test(test_imdb.num_data * n_classes);
    std::vector<float> test_epoch_prob_class(test_imdb.num_data * n_classes *
                                             n_epochs);
    std::vector<float> ma_output(
        net.batch_size * net.nodes[net.nodes.size() - 1], 0);
    std::vector<float> Sa_output(
        net.batch_size * net.nodes[net.nodes.size() - 1], 0);

    for (int e = 0; e < n_epochs; e++) {
        // Shufle data
        std::random_shuffle(data_idx.begin(), data_idx.end());

        // Timer
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "\n";
        std::cout << "Training...\n";
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; i++) {
            // TODO: Make a cleaner way to handle both cases
            if (i == 0 && e == 0) {
                net.ra_mt = 0.0f;
            } else {
                net.ra_mt = 0.9f;
            }

            // Load data
            get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                          batch_idx);
            get_batch_data(imdb.images, batch_idx, net.nodes[0], x_batch);
            get_batch_data(imdb.obs_label, batch_idx, hrs.n_obs, y_batch);
            get_batch_data(imdb.obs_idx, batch_idx, hrs.n_obs, idx_ud_batch);
            get_batch_data(imdb.labels, batch_idx, 1, label_batch);
            ip_gpu.copy_host_to_device(x_batch, Sx_batch);
            op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

            // Initialize input
            initializeStates<<<BLOCKS, THREADS>>>(
                ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa, state_gpu.d_J,
                net.batch_size * net.nodes[0]);

            // Feed forward
            feedForward(net, theta_gpu, idx_gpu, state_gpu);

            // Feed backward for hidden states
            stateBackward(net, theta_gpu, state_gpu, idx_gpu, op_gpu,
                          d_state_gpu);

            // Feed backward for parameters
            paramBackward(net, theta_gpu, state_gpu, d_state_gpu, idx_gpu,
                          d_theta_gpu);

            // Update model parameters.
            globalParamUpdate(d_theta_gpu, wN, bN, wN_sc, bN_sc, THREADS,
                              theta_gpu);

            // Compute error rate
            state_gpu.copy_device_to_host(state);
            get_output_states(state.ma, state.Sa, ma_output, Sa_output,
                              net.z_pos[net.nodes.size() - 1]);
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(ma_output, Sa_output, label_batch, hrs, n_classes,
                          net.batch_size);
            mt_idx = i * net.batch_size;
            update_vector(error_rate, error_rate_batch, mt_idx, 1);
            // update_vector(prob_class, prob_class_batch, mt_idx,
            // n_classes);

            if (i % 1000 == 0) {
                int curr_idx = mt_idx + net.batch_size;
                auto avg_error =
                    compute_average_error_rate(error_rate, curr_idx, 100);

                std::cout << "\tError rate for last 100 observation: ";
                std::cout << std::fixed;
                std::cout << std::setprecision(3);
                std::cout << avg_error << "\n";
            }
        }
        std::cout << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto run_time =
            std::chrono::duration_cast<std::chrono::seconds>(end - start)
                .count();
        std::cout << " Time per epoch: " << run_time << " sec\n";
        std::cout << " Time left     : " << run_time * (n_epochs - e - 1) / 60
                  << " mins\n";
        std::cout << "Testing...\n";
        for (int i = 0; i < test_n_iter; i++) {
            // TODO: set = 0.9 when i > 0 or disable mean and variance in
            // feed forward
            net.ra_mt = 0.0f;

            // Load data
            get_batch_idx(test_data_idx, i, net.batch_size, batch_idx);
            get_batch_data(test_imdb.images, batch_idx, net.nodes[0], x_batch);
            get_batch_data(test_imdb.obs_label, batch_idx, hrs.n_obs, y_batch);
            get_batch_data(test_imdb.obs_idx, batch_idx, hrs.n_obs,
                           idx_ud_batch);
            get_batch_data(test_imdb.labels, batch_idx, 1, label_batch);
            ip_gpu.copy_host_to_device(x_batch, Sx_batch);
            op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

            // Initialize input
            initializeStates<<<BLOCKS, THREADS>>>(
                ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa, state_gpu.d_J,
                net.batch_size * net.nodes[0]);

            // Feed forward
            feedForward(net, theta_gpu, idx_gpu, state_gpu);

            // Compute error rate
            state_gpu.copy_device_to_host(state);
            get_output_states(state.ma, state.Sa, ma_output, Sa_output,
                              net.z_pos[net.nodes.size() - 1]);
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(ma_output, Sa_output, label_batch, hrs, n_classes,
                          net.batch_size);
            mt_idx = i * net.batch_size;
            update_vector(test_error_rate, error_rate_batch, mt_idx, 1);
            // update_vector(prob_class, prob_class_batch, mt_idx,
            // n_classes);
        }

        auto test_avg_error = compute_average_error_rate(
            test_error_rate, test_imdb.num_data, test_imdb.num_data);
        test_epoch_error_rate[e] = test_avg_error;
        std::cout << "\n";
        std::cout << "\tError rate: ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << test_avg_error << "\n" << std::endl;
    }
    d_state_gpu.copy_device_to_host();
    theta_gpu.copy_device_to_host(theta);
    d_theta_gpu.copy_device_to_host();

    // Save error rate
    std::string suffix = "test";
    save_error_rate(path.saved_inference_path, test_epoch_error_rate, suffix);
    // Save debugging data
    if (debug) {
        std::string res_path = path.debug_path + "/saved_results/";
        save_inference_results(res_path, d_state_gpu, theta);
    }
}

///////////////////////////////////////////////////////////////////////
// REGRESSION
///////////////////////////////////////////////////////////////////////
void regression(Network &net, IndexOut &idx, NetState &state, Param &theta,
                Dataloader &db, int n_epochs, SavePath &path, bool train_mode,
                bool debug)
/* Regression task

Args:
    Net: Network architecture
    idx: Indices of network
    theta: Weights & biases of network
    db: database
    n_epochs: Number of epochs
    path: Directory stored the final results
    train_mode: Whether to train the network
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
*/
{
    // Compute number of data
    int n_iter = db.num_data / net.batch_size;

    // Number of bytes
    size_t id_bytes, od_bytes, ode_bytes, max_n_s_bytes;
    compute_net_memory(net, id_bytes, od_bytes, ode_bytes, max_n_s_bytes);

    // Copie data
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> data_idx = create_range(db.num_data);
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);

    x_batch.resize(net.batch_size * net.nodes.front(), 0);
    Sx_batch.resize(net.batch_size * net.nodes.front(), 0);
    y_batch.resize(net.batch_size * net.nodes.back(), 0);
    V_batch.resize(net.batch_size * net.nodes.back(), pow(net.sigma_v, 2));

    // Data transfer
    StateGPU state_gpu;
    ParamGPU theta_gpu;
    IndexGPU idx_gpu;
    DeltaStateGPU d_state_gpu;
    DeltaParamGPU d_theta_gpu;

    initialize_network_to_device(net, idx, state, theta, idx_gpu, state_gpu,
                                 theta_gpu, d_state_gpu, d_theta_gpu);

    // Data transfer for input and output data
    InputGPU ip_gpu(net.nodes.front(), net.batch_size);
    ip_gpu.allocate_cuda_memory();

    ObsGPU op_gpu(net.nodes.back(), net.nye, net.batch_size);
    op_gpu.allocate_cuda_memory();

    int wN = theta.mw.size();
    int bN = theta.mb.size();
    int wN_sc = theta.mw_sc.size();
    int bN_sc = theta.mb_sc.size();

    int THREADS = 16;
    unsigned int BLOCKS =
        (net.batch_size * net.nodes.front() + THREADS - 1) / THREADS;

    if (train_mode) {
        // Shufle data
        std::random_shuffle(data_idx.begin(), data_idx.end());
        for (int e = 0; e < n_epochs; e++) {
            // Timer
            std::cout << "################\n";
            std::cout << "Epoch #" << e + 1 << "\n";
            std::cout << "Training...\n";
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // Load data
                get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                              batch_idx);
                get_batch_data(db.x, batch_idx, net.nodes.front(), x_batch);
                get_batch_data(db.y, batch_idx, net.nodes.back(), y_batch);
                ip_gpu.copy_host_to_device(x_batch, Sx_batch);
                op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

                // Initialize input
                initializeStates<<<BLOCKS, THREADS>>>(
                    ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                    state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa,
                    state_gpu.d_J, net.batch_size * net.nodes.front());

                // Feed forward
                feedForward(net, theta_gpu, idx_gpu, state_gpu);

                // Feed backward for hidden states
                stateBackward(net, theta_gpu, state_gpu, idx_gpu, op_gpu,
                              d_state_gpu);

                // Feed backward for parameters
                paramBackward(net, theta_gpu, state_gpu, d_state_gpu, idx_gpu,
                              d_theta_gpu);

                // Update model parameters
                globalParamUpdate(d_theta_gpu, wN, bN, wN_sc, bN_sc, THREADS,
                                  theta_gpu);
            }

            // Report running time
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::seconds>(end - start)
                    .count();
            std::cout << " Time per epoch: " << run_time << " sec\n";
            std::cout << " Time left     : "
                      << run_time * (n_epochs - e - 1) / 60 << " mins\n";
        }
        // state_gpu.copy_device_to_host(state);
        theta_gpu.copy_device_to_host(theta);

    } else {
        std::cout << "Testing...\n";
        std::vector<float> ma_batch_out(net.batch_size * net.nodes.back(), 0);
        std::vector<float> Sa_batch_out(net.batch_size * net.nodes.back(), 0);
        std::vector<float> ma_out(db.num_data * net.nodes.back(), 0);
        std::vector<float> Sa_out(db.num_data * net.nodes.back(), 0);
        int mt_idx = 0;

        // Prediction
        for (int i = 0; i < n_iter; i++) {
            // Load data
            get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                          batch_idx);
            get_batch_data(db.x, batch_idx, net.nodes.front(), x_batch);
            get_batch_data(db.y, batch_idx, net.nodes.back(), y_batch);
            ip_gpu.copy_host_to_device(x_batch, Sx_batch);
            op_gpu.copy_host_to_device(y_batch, idx_ud_batch, V_batch);

            // Initialize input
            initializeStates<<<BLOCKS, THREADS>>>(
                ip_gpu.d_x_batch, ip_gpu.d_Sx_batch, state_gpu.d_mz,
                state_gpu.d_Sz, state_gpu.d_ma, state_gpu.d_Sa, state_gpu.d_J,
                net.batch_size * net.nodes.front());

            // Feed forward
            feedForward(net, theta_gpu, idx_gpu, state_gpu);

            // Get hidden states for output layers
            state_gpu.copy_device_to_host(state);
            get_output_states(state.ma, state.Sa, ma_batch_out, Sa_batch_out,
                              net.z_pos.back());

            // Update the final hidden state vector for last layer
            mt_idx = i * net.batch_size * net.nodes.back();
            update_vector(ma_out, ma_batch_out, mt_idx, net.nodes.back());
            update_vector(Sa_out, Sa_batch_out, mt_idx, net.nodes.back());
        }
        // Denormalize data
        std::vector<float> sy_norm(db.y.size(), 0);
        std::vector<float> my(sy_norm.size(), 0);
        std::vector<float> sy(sy_norm.size(), 0);
        std::vector<float> y_test(sy_norm.size(), 0);
        for (int k = 0; k < db.y.size(); k++) {
            sy_norm[k] = pow(Sa_out[k], 0.5) + net.sigma_v;
        }
        denormalize_mean(ma_out, db.mu_y, db.sigma_y, net.nodes.back(), my);
        denormalize_mean(db.y, db.mu_y, db.sigma_y, net.nodes.back(), y_test);
        denormalize_std(sy_norm, db.mu_y, db.sigma_y, net.nodes.back(), sy);

        // Compute metrics
        auto mse = mean_squared_error(my, y_test);

        // Compute log-likelihood
        auto log_lik = avg_univar_log_lik(my, y_test, sy);

        // Display results
        std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "RMSE           : " << pow(mse, 0.5) << "\n";
        std::cout << "Log likelihood: " << log_lik;
        std::cout << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////
// TASK MAIN
///////////////////////////////////////////////////////////////////////
void task_command(std::string &user_input_file, SavePath &path) {
    /* Assign different tasks and its parameters

    Args:
        user_input_file: User-specified inputs
        res_path: Directory path where results are stored under *.csv file
    */
    auto user_input = load_userinput(user_input_file);

    if (user_input.task_name == "classification") {
        // Network
        bool train_mode = true;
        IndexOut idx;
        Network net;
        Param theta;
        NetState state;
        net_init(user_input.net_name, net, theta, state, idx);
        net.is_idx_ud = true;

        // Data
        auto hrs = class_to_obs(user_input.num_classes);
        net.nye = hrs.n_obs;
        auto imdb = get_images(user_input.data_name, user_input.x_train_dir,
                               user_input.y_train_dir, user_input.mu,
                               user_input.sigma, net.widths[0], net.heights[0],
                               net.filters[0], hrs, user_input.num_train_data);
        auto test_imdb = get_images(
            user_input.data_name, user_input.x_test_dir, user_input.y_test_dir,
            user_input.mu, user_input.sigma, net.widths[0], net.heights[0],
            net.filters[0], hrs, user_input.num_test_data);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, theta);
        }

        // Saved debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "/saved_param/";
            std::string idx_path = path.debug_path + "/saved_idx/";
            save_net_prop(param_path, idx_path, theta, idx);
        }

        std::cout << "Training...\n" << std::endl;
        classification(net, idx, state, theta, imdb, test_imdb,
                       user_input.num_epochs, user_input.num_classes, path,
                       train_mode, user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, theta);

    } else if (user_input.task_name == "autoencoder") {
        // Encoder
        IndexOut idx_e;
        Network net_e;
        Param theta_e;
        NetState state_e;
        net_init(user_input.encoder_net_name, net_e, theta_e, state_e, idx_e);
        net_e.is_output_ud = false;

        // Decoder
        IndexOut idx_d;
        Network net_d;
        Param theta_d;
        NetState state_d;
        net_init(user_input.decoder_net_name, net_d, theta_d, state_d, idx_d);
        net_d.is_idx_ud = false;
        // It eable to infer the input's hidden states
        net_d.last_backward_layer = 0;

        // Load data
        auto hrs = class_to_obs(user_input.num_classes);
        auto imdb =
            get_images(user_input.data_name, user_input.x_train_dir,
                       user_input.y_train_dir, user_input.mu, user_input.sigma,
                       net_e.widths[0], net_e.heights[0], net_e.filters[0], hrs,
                       user_input.num_train_data);
        auto test_imdb = get_images(
            user_input.data_name, user_input.x_test_dir, user_input.y_test_dir,
            user_input.mu, user_input.sigma, net_e.widths[0], net_e.heights[0],
            net_e.filters[0], hrs, user_input.num_test_data);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.encoder_net_name,
                           path.saved_param_path, theta_e);
            load_net_param(user_input.model_name, user_input.decoder_net_name,
                           path.saved_param_path, theta_d);
        }

        // Save data for debugging
        if (user_input.debug) {
            save_autoencoder_net_prop(theta_e, theta_d, idx_e, idx_d,
                                      path.debug_path);
        }

        // Train network
        bool train_mode = true;
        autoencoder(net_e, idx_e, state_e, theta_e, net_d, idx_d, state_d,
                    theta_d, imdb, user_input.num_epochs,
                    user_input.num_classes, path, train_mode, user_input.debug);

        save_net_param(user_input.model_name, user_input.encoder_net_name,
                       path.saved_param_path, theta_e);
        save_net_param(user_input.model_name, user_input.decoder_net_name,
                       path.saved_param_path, theta_d);

        train_mode = false;
        autoencoder(net_e, idx_e, state_e, theta_e, net_d, idx_d, state_d,
                    theta_d, test_imdb, user_input.num_epochs,
                    user_input.num_classes, path, train_mode, user_input.debug);

    } else if (user_input.task_name == "regression") {
        // Train network
        IndexOut idx;
        Network net;
        Param theta;
        NetState state;

        // Test network
        IndexOut test_idx;
        Network test_net;
        NetState test_state;
        int test_batch_size = 1;

        net_init(user_input.net_name, net, theta, state, idx);
        reset_net_batchsize(user_input.net_name, test_net, test_state, test_idx,
                            test_batch_size);

        // Train data
        std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
        auto train_db =
            get_dataloader(user_input.x_train_dir, user_input.y_train_dir, mu_x,
                           sigma_x, mu_y, sigma_y, user_input.num_train_data,
                           net.nodes.front(), net.nodes.back());
        // Test data
        auto test_db = get_dataloader(
            user_input.x_test_dir, user_input.y_test_dir, train_db.mu_x,
            train_db.sigma_x, train_db.mu_y, train_db.sigma_y,
            user_input.num_test_data, net.nodes.front(), net.nodes.back());

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, theta);
        }

        // Save network's parameter to debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "/saved_param/";
            save_param(param_path, theta);
        }

        // Training
        bool train_mode = true;
        regression(net, idx, state, theta, train_db, user_input.num_epochs,
                   path, train_mode, user_input.debug);

        // Testing
        train_mode = false;
        regression(test_net, test_idx, test_state, theta, test_db,
                   user_input.num_epochs, path, train_mode, user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, theta);
    } else {
        throw std::invalid_argument("Task name does not exist - task.cu");
    }
}

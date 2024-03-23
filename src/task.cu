///////////////////////////////////////////////////////////////////////////////
// File:         task.cu
// Description:  providing different tasks such as regression, classification
//               that uses TAGI approach.
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 23, 2022
// Updated:      March 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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
    id_bytes = net.batch_size * net.n_x * sizeof(float);
    od_bytes = net.batch_size * net.n_y * sizeof(float);
    ode_bytes = net.batch_size * net.nye * sizeof(int);
    max_n_s_bytes = net.n_max_state * sizeof(float);
}

///////////////////////////////////////////////////////////////////////
// AUTOENCODER
///////////////////////////////////////////////////////////////////////
void autoencoder(TagiNetwork &net_e, TagiNetwork &net_d, ImageData &imdb,
                 int n_epochs, int n_classes, SavePath &path, bool train_mode,
                 bool debug)
/* Autoencoder network for generating images

Args:
    net_e: Encoder
    net_d: Decoder
    theta_d: Parameters of network for decoder
    imdb: Image database
    n_iter: Number of iteration for each epoch
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
 */
{
    // Seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);

    // Batch size check
    if (net_e.prop.batch_size != net_d.prop.batch_size) {
        throw std::invalid_argument(
            " Batch size is not equal - Task - Autoencoder");
    }

    // Compute number of data
    int n_iter = imdb.num_data / net_d.prop.batch_size;

    // Input and output layer
    std::vector<float> x_batch, Sx_batch, y_batch, y_batch_e, V_batch_e,
        delta_mz, delta_Sz;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(imdb.num_data);
    std::vector<int> batch_idx(net_d.prop.batch_size);
    std::vector<int> idx_ud_batch(net_d.prop.nye * net_d.prop.batch_size, 0);
    std::vector<int> idx_ud_batch_e(net_e.prop.nye * net_e.prop.batch_size, 0);
    std::vector<int> label_batch(net_d.prop.batch_size, 0);

    x_batch.resize(net_e.prop.batch_size * net_e.prop.n_x, 0);
    Sx_batch.resize(net_e.prop.batch_size * net_e.prop.n_x,
                    powf(net_e.prop.sigma_x, 2));
    y_batch.resize(net_d.prop.batch_size * net_d.prop.n_y, 0);
    y_batch_e.resize(net_e.prop.batch_size * net_e.prop.n_y, 0);
    V_batch_e.resize(net_e.prop.batch_size * net_e.prop.n_y, 0);

    // *TODO: Is there any better way?
    std::vector<float> Sx_f_batch;
    if (net_e.prop.is_full_cov) {
        float var_x = powf(net_e.prop.sigma_x, 2);
        auto Sx_f = initialize_upper_triu(var_x, net_e.prop.n_x);
        Sx_f_batch = repmat_vector(Sx_f, net_e.prop.batch_size);
    }

    /* TRAINING */
    if (train_mode) {
        std::cout << "Training...\n";
        for (int e = 0; e < n_epochs; e++) {
            std::cout << "################\n";
            std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";

            if (e > 0) {
                // Shufle data
                std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

                // Decay observation noise
                decay_obs_noise(net_d.prop.sigma_v,
                                net_d.prop.decay_factor_sigma_v,
                                net_d.prop.sigma_v_min);
            }
            std::vector<float> V_batch(net_d.prop.batch_size * net_d.prop.n_y,
                                       powf(net_d.prop.sigma_v, 2));

            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                ;
                // TODO: Make a cleaner way to handle both cases
                if (i == 0 && e == 0) {
                    net_e.prop.ra_mt = 0.0f;
                    net_d.prop.ra_mt = 0.0f;
                } else {
                    net_e.prop.ra_mt = 0.9f;
                    net_d.prop.ra_mt = 0.9f;
                }

                // Load input data for encoder and output data for decoder
                get_batch_idx(data_idx, i * net_d.prop.batch_size,
                              net_e.prop.batch_size, batch_idx);
                get_batch_data(imdb.images, batch_idx, net_e.prop.n_x, x_batch);
                get_batch_data(imdb.labels, batch_idx, 1, label_batch);

                // Feed forward for encoder
                net_e.feed_forward(x_batch, Sx_batch, Sx_f_batch);

                // Get all output's states of encoder
                net_e.get_all_network_outputs();

                // Feed forward for decoder
                net_d.connected_feed_forward(net_e.ma, net_e.Sa, net_e.mz,
                                             net_e.Sz, net_e.J);

                // Feed backward for decoder
                net_d.state_feed_backward(x_batch, V_batch, idx_ud_batch);
                net_d.param_feed_backward();

                // Feed backward for encoder
                std::tie(delta_mz, delta_Sz) = net_d.get_state_delta_mean_var();
                net_e.state_feed_backward(delta_mz, delta_Sz, idx_ud_batch);
                net_e.param_feed_backward();

                ///////////////////////////
                // DEBUG ONLY
                if (debug) {
                    // Transfer data from device to host
                    net_e.state_gpu.copy_device_to_host();
                    net_d.state_gpu.copy_device_to_host();
                    net_e.d_theta_gpu.copy_device_to_host();
                    net_d.theta_gpu.copy_device_to_host();

                    // Save results
                    std::string hs_path_d =
                        path.debug_path + "/saved_hidden_state_dec/";
                    std::string hs_path_e =
                        path.debug_path + "/saved_hidden_state_enc/";

                    std::string dp_path_d =
                        path.debug_path + "/saved_delta_param_dec/";
                    std::string dp_path_e =
                        path.debug_path + "/saved_delta_param_enc/";

                    save_hidden_states(hs_path_e, net_e.state);
                    save_hidden_states(hs_path_d, net_d.state);
                    save_delta_param(dp_path_e, net_e.d_theta_gpu);
                    save_delta_param(dp_path_d, net_d.d_theta_gpu);
                }
            }
            // Report computational time
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
            std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
            std::cout << " Time left     : ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60
                      << " mins\n";
        }
        net_e.theta_gpu.copy_device_to_host();
        net_d.theta_gpu.copy_device_to_host();

        // Save results
        if (debug) {
            net_e.state_gpu.copy_device_to_host();
            net_d.state_gpu.copy_device_to_host();
            net_e.d_state_gpu.copy_device_to_host();
            net_d.d_state_gpu.copy_device_to_host();
            std::string res_path_e = path.debug_path + "/saved_result_enc/";
            save_inference_results(res_path_e, net_e.d_state_gpu, net_e.theta);

            std::string res_path_d = path.debug_path + "/saved_result_dec/";
            save_inference_results(res_path_d, net_d.d_state_gpu, net_d.theta);
        }
    } else {
        /* TESTING */
        std::cout << "Testing...\n";
        std::vector<float> ma_d_batch_out(
            net_d.prop.batch_size * net_d.prop.n_y, 0);
        std::vector<float> Sa_d_batch_out(
            net_d.prop.batch_size * net_d.prop.n_y, 0);
        std::vector<float> ma_d_out(imdb.num_data * net_d.prop.n_y, 0);
        std::vector<float> Sa_d_out(imdb.num_data * net_d.prop.n_y, 0);
        std::vector<float> V_batch(net_d.prop.batch_size * net_d.prop.n_y,
                                   powf(net_d.prop.sigma_v, 2));
        int mt_idx = 0;

        // Generate image from test set
        for (int i = 0; i < n_iter; i++) {
            // TODO: set momentum for normalization layer when i > i
            net_e.prop.ra_mt = 1.0f;
            net_d.prop.ra_mt = 1.0f;

            // Load input data for encoder and output data for decoder
            get_batch_idx(data_idx, i, net_e.prop.batch_size, batch_idx);
            get_batch_data(imdb.images, batch_idx, net_e.prop.n_x, x_batch);
            get_batch_data(imdb.labels, batch_idx, 1, label_batch);

            // Feed forward for encoder
            net_e.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Get all output's states of encoder
            net_e.get_all_network_outputs();

            // Feed forward for decoder
            net_d.connected_feed_forward(net_e.ma, net_e.Sa, net_e.mz, net_e.Sz,
                                         net_e.J);

            // Get hidden states for output layers
            net_d.state_gpu.copy_device_to_host();
            output_hidden_states(net_d.state, net_d.prop, ma_d_batch_out,
                                 Sa_d_batch_out);

            // Update the final hidden state vector for last layer
            mt_idx = i * net_d.prop.batch_size * net_d.prop.n_y;
            update_vector(ma_d_out, ma_d_batch_out, mt_idx, net_d.prop.n_y);
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
void classification(TagiNetwork &net, ImageData &imdb, ImageData &test_imdb,
                    int n_epochs, int n_classes, SavePath &path,
                    bool train_mode, bool debug)
/*Classification task

Args:
    Net: Tagi network
    imdb: Image database
    n_epochs: Number of epochs
    n_classes: Number of classes of image data
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
 */
{
    // Seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(123456);

    // Compute number of data points
    int n_iter = imdb.num_data / net.prop.batch_size;
    int test_n_iter = test_imdb.num_data / net.prop.batch_size;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(test_imdb.num_data);

    // Input and output layer
    net.prop.nye = imdb.output_len;
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);
    std::vector<int> label_batch(net.prop.batch_size, 0);

    x_batch.resize(net.prop.batch_size * net.prop.n_x, 0);
    Sx_batch.resize(net.prop.batch_size * net.prop.n_x,
                    powf(net.prop.sigma_x, 2));
    y_batch.resize(net.prop.batch_size * imdb.output_len, 0);
    V_batch.resize(net.prop.batch_size * imdb.output_len,
                   powf(net.prop.sigma_v, 2));

    // *TODO: Is there any better way?
    std::vector<float> Sx_f_batch;
    if (net.prop.is_full_cov) {
        float var_x = powf(net.prop.sigma_x, 2);
        auto Sx_f = initialize_upper_triu(var_x, net.prop.n_x);
        Sx_f_batch = repmat_vector(Sx_f, net.prop.batch_size);
    }

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
    int mt_idx = 0;

    for (int e = 0; e < n_epochs; e++) {
        /* TRAINING */
        if (e > 0) {
            // Shufle data
            std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
            // Decay observation noise
            decay_obs_noise(net.prop.sigma_v, net.prop.decay_factor_sigma_v,
                            net.prop.sigma_v_min);
        }
        std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                                   powf(net.prop.sigma_v, 2));
        // Timer
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
        std::cout << "Training...\n";
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < n_iter; i++) {
            // TODO: Make a cleaner way to handle both cases
            if (i == 0 && e == 0) {
                net.prop.ra_mt = 0.0f;
            } else {
                net.prop.ra_mt = 0.9f;
            }

            // Load data
            get_batch_images_labels(imdb, data_idx, net.prop.batch_size, i,
                                    x_batch, y_batch, idx_ud_batch,
                                    label_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Feed backward for hidden states
            net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

            // Feed backward for parameters
            net.param_feed_backward();

            // Compute error rate
            net.get_network_outputs();
            if (net.prop.activations.back() == net.prop.act_names.hr_softmax) {
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(net.ma, net.Sa, label_batch, n_classes,
                              net.prop.batch_size);
            } else {
                error_rate_batch = get_class_error(
                    net.ma, label_batch, n_classes, net.prop.batch_size);
            }
            // std::tie(error_rate_batch, prob_class_batch) = get_error(
            //     net.ma, net.Sa, label_batch, n_classes, net.prop.batch_size);
            mt_idx = i * net.prop.batch_size;
            update_vector(error_rate, error_rate_batch, mt_idx, 1);

            if (i % 1000 == 0) {
                int curr_idx = mt_idx + net.prop.batch_size;
                auto avg_error =
                    compute_average_error_rate(error_rate, curr_idx, 100);

                std::cout << "\tError rate for last 100 observation: ";
                std::cout << std::fixed;
                std::cout << std::setprecision(3);
                std::cout << avg_error << "\n";
            }
        }
        // Report computational time
        std::cout << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto run_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
        std::cout << " Time left     : ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60 << " mins\n";

        /* TESTING */
        std::cout << "Testing...\n";
        for (int i = 0; i < test_n_iter; i++) {
            // TODO: set = 0.9 when i > 0 or disable mean and variance in
            // feed forward
            net.prop.ra_mt = 0.0f;

            // Load data
            get_batch_images(test_imdb, test_data_idx, net.prop.batch_size, i,
                             x_batch, label_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Compute error rate
            net.get_network_outputs();
            if (net.prop.activations.back() == net.prop.act_names.hr_softmax) {
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(net.ma, net.Sa, label_batch, n_classes,
                              net.prop.batch_size);
            } else {
                error_rate_batch = get_class_error(
                    net.ma, label_batch, n_classes, net.prop.batch_size);
            }
            mt_idx = i * net.prop.batch_size;
            update_vector(test_error_rate, error_rate_batch, mt_idx, 1);
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
    net.theta_gpu.copy_device_to_host();

    // Save error rate
    std::string suffix = "test";
    save_error_rate(path.saved_inference_path, test_epoch_error_rate, suffix);
    // Save debugging data
    if (debug) {
        net.d_state_gpu.copy_device_to_host();
        std::string res_path = path.debug_path + "/saved_results/";
        save_inference_results(res_path, net.d_state_gpu, net.theta);
    }
}

///////////////////////////////////////////////////////////////////////
// REGRESSION
///////////////////////////////////////////////////////////////////////
void regression(TagiNetwork &net, Dataloader &db, int n_epochs, SavePath &path,
                bool train_mode, bool debug)
/* Regression task

Args:
    net: Tagi network
    db: database
    n_epochs: Number of epochs
    path: Directory stored the final results
    train_mode: Whether to train the network
    path: Directory stored the final results
    debug: Debugging mode allows saving inference data
*/
{
    // Seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(123456);
    int derivative_layer = 0;

    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(net.prop.batch_size * net.prop.n_x, 0);
    std::vector<float> Sx_batch(net.prop.batch_size * net.prop.n_x,
                                pow(net.prop.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                               pow(net.prop.sigma_v, 2));
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);

    // *TODO: Is there any better way?
    if (net.prop.is_full_cov) {
        float var_x = powf(net.prop.sigma_x, 2);
        auto Sx_f = initialize_upper_triu(var_x, net.prop.n_x);
        Sx_f_batch = repmat_vector(Sx_f, net.prop.batch_size);
    }

    /* TRAINING */
    if (train_mode) {
        for (int e = 0; e < n_epochs; e++) {
            // Shufle data
            if (e > 0) {
                // Shufle data
                std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

                // Decay observation noise
                decay_obs_noise(net.prop.sigma_v, net.prop.decay_factor_sigma_v,
                                net.prop.sigma_v_min);
            }
            std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                                       powf(net.prop.sigma_v, 2));

            // Timer
            std::cout << "################\n";
            std::cout << "Epoch PP #" << e + 1 << "/" << n_epochs << "\n";
            std::cout << "Training...\n";
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // Load data
                get_batch_idx(data_idx, i * net.prop.batch_size,
                              net.prop.batch_size, batch_idx);
                get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);
                get_batch_data(db.y, batch_idx, net.prop.n_y, y_batch);

                // Feed forward
                net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

                net.get_network_outputs();

                // Feed backward for hidden states
                net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

                // Feed backward for parameters
                net.param_feed_backward();
            }

            // Report running time
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
            std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
            std::cout << " Time left     : ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60
                      << " mins\n";
        }
        // state_gpu.copy_device_to_host(state);
        net.theta_gpu.copy_device_to_host();

        // Retrieve homocesdastic noise distribution's parameter
        if (net.prop.noise_type.compare("homosce") == 0) {
            net.state_gpu.copy_device_to_host();
            get_homosce_noise_param(net.state.noise_state.ma_v2b_prior,
                                    net.state.noise_state.Sa_v2b_prior,
                                    net.prop.mu_v2b, net.prop.sigma_v2b);
        }

    } else {
        /* TESTING */
        std::cout << "Testing...\n";
        std::vector<float> ma_batch_out(net.prop.batch_size * net.prop.n_y, 0);
        std::vector<float> Sa_batch_out(net.prop.batch_size * net.prop.n_y, 0);
        std::vector<float> ma_out(db.num_data * net.prop.n_y, 0);
        std::vector<float> Sa_out(db.num_data * net.prop.n_y, 0);
        int mt_idx = 0;

        // Derivative results for the input layers
        std::vector<float> mdy_batch_in, Sdy_batch_in, mdy_in, Sdy_in;
        if (net.prop.collect_derivative) {
            mdy_batch_in.resize(net.prop.batch_size * net.prop.n_x, 0);
            Sdy_batch_in.resize(net.prop.batch_size * net.prop.n_x, 0);
            mdy_in.resize(db.num_data * net.prop.n_x, 0);
            Sdy_in.resize(db.num_data * net.prop.n_x, 0);
        }

        // Prediction
        for (int i = 0; i < n_iter; i++) {
            // Load data
            get_batch_idx(data_idx, i * net.prop.batch_size,
                          net.prop.batch_size, batch_idx);
            get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);
            get_batch_data(db.y, batch_idx, net.prop.n_y, y_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            if (net.prop.collect_derivative) {
                compute_network_derivatives(net.prop, net.theta_gpu,
                                            net.state_gpu, derivative_layer);
            }

            // Get hidden states for output layers
            net.state_gpu.copy_device_to_host();
            output_hidden_states(net.state, net.prop, ma_batch_out,
                                 Sa_batch_out);

            if (net.prop.collect_derivative) {
                get_input_derv_states(net.state.derv_state.md_layer,
                                      net.state.derv_state.Sd_layer,
                                      mdy_batch_in, Sdy_batch_in);
                update_vector(mdy_in, mdy_batch_in, mt_idx, net.prop.n_x);
                update_vector(Sdy_in, Sdy_batch_in, mt_idx, net.prop.n_x);
            }

            // Update the final hidden state vector for last layer
            mt_idx = i * net.prop.batch_size * net.prop.n_y;
            update_vector(ma_out, ma_batch_out, mt_idx, net.prop.n_y);
            update_vector(Sa_out, Sa_batch_out, mt_idx, net.prop.n_y);
        }
        // Denormalize data
        std::vector<float> sy_norm(db.y.size(), 0);
        std::vector<float> my(sy_norm.size(), 0);
        std::vector<float> sy(sy_norm.size(), 0);
        std::vector<float> y_test(sy_norm.size(), 0);

        // Compute log-likelihood
        for (int k = 0; k < db.y.size(); k++) {
            sy_norm[k] = powf(Sa_out[k] + powf(net.prop.sigma_v, 2), 0.5);
        }
        denormalize_mean(ma_out, db.mu_y, db.sigma_y, net.prop.n_y, my);
        denormalize_mean(db.y, db.mu_y, db.sigma_y, net.prop.n_y, y_test);
        denormalize_std(sy_norm, db.mu_y, db.sigma_y, net.prop.n_y, sy);

        // Compute metrics
        auto mse = mean_squared_error(my, y_test);
        auto log_lik = avg_univar_log_lik(my, y_test, sy);

        // Display results
        std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "RMSE           : ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << powf(mse, 0.5) << "\n";
        std::cout << "Log likelihood: ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << log_lik;
        std::cout << std::endl;

        // Save predictions
        std::string suffix = "prediction";
        save_predictions(path.saved_inference_path, my, sy, suffix);
        if (net.prop.collect_derivative) {
            save_derivatives(path.saved_inference_path, mdy_in, Sdy_in, suffix);
        }
    }
}

///////////////////////////////////////////////////////////////////////
// TIME SERIES FORECASTING
///////////////////////////////////////////////////////////////////////
void time_series_forecasting(TagiNetwork &net, Dataloader &db, int n_epochs,
                             SavePath &path, bool train_mode, bool debug)
/* Time series forecasting

Args:
    Net: Tagi network
    db: database
    n_epochs: Number of epochs
    path: Directory stored the final results
    path: Directory stored the final results
    train_mode: Whether to train the network
    debug: Debugging mode allows saving inference data
*/
{
    // Seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);

    // Compute number of data
    int n_iter = db.num_data / net.prop.batch_size;
    int n_input_ts = net.prop.n_x * net.prop.input_seq_len;
    int n_output_ts = net.prop.n_y;

    // Initialize the data's variables
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> data_idx = create_range(db.num_data);
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);

    x_batch.resize(net.prop.batch_size * n_input_ts, 0);
    Sx_batch.resize(net.prop.batch_size * n_input_ts,
                    powf(net.prop.sigma_x, 2));
    y_batch.resize(net.prop.batch_size * n_output_ts, 0);
    V_batch.resize(net.prop.batch_size * n_output_ts,
                   powf(net.prop.sigma_v, 2));

    // *TODO: Is there any better way?
    std::vector<float> Sx_f_batch;
    if (net.prop.is_full_cov) {
        float var_x = powf(net.prop.sigma_x, 2);
        auto Sx_f = initialize_upper_triu(var_x, n_input_ts);
        Sx_f_batch = repmat_vector(Sx_f, net.prop.batch_size);
    }

    /* TRAINING */
    if (train_mode) {
        for (int e = 0; e < n_epochs; e++) {
            // Shufle data
            if (e > 0) {
                // Shufle data
                std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

                // Decay observation noise
                decay_obs_noise(net.prop.sigma_v, net.prop.decay_factor_sigma_v,
                                net.prop.sigma_v_min);
            }
            std::vector<float> V_batch(net.prop.batch_size * n_output_ts,
                                       powf(net.prop.sigma_v, 2));

            // Timer
            std::cout << "################\n";
            std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
            std::cout << "Training...\n";
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // Load data
                get_batch_idx(data_idx, i * net.prop.batch_size,
                              net.prop.batch_size, batch_idx);
                get_batch_data(db.x, batch_idx, n_input_ts, x_batch);
                get_batch_data(db.y, batch_idx, n_output_ts, y_batch);

                // Feed forward
                net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

                // Feed backward for hidden states
                net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

                // Feed backward for parameters
                net.param_feed_backward();

                // Save current cell & hidden states for next step.
                if (net.prop.batch_size == 1 && net.prop.input_seq_len == 1) {
                    save_prev_states(net.prop, net.state_gpu);
                }
            }

            // Report running time
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
            std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
            std::cout << " Time left     : ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60
                      << " mins\n";
        }
        // state_gpu.copy_device_to_host(state);
        net.theta_gpu.copy_device_to_host();

        // Retrieve homocesdastic noise distribution's parameter
        if (net.prop.noise_type.compare("homosce") == 0) {
            net.state_gpu.copy_device_to_host();
            get_homosce_noise_param(net.state.noise_state.ma_v2b_prior,
                                    net.state.noise_state.Sa_v2b_prior,
                                    net.prop.mu_v2b, net.prop.sigma_v2b);
        }

    } else {
        /* TESTING */
        std::cout << "Testing...\n";
        std::vector<float> ma_batch_out(net.prop.batch_size * n_output_ts, 0);
        std::vector<float> Sa_batch_out(net.prop.batch_size * n_output_ts, 0);
        std::vector<float> ma_out(db.num_data * n_output_ts, 0);
        std::vector<float> Sa_out(db.num_data * n_output_ts, 0);
        int mt_idx = 0;

        // Prediction
        for (int i = 0; i < n_iter; i++) {
            mt_idx = i * net.prop.batch_size * n_output_ts;

            // Load data
            get_batch_idx(data_idx, i * net.prop.batch_size,
                          net.prop.batch_size, batch_idx);
            get_batch_data(db.x, batch_idx, n_input_ts, x_batch);
            get_batch_data(db.y, batch_idx, n_output_ts, y_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Save current cell & hidden states for next step.
            if (net.prop.batch_size == 1 && net.prop.input_seq_len == 1) {
                save_prev_states(net.prop, net.state_gpu);
            }

            // Get hidden states for output layers
            net.get_network_outputs();

            // Update the final hidden state vector for last layer
            update_vector(ma_out, net.ma, mt_idx, n_output_ts);
            update_vector(Sa_out, net.Sa, mt_idx, n_output_ts);
        }
        // Retrive predictions (i.e., 1st column)
        int n_y = net.prop.n_y / net.prop.output_seq_len;
        std::vector<float> my_1(db.num_data, 0), Sy_1(db.num_data, 0),
            y_1(db.num_data, 0);
        get_1st_column_data(ma_out, net.prop.output_seq_len, n_y, my_1);
        get_1st_column_data(Sa_out, net.prop.output_seq_len, n_y, Sy_1);
        get_1st_column_data(db.y, net.prop.output_seq_len, n_y, y_1);

        // Unnormalize data
        std::vector<float> sy_norm(db.num_data, 0), my(db.num_data, 0),
            sy(db.num_data, 0), y_test(db.num_data, 0);

        // Compute log-likelihood
        for (int k = 0; k < db.num_data; k++) {
            sy_norm[k] = pow(Sy_1[k] + pow(net.prop.sigma_v, 2), 0.5);
        }
        denormalize_mean(my_1, db.mu_y, db.sigma_y, n_y, my);
        denormalize_mean(y_1, db.mu_y, db.sigma_y, n_y, y_test);
        denormalize_std(sy_norm, db.mu_y, db.sigma_y, n_y, sy);

        // Compute metrics
        auto mse = mean_squared_error(my, y_test);
        auto log_lik = avg_univar_log_lik(my, y_test, sy);

        // Display results
        std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "RMSE           : ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << pow(mse, 0.5) << "\n";
        std::cout << "Log likelihood: ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << log_lik;
        std::cout << std::endl;

        // Save predictions
        std::string suffix = "time_series_prediction";
        save_predictions(path.saved_inference_path, my, sy, suffix);
    }
}

///////////////////////////////////////////////////////////////////////
// TASK MAIN
///////////////////////////////////////////////////////////////////////
void task_command(UserInput &user_input, SavePath &path) {
    /* Assign different tasks and its parameters

    Args:
        user_input: User-specified inputs
        res_path: Directory path where results are stored under *.csv file
    */

    if (user_input.task_name == "classification") {
        // Network
        Network net_prop;
        bool train_mode = true;

        // Add extestion to file name
        std::string net_file_ext = user_input.net_name + ".txt";

        // Initialize network
        load_cfg(net_file_ext, net_prop);
        net_prop.device = user_input.device;
        if (net_prop.activations.back() == net_prop.act_names.hr_softmax) {
            net_prop.is_idx_ud = true;
            auto hrs = class_to_obs(user_input.num_classes);
            net_prop.nye = hrs.n_obs;
        }
        TagiNetwork net(net_prop);

        // Data
        auto imdb = get_images(user_input.data_name, user_input.x_train_dir,
                               user_input.y_train_dir, user_input.mu,
                               user_input.sigma, user_input.num_train_data,
                               user_input.num_classes, net.prop);

        auto test_imdb = get_images(user_input.data_name, user_input.x_test_dir,
                                    user_input.y_test_dir, user_input.mu,
                                    user_input.sigma, user_input.num_test_data,
                                    user_input.num_classes, net.prop);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, net.theta);

            net.theta_gpu.copy_host_to_device();
        }

        // Saved debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "saved_param/";
            save_net_param(user_input.model_name, user_input.net_name,
                           param_path, net.theta);
        }

        std::cout << "Training...\n" << std::endl;
        classification(net, imdb, test_imdb, user_input.num_epochs,
                       user_input.num_classes, path, train_mode,
                       user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, net.theta);

    } else if (user_input.task_name == "autoencoder") {
        // Encoder
        Network net_prop_e;
        std::string net_file_ext_e = user_input.encoder_net_name + ".txt";
        load_cfg(net_file_ext_e, net_prop_e);
        net_prop_e.device = user_input.device;
        TagiNetwork net_e(net_prop_e);
        net_e.prop.is_output_ud = false;

        // Decoder
        Network net_prop_d;
        std::string net_file_ext_d = user_input.decoder_net_name + ".txt";
        load_cfg(net_file_ext_d, net_prop_d);
        net_prop_d.device = user_input.device;
        TagiNetwork net_d(net_prop_d);
        net_d.prop.last_backward_layer = 0;

        // Load data
        auto imdb = get_images(user_input.data_name, user_input.x_train_dir,
                               user_input.y_train_dir, user_input.mu,
                               user_input.sigma, user_input.num_train_data,
                               user_input.num_classes, net_e.prop);

        auto test_imdb = get_images(user_input.data_name, user_input.x_test_dir,
                                    user_input.y_test_dir, user_input.mu,
                                    user_input.sigma, user_input.num_test_data,
                                    user_input.num_classes, net_e.prop);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.encoder_net_name,
                           path.saved_param_path, net_e.theta);
            load_net_param(user_input.model_name, user_input.decoder_net_name,
                           path.saved_param_path, net_d.theta);
            // Send to device
            net_e.theta_gpu.copy_host_to_device();
            net_d.theta_gpu.copy_host_to_device();
        }

        // Save data for debugging
        if (user_input.debug) {
            save_autoencoder_net_prop(net_e.theta, net_d.theta, net_e.idx,
                                      net_d.idx, path.debug_path);
        }

        // Train network
        bool train_mode = true;
        autoencoder(net_e, net_d, imdb, user_input.num_epochs,
                    user_input.num_classes, path, train_mode, user_input.debug);

        save_net_param(user_input.model_name, user_input.encoder_net_name,
                       path.saved_param_path, net_e.theta);
        save_net_param(user_input.model_name, user_input.decoder_net_name,
                       path.saved_param_path, net_d.theta);

        train_mode = false;
        autoencoder(net_e, net_d, test_imdb, user_input.num_epochs,
                    user_input.num_classes, path, train_mode, user_input.debug);

    } else if (user_input.task_name == "regression") {
        // Train network
        Network net_prop;

        // Add extestion to file name
        std::string net_file_ext = user_input.net_name + ".txt";

        // Initialize network
        load_cfg(net_file_ext, net_prop);
        net_prop.device = user_input.device;
        TagiNetwork net(net_prop);

        // Train data. TODO: refactor the dataloader
        std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
        auto train_db =
            get_dataloader(user_input.x_train_dir, user_input.y_train_dir, mu_x,
                           sigma_x, mu_y, sigma_y, user_input.num_train_data,
                           net.prop.n_x, net.prop.n_y, user_input.data_norm);

        // Test data
        auto test_db =
            get_dataloader(user_input.x_test_dir, user_input.y_test_dir,
                           train_db.mu_x, train_db.sigma_x, train_db.mu_y,
                           train_db.sigma_y, user_input.num_test_data,
                           net.prop.n_x, net.prop.n_y, user_input.data_norm);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, net.theta);

            net.theta_gpu.copy_host_to_device();
        }

        // Save network's parameter to debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "saved_param/";
            save_net_param(user_input.model_name, user_input.net_name,
                           param_path, net.theta);
        }

        // Training
        bool train_mode = true;
        regression(net, train_db, user_input.num_epochs, path, train_mode,
                   user_input.debug);

        train_mode = false;
        regression(net, test_db, user_input.num_epochs, path, train_mode,
                   user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, net.theta);
    } else if (user_input.task_name == "time_series") {
        // Train network
        Network net_prop;

        // Add extestion to file name
        std::string net_file_ext = user_input.net_name + ".txt";

        // Initialize network
        load_cfg(net_file_ext, net_prop);
        net_prop.device = user_input.device;
        TagiNetwork net(net_prop);

        // Train data
        std::string train_dataloader_name = "train";
        auto train_db = make_time_series_dataloader(user_input, net.prop,
                                                    train_dataloader_name);

        // Test data
        std::string test_dataloader_name = "test";
        auto test_db = make_time_series_dataloader(user_input, net.prop,
                                                   test_dataloader_name);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, net.theta);
            net.theta_gpu.copy_host_to_device();
        }

        // Save network's parameter to debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "saved_param/";
            save_net_param(user_input.model_name, user_input.net_name,
                           param_path, net.theta);
        }

        // Training
        bool train_mode = true;
        time_series_forecasting(net, train_db, user_input.num_epochs, path,
                                train_mode, user_input.debug);

        train_mode = false;
        time_series_forecasting(net, test_db, user_input.num_epochs, path,
                                train_mode, user_input.debug);
        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, net.theta);

    } else {
        throw std::invalid_argument("Task name does not exist - task.cu");
    }
}

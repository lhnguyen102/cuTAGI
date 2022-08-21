///////////////////////////////////////////////////////////////////////////////
// File:         task_cpu.cpp
// Description:  CPU version for task command providing different tasks
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 21, 2022
// Updated:      August 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/task_cpu.h"

///////////////////////////////////////////////////////////////////////
// CLASSIFICATION
///////////////////////////////////////////////////////////////////////
void classification_cpu(Network &net, IndexOut &idx, NetState &state,
                        Param &theta, ImageData &imdb, ImageData &test_imdb,
                        int n_epochs, int n_classes, SavePath &path,
                        bool train_mode, bool debug)
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
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);

    // Compute number of data points
    int n_iter = imdb.num_data / net.batch_size;
    int test_n_iter = test_imdb.num_data / net.batch_size;
    int n_w = theta.mw.size();
    int n_b = theta.mb.size();
    int n_w_sc = theta.mw_sc.size();
    int n_b_sc = theta.mb_sc.size();
    int ni_B = net.batch_size * net.nodes.front();
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(test_imdb.num_data);

    // Input and output layer
    auto hrs = class_to_obs(n_classes);
    std::vector<float> x_batch(net.batch_size * net.n_x, 0);
    std::vector<float> Sx_batch(net.batch_size * net.n_x, pow(net.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(net.batch_size * hrs.n_obs, 0);
    std::vector<float> V_batch(net.batch_size * hrs.n_obs, pow(net.sigma_v, 2));
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);
    std::vector<int> label_batch(net.batch_size, 0);

    // Input & output
    Input ip;
    Obs op;

    // Updated quantities for state & parameters
    DeltaState d_state;
    DeltaParam d_theta;
    d_state.set_values(net.n_state, state.msc.size(), state.mdsc.size(),
                       net.n_max_state);
    d_theta.set_values(n_w, n_b, n_w_sc, n_b_sc);

    // Error rate for training
    int mt_idx = 0;
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
    std::vector<float> ma_output(net.batch_size * net.n_y, 0);
    std::vector<float> Sa_output(net.batch_size * net.n_y, 0);

    for (int e = 0; e < n_epochs; e++) {
        /* TRAINNING */
        if (e > 0) {
            // Shufle data
            std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

            // Decay observation noise
            decay_obs_noise(net.sigma_v, net.decay_factor_sigma_v,
                            net.sigma_v_min);
        }

        std::vector<float> V_batch(net.batch_size * hrs.n_obs,
                                   pow(net.sigma_v, 2));

        // Timer
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
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
            get_batch_data(imdb.images, batch_idx, net.n_x, x_batch);
            get_batch_data(imdb.obs_label, batch_idx, hrs.n_obs, y_batch);
            get_batch_data(imdb.obs_idx, batch_idx, hrs.n_obs, idx_ud_batch);
            get_batch_data(imdb.labels, batch_idx, 1, label_batch);
            ip.set_values(x_batch, Sx_batch, Sx_f_batch);
            op.set_values(y_batch, V_batch, idx_ud_batch);

            // Initialize input
            initialize_states_cpu(ip.x_batch, ip.Sx_batch, ip.Sx_f_batch,
                                  net.n_x, net.batch_size, state);

            // Feed forward
            feed_forward_cpu(net, theta, idx, state);

            // Feed backward for hidden states
            state_backward_cpu(net, theta, state, idx, op, d_state);

            // Feed backward for parameters
            param_backward_cpu(net, theta, state, d_state, idx, d_theta);

            // Update model parameters
            global_param_update_cpu(d_theta, n_w, n_b, n_w_sc, n_b_sc, theta);

            // Compute error rate
            output_hidden_states(state, net, ma_output, Sa_output);
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(ma_output, Sa_output, label_batch, hrs, n_classes,
                          net.batch_size);
            mt_idx = i * net.batch_size;
            update_vector(error_rate, error_rate_batch, mt_idx, 1);

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
        // Report computational time
        std::cout << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto run_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
        std::cout << " Time left     : "
                  << (run_time * 1e-9) * (n_epochs - e - 1) / 60 << " mins\n";

        /* TESTING */
        std::cout << "Testing...\n";
        for (int i = 0; i < test_n_iter; i++) {
            // TODO: set = 0.9 when i > 0 or disable mean and variance in
            // feed forward
            net.ra_mt = 0.0f;

            // Load data
            get_batch_idx(test_data_idx, i, net.batch_size, batch_idx);
            get_batch_data(test_imdb.images, batch_idx, net.n_x, x_batch);
            get_batch_data(test_imdb.obs_label, batch_idx, hrs.n_obs, y_batch);
            get_batch_data(test_imdb.obs_idx, batch_idx, hrs.n_obs,
                           idx_ud_batch);
            get_batch_data(test_imdb.labels, batch_idx, 1, label_batch);
            ip.set_values(x_batch, Sx_batch, Sx_f_batch);
            op.set_values(y_batch, V_batch, idx_ud_batch);

            // Initialize input
            initialize_states_cpu(ip.x_batch, ip.Sx_batch, ip.Sx_f_batch,
                                  net.n_x, net.batch_size, state);

            // Feed forward
            feed_forward_cpu(net, theta, idx, state);

            // Compute error rate
            output_hidden_states(state, net, ma_output, Sa_output);
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(ma_output, Sa_output, label_batch, hrs, n_classes,
                          net.batch_size);
            mt_idx = i * net.batch_size;
            update_vector(test_error_rate, error_rate_batch, mt_idx, 1);
        }
        auto test_avg_error = compute_average_error_rate(
            test_error_rate, test_imdb.num_data, test_imdb.num_data);
        test_epoch_error_rate[e] = test_avg_error;
        std::cout << "\n";
        std::cout << "\tError rate: ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << test_avg_error << "\n";
        std::cout << state.noise_state.ma_v2b_prior[0] << "\n" << std::endl;
    }

    // Save error rate
    std::string suffix = "test";
    save_error_rate(path.saved_inference_path, test_epoch_error_rate, suffix);
}

///////////////////////////////////////////////////////////////////////
// REGRESSION
///////////////////////////////////////////////////////////////////////
void regression_cpu(Network &net, IndexOut &idx, NetState &state, Param &theta,
                    Dataloader &db, int n_epochs, SavePath &path,
                    bool train_mode, bool debug)
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
    // Seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int derivative_layer = 0;

    // Number of data points
    int n_iter = db.num_data / net.batch_size;
    int n_w = theta.mw.size();
    int n_b = theta.mb.size();
    int n_w_sc = theta.mw_sc.size();
    int n_b_sc = theta.mb_sc.size();
    int ni_B = net.batch_size * net.n_x;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(net.batch_size * net.n_x, 0);
    std::vector<float> Sx_batch(net.batch_size * net.n_x, pow(net.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(net.batch_size * net.n_y, 0);
    std::vector<float> V_batch(net.batch_size * net.n_y, pow(net.sigma_v, 2));
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);

    // *TODO: Is there any better way?
    if (net.is_full_cov) {
        float var_x = pow(net.sigma_x, 2);
        auto Sx_f = initialize_upper_triu(var_x, net.n_x);
        Sx_f_batch = repmat_vector(Sx_f, net.batch_size);
    }

    // Input & output
    Input ip;
    Obs op;

    // Updated quantities for state & parameters
    DeltaState d_state;
    DeltaParam d_theta;
    d_state.set_values(net.n_state, state.msc.size(), state.mdsc.size(),
                       net.n_max_state);
    d_theta.set_values(n_w, n_b, n_w_sc, n_b_sc);

    if (train_mode) {
        for (int e = 0; e < n_epochs; e++) {
            if (e > 0) {
                // Shuffle data
                std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

                // Decay observation noise
                decay_obs_noise(net.sigma_v, net.decay_factor_sigma_v,
                                net.sigma_v_min);
            }

            std::vector<float> V_batch(net.batch_size * net.n_y,
                                       pow(net.sigma_v, 2));

            // Timer
            std::cout << "################\n";
            std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
            std::cout << "Training...\n";
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // Load data
                get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                              batch_idx);
                get_batch_data(db.x, batch_idx, net.n_x, x_batch);
                get_batch_data(db.y, batch_idx, net.n_y, y_batch);
                ip.set_values(x_batch, Sx_batch, Sx_f_batch);
                op.set_values(y_batch, V_batch, idx_ud_batch);

                // Initialize input
                initialize_states_cpu(ip.x_batch, ip.Sx_batch, ip.Sx_f_batch,
                                      net.n_x, net.batch_size, state);

                // Feed forward
                feed_forward_cpu(net, theta, idx, state);

                // Feed backward for hidden states
                state_backward_cpu(net, theta, state, idx, op, d_state);

                // Feed backward for parameters
                param_backward_cpu(net, theta, state, d_state, idx, d_theta);

                // Update model parameters
                global_param_update_cpu(d_theta, n_w, n_b, n_w_sc, n_b_sc,
                                        theta);
            }

            // Report running time
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
            std::cout << " Time per epoch: ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << run_time * 1e-9 << " sec\n";
            std::cout << " Time left     : ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60
                      << " mins\n";
        }

        // Retrieve homocesdastic noise distribution's parameter
        if (net.noise_type.compare("homosce") == 0) {
            get_homosce_noise_param(state.noise_state.ma_v2b_prior,
                                    state.noise_state.Sa_v2b_prior, net.mu_v2b,
                                    net.sigma_v2b);
        }
    } else {
        std::cout << "Testing...\n";

        // Output results
        std::vector<float> ma_batch_out(net.batch_size * net.n_y, 0);
        std::vector<float> Sa_batch_out(net.batch_size * net.n_y, 0);
        std::vector<float> ma_out(db.num_data * net.n_y, 0);
        std::vector<float> Sa_out(db.num_data * net.n_y, 0);

        // Derivative results for the input layers
        std::vector<float> mdy_batch_in, Sdy_batch_in, mdy_in, Sdy_in;
        if (net.collect_derivative) {
            mdy_batch_in.resize(net.batch_size * net.n_x, 0);
            Sdy_batch_in.resize(net.batch_size * net.n_x, 0);
            mdy_in.resize(db.num_data * net.n_x, 0);
            Sdy_in.resize(db.num_data * net.n_x, 0);
        }

        int mt_idx = 0;

        // Prediction
        for (int i = 0; i < n_iter; i++) {
            mt_idx = i * net.batch_size * net.n_y;

            // Load data
            get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                          batch_idx);
            get_batch_data(db.x, batch_idx, net.n_x, x_batch);
            get_batch_data(db.y, batch_idx, net.n_y, y_batch);
            ip.set_values(x_batch, Sx_batch, Sx_f_batch);
            op.set_values(y_batch, V_batch, idx_ud_batch);

            // Initialize input
            initialize_states_cpu(ip.x_batch, ip.Sx_batch, ip.Sx_f_batch,
                                  net.n_x, net.batch_size, state);

            // Feed forward
            feed_forward_cpu(net, theta, idx, state);

            // Derivatives
            if (net.collect_derivative) {
                compute_network_derivatives_cpu(net, theta, state,
                                                derivative_layer);
                get_input_derv_states(state.derv_state.md_layer,
                                      state.derv_state.Sd_layer, mdy_batch_in,
                                      Sdy_batch_in);
                update_vector(mdy_in, mdy_batch_in, mt_idx, net.n_x);
                update_vector(Sdy_in, Sdy_batch_in, mt_idx, net.n_x);
            }

            // Get hidden states for output layers
            output_hidden_states(state, net, ma_batch_out, Sa_batch_out);

            // Update the final hidden state vector for last layer
            update_vector(ma_out, ma_batch_out, mt_idx, net.n_y);
            update_vector(Sa_out, Sa_batch_out, mt_idx, net.n_y);
        }
        // Denormalize data
        std::vector<float> sy_norm(db.y.size(), 0);
        std::vector<float> my(sy_norm.size(), 0);
        std::vector<float> sy(sy_norm.size(), 0);
        std::vector<float> y_test(sy_norm.size(), 0);

        // Compute log-likelihood
        for (int k = 0; k < db.y.size(); k++) {
            sy_norm[k] = pow(Sa_out[k] + pow(net.sigma_v, 2), 0.5);
        }
        denormalize_mean(ma_out, db.mu_y, db.sigma_y, net.n_y, my);
        denormalize_mean(db.y, db.mu_y, db.sigma_y, net.n_y, y_test);
        denormalize_std(sy_norm, db.mu_y, db.sigma_y, net.n_y, sy);

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
        std::string suffix = "prediction";
        save_predictions(path.saved_inference_path, my, sy, suffix);
        if (net.collect_derivative) {
            save_derivatives(path.saved_inference_path, mdy_in, Sdy_in, suffix);
        }
    }
}

void time_series_forecasting(Network &net, IndexOut &idx, NetState &state,
                             Param &theta, Dataloader &db, int n_epochs,
                             SavePath &path, bool train_mode)
/*Time series forecasting*/
{
    // Seed
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);

    // Number of data points
    int n_iter = db.num_data / net.batch_size;
    int n_w = theta.mw.size();
    int n_b = theta.mb.size();
    int n_w_sc = theta.mw_sc.size();
    int n_b_sc = theta.mb_sc.size();
    int n_input_ts = net.n_x * net.input_seq_len;
    int n_output_ts = net.n_y * net.output_seq_len;
    int n_input_seq_batch = net.batch_size * net.input_seq_len;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(n_input_ts * net.batch_size, 0);
    std::vector<float> Sx_batch(n_input_ts * net.batch_size,
                                pow(net.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(n_output_ts * net.batch_size, 0);
    std::vector<float> V_batch(n_output_ts * net.batch_size,
                               pow(net.sigma_v, 2));
    std::vector<int> batch_idx(net.batch_size);
    std::vector<int> idx_ud_batch(net.nye * net.batch_size, 0);

    // Input & output
    Input ip;
    Obs op;

    // Updated quantities for state & parameters
    DeltaState d_state;
    DeltaParam d_theta;
    d_state.set_values(net.n_state, state.msc.size(), state.mdsc.size(),
                       net.n_max_state);
    d_theta.set_values(n_w, n_b, n_w_sc, n_b_sc);

    if (train_mode) {
        for (int e = 0; e < n_epochs; e++) {
            if (e > 0) {
                // Shuffle data
                std::shuffle(data_idx.begin(), data_idx.end(), seed_e);

                // Decay observation noise
                decay_obs_noise(net.sigma_v, net.decay_factor_sigma_v,
                                net.sigma_v_min);
            }

            std::vector<float> V_batch(net.batch_size * n_output_ts,
                                       pow(net.sigma_v, 2));

            // Timer
            std::cout << "################\n";
            std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
            std::cout << "Training...\n";
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // Load data
                get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                              batch_idx);
                get_batch_data(db.x, batch_idx, n_input_ts, x_batch);
                get_batch_data(db.y, batch_idx, n_output_ts, y_batch);
                ip.set_values(x_batch, Sx_batch, Sx_f_batch);
                op.set_values(y_batch, V_batch, idx_ud_batch);

                // Initialize input
                initialize_states_cpu(ip.x_batch, ip.Sx_batch, ip.Sx_f_batch,
                                      net.n_x, n_input_seq_batch, state);

                // Feed forward
                feed_forward_cpu(net, theta, idx, state);

                // Feed backward for hidden states
                state_backward_cpu(net, theta, state, idx, op, d_state);

                // Feed backward for parameters
                param_backward_cpu(net, theta, state, d_state, idx, d_theta);

                // Update model parameters
                global_param_update_cpu(d_theta, n_w, n_b, n_w_sc, n_b_sc,
                                        theta);
            }

            // Report running time
            std::cout << std::endl;
            auto end = std::chrono::steady_clock::now();
            auto run_time =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                     start)
                    .count();
            std::cout << " Time per epoch: ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << run_time * 1e-9 << " sec\n";
            std::cout << " Time left     : ";
            std::cout << std::fixed;
            std::cout << std::setprecision(3);
            std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60
                      << " mins\n";
        }

        // Retrieve homocesdastic noise distribution's parameters
        if (net.noise_type.compare("homosce") == 0) {
            get_homosce_noise_param(state.noise_state.ma_v2b_prior,
                                    state.noise_state.Sa_v2b_prior, net.mu_v2b,
                                    net.sigma_v2b);
        }
    } else {
        std::cout << "Testing...\n";

        // Output results
        std::vector<float> ma_batch_out(n_output_ts * net.batch_size, 0);
        std::vector<float> Sa_batch_out(n_output_ts * net.batch_size, 0);
        std::vector<float> ma_out(db.num_data * n_output_ts, 0);
        std::vector<float> Sa_out(db.num_data * n_output_ts, 0);

        int mt_idx = 0;

        // Prediction
        for (int i = 0; i < n_iter; i++) {
            mt_idx = i * n_output_ts * net.batch_size;

            // Load data
            get_batch_idx(data_idx, i * net.batch_size, net.batch_size,
                          batch_idx);
            get_batch_data(db.x, batch_idx, n_input_ts, x_batch);
            get_batch_data(db.y, batch_idx, n_output_ts, y_batch);
            ip.set_values(x_batch, Sx_batch, Sx_f_batch);
            op.set_values(y_batch, V_batch, idx_ud_batch);

            // Initialize input
            initialize_states_cpu(ip.x_batch, ip.Sx_batch, ip.Sx_f_batch,
                                  net.n_x, n_input_seq_batch, state);

            // Feed forward
            feed_forward_cpu(net, theta, idx, state);

            // Get hidden states for output layers
            output_hidden_states(state, net, ma_batch_out, Sa_batch_out);

            // Update the final hidden state vector for last layer
            update_vector(ma_out, ma_batch_out, mt_idx, n_output_ts);
            update_vector(Sa_out, Sa_batch_out, mt_idx, n_output_ts);
        }
        // Retrive predictions (i.e., 1st column)
        std::vector<float> my_1(db.num_data, 0), Sy_1(db.num_data, 0),
            y_1(db.num_data, 0);
        get_1st_column_data(ma_out, net.output_seq_len, net.n_y, my_1);
        get_1st_column_data(Sa_out, net.output_seq_len, net.n_y, Sy_1);
        get_1st_column_data(db.y, net.output_seq_len, net.n_y, y_1);

        // Unnormalize data
        std::vector<float> sy_norm(db.y.size(), 0), my(sy_norm.size(), 0),
            sy(sy_norm.size(), 0), y_test(sy_norm.size(), 0);

        // Compute log-likelihood
        for (int k = 0; k < db.y.size(); k++) {
            sy_norm[k] = pow(Sy_1[k] + pow(net.sigma_v, 2), 0.5);
        }
        denormalize_mean(my_1, db.mu_y, db.sigma_y, net.n_y, my);
        denormalize_mean(y_1, db.mu_y, db.sigma_y, net.n_y, y_test);
        denormalize_std(sy_norm, db.mu_y, db.sigma_y, net.n_y, sy);

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

void task_command_cpu(UserInput &user_input, SavePath &path)
/* Assign different tasks and its parameters

    Args:
        user_input: User-specified inputs
        res_path: Directory path where results are stored under *.csv file
*/
{
    if (user_input.task_name == "classification") {
        // Network
        bool train_mode = true;
        IndexOut idx;
        Network net;
        Param theta;
        NetState state;
        net_init(user_input.net_name, user_input.device, net, theta, state,
                 idx);
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
        classification_cpu(net, idx, state, theta, imdb, test_imdb,
                           user_input.num_epochs, user_input.num_classes, path,
                           train_mode, user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, theta);

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

        net_init(user_input.net_name, user_input.device, net, theta, state,
                 idx);
        reset_net_batchsize(user_input.net_name, user_input.device, test_net,
                            test_state, test_idx, test_batch_size);

        // Train data. TODO: refactor the dataloader
        std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
        auto train_db =
            get_dataloader(user_input.x_train_dir, user_input.y_train_dir, mu_x,
                           sigma_x, mu_y, sigma_y, user_input.num_train_data,
                           net.n_x, net.n_y, user_input.data_norm);
        // Test data
        auto test_db = get_dataloader(
            user_input.x_test_dir, user_input.y_test_dir, train_db.mu_x,
            train_db.sigma_x, train_db.mu_y, train_db.sigma_y,
            user_input.num_test_data, net.n_x, net.n_y, user_input.data_norm);

        // Load param
        if (user_input.load_param) {
            load_net_param(user_input.model_name, user_input.net_name,
                           path.saved_param_path, theta);
        }

        // Save network's parameter to debug data
        if (user_input.debug) {
            std::string param_path = path.debug_path + "saved_param/";
            save_net_param(user_input.model_name, user_input.net_name,
                           param_path, theta);
        }

        // Training
        bool train_mode = true;
        regression_cpu(net, idx, state, theta, train_db, user_input.num_epochs,
                       path, train_mode, user_input.debug);

        // Testing
        if (net.noise_type.compare("homosce") == 0) {
            test_net.mu_v2b = net.mu_v2b;
            test_net.sigma_v2b = net.sigma_v2b;
        }
        test_net.sigma_v = net.sigma_v;
        train_mode = false;
        regression_cpu(test_net, test_idx, test_state, theta, test_db,
                       user_input.num_epochs, path, train_mode,
                       user_input.debug);

        // Save net's parameters
        save_net_param(user_input.model_name, user_input.net_name,
                       path.saved_param_path, theta);
    } else {
        throw std::invalid_argument("Task name does not exist - task_cpu.cpp");
    }
}
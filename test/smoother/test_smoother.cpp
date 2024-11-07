///////////////////////////////////////////////////////////////////////////////
// File:         test_smoother.cpp
// Description:  Test smoother for a toy problem using
// Smoothing in LSTM networks.
// Authors : Van-Dai Vuong, Luong - Ha Nguyen& James - A.Goulet
// Created : August 21, 2024
// Updated:  August 21, 2024
// Contact:  van-dai.vuong@polymtl.ca, luongha.nguyen@gmail.com &
// james.goulet@polymtl.ca
// License : This code is released under the MIT
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include "test_smoother.h"

#include <chrono>
#include <vector>

#include "../../include/common.h"
#include "../../include/dataloader.h"
// #include "../../include/debugger.h"
#include "../../include/sequential.h"
#include "../../include/slinear_layer.h"
#include "../../include/slstm_layer.h"
// #include "../../include/state_feed_backward_cpu.h"
// #include "../../include/struct_var.h"
// #ifdef USE_CUDA
// #include "../../include/tagi_network.cuh"
// #endif

Dataloader get_time_series_dataloader_smoother(
    std::vector<std::string> &data_file, int num_data, int num_features,
    std::vector<int> &output_col, bool data_norm, int input_seq_len,
    int output_seq_len, int seq_stride, std::vector<float> &mu_x,
    std::vector<float> &sigma_x)
/* Get dataloader for input and output data.

Args:

Returns:
    dataset: Dataloader
 */
{
    Dataloader db;
    int num_outputs = output_col.size();
    std::vector<float> x(num_features * num_data, 0), cat_x;

    // Load input data from csv file that contains the input & output data.
    // NOTE: csv file need to have a header for each columns
    for (int i = 0; i < data_file.size(); i++) {
        read_csv(data_file[i], x, num_features, true);
        cat_x.insert(cat_x.end(), x.begin(), x.end());
    };

    // Compute sample mean and std for dataset
    if (mu_x.size() == 0) {
        mu_x.resize(num_features, 0);
        sigma_x.resize(num_features, 1);
        compute_mean_std(cat_x, mu_x, sigma_x, num_features);
    }
    std::vector<float> mu_y(num_outputs, 0);
    std::vector<float> sigma_y(num_outputs, 1);
    if (data_norm) {
        normalize_data(cat_x, mu_x, sigma_x, num_features);
        for (int i = 0; i < num_outputs; i++) {
            mu_y[i] = mu_x[output_col[i]];
            sigma_y[i] = sigma_x[output_col[i]];
        }
    }

    // Create rolling windows
    int num_samples =
        (cat_x.size() / num_features - input_seq_len - output_seq_len) /
            seq_stride +
        1;
    std::vector<float> input_data(input_seq_len * num_features * num_samples);
    std::vector<float> output_data(output_seq_len * num_outputs * num_samples);
    create_rolling_windows(cat_x, output_col, input_seq_len, output_seq_len,
                           num_features, seq_stride, input_data, output_data);

    // Set data to output variable
    db.x = input_data;
    db.mu_x = mu_x;
    db.sigma_x = sigma_x;
    db.nx = num_features * input_seq_len;

    db.y = output_data;
    db.mu_y = mu_y;
    db.sigma_y = sigma_y;
    db.ny = num_outputs * output_seq_len;
    db.num_data = num_samples;

    return db;
}

void replace_with_prediction(std::vector<float> &mu_sequence,
                             std::vector<float> &x) {
    // Replace nan in input x by the lstm_prediction
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] == 0) {
            x[i] = mu_sequence[i];
        }
    }
}

void add_prediction_to_sequence(size_t input_seq_len, float m_pred,
                                std::vector<float> &mu_sequence) {
    // Add new prediction to mu_sequence
    mu_sequence.push_back(m_pred);

    // Ensure mu_sequence has only the last `input_seq_len` elements
    if (mu_sequence.size() > input_seq_len) {
        mu_sequence.erase(mu_sequence.begin(),
                          mu_sequence.end() - input_seq_len);
    }
}

void smoother_v1()
/**/
{
    // Data
    int num_train_data = 240;
    int num_test_data = 48;
    std::vector<int> output_col{0};
    int num_features = 1;
    std::vector<std::string> x_train_path{
        "data/toy_time_series_smoother/x_train_sin_smoother_zero.csv"};
    std::vector<std::string> x_test_path{
        "data/toy_time_series_smoother/x_test_sin_smoother.csv"};

    int input_seq_len = 24;
    int output_seq_len = 1;
    int seq_stride = 1;
    std::vector<float> mu_x, sigma_x;

    auto train_db = get_time_series_dataloader_smoother(
        x_train_path, num_train_data, num_features, output_col, true,
        input_seq_len, output_seq_len, seq_stride, mu_x, sigma_x);

    auto test_db = get_time_series_dataloader_smoother(
        x_test_path, num_test_data, num_features, output_col, true,
        input_seq_len, output_seq_len, seq_stride, train_db.mu_x,
        train_db.sigma_x);

    // Set seed
    manual_seed(0);

    // Model
    Sequential model(SLSTM(num_features + input_seq_len - 1, 40, 1),
                     SLSTM(40, 40, 1), SLinear(40 * 1, 1));
    model.input_state_update = true;
    model.num_samples = train_db.num_data;

    // model.to_device("cuda");
    model.set_threads(1);

    OutputUpdater output_updater(model.device);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    int n_epochs = 50;
    int batch_size = 1;
    float sigma_obs = 1.0;

    int iters = train_db.num_data / batch_size;
    std::cout << "num_iter: " << iters << "\n";
    std::vector<float> x_batch(batch_size * train_db.nx, 0.0f);
    std::vector<float> var_obs(batch_size * train_db.ny, pow(sigma_obs, 2));
    std::vector<float> y_batch(batch_size * train_db.ny, 0.0f);
    std::vector<int> batch_idx(batch_size);
    std::vector<float> mu_a_output(batch_size * train_db.ny, 0);
    std::vector<float> var_a_output(batch_size * train_db.ny, 0);
    auto data_idx = create_range(train_db.num_data);
    float decay_factor = 0.95f;
    float min_sigma_obs = 0.3f;
    // for smoother
    std::vector<float> mu_zo_smooths(0.0f, input_seq_len),
        var_zo_smooths(0.0f, input_seq_len);
    std::vector<float> mu_sequence(input_seq_len, 1.0f);
    float mu_zo = 0;
    // Number of observations before training time to be inferred. These
    // obervations are 0 in training data.
    int infer_window_len = 48;

    for (int e = 0; e < n_epochs; e++) {
        if (e > 0) {
            // Decay observation noise
            decay_obs_noise(sigma_obs, decay_factor, min_sigma_obs);
            std::vector<float> var_obs(batch_size * train_db.ny,
                                       pow(sigma_obs, 1));
        }
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
        std::cout << "Training...\n";
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            // Load data
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(train_db.x, batch_idx, train_db.nx, x_batch);
            get_batch_data(train_db.y, batch_idx, train_db.ny, y_batch);

            //  replace nan in input x by the lstm_prediction
            if (i < input_seq_len + infer_window_len) {
                replace_with_prediction(mu_sequence, x_batch);
            }

            // Forward
            model.forward(x_batch);
            output_updater.update(*model.output_z_buffer, y_batch, var_obs,
                                  *model.input_delta_z_buffer);

            // Backward pass
            model.backward();
            model.step();

            // Output the prediction
            mu_zo = model.output_z_buffer->mu_a[0];
            // Add new prediction to mu_sequence
            add_prediction_to_sequence(input_seq_len, mu_zo, mu_sequence);
        }

        // Smoother
        std::tie(mu_zo_smooths, var_zo_smooths) = model.smoother();
        mu_sequence.assign(mu_zo_smooths.begin(),
                           mu_zo_smooths.begin() + input_seq_len);

        // Report running time
        std::cout << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto run_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        std::cout << " Time per epoch: ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << run_time * 1e-9 << " sec\n";
        std::cout << " Time left     : ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60 << " mins\n";
    }

    ////////////////////////////////////////////////////////////////////
    // Testing
    ////////////////////////////////////////////////////////////////////
    std::cout << "Testing...\n";
    // Output results
    std::vector<float> mu_a_output_test(test_db.num_data * test_db.ny, 0);
    std::vector<float> var_a_output_test(test_db.num_data * test_db.ny, 0);
    auto test_data_idx = create_range(test_db.num_data);

    int n_iter =
        static_cast<float>(test_db.num_data) / static_cast<float>(batch_size);
    // int n_iter = ceil(n_iter_round);
    int mt_idx = 0;

    for (int i = 0; i < n_iter; i++) {
        mt_idx = i * test_db.ny * batch_size;

        // Data
        get_batch_idx(test_data_idx, i * batch_size, batch_size, batch_idx);
        get_batch_data(test_db.x, batch_idx, test_db.nx, x_batch);
        get_batch_data(test_db.y, batch_idx, test_db.ny, y_batch);

        // Forward
        model.forward(x_batch);

        // Extract output
        if (model.device == "cuda") {
            model.output_to_host();
        }

        for (int j = 0; j < batch_size * test_db.ny; j++) {
            mu_a_output_test[j + mt_idx] = model.output_z_buffer->mu_a[j];
            var_a_output_test[j + mt_idx] = model.output_z_buffer->var_a[j];
        }
    }
    // Retrive predictions (i.e., 1st column)
    int n_y = test_db.ny / output_seq_len;
    std::vector<float> mu_y_1(test_db.num_data, 0);
    std::vector<float> var_y_1(test_db.num_data, 0);
    std::vector<float> y_1(test_db.num_data, 0);
    get_1st_column_data(mu_a_output_test, output_seq_len, n_y, mu_y_1);
    get_1st_column_data(var_a_output_test, output_seq_len, n_y, var_y_1);
    get_1st_column_data(test_db.y, output_seq_len, n_y, y_1);

    // Unnormalize data
    std::vector<float> std_y_norm(test_db.num_data, 0);
    std::vector<float> mu_y(test_db.num_data, 0);
    std::vector<float> std_y(test_db.num_data, 0);
    std::vector<float> y_test(test_db.num_data, 0);

    // Compute log-likelihood
    for (int k = 0; k < test_db.num_data; k++) {
        std_y_norm[k] = pow(var_y_1[k] + pow(sigma_obs, 2), 0.5);
    }

    denormalize_mean(mu_y_1, test_db.mu_y, test_db.sigma_y, n_y, mu_y);
    denormalize_mean(y_1, test_db.mu_y, test_db.sigma_y, n_y, y_test);
    denormalize_std(std_y_norm, test_db.mu_y, test_db.sigma_y, n_y, std_y);

    // // Compute metrics
    auto mse = mean_squared_error(mu_y, y_test);
    auto log_lik = avg_univar_log_lik(mu_y, y_test, std_y);

    // Display results
    std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "MSE           : ";
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << mse << "\n";
    std::cout << "Log likelihood: ";
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << log_lik;
    std::cout << std::endl;

    // // Save predictions
    // std::string suffix = "time_series_prediction_test";
    // std::string saved_inference_path = "./saved_results/";
    // save_predictions(saved_inference_path, mu_y, std_y, suffix);
}

int test_smoother()
/*
 */
{
    smoother_v1();
    return 0;
}
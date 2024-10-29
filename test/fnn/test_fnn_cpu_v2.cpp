///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 25, 2023
// Updated:      November 25, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_cpu_v2.h"

void fnn_v2()
/*
 */
{
    //////////////////////////////////////////////////////////////////////
    // Data preprocessing
    //////////////////////////////////////////////////////////////////////
    std::string x_train_dir, y_train_dir, x_test_dir, y_test_dir;
    std::vector<std::string> x_train_path, y_train_path, x_test_path,
        y_test_path;
    std::string data_path = "/test/data/1D";
    x_train_dir = data_path + "/x_train.csv";
    y_train_dir = data_path + "/y_train.csv";
    x_test_dir = data_path + "/x_test.csv";
    y_test_dir = data_path + "/y_test.csv";
    x_train_path.push_back(x_train_dir);
    y_train_path.push_back(y_train_dir);
    x_test_path.push_back(x_test_dir);
    y_test_path.push_back(y_test_dir);

    int n_x = 1;
    int n_y = 1;
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    auto train_db = get_dataloader(x_train_path, y_train_path, mu_x, sigma_x,
                                   mu_y, sigma_y, 20, n_x, n_y, true);
    auto test_db = get_dataloader(x_test_path, y_test_path, train_db.mu_x,
                                  train_db.sigma_x, train_db.mu_y,
                                  train_db.sigma_y, 100, n_x, n_y, true);

    Sequential model(Linear(1, 50), ReLU(), Linear(50, 1));
    // model.load("test_model/test.bin");
    model.set_threads(1);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int batch_size = 10;
    float sigma_obs = 0.06;
    int iters = test_db.num_data / batch_size;
    std::vector<float> var_obs(batch_size * n_y, pow(sigma_obs, 2));
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> y_batch(batch_size * n_y, 0.0f);
    std::vector<int> batch_idx(batch_size);
    auto data_idx = create_range(train_db.num_data);
    for (int e = 0; e < 1; e++) {
        // if (e > 0) {
        //     // Shuffle data
        //     std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
        // }
        for (int i = 0; i < 2; i++) {
            // Load data
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(test_db.x, batch_idx, n_x, x_batch);
            get_batch_data(test_db.y, batch_idx, n_y, y_batch);

            // Forward pass
            model.forward(x_batch);
            int check = 1;

            // // Output layer
            // update_output_delta_z(*model.output_z_buffer, y_batch, var_obs,
            //                       model.input_delta_z_buffer->delta_mu,
            //                       model.input_delta_z_buffer->delta_var);

            // // Backward pass
            // model.backward();
            // model.step();
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Testing
    //////////////////////////////////////////////////////////////////////
    int test_batch_size = 1;
    int test_iters = test_db.num_data / test_batch_size;
    std::vector<float> test_var_obs(test_batch_size * n_y, pow(sigma_obs, 2));
    std::vector<float> test_x_batch(test_batch_size * n_x, 0.0f);
    std::vector<float> test_y_batch(test_batch_size * n_y, 0.0f);
    std::vector<int> test_batch_idx(test_batch_size);
    auto test_data_idx = create_range(test_db.num_data);

    // Output results
    std::vector<float> mu_a_batch_out(test_batch_size * n_y, 0.0f);
    std::vector<float> var_a_batch_out(test_batch_size * n_y, 0.0f);
    std::vector<float> mu_a_out(test_db.num_data * n_y, 0);
    std::vector<float> var_a_out(test_db.num_data * n_y, 0);

    for (int i = 0; i < test_iters; i++) {
        int mt_idx = i * test_batch_size * n_y;

        // Load data
        get_batch_idx(test_data_idx, i * test_batch_size, test_batch_size,
                      test_batch_idx);
        get_batch_data(test_db.x, test_batch_idx, n_x, test_x_batch);

        // Forward pass
        model.forward(test_x_batch);

        // Collect the output data
        for (int j = 0; j < n_y * test_batch_size; j++) {
            mu_a_batch_out[j] = model.output_z_buffer->mu_a[j];
            var_a_batch_out[j] = model.output_z_buffer->var_a[j];
        }
        update_vector(mu_a_out, mu_a_batch_out, mt_idx, n_y);
        update_vector(var_a_out, var_a_batch_out, mt_idx, n_y);
    }
    // Denormalize data
    std::vector<float> sy_norm(test_db.y.size(), 0);
    std::vector<float> my(sy_norm.size(), 0);
    std::vector<float> sy(sy_norm.size(), 0);
    std::vector<float> y_test(sy_norm.size(), 0);

    // Compute log-likelihood
    for (int k = 0; k < test_db.y.size(); k++) {
        sy_norm[k] = pow(var_a_out[k] + pow(sigma_obs, 2), 0.5);
    }
    denormalize_mean(mu_a_out, test_db.mu_y, test_db.sigma_y, n_y, my);
    denormalize_mean(test_db.y, test_db.mu_y, test_db.sigma_y, n_y, y_test);
    denormalize_std(sy_norm, test_db.mu_y, test_db.sigma_y, n_y, sy);

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
    std::string suffix = "prediction_fc_v2";
    std::string save_path = get_current_dir() + "/saved_results/";
    save_predictions(save_path, my, sy, suffix);

    int check = 1;
}

int test_fnn_cpu_v2() {
    // debug_fnn();
    fnn_v2();
    return 0;
}
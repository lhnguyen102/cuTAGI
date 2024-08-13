///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_heteros_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 13, 2024
// Updated:      August 13, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_heteros_cpu_v2.h"

#include <chrono>
#include <vector>

#include "../../include/activation.h"
#include "../../include/common.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/sequential.h"

void fnn_heteros_v2()
/*
 */
{
    //////////////////////////////////////////////////////////////////////
    // Data preprocessing
    //////////////////////////////////////////////////////////////////////
    std::string x_train_dir, y_train_dir, x_test_dir, y_test_dir;
    std::vector<std::string> x_train_path, y_train_path, x_test_path,
        y_test_path;
    std::string data_path = "./data/UCI/Boston_housing";
    x_train_dir = data_path + "/x_train.csv";
    y_train_dir = data_path + "/y_train.csv";
    x_test_dir = data_path + "/x_test.csv";
    y_test_dir = data_path + "/y_test.csv";
    x_train_path.push_back(x_train_dir);
    y_train_path.push_back(y_train_dir);
    x_test_path.push_back(x_test_dir);
    y_test_path.push_back(y_test_dir);

    int n_x = 13;
    int n_y = 1;
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    auto train_db = get_dataloader(x_train_path, y_train_path, mu_x, sigma_x,
                                   mu_y, sigma_y, 455, n_x, n_y, true);
    auto test_db = get_dataloader(x_test_path, y_test_path, train_db.mu_x,
                                  train_db.sigma_x, train_db.mu_y,
                                  train_db.sigma_y, 51, n_x, n_y, true);

    Sequential model(Linear(13, 50), ReLU(), Linear(50, 2), AGVI());
    model.set_threads(8);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////

    int batch_size = 10;
    int iters = train_db.num_data / batch_size;
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> y_batch(batch_size * n_y, 0.0f);
    std::vector<int> batch_idx(batch_size);
    auto data_idx = create_range(train_db.num_data);

    NoiseOutputUpdater output_updater(model.device);

    for (int e = 0; e < 30; e++) {
        // if (e > 0) {
        //     // Shuffle data
        //     std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
        // }
        for (int i = 0; i < iters; i++) {
            // Load data
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(train_db.x, batch_idx, n_x, x_batch);
            get_batch_data(train_db.y, batch_idx, n_y, y_batch);

            // Forward pass
            model.forward(x_batch);

            output_updater.update(*model.output_z_buffer, y_batch,
                                  *model.input_delta_z_buffer);

            // Backward pass
            model.backward();
            model.step();
        }
    }

    //////////////////////////////////////////////////////////////////////
    // Testing
    //////////////////////////////////////////////////////////////////////
    int test_batch_size = 1;
    int test_iters = test_db.num_data / test_batch_size;
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
        for (int j = 0; j < n_y * 2 * test_batch_size; j += 2) {
            mu_a_batch_out[j / 2] = model.output_z_buffer->mu_a[j];
            var_a_batch_out[j / 2] = model.output_z_buffer->var_a[j] +
                                     model.output_z_buffer->mu_a[j + 1];
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
        sy_norm[k] = pow(var_a_out[k], 0.5);
    }

    denormalize_mean(mu_a_out, test_db.mu_y, test_db.sigma_y, n_y, my);
    denormalize_mean(test_db.y, test_db.mu_y, test_db.sigma_y, n_y, y_test);
    denormalize_std(sy_norm, test_db.mu_y, test_db.sigma_y, n_y, sy);

    // Compute metrics
    auto mse = mean_squared_error(my, y_test);
    auto log_lik = avg_univar_log_lik(my, y_test, sy);

    // Display results
    std::cout << "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    std::cout << "RMSE          : ";
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << pow(mse, 0.5) << "\n";
    std::cout << "Log likelihood: ";
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << log_lik;
    std::cout << std::endl;

    // Save predictions
    std::string suffix = "prediction_fc_heteros_v2";
    std::string save_path = get_current_dir() + "/saved_results/";
    save_predictions(save_path, my, sy, suffix);
}

int test_fnn_heteros_cpu_v2() {
    std::cout << "Test FNN Heteros CPU v2\n";
    // fnn_heteros_old();
    fnn_heteros_v2();
    return 0;
}
#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "../../include/activation.h"
#include "../../include/common.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/sequential.h"

extern bool g_gpu_enabled;

void heteros_test_runner(Sequential &model, float &mse, float &log_lik)
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

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    int batch_size = 2;
    int iters = train_db.num_data / batch_size;
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> y_batch(batch_size * n_y, 0.0f);
    std::vector<int> batch_idx(batch_size);
    auto data_idx = create_range(train_db.num_data);

    OutputUpdater output_updater(model.device);

    for (int e = 0; e < 5; e++) {
        for (int i = 0; i < iters; i++) {
            // Load data
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(train_db.x, batch_idx, n_x, x_batch);
            get_batch_data(train_db.y, batch_idx, n_y, y_batch);

            // Forward pass
            model.forward(x_batch);

            output_updater.update_heteros(*model.output_z_buffer, y_batch,
                                          *model.input_delta_z_buffer);

            // Backward pass
            model.backward();
            model.step();

            // Extract output
            if (model.device == "cuda") {
                model.output_to_host();
            }
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

        // Extract output
        if (model.device == "cuda") {
            model.output_to_host();
        }

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
    mse = mean_squared_error(my, y_test);
    log_lik = avg_univar_log_lik(my, y_test, sy);
}

class SineSignalHeterosTest : public ::testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SineSignalHeterosTest, HeterosNoiseTest_CPU) {
    Sequential model(Linear(13, 16), ReLU(), Linear(16, 2), EvenExp());
    model.set_threads(2);

    float avg_error;
    float log_lik;
    float mse_threshold = 8.0f;
    float log_lik_threshold = -3.0f;
    heteros_test_runner(model, avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}

#ifdef USE_CUDA
TEST_F(SineSignalHeterosTest, HeterosNoiseTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Linear(13, 16), ReLU(), Linear(16, 2), EvenExp());
    model.to_device("cuda");

    float avg_error;
    float log_lik;
    float mse_threshold = 8.0f;
    float log_lik_threshold = -3.0f;
    heteros_test_runner(model, avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}
#endif
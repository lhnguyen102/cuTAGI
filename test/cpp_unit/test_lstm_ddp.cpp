#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../include/common.h"
#include "../../include/custom_logger.h"
#include "../../include/data_struct.h"
#include "../../include/ddp.h"
#include "../../include/linear_layer.h"
#include "../../include/lstm_layer.h"
#include "../../include/sequential.h"
#include "test_utils.h"

#if defined(USE_NCCL) && defined(USE_CUDA) && defined(USE_MPI)
#define DISTRIBUTED_TEST_AVAILABLE 1
#include <mpi.h>

#ifdef USE_CUDA
#include "../../include/base_layer_cuda.cuh"
#endif
#endif

extern bool g_gpu_enabled;

/**
 * Distributed LSTM test runner
 */
void distributed_sin_signal_lstm_test_runner(DDPSequential &dist_model,
                                             int input_seq_len, float &mse,
                                             float &log_lik) {
    // Get the underlying model and configuration
    auto model = dist_model.get_model();
    auto config = dist_model.get_config();
    int rank = config.rank;
    int world_size = config.world_size;

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting distributed training");

    // Set seed
    manual_seed(42 + rank);

    // Data
    int num_train_data = 924;
    int num_test_data = 232;
    std::vector<int> output_col{0};
    int num_features = 1;
    std::vector<std::string> x_train_path{
        "data/toy_time_series/x_train_sin_data.csv"};
    std::vector<std::string> x_test_path{
        "data/toy_time_series/x_test_sin_data.csv"};

    int output_seq_len = 1;
    int seq_stride = 1;
    std::vector<float> mu_x, sigma_x;

    auto train_db = get_time_series_dataloader(
        x_train_path, num_train_data, num_features, output_col, true,
        input_seq_len, output_seq_len, seq_stride, mu_x, sigma_x);

    auto test_db = get_time_series_dataloader(
        x_test_path, num_test_data, num_features, output_col, true,
        input_seq_len, output_seq_len, seq_stride, train_db.mu_x,
        train_db.sigma_x);

    ////////////////////////////////////////////////////////////////////////////
    // Training
    ////////////////////////////////////////////////////////////////////////////
    std::string device =
        model->device + ":" + std::to_string(model->device_idx);
    OutputUpdater output_updater(device);
    unsigned seed = 42 + rank;
    std::default_random_engine seed_e(seed);
    int n_epochs = 2;
    int batch_size = 8;
    float sigma_obs = 0.02;

    // Calculate data partition for this process
    int data_per_process = train_db.num_data / world_size;
    int start_idx = rank * data_per_process;
    int end_idx = (rank == world_size - 1) ? train_db.num_data
                                           : (rank + 1) * data_per_process;
    int local_data_size = end_idx - start_idx;

    LOG(LogLevel::INFO,
        "Process " + std::to_string(rank) + " handling data range [" +
            std::to_string(start_idx) + ", " + std::to_string(end_idx) + ")");

    int iters = local_data_size / batch_size;
    std::vector<float> x_batch(batch_size * train_db.nx, 0.0f);
    std::vector<float> var_obs(batch_size * train_db.ny, pow(sigma_obs, 2));
    std::vector<float> y_batch(batch_size * train_db.ny, 0.0f);
    std::vector<int> batch_idx(batch_size);
    std::vector<float> mu_a_output(batch_size * train_db.ny, 0);
    std::vector<float> var_a_output(batch_size * train_db.ny, 0);

    // Create a range for this process's data
    auto full_data_idx = create_range(train_db.num_data);
    std::shuffle(full_data_idx.begin(), full_data_idx.end(), seed_e);

    // Extract this process's portion of the data
    std::vector<int> local_data_idx(full_data_idx.begin() + start_idx,
                                    full_data_idx.begin() + end_idx);

    float decay_factor = 0.95f;
    float min_sigma_obs = 0.3f;

    for (int e = 0; e < n_epochs; e++) {
        if (e > 0) {
            // Shuffle data
            std::shuffle(local_data_idx.begin(), local_data_idx.end(), seed_e);
            decay_obs_noise(sigma_obs, decay_factor, min_sigma_obs);
            std::vector<float> var_obs(batch_size * train_db.ny,
                                       pow(sigma_obs, 2));
        }
        for (int i = 0; i < iters; i++) {
            // Load data
            get_batch_idx(local_data_idx, i * batch_size, batch_size,
                          batch_idx);
            get_batch_data(train_db.x, batch_idx, train_db.nx, x_batch);
            get_batch_data(train_db.y, batch_idx, train_db.ny, y_batch);

            // Forward
            dist_model.forward(x_batch);
            output_updater.update(*model->output_z_buffer, y_batch, var_obs,
                                  *model->input_delta_z_buffer);

            // Backward pass
            dist_model.backward();
            dist_model.step();
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Testing (only on rank 0)
    ///////////////////////////////////////////////////////////////////////////
    if (rank == 0) {
        std::vector<float> mu_a_output_test(test_db.num_data * test_db.ny, 0);
        std::vector<float> var_a_output_test(test_db.num_data * test_db.ny, 0);
        auto test_data_idx = create_range(test_db.num_data);

        int n_iter = static_cast<float>(test_db.num_data) /
                     static_cast<float>(batch_size);
        int mt_idx = 0;

        for (int i = 0; i < n_iter; i++) {
            mt_idx = i * test_db.ny * batch_size;

            // Data
            get_batch_idx(test_data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(test_db.x, batch_idx, test_db.nx, x_batch);
            get_batch_data(test_db.y, batch_idx, test_db.ny, y_batch);

            // Forward
            dist_model.forward(x_batch);

            // Extract output
            if (model->device == "cuda") {
                model->output_to_host();
            }

            for (int j = 0; j < batch_size * test_db.ny; j++) {
                mu_a_output_test[j + mt_idx] = model->output_z_buffer->mu_a[j];
                var_a_output_test[j + mt_idx] =
                    model->output_z_buffer->var_a[j];
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

        // Compute metrics
        mse = mean_squared_error(mu_y, y_test);
        log_lik = avg_univar_log_lik(mu_y, y_test, std_y);

        LOG(LogLevel::INFO, "Final MSE: " + std::to_string(mse));
        LOG(LogLevel::INFO, "Final log likelihood: " + std::to_string(log_lik));
    }
}

class LSTMDDPTest : public DistributedTestFixture {
   protected:
    void SetUp() override {
        DistributedTestFixture::SetUp();

        // Check if CUDA is available
        if (!g_gpu_enabled) {
            GTEST_SKIP() << "CUDA is not available, skipping distributed tests";
        }

        // Check if we have at least 2 GPUs
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count < 2) {
            GTEST_SKIP() << "At least 2 GPUs are required for distributed "
                            "tests, but only "
                         << device_count << " found";
        }
    }
};

#ifdef DISTRIBUTED_TEST_AVAILABLE
/**
 * Test distributed training with LSTM model using NCCL backend
 */
TEST_F(LSTMDDPTest, LSTM_NCCL) {
    // Log process information
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) + " starting test");

    int input_seq_len = 4;
    auto model = std::make_shared<Sequential>(LSTM(1, 8, input_seq_len),
                                              LSTM(8, 8, input_seq_len),
                                              Linear(8 * input_seq_len, 1));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    DDPSequential dist_model(model, config);

    // Run distributed training
    float mse, log_lik;
    distributed_sin_signal_lstm_test_runner(dist_model, input_seq_len, mse,
                                            log_lik);

    // Only rank 0 should assert the results
    if (rank == 0) {
        float mse_threshold = 0.5f;
        float log_lik_threshold = -3.0f;
        EXPECT_LT(mse, mse_threshold) << "MSE is higher than threshold";
        EXPECT_GT(log_lik, log_lik_threshold)
            << "Log likelihood is lower than threshold";
        LOG(LogLevel::INFO, "Final MSE: " + std::to_string(mse));
        LOG(LogLevel::INFO, "Final log likelihood: " + std::to_string(log_lik));
    }
}
#else
TEST_F(LSTMDDPTest, LSTM_NCCL) {
    GTEST_SKIP() << "Distributed functionality is not available. Skipping "
                    "distributed tests.";
}
#endif
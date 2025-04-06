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

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/batchnorm_layer.h"
#include "../../include/common.h"
#include "../../include/conv2d_layer.h"
#include "../../include/cuda_utils.h"
#include "../../include/custom_logger.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/ddp.h"
#include "../../include/linear_layer.h"
#include "../../include/max_pooling_layer.h"
#include "../../include/pooling_layer.h"
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
 * Distributed MNIST test runner
 */
#ifdef DISTRIBUTED_TEST_AVAILABLE
void distributed_mnist_test_runner(DDPSequential &dist_model,
                                   float &avg_error_output,
                                   float &test_error_output) {
    // Get the underlying model and configuration
    auto config = dist_model.get_config();
    int rank = config.rank;
    int world_size = config.world_size;

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting distributed training");

    // Set seed
    int seed = 42 + rank;
    manual_seed(seed);
    auto seed_engine = get_random_engine();

    //////////////////////////////////////////////////////////////////////
    // Data preprocessing
    //////////////////////////////////////////////////////////////////////
    std::vector<std::string> x_train_paths = {
        "./data/mnist/train-images-idx3-ubyte"};
    std::vector<std::string> y_train_paths = {
        "./data/mnist/train-labels-idx1-ubyte"};
    std::vector<std::string> x_test_paths = {
        "./data/mnist/t10k-images-idx3-ubyte"};
    std::vector<std::string> y_test_paths = {
        "./data/mnist/t10k-labels-idx1-ubyte"};

    std::string data_name = "mnist";
    std::vector<float> mu = {0.1309};
    std::vector<float> sigma = {1.0f};
    int num_train_data = 60000;
    int num_test_data = 10000;
    int num_classes = 10;
    int width = 28;
    int height = 28;
    int channel = 1;
    int n_x = width * height;
    int n_y = 11;

    auto train_db = get_images_v2(data_name, x_train_paths, y_train_paths, mu,
                                  sigma, num_train_data, num_classes, width,
                                  height, channel, true);
    auto test_db =
        get_images_v2(data_name, x_test_paths, y_test_paths, mu, sigma,
                      num_test_data, num_classes, width, height, channel, true);

    ////////////////////////////////////////////////////////////////////////////
    // Training
    ////////////////////////////////////////////////////////////////////////////
    std::string device_with_index = dist_model.get_device_with_index();
    OutputUpdater output_updater(device_with_index);

    int n_epochs = 2;
    int batch_size = 64;
    float sigma_obs = 0.1;

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
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> var_obs(batch_size * train_db.output_len,
                               pow(sigma_obs, 2));
    std::vector<float> y_batch(batch_size * train_db.output_len, 0.0f);
    std::vector<int> batch_idx(batch_size);
    std::vector<int> idx_ud_batch(train_db.output_len * batch_size, 0);
    std::vector<int> label_batch(batch_size, 0);
    std::vector<float> mu_a_output(batch_size * n_y, 0);
    std::vector<float> var_a_output(batch_size * n_y, 0);

    // Create a range for this process's data
    auto full_data_idx = create_range(train_db.num_data);
    std::shuffle(full_data_idx.begin(), full_data_idx.end(), seed_engine);

    // Extract this process's portion of the data
    std::vector<int> local_data_idx(full_data_idx.begin() + start_idx,
                                    full_data_idx.begin() + end_idx);

    int mt_idx = 0;
    std::vector<int> error_rate(local_data_size, 0);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;

    // Train for a fixed number of iterations by adjusting iterations based on
    // world size
    int num_iterations = train_db.num_data / world_size / batch_size;
    int total_train_error = 0;
    int total_train_samples = 0;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        std::shuffle(local_data_idx.begin(), local_data_idx.end(), seed_engine);
        for (int i = 0; i < num_iterations; i++) {
            int mt_idx = i * batch_size;
            // Get batch for this process
            get_batch_images_labels(train_db, local_data_idx, batch_size,
                                    mt_idx, x_batch, y_batch, idx_ud_batch,
                                    label_batch);

            // Forward pass
            dist_model.forward(x_batch);

            // Update output layer
            output_updater.update_using_indices(
                *dist_model.model->output_z_buffer, y_batch, var_obs,
                idx_ud_batch, *dist_model.model->input_delta_z_buffer);

            // Backward pass and parameter update
            dist_model.backward();
            dist_model.step();

            // Cuda device to host
            dist_model.output_to_host();

            for (int j = 0; j < batch_size * n_y; j++) {
                mu_a_output[j] = dist_model.model->output_z_buffer->mu_a[j];
                var_a_output[j] = dist_model.model->output_z_buffer->var_a[j];
                // check if nan or inf
                if (std::isnan(mu_a_output[j]) || std::isinf(mu_a_output[j]) ||
                    std::isnan(var_a_output[j]) ||
                    std::isinf(var_a_output[j])) {
                    LOG(LogLevel::ERROR, "NaN or inf detected in output");
                }
            }

            // Calculate error rate
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(mu_a_output, var_a_output, label_batch, num_classes,
                          batch_size);
            total_train_error += std::accumulate(error_rate_batch.begin(),
                                                 error_rate_batch.end(), 0);
            total_train_samples += batch_size;

            if (i % 100 == 0 && i > 0) {
                float train_error_rate_float =
                    static_cast<float>(total_train_error) / total_train_samples;

                LOG(LogLevel::INFO,
                    "Process " + std::to_string(rank) +
                        " completed iteration " + std::to_string(i) + " of " +
                        std::to_string(num_iterations) + " with error rate: " +
                        std::to_string(train_error_rate_float));
            }
        }
    }

    avg_error_output =
        static_cast<float>(total_train_error) / total_train_samples;

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) +
                            " finished with average error rate: " +
                            std::to_string(avg_error_output));

    ////////////////////////////////////////////////////////////////////////////
    // Testing (distributed across all devices)
    ////////////////////////////////////////////////////////////////////////////
    int test_data_per_process = test_db.num_data / world_size;
    int test_start_idx = rank * test_data_per_process;
    int test_end_idx = (rank == world_size - 1)
                           ? test_db.num_data
                           : (rank + 1) * test_data_per_process;
    int local_test_size = test_end_idx - test_start_idx;

    // Prepare test data for this process
    std::vector<float> x_test_batch(batch_size * n_x, 0.0f);
    std::vector<float> y_test_batch(batch_size * train_db.output_len, 0.0f);
    std::vector<int> idx_ud_test_batch(train_db.output_len * batch_size, 0);
    std::vector<int> label_test_batch(batch_size, 0);

    // Create test data indices for this process
    auto test_data_idx = create_range(test_db.num_data);
    std::vector<int> local_test_idx(test_data_idx.begin() + test_start_idx,
                                    test_data_idx.begin() + test_end_idx);

    int local_test_iterations = local_test_size / batch_size;
    int local_test_error = 0;
    int local_test_samples = 0;

    // Run testing on local partition
    for (int i = 0; i < local_test_iterations; i++) {
        int mt_test_idx = i * batch_size;
        // Get test batch
        get_batch_images_labels(test_db, local_test_idx, batch_size,
                                mt_test_idx, x_test_batch, y_test_batch,
                                idx_ud_test_batch, label_test_batch);

        // Forward pass
        dist_model.forward(x_test_batch);
        dist_model.output_to_host();

        // Extract output
        for (int j = 0; j < batch_size * n_y; j++) {
            mu_a_output[j] = dist_model.model->output_z_buffer->mu_a[j];
            var_a_output[j] = dist_model.model->output_z_buffer->var_a[j];
        }

        // Calculate error rate
        std::tie(error_rate_batch, prob_class_batch) =
            get_error(mu_a_output, var_a_output, label_test_batch, num_classes,
                      batch_size);

        // Accumulate errors
        local_test_error += std::accumulate(error_rate_batch.begin(),
                                            error_rate_batch.end(), 0);
        local_test_samples += batch_size;
    }
    // print local test error and samples and its error rate
    float local_test_error_rate =
        static_cast<float>(local_test_error) / local_test_samples;
    LOG(LogLevel::INFO,
        "Process " + std::to_string(rank) +
            " local test error: " + std::to_string(local_test_error) +
            " local test samples: " + std::to_string(local_test_samples) +
            " local test error rate: " + std::to_string(local_test_error_rate));

    // Gather results from all processes
    int total_test_error = 0;
    int total_test_samples = 0;

    // Use MPI_Reduce to sum up errors and samples
    MPI_Reduce(&local_test_error, &total_test_error, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&local_test_samples, &total_test_samples, 1, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // Calculate final test error rate on rank 0
    if (rank == 0) {
        test_error_output =
            static_cast<float>(total_test_error) / total_test_samples;
        LOG(LogLevel::INFO, "Final test error rate across all processes: " +
                                std::to_string(test_error_output));
    }
}
#endif
class MNISTDDPTest : public DistributedTestFixture {
   protected:
    void SetUp() override {
        DistributedTestFixture::SetUp();
        // Check if CUDA is available
        if (!g_gpu_enabled) {
            GTEST_SKIP() << "CUDA is not available, skipping distributed tests";
        }

        // Check if we have at least 2 GPUs
#ifdef USE_CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count < 2) {
            GTEST_SKIP() << "At least 2 GPUs are required for distributed "
                            "tests, but only "
                         << device_count << " found";
        }
#endif
        // Check if MNIST data exists, download if needed
        const std::string x_train_path = "./data/mnist/train-images-idx3-ubyte";
        const std::string y_train_path = "./data/mnist/train-labels-idx1-ubyte";
        const std::string x_test_path = "./data/mnist/t10k-images-idx3-ubyte";
        const std::string y_test_path = "./data/mnist/t10k-labels-idx1-ubyte";

        if (!file_exists(x_train_path) || !file_exists(y_train_path) ||
            !file_exists(x_test_path) || !file_exists(y_test_path)) {
            std::cout
                << "One or more MNIST data files are missing. Downloading..."
                << std::endl;
            if (!download_mnist_data()) {
                std::cerr << "Failed to download MNIST data files."
                          << std::endl;
            }
        }
    }

    bool file_exists(const std::string &path) {
        std::ifstream file(path);
        return file.good();
    }

    bool download_mnist_data() {
        if (system("mkdir -p ./data/mnist") != 0) {
            std::cerr << "Failed to create directory ./data/mnist" << std::endl;
            return false;
        }

        std::string base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/";
        if (!download_and_extract("train-images-idx3-ubyte", base_url) ||
            !download_and_extract("train-labels-idx1-ubyte", base_url) ||
            !download_and_extract("t10k-images-idx3-ubyte", base_url) ||
            !download_and_extract("t10k-labels-idx1-ubyte", base_url)) {
            return false;
        }

        std::cout << "MNIST data files downloaded successfully." << std::endl;
        return true;
    }

    bool download_and_extract(const std::string &filename,
                              const std::string &base_url) {
        // Download file with curl, handle failure with --fail option
        std::string command = "curl --fail -o ./data/mnist/" + filename +
                              ".gz " + base_url + filename + ".gz";
        if (system(command.c_str()) != 0) {
            std::cerr << "Failed to download " << filename << ".gz"
                      << std::endl;
            return false;
        }

        // Extract gzip file
        command = "gunzip -f ./data/mnist/" + filename + ".gz";
        if (system(command.c_str()) != 0) {
            std::cerr << "Failed to extract " << filename << ".gz" << std::endl;
            return false;
        }
        return true;
    }
};

/**
 * Test distributed training with a simple CNN model using NCCL backend
 */
#ifdef DISTRIBUTED_TEST_AVAILABLE
TEST_F(MNISTDDPTest, SimpleCNN_NCCL) {
    if (!is_mpi_initialized()) {
        GTEST_SKIP() << "MPI is not initialized. Run with mpirun.";
        return;
    }
    // check if NCCL is available
    if (!is_nccl_available()) {
        GTEST_SKIP() << "NCCL is not available. Skipping distributed tests.";
        return;
    }
    // Log process information
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) + " starting test");

    auto model = std::make_shared<Sequential>(
        Conv2d(1, 16, 4, true, 1, 1, 1, 28, 28), ReLU(), MaxPool2d(3, 2),
        Conv2d(16, 32, 5), ReLU(), MaxPool2d(3, 2), Linear(32 * 4 * 4, 128),
        ReLU(), Linear(128, 11));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    DDPSequential dist_model(model, config);

    // Run distributed training
    float avg_error, test_error;
    distributed_mnist_test_runner(dist_model, avg_error, test_error);

    // Only rank 0 should assert the results
    if (rank == 0) {
        float threshold = 0.5;
        EXPECT_LT(avg_error, threshold)
            << "Training error rate should be below " << threshold;
        EXPECT_LT(test_error, threshold)
            << "Test error rate should be below " << threshold;
        LOG(LogLevel::INFO,
            "Final training error rate: " + std::to_string(avg_error));
        LOG(LogLevel::INFO,
            "Final test error rate: " + std::to_string(test_error));
    }
}
#else
TEST_F(MNISTDDPTest, SimpleCNN_NCCL) {
    GTEST_SKIP() << "Distributed functionality is not available. Skipping "
                    "distributed tests.";
}
#endif

#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>

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
#include "../../include/conv2d_layer.h"
#include "../../include/custom_logger.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/ddp.h"
#include "../../include/layer_block.h"
#include "../../include/layernorm_layer.h"
#include "../../include/linear_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/resnet_block.h"
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
 * Distributed ResNet test runner
 */
#ifdef DISTRIBUTED_TEST_AVAILABLE
void distributed_resnet_cifar10_runner(DDPSequential& dist_model,
                                       float& avg_error_output) {
    // Get the underlying model and configuration
    auto config = dist_model.get_config();
    int rank = config.rank;
    int world_size = config.world_size;

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting distributed training");

    // Set seed
    manual_seed(42 + rank);

    // Data preprocessing
    std::vector<std::string> y_train_paths, y_test_paths;
    std::vector<std::string> x_train_paths = {
        "./data/cifar/cifar-10-batches-c/data_batch_1.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_2.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_3.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_4.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_5.bin"};
    std::vector<std::string> x_test_paths = {
        "./data/cifar/cifar-10-batches-c/test_batch.bin"};

    std::string data_name = "cifar";
    std::vector<float> mu = {0.4914, 0.4822, 0.4465};
    std::vector<float> sigma = {0.2023, 0.1994, 0.2010};
    int num_train_data = 50000;
    int num_test_data = 10000;
    int num_classes = 10;
    int width = 32;
    int height = 32;
    int channel = 3;
    int n_x = width * height * channel;
    int n_y = 11;

    // Calculate data partition for this process
    int data_per_process = num_train_data / world_size;
    int start_idx = rank * data_per_process;
    int end_idx = (rank == world_size - 1) ? num_train_data
                                           : (rank + 1) * data_per_process;
    int local_data_size = end_idx - start_idx;

    LOG(LogLevel::INFO,
        "Process " + std::to_string(rank) + " handling data range [" +
            std::to_string(start_idx) + ", " + std::to_string(end_idx) + ")");

    auto train_db = get_images_v2(data_name, x_train_paths, y_train_paths, mu,
                                  sigma, num_train_data, num_classes, width,
                                  height, channel, true);
    auto test_db =
        get_images_v2(data_name, x_test_paths, y_test_paths, mu, sigma,
                      num_test_data, num_classes, width, height, channel, true);

    // Updater
    std::string device_with_index = dist_model.get_device_with_index();
    OutputUpdater output_updater(device_with_index);

    // Training setup
    unsigned seed = 42 + rank;
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 128;
    float sigma_obs = 0.1;
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
    std::shuffle(full_data_idx.begin(), full_data_idx.end(), seed_e);

    // Extract this process's portion of the data
    std::vector<int> local_data_idx(full_data_idx.begin() + start_idx,
                                    full_data_idx.begin() + end_idx);

    // Error rate tracking
    int mt_idx = 0;
    std::vector<int> error_rate(local_data_size, 0);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;

    int total_train_error = 0;
    int total_train_samples = 0;

    // Training loop
    for (int e = 0; e < n_epochs; e++) {
        if (e > 0) {
            std::shuffle(local_data_idx.begin(), local_data_idx.end(), seed_e);
        }
        for (int i = 0; i < iters; i++) {
            mt_idx = i * batch_size;
            // Load data
            get_batch_images_labels(train_db, local_data_idx, batch_size,
                                    mt_idx, x_batch, y_batch, idx_ud_batch,
                                    label_batch);

            // Forward pass
            dist_model.forward(x_batch);
            dist_model.output_to_host();

            for (int j = 0; j < batch_size * n_y; j++) {
                mu_a_output[j] = dist_model.model->output_z_buffer->mu_a[j];
                var_a_output[j] = dist_model.model->output_z_buffer->var_a[j];
            }

            std::tie(error_rate_batch, prob_class_batch) =
                get_error(mu_a_output, var_a_output, label_batch, num_classes,
                          batch_size);

            total_train_error += std::accumulate(error_rate_batch.begin(),
                                                 error_rate_batch.end(), 0);
            total_train_samples += batch_size;

            // Output layer update
            output_updater.update_using_indices(
                *dist_model.model->output_z_buffer, y_batch, var_obs,
                idx_ud_batch, *dist_model.model->input_delta_z_buffer);

            // Backward pass
            dist_model.backward();
            dist_model.step();
            if (i % 100 == 0) {
                float train_error_rate_float =
                    static_cast<float>(total_train_error) / total_train_samples;
                LOG(LogLevel::INFO,
                    "Process " + std::to_string(rank) +
                        " completed iteration " + std::to_string(i) + " of " +
                        std::to_string(iters) + " with error rate: " +
                        std::to_string(train_error_rate_float));
            }
        }
    }

    // Testing (only on rank 0)
    int test_data_per_process = test_db.num_data / world_size;
    int test_start_idx = rank * test_data_per_process;
    int test_end_idx = (rank == world_size - 1)
                           ? test_db.num_data
                           : (rank + 1) * test_data_per_process;
    int local_test_size = test_end_idx - test_start_idx;

    // prepare test data
    std::vector<float> x_test_batch(batch_size * n_x, 0.0f);
    std::vector<float> y_test_batch(batch_size * train_db.output_len, 0.0f);
    std::vector<int> idx_ud_test_batch(train_db.output_len * batch_size, 0);
    std::vector<int> label_test_batch(batch_size, 0);

    // create test data indices
    auto test_data_idx = create_range(test_db.num_data);
    std::vector<int> local_test_idx(test_data_idx.begin() + test_start_idx,
                                    test_data_idx.begin() + test_end_idx);

    int local_test_iterations = local_test_size / batch_size;
    int local_test_error = 0;
    int local_test_samples = 0;

    for (int i = 0; i < local_test_iterations; i++) {
        int mt_test_idx = i * batch_size;
        // get test batch
        get_batch_images_labels(test_db, local_test_idx, batch_size,
                                mt_test_idx, x_test_batch, y_test_batch,
                                idx_ud_test_batch, label_test_batch);

        // forward pass
        dist_model.forward(x_test_batch);
        dist_model.output_to_host();

        // extract output
        for (int j = 0; j < batch_size * n_y; j++) {
            mu_a_output[j] = dist_model.model->output_z_buffer->mu_a[j];
            var_a_output[j] = dist_model.model->output_z_buffer->var_a[j];
        }

        // calculate error rate
        std::tie(error_rate_batch, prob_class_batch) =
            get_error(mu_a_output, var_a_output, label_test_batch, num_classes,
                      batch_size);

        // accumulate errors
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

    // Calculate final test error rate
    avg_error_output =
        static_cast<float>(total_test_error) / total_test_samples;
}

class ResNetDDPTest : public DistributedTestFixture {
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
    }
};

/**
 * Test distributed training with ResNet model using NCCL backend
 */
TEST_F(ResNetDDPTest, ResNet_NCCL) {
    // Log process information
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) + " starting test");

    // Create ResNet blocks
    auto block_1 = create_layer_block(64, 64);
    auto block_2 = create_layer_block(64, 64);
    auto block_3 = create_layer_block(64, 128, 2, 2);
    auto block_4 = create_layer_block(128, 128);
    auto block_5 = create_layer_block(128, 256, 2, 2);
    auto block_6 = create_layer_block(256, 256);
    auto block_7 = create_layer_block(256, 512, 2, 2);
    auto block_8 = create_layer_block(512, 512);

    ResNetBlock resnet_block_1(block_1);
    ResNetBlock resnet_block_2(block_2);
    ResNetBlock resnet_block_3(block_3, LayerBlock(Conv2d(64, 128, 2, false, 2),
                                                   ReLU(), BatchNorm2d(128)));
    ResNetBlock resnet_block_4(block_4);
    ResNetBlock resnet_block_5(
        block_5,
        LayerBlock(Conv2d(128, 256, 2, false, 2), ReLU(), BatchNorm2d(256)));
    ResNetBlock resnet_block_6(block_6);
    ResNetBlock resnet_block_7(
        block_7,
        LayerBlock(Conv2d(256, 512, 2, false, 2), ReLU(), BatchNorm2d(512)));
    ResNetBlock resnet_block_8(block_8);

    // Create the base model
    auto model = std::make_shared<Sequential>(
        // Input block
        Conv2d(3, 64, 3, false, 1, 1, 1, 32, 32), ReLU(), BatchNorm2d(64),
        // Residual blocks
        resnet_block_1, resnet_block_2, resnet_block_3, resnet_block_4,
        resnet_block_5, resnet_block_6, resnet_block_7, resnet_block_8,
        // Output block
        AvgPool2d(4), Linear(512, 11));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    DDPSequential dist_model(model, config);

    // Run distributed training
    float avg_error_output = 1.0f;
    distributed_resnet_cifar10_runner(dist_model, avg_error_output);

    // Only rank 0 should assert the results
    if (rank == 0) {
        float error_threshold = 0.8f;
        EXPECT_LT(avg_error_output, error_threshold)
            << "Average error rate is higher than threshold";
        LOG(LogLevel::INFO,
            "Final average error rate: " + std::to_string(avg_error_output));
    }
}
#else
TEST_F(ResNetDDPTest, ResNet_NCCL) {
    GTEST_SKIP() << "Distributed functionality is not available. Skipping "
                    "distributed tests.";
}
#endif

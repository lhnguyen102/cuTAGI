

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
#include "../../include/custom_logger.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/ddp.h"
#include "../../include/linear_layer.h"
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

class DDPOpsTest : public DistributedTestFixture {
   protected:
    void SetUp() override {
        DistributedTestFixture::SetUp();
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
 * Test parameter synchronization in DDPSequential with summing mode
 * This test verifies that the sync_parameters() method correctly
 synchronizes and sums model parameters across processes
 */
#ifdef DISTRIBUTED_TEST_AVAILABLE
TEST_F(DDPOpsTest, ParameterSynchronization_NCCL_Sum) {
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting parameter sync test (sum mode)");

    // Create a simple model
    auto model = std::make_shared<Sequential>(Linear(10, 5, true));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    // Set average to false for sum mode
    DDPSequential dist_model(model, config, false);

    // Get the underlying model
    auto sequential_model = dist_model.get_model();

    // Set different delta parameters on each process
    auto &linear_layer = sequential_model->layers[0];

    // Fill delta parameters with rank-specific values
    float rank_value = static_cast<float>(rank + 1);
    std::fill(linear_layer->delta_mu_w.begin(), linear_layer->delta_mu_w.end(),
              rank_value);
    std::fill(linear_layer->delta_var_w.begin(),
              linear_layer->delta_var_w.end(), rank_value * 2);

    if (linear_layer->bias) {
        std::fill(linear_layer->delta_mu_b.begin(),
                  linear_layer->delta_mu_b.end(), rank_value * 3);
        std::fill(linear_layer->delta_var_b.begin(),
                  linear_layer->delta_var_b.end(), rank_value * 4);
    }

    // Make sure all parameters are copied to the device for CUDA layers
#ifdef USE_CUDA
    auto cuda_layer = dynamic_cast<BaseLayerCuda *>(linear_layer.get());
    if (cuda_layer) {
        cuda_layer->delta_params_to_device();
    }
#endif

    // Remember the values before sync for comparison
    std::vector<float> original_delta_mu_w = linear_layer->delta_mu_w;
    std::vector<float> original_delta_var_w = linear_layer->delta_var_w;
    std::vector<float> original_delta_mu_b;
    std::vector<float> original_delta_var_b;
    if (linear_layer->bias) {
        original_delta_mu_b = linear_layer->delta_mu_b;
        original_delta_var_b = linear_layer->delta_var_b;
    }

    // Synchronize parameters
    dist_model.sync_parameters();
    dist_model.barrier();

    // Copy any device parameters back to host for checking
#ifdef USE_CUDA
    if (cuda_layer) {
        cuda_layer->delta_params_to_host();
    }
#endif

    // Expected values after synchronization - sum of all ranks' values
    // Since we're in sum mode, we expect the sum of all ranks' values
    float sum_rank_values = 0.0f;
    for (int i = 0; i < world_size; i++) {
        sum_rank_values += (i + 1);  // Sum of rank+1 for all ranks
    }

    float expected_mu_w = sum_rank_values;
    float expected_var_w = sum_rank_values * 2;
    float expected_mu_b = sum_rank_values * 3;
    float expected_var_b = sum_rank_values * 4;

    // Check if parameters were properly synchronized
    bool sync_successful = true;

    // Check weights
    for (size_t i = 0; i < linear_layer->delta_mu_w.size(); i++) {
        if (std::abs(linear_layer->delta_mu_w[i] - expected_mu_w) > 1e-5) {
            sync_successful = false;
            LOG(LogLevel::ERROR,
                "Process " + std::to_string(rank) + " delta_mu_w[" +
                    std::to_string(i) +
                    "] = " + std::to_string(linear_layer->delta_mu_w[i]) +
                    ", expected " + std::to_string(expected_mu_w) +
                    " (original was " + std::to_string(original_delta_mu_w[i]) +
                    ")");
            break;
        }
    }

    for (size_t i = 0; i < linear_layer->delta_var_w.size(); i++) {
        if (std::abs(linear_layer->delta_var_w[i] - expected_var_w) > 1e-5) {
            sync_successful = false;
            LOG(LogLevel::ERROR,
                "Process " + std::to_string(rank) + " delta_var_w[" +
                    std::to_string(i) +
                    "] = " + std::to_string(linear_layer->delta_var_w[i]) +
                    ", expected " + std::to_string(expected_var_w) +
                    " (original was " +
                    std::to_string(original_delta_var_w[i]) + ")");
            break;
        }
    }

    // Check biases if present
    if (linear_layer->bias) {
        for (size_t i = 0; i < linear_layer->delta_mu_b.size(); i++) {
            if (std::abs(linear_layer->delta_mu_b[i] - expected_mu_b) > 1e-5) {
                sync_successful = false;
                LOG(LogLevel::ERROR,
                    "Process " + std::to_string(rank) + " delta_mu_b[" +
                        std::to_string(i) +
                        "] = " + std::to_string(linear_layer->delta_mu_b[i]) +
                        ", expected " + std::to_string(expected_mu_b) +
                        " (original was " +
                        std::to_string(original_delta_mu_b[i]) + ")");
                break;
            }
        }

        for (size_t i = 0; i < linear_layer->delta_var_b.size(); i++) {
            if (std::abs(linear_layer->delta_var_b[i] - expected_var_b) >
                1e-5) {
                sync_successful = false;
                LOG(LogLevel::ERROR,
                    "Process " + std::to_string(rank) + " delta_var_b[" +
                        std::to_string(i) +
                        "] = " + std::to_string(linear_layer->delta_var_b[i]) +
                        ", expected " + std::to_string(expected_var_b) +
                        " (original was " +
                        std::to_string(original_delta_var_b[i]) + ")");
                break;
            }
        }
    }

    // Only rank 0 should assert the results to avoid multiple test failures
    if (rank == 0) {
        EXPECT_TRUE(sync_successful)
            << "Parameter synchronization (sum mode) failed";
    }
}

/**
 * Test parameter synchronization in DDPSequential with averaging mode
 * This test verifies that the sync_parameters() method correctly
 synchronizes
 * and averages model parameters across processes
 */
TEST_F(DDPOpsTest, ParameterSynchronization_NCCL_Average) {
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting parameter sync test (average mode)");

    // Create a simple model
    auto model = std::make_shared<Sequential>(Linear(10, 5, true));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    // Set average to true for averaging mode
    DDPSequential dist_model(model, config, true);

    // Get the underlying model
    auto sequential_model = dist_model.get_model();

    // Set different delta parameters on each process
    auto &linear_layer = sequential_model->layers[0];

    // Fill delta parameters with rank-specific values
    float rank_value = static_cast<float>(rank + 1);
    std::fill(linear_layer->delta_mu_w.begin(), linear_layer->delta_mu_w.end(),
              rank_value);
    std::fill(linear_layer->delta_var_w.begin(),
              linear_layer->delta_var_w.end(), rank_value * 2);

    if (linear_layer->bias) {
        std::fill(linear_layer->delta_mu_b.begin(),
                  linear_layer->delta_mu_b.end(), rank_value * 3);
        std::fill(linear_layer->delta_var_b.begin(),
                  linear_layer->delta_var_b.end(), rank_value * 4);
    }

    // Make sure all parameters are copied to the device for CUDA layers
#ifdef USE_CUDA
    auto cuda_layer = dynamic_cast<BaseLayerCuda *>(linear_layer.get());
    if (cuda_layer) {
        cuda_layer->delta_params_to_device();
    }
#endif

    // Remember the values before sync for comparison
    std::vector<float> original_delta_mu_w = linear_layer->delta_mu_w;
    std::vector<float> original_delta_var_w = linear_layer->delta_var_w;
    std::vector<float> original_delta_mu_b;
    std::vector<float> original_delta_var_b;
    if (linear_layer->bias) {
        original_delta_mu_b = linear_layer->delta_mu_b;
        original_delta_var_b = linear_layer->delta_var_b;
    }

    // Synchronize parameters
    dist_model.sync_parameters();
    dist_model.barrier();

    // Copy any device parameters back to host for checking
#ifdef USE_CUDA
    if (cuda_layer) {
        cuda_layer->delta_params_to_host();
    }
#endif

    // Expected values after synchronization - average of all ranks' values
    // Since we're in average mode, we expect the average of all ranks'
    // values
    float sum_rank_values = 0.0f;
    for (int i = 0; i < world_size; i++) {
        sum_rank_values += (i + 1);  // Sum of rank+1 for all ranks
    }

    float expected_mu_w = sum_rank_values / world_size;
    float expected_var_w = (sum_rank_values * 2) / world_size;
    float expected_mu_b = (sum_rank_values * 3) / world_size;
    float expected_var_b = (sum_rank_values * 4) / world_size;

    // Check if parameters were properly synchronized
    bool sync_successful = true;

    // Check weights
    for (size_t i = 0; i < linear_layer->delta_mu_w.size(); i++) {
        if (std::abs(linear_layer->delta_mu_w[i] - expected_mu_w) > 1e-5) {
            sync_successful = false;
            LOG(LogLevel::ERROR,
                "Process " + std::to_string(rank) + " delta_mu_w[" +
                    std::to_string(i) +
                    "] = " + std::to_string(linear_layer->delta_mu_w[i]) +
                    ", expected " + std::to_string(expected_mu_w) +
                    " (original was " + std::to_string(original_delta_mu_w[i]) +
                    ")");
            break;
        }
    }

    for (size_t i = 0; i < linear_layer->delta_var_w.size(); i++) {
        if (std::abs(linear_layer->delta_var_w[i] - expected_var_w) > 1e-5) {
            sync_successful = false;
            LOG(LogLevel::ERROR,
                "Process " + std::to_string(rank) + " delta_var_w[" +
                    std::to_string(i) +
                    "] = " + std::to_string(linear_layer->delta_var_w[i]) +
                    ", expected " + std::to_string(expected_var_w) +
                    " (original was " +
                    std::to_string(original_delta_var_w[i]) + ")");
            break;
        }
    }

    // Check biases if present
    if (linear_layer->bias) {
        for (size_t i = 0; i < linear_layer->delta_mu_b.size(); i++) {
            if (std::abs(linear_layer->delta_mu_b[i] - expected_mu_b) > 1e-5) {
                sync_successful = false;
                LOG(LogLevel::ERROR,
                    "Process " + std::to_string(rank) + " delta_mu_b[" +
                        std::to_string(i) +
                        "] = " + std::to_string(linear_layer->delta_mu_b[i]) +
                        ", expected " + std::to_string(expected_mu_b) +
                        " (original was " +
                        std::to_string(original_delta_mu_b[i]) + ")");
                break;
            }
        }

        for (size_t i = 0; i < linear_layer->delta_var_b.size(); i++) {
            if (std::abs(linear_layer->delta_var_b[i] - expected_var_b) >
                1e-5) {
                sync_successful = false;
                LOG(LogLevel::ERROR,
                    "Process " + std::to_string(rank) + " delta_var_b[" +
                        std::to_string(i) +
                        "] = " + std::to_string(linear_layer->delta_var_b[i]) +
                        ", expected " + std::to_string(expected_var_b) +
                        " (original was " +
                        std::to_string(original_delta_var_b[i]) + ")");
                break;
            }
        }
    }

    // Only rank 0 should assert the results to avoid multiple test failures
    if (rank == 0) {
        EXPECT_TRUE(sync_successful)
            << "Parameter synchronization (average mode) failed";
    }
}

/**
 * Test base parameter synchronization in DDPSequential using broadcast
 from
 * rank 0 This test verifies that the sync_base_parameters() method
 correctly
 * synchronizes model parameters across processes by broadcasting from
 rank 0
 */
TEST_F(DDPOpsTest, BaseParameterSynchronization_NCCL_Broadcast) {
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting base parameter sync test");

    // Create a simple model
    auto model = std::make_shared<Sequential>(Linear(10, 5, true));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    DDPSequential dist_model(model, config);

    // Get the underlying model
    auto sequential_model = dist_model.get_model();
    auto &linear_layer = sequential_model->layers[0];

    // Set different parameters on each process
    float rank_value = static_cast<float>(rank + 1);
    std::fill(linear_layer->mu_w.begin(), linear_layer->mu_w.end(), rank_value);
    std::fill(linear_layer->var_w.begin(), linear_layer->var_w.end(),
              rank_value * 2);

    if (linear_layer->bias) {
        std::fill(linear_layer->mu_b.begin(), linear_layer->mu_b.end(),
                  rank_value * 3);
        std::fill(linear_layer->var_b.begin(), linear_layer->var_b.end(),
                  rank_value * 4);
    }

    // Make sure all parameters are copied to the device for CUDA layers
#ifdef USE_CUDA
    auto cuda_layer = dynamic_cast<BaseLayerCuda *>(linear_layer.get());
    if (cuda_layer) {
        cuda_layer->params_to_device();
    }
#endif

    // Remember rank 0's values as they should be broadcast to all processes
    float rank0_value = 1.0f;  // rank 0 + 1
    float expected_mu_w = rank0_value;
    float expected_var_w = rank0_value * 2;
    float expected_mu_b = rank0_value * 3;
    float expected_var_b = rank0_value * 4;

    // Synchronize parameters
    dist_model.sync_base_parameters();
    dist_model.barrier();

    // Copy any device parameters back to host for checking
#ifdef USE_CUDA
    if (cuda_layer) {
        cuda_layer->params_to_host();
    }
#endif

    // Check if parameters were properly synchronized
    bool sync_successful = true;

    // Check weights
    for (size_t i = 0; i < linear_layer->mu_w.size(); i++) {
        if (std::abs(linear_layer->mu_w[i] - expected_mu_w) > 1e-5) {
            sync_successful = false;
            LOG(LogLevel::ERROR,
                "Process " + std::to_string(rank) + " mu_w[" +
                    std::to_string(i) +
                    "] = " + std::to_string(linear_layer->mu_w[i]) +
                    ", expected " + std::to_string(expected_mu_w));
            break;
        }
    }

    for (size_t i = 0; i < linear_layer->var_w.size(); i++) {
        if (std::abs(linear_layer->var_w[i] - expected_var_w) > 1e-5) {
            sync_successful = false;
            LOG(LogLevel::ERROR,
                "Process " + std::to_string(rank) + " var_w[" +
                    std::to_string(i) +
                    "] = " + std::to_string(linear_layer->var_w[i]) +
                    ", expected " + std::to_string(expected_var_w));
            break;
        }
    }

    // Check biases if present
    if (linear_layer->bias) {
        for (size_t i = 0; i < linear_layer->mu_b.size(); i++) {
            if (std::abs(linear_layer->mu_b[i] - expected_mu_b) > 1e-5) {
                sync_successful = false;
                LOG(LogLevel::ERROR,
                    "Process " + std::to_string(rank) + " mu_b[" +
                        std::to_string(i) +
                        "] = " + std::to_string(linear_layer->mu_b[i]) +
                        ", expected " + std::to_string(expected_mu_b));
                break;
            }
        }

        for (size_t i = 0; i < linear_layer->var_b.size(); i++) {
            if (std::abs(linear_layer->var_b[i] - expected_var_b) > 1e-5) {
                sync_successful = false;
                LOG(LogLevel::ERROR,
                    "Process " + std::to_string(rank) + " var_b[" +
                        std::to_string(i) +
                        "] = " + std::to_string(linear_layer->var_b[i]) +
                        ", expected " + std::to_string(expected_var_b));
                break;
            }
        }
    }

    // Only rank 0 should assert the results to avoid multiple test failures
    if (rank == 0) {
        EXPECT_TRUE(sync_successful)
            << "Base parameter synchronization (broadcast from rank 0) failed ";
    }
}
#else
// Add a dummy test when distributed functionality is not available
TEST_F(DDPOpsTest, DDPNotAvailable) {
    LOG(LogLevel::INFO,
        "Distributed functionality is not available. Skipping distributed "
        "tests.");
    GTEST_SKIP() << "Distributed functionality is not available (requires "
                    "NCCL, CUDA, and MPI)";
}
#endif

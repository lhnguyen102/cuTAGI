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
#include "../../include/conv2d_layer.h"
#include "../../include/custom_logger.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/ddp.h"
#include "../../include/linear_layer.h"
#include "../../include/max_pooling_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

// Check if all required dependencies for distributed training are available
#if defined(USE_NCCL) && defined(USE_CUDA) && defined(USE_MPI)
#define DISTRIBUTED_TEST_AVAILABLE 1
#include <mpi.h>
#endif

extern bool g_gpu_enabled;

// Global flag to track if MPI is initialized by our tests
static bool g_mpi_initialized_by_test = false;

#ifdef DISTRIBUTED_TEST_AVAILABLE
/**
 * Initialize MPI if not already initialized
 * Returns true if MPI was initialized by this function
 */
bool initialize_mpi_if_needed() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        return true;
    }
    return false;
}

/**
 * Finalize MPI if it was initialized by our tests
 */
void finalize_mpi_if_needed() {
    if (g_mpi_initialized_by_test) {
        MPI_Finalize();
        g_mpi_initialized_by_test = false;
    }
}

/**
 * Check if MPI is initialized
 */
bool is_mpi_initialized() {
    int initialized;
    MPI_Initialized(&initialized);
    return initialized != 0;
}

/**
 * Get MPI rank
 */
int get_mpi_rank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

/**
 * Get MPI world size
 */
int get_mpi_world_size() {
    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}
#else
// Stub implementations for non-MPI builds
bool initialize_mpi_if_needed() { return false; }
void finalize_mpi_if_needed() {}
bool is_mpi_initialized() { return false; }
int get_mpi_rank() { return 0; }
int get_mpi_world_size() { return 1; }
#endif

/**
 * Distributed MNIST test runner
 */
void distributed_mnist_test_runner(DDPSequential &dist_model,
                                   float &avg_error_output) {
    // Get the underlying model and configuration
    auto model = dist_model.get_model();
    auto config = dist_model.get_config();
    int rank = config.rank;
    int world_size = config.world_size;

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting distributed training");

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
    std::vector<float> sigma = {2.0f};
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
    std::string device =
        model->device + ":" + std::to_string(model->device_idx);
    OutputUpdater output_updater(device);

    unsigned seed = 42 + rank;  // Different seed for each process
    std::default_random_engine seed_e(seed);
    int n_epochs = 10;
    int batch_size = 128;
    float sigma_obs = 1.0;

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
    std::shuffle(full_data_idx.begin(), full_data_idx.end(), seed_e);

    // Extract this process's portion of the data
    std::vector<int> local_data_idx(full_data_idx.begin() + start_idx,
                                    full_data_idx.begin() + end_idx);

    int mt_idx = 0;
    std::vector<int> error_rate(local_data_size, 0);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;

    // Train for a fixed number of iterations by adjusting iterations based on
    // world size
    int num_iterations = 100 / world_size;
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        for (int i = 0; i < num_iterations; i++) {
            // Get batch for this process
            get_batch_images_labels(train_db, local_data_idx, batch_size, i,
                                    x_batch, y_batch, idx_ud_batch,
                                    label_batch);

            // Forward pass
            dist_model.forward(x_batch);

            // Update output layer
            output_updater.update_using_indices(*model->output_z_buffer,
                                                y_batch, var_obs, idx_ud_batch,
                                                *model->input_delta_z_buffer);

            // Backward pass and parameter update
            dist_model.backward();
            dist_model.step();

            // Extract output for error calculation
            model->output_to_host();

            for (int j = 0; j < batch_size * n_y; j++) {
                mu_a_output[j] = model->output_z_buffer->mu_a[j];
                var_a_output[j] = model->output_z_buffer->var_a[j];
            }

            // Calculate error rate
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(mu_a_output, var_a_output, label_batch, num_classes,
                          batch_size);
            mt_idx = i * batch_size;
            update_vector(error_rate, error_rate_batch, mt_idx, 1);

            if (i % 10 == 0) {
                LOG(LogLevel::INFO, "Process " + std::to_string(rank) +
                                        " completed iteration " +
                                        std::to_string(i) + " of " +
                                        std::to_string(num_iterations));
            }
        }
    }

    int curr_idx = mt_idx + batch_size;
    avg_error_output = compute_average_error_rate(error_rate, curr_idx,
                                                  num_iterations * batch_size);

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) +
                            " finished with average error rate: " +
                            std::to_string(avg_error_output));
}

class DistributedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize MPI if needed
        g_mpi_initialized_by_test = initialize_mpi_if_needed();

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

    void TearDown() override {
        // Finalize MPI if we initialized it
        finalize_mpi_if_needed();
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

#ifdef DISTRIBUTED_TEST_AVAILABLE
/**
 * Test distributed training with a simple CNN model using NCCL backend
 */
TEST_F(DistributedTest, SimpleCNN_NCCL) {
    // This test requires MPI to be initialized e.g., command: mpirun -np 2
    if (!is_mpi_initialized()) {
        GTEST_SKIP() << "MPI is not initialized. Run with mpirun.";
    }

    // Get MPI rank and world size
    int rank = get_mpi_rank();
    int world_size = get_mpi_world_size();

    if (world_size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI processes";
    }

    // Log process information
    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) + " starting test");

    auto model = std::make_shared<Sequential>(
        Conv2d(1, 16, 4, true, 1, 1, 1, 28, 28), ReLU(), AvgPool2d(3, 2),
        Conv2d(16, 16, 5), ReLU(), AvgPool2d(3, 2), Linear(16 * 4 * 4, 128),
        ReLU(), Linear(128, 11));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    DDPSequential dist_model(model, config);

    // Run distributed training
    float avg_error;
    distributed_mnist_test_runner(dist_model, avg_error);

    // Only rank 0 should assert the results
    if (rank == 0) {
        float threshold = 0.5;
        EXPECT_LT(avg_error, threshold)
            << "Error rate should be below " << threshold;
        LOG(LogLevel::INFO,
            "Final average error rate: " + std::to_string(avg_error));
    }
}

/**
 * Test parameter synchronization in DDPSequential
 * This test verifies that the sync_parameters() method correctly synchronizes
 * model parameters across processes
 */
TEST_F(DistributedTest, ParameterSynchronization_NCCL) {
    // This test requires MPI to be initialized
    if (!is_mpi_initialized()) {
        GTEST_SKIP() << "MPI is not initialized. Run with mpirun.";
    }

    // Get MPI rank and world size
    int rank = get_mpi_rank();
    int world_size = get_mpi_world_size();

    if (world_size < 2) {
        GTEST_SKIP() << "This test requires at least 2 MPI processes";
    }

    LOG(LogLevel::INFO, "Process " + std::to_string(rank) + " of " +
                            std::to_string(world_size) +
                            " starting parameter sync test");

    // Create a simple model
    auto model = std::make_shared<Sequential>(Linear(10, 5, true));

    // Configure distributed training
    std::vector<int> device_ids;
    for (int i = 0; i < world_size; i++) {
        device_ids.push_back(i % 2);  // Use GPUs 0 and 1 in round-robin fashion
    }

    DDPConfig config(device_ids, "nccl", rank, world_size);
    DDPSequential dist_model(model, config, true);  // Enable averaging

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

    // Synchronize parameters
    dist_model.sync_parameters();
    dist_model.barrier();

    // Expected values after synchronization (with averaging enabled)
    float expected_mu_w = (world_size + 1) / 2.0f;
    float expected_var_w = expected_mu_w * 2;
    float expected_mu_b = expected_mu_w * 3;
    float expected_var_b = expected_mu_w * 4;

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
                    ", expected " + std::to_string(expected_mu_w));
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
                    ", expected " + std::to_string(expected_var_w));
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
                        ", expected " + std::to_string(expected_mu_b));
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
                        ", expected " + std::to_string(expected_var_b));
                break;
            }
        }
    }

    // Only rank 0 should assert the results to avoid multiple test failures
    if (rank == 0) {
        EXPECT_TRUE(sync_successful) << "Parameter synchronization failed";
    }
}
#else
// Add a dummy test when distributed functionality is not available
TEST(DistributedTest, DistributedNotAvailable) {
    LOG(LogLevel::INFO,
        "Distributed functionality is not available. Skipping distributed "
        "tests.");
    GTEST_SKIP() << "Distributed functionality is not available (requires "
                    "NCCL, CUDA, and MPI)";
}
#endif

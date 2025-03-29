#pragma once
#include <gtest/gtest.h>

#include <string>

#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/layer_block.h"
#include "../../include/sequential.h"

#if defined(USE_NCCL) && defined(USE_CUDA) && defined(USE_MPI)
#define DISTRIBUTED_TEST_AVAILABLE 1
#include <mpi.h>
#endif

// Global flag to track if MPI is initialized by our tests
extern bool g_mpi_initialized_by_test;

/**
 * Initialize MPI if not already initialized
 * Returns true if MPI was initialized by this function
 */
bool initialize_mpi_if_needed();

/**
 * Finalize MPI if it was initialized by our tests
 */
void finalize_mpi_if_needed();

/**
 * Check if MPI is initialized
 */
bool is_mpi_initialized();

/**
 * Get MPI rank
 */
int get_mpi_rank();

/**
 * Get MPI world size
 */
int get_mpi_world_size();

/**
 * Base test fixture for distributed tests
 */
class DistributedTestFixture : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        // Initialize MPI if needed
        g_mpi_initialized_by_test = initialize_mpi_if_needed();
    }

    static void TearDownTestSuite() {
        // Finalize MPI if we initialized it
        finalize_mpi_if_needed();
    }

    void SetUp() override {
        // Check if MPI is initialized
        if (!is_mpi_initialized()) {
            GTEST_SKIP() << "MPI is not initialized. Run with mpirun.";
            return;
        }

        // Get MPI rank and world size
        rank = get_mpi_rank();
        world_size = get_mpi_world_size();

        if (world_size < 2) {
            GTEST_SKIP() << "This test requires at least 2 MPI processes";
            return;
        }
    }

    int rank;
    int world_size;
};

LayerBlock create_layer_block(int in_channels, int out_channels, int stride = 1,
                              int padding_type = 1);

Dataloader get_time_series_dataloader(std::vector<std::string> &data_file,
                                      int num_data, int num_features,
                                      std::vector<int> &output_col,
                                      bool data_norm, int input_seq_len,
                                      int output_seq_len, int seq_stride,
                                      std::vector<float> &mu_x,
                                      std::vector<float> &sigma_x);

void sin_signal_lstm_test_runner(Sequential &model, int input_seq_len,
                                 float &mse, float &log_lik);

void sin_signal_smoother_test_runner(Sequential &model, int input_seq_len,
                                     int num_features, float &mse,
                                     float &log_lik);

#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "../../include/activation.h"
#include "../../include/activation_cuda.cuh"
#include "../../include/data_struct.h"
#include "../../include/data_struct_cuda.cuh"

extern bool g_gpu_enabled;

TEST(CfsoftmaxMinimal, CPUvsCUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    int batch_size = 4;
    int hidden_size = 16;
    int input_size = hidden_size;
    std::vector<float> input_data(batch_size * input_size);
    std::vector<float> input_var(batch_size * input_size);
    std::vector<float> input_jcb(batch_size * input_size, 1.0f);
    std::default_random_engine gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < batch_size * input_size; i++) {
        input_data[i] = dist(gen);
        input_var[i] = 0.1f + 0.1f * std::abs(dist(gen));
    }

    // Create input states for CPU
    BaseHiddenStates input_states;
    input_states.mu_a = input_data;
    input_states.var_a = input_var;
    input_states.jcb = input_jcb;
    input_states.block_size = batch_size;
    input_states.actual_size = input_size;
    input_states.size = batch_size * input_size;

    // Create output states for CPU
    BaseHiddenStates cpu_output_states;
    cpu_output_states.mu_a.resize(batch_size * input_size);
    cpu_output_states.var_a.resize(batch_size * input_size);
    cpu_output_states.jcb.resize(batch_size * input_size);
    cpu_output_states.block_size = batch_size;
    cpu_output_states.actual_size = input_size;
    cpu_output_states.size = batch_size * input_size;

    // Create temp states for CPU
    BaseTempStates temp_states;

    // Create CPU ClosedFormSoftmax layer
    ClosedFormSoftmax cpu_cfsoftmax;
    cpu_cfsoftmax.input_size = input_size;
    cpu_cfsoftmax.output_size = input_size;

    // Run CPU forward pass
    cpu_cfsoftmax.forward(input_states, cpu_output_states, temp_states);

#ifdef USE_CUDA
    // Create CUDA input states
    HiddenStateCuda cuda_input_states(batch_size * input_size, batch_size);
    cuda_input_states.mu_a = input_data;
    cuda_input_states.var_a = input_var;
    cuda_input_states.jcb = input_jcb;
    cuda_input_states.block_size = batch_size;
    cuda_input_states.actual_size = input_size;
    cuda_input_states.size = batch_size * input_size;
    cuda_input_states.to_device();

    // Create CUDA output states
    HiddenStateCuda cuda_output_states(batch_size * input_size, batch_size);
    cuda_output_states.block_size = batch_size;
    cuda_output_states.actual_size = input_size;
    cuda_output_states.size = batch_size * input_size;

    // Create CUDA ClosedFormSoftmax layer
    ClosedFormSoftmaxCuda cuda_cfsoftmax;
    cuda_cfsoftmax.input_size = input_size;
    cuda_cfsoftmax.output_size = input_size;

    // Run CUDA forward pass
    cuda_cfsoftmax.forward(cuda_input_states, cuda_output_states, temp_states);

    // Transfer CUDA output to host for comparison
    cuda_output_states.to_host();

    // Transfer CUDA layer internal data to host for comparison
    cuda_cfsoftmax.data_to_host();
#else
    // For CPU-only builds, just use the same CPU results
    BaseHiddenStates cuda_output_states = cpu_output_states;
    ClosedFormSoftmax cuda_cfsoftmax;
    cuda_cfsoftmax.input_size = input_size;
    cuda_cfsoftmax.output_size = input_size;
    cuda_cfsoftmax.forward(input_states, cuda_output_states, temp_states);
#endif

    // Compare output values
    float tol = 1e-3f;

    // Compare internal variables - exponential sums
    ASSERT_EQ(cpu_cfsoftmax.mu_e_sum.size(), cuda_cfsoftmax.mu_e_sum.size());
    for (size_t i = 0; i < cpu_cfsoftmax.mu_e_sum.size(); ++i) {
        EXPECT_NEAR(cpu_cfsoftmax.mu_e_sum[i], cuda_cfsoftmax.mu_e_sum[i], tol)
            << "mu_e_sum mismatch at index " << i;
        EXPECT_NEAR(cpu_cfsoftmax.var_e_sum[i], cuda_cfsoftmax.var_e_sum[i],
                    tol)
            << "var_e_sum mismatch at index " << i;
    }

    // Compare internal variables - log exponential sums
    ASSERT_EQ(cpu_cfsoftmax.mu_log_e_sum.size(),
              cuda_cfsoftmax.mu_log_e_sum.size());
    for (size_t i = 0; i < cpu_cfsoftmax.mu_log_e_sum.size(); ++i) {
        EXPECT_NEAR(cpu_cfsoftmax.mu_log_e_sum[i],
                    cuda_cfsoftmax.mu_log_e_sum[i], tol)
            << "mu_log_e_sum mismatch at index " << i;
        EXPECT_NEAR(cpu_cfsoftmax.var_log_e_sum[i],
                    cuda_cfsoftmax.var_log_e_sum[i], tol)
            << "var_log_e_sum mismatch at index " << i;
    }

    // Compare internal variables - log activation values
    ASSERT_EQ(cpu_cfsoftmax.mu_log_a.size(), cuda_cfsoftmax.mu_log_a.size());
    for (size_t i = 0; i < cpu_cfsoftmax.mu_log_a.size(); ++i) {
        EXPECT_NEAR(cpu_cfsoftmax.mu_log_a[i], cuda_cfsoftmax.mu_log_a[i], tol)
            << "mu_log_a mismatch at index " << i;
        EXPECT_NEAR(cpu_cfsoftmax.var_log_a[i], cuda_cfsoftmax.var_log_a[i],
                    tol)
            << "var_log_a mismatch at index " << i;
        EXPECT_NEAR(cpu_cfsoftmax.cov_log_a_z[i], cuda_cfsoftmax.cov_log_a_z[i],
                    tol)
            << "cov_log_a_z mismatch at index " << i;
    }

    // Compare output sizes
    ASSERT_EQ(cpu_output_states.mu_a.size(), cuda_output_states.mu_a.size())
        << "Output mu sizes differ";
    ASSERT_EQ(cpu_output_states.var_a.size(), cuda_output_states.var_a.size())
        << "Output var sizes differ";

    // Compare output values
    for (size_t i = 0; i < cpu_output_states.mu_a.size(); ++i) {
        EXPECT_NEAR(cpu_output_states.mu_a[i], cuda_output_states.mu_a[i], tol)
            << "Output mu mismatch at index " << i;
        EXPECT_NEAR(cpu_output_states.var_a[i], cuda_output_states.var_a[i],
                    tol)
            << "Output var mismatch at index " << i;
        EXPECT_NEAR(cpu_output_states.jcb[i], cuda_output_states.jcb[i], tol)
            << "Output jcb mismatch at index " << i;
    }
}

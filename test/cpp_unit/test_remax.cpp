#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "../../include/activation.h"
#include "../../include/activation_cuda.cuh"
#include "../../include/data_struct.h"
#include "../../include/data_struct_cuda.cuh"

extern bool g_gpu_enabled;

TEST(RemaxMinimal, CPUvsCUDA_InternalVars) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    int batch_size = 4;
    int hidden_size = 8;
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

    // Create CPU Remax layer
    Remax cpu_remax;
    cpu_remax.input_size = input_size;
    cpu_remax.output_size = input_size;

    // Run CPU forward pass
    cpu_remax.forward(input_states, cpu_output_states, temp_states);

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

    // Create CUDA Remax layer
    RemaxCuda cuda_remax;
    cuda_remax.input_size = input_size;
    cuda_remax.output_size = input_size;

    // Run CUDA forward pass
    cuda_remax.forward(cuda_input_states, cuda_output_states, temp_states);

    // Transfer CUDA output to host for comparison
    cuda_output_states.to_host();

    // Transfer CUDA layer internal data to host for comparison
    cuda_remax.data_to_host();
#else
    // For CPU-only builds, just use the same CPU results
    BaseHiddenStates cuda_output_states = cpu_output_states;
    Remax cuda_remax;
    cuda_remax.input_size = input_size;
    cuda_remax.output_size = input_size;
    cuda_remax.forward(input_states, cuda_output_states, temp_states);
#endif

    // Compare output values
    float tol = 1e-4f;

    // Compare internal variables
    ASSERT_EQ(cpu_remax.mu_m.size(), cuda_remax.mu_m.size());
    for (size_t i = 0; i < cpu_remax.mu_m.size(); ++i) {
        EXPECT_NEAR(cpu_remax.mu_m[i], cuda_remax.mu_m[i], tol)
            << "mu_m mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.var_m[i], cuda_remax.var_m[i], tol)
            << "var_m mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.jcb_m[i], cuda_remax.jcb_m[i], tol)
            << "jcb_m mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.mu_log_m[i], cuda_remax.mu_log_m[i], tol)
            << "mu_log_m mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.var_log_m[i], cuda_remax.var_log_m[i], tol)
            << "var_log_m mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.cov_log_m_mt[i], cuda_remax.cov_log_m_mt[i], tol)
            << "cov_log_m_mt mismatch at index " << i;
    }

    for (size_t i = 0; i < cpu_remax.mu_mt.size(); ++i) {
        EXPECT_NEAR(cpu_remax.mu_mt[i], cuda_remax.mu_mt[i], tol)
            << "mu_mt mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.var_mt[i], cuda_remax.var_mt[i], tol)
            << "var_mt mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.mu_log_mt[i], cuda_remax.mu_log_mt[i], tol)
            << "mu_log_mt mismatch at index " << i;
        EXPECT_NEAR(cpu_remax.var_log_mt[i], cuda_remax.var_log_mt[i], tol)
            << "var_log_mt mismatch at index " << i;
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

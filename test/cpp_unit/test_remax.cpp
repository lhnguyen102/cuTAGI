#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "../../include/activation.h"
#include "../../include/activation_cuda.cuh"
#include "../../include/data_struct.h"
#include "../../include/data_struct_cuda.cuh"
#include "../../include/sequential.h"

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
    BaseHiddenStates input_states;
    input_states.mu_a = input_data;
    input_states.var_a = input_var;
    input_states.jcb = input_jcb;
    input_states.block_size = batch_size;
    input_states.actual_size = input_size;
    input_states.size = batch_size * input_size;

#ifdef USE_CUDA
    // For CUDA version, use HiddenStateCuda for both input and output
    HiddenStateCuda cuda_input_states(batch_size * input_size, batch_size);
    HiddenStateCuda cuda_out(batch_size * input_size, batch_size);

    // Copy data to CUDA input states
    cuda_input_states.mu_a = input_data;
    cuda_input_states.var_a = input_var;
    cuda_input_states.jcb = input_jcb;
    cuda_input_states.block_size = batch_size;
    cuda_input_states.actual_size = input_size;
    cuda_input_states.size = batch_size * input_size;
    cuda_input_states.to_device();
#else
    // For CPU version, use BaseHiddenStates
    BaseHiddenStates cuda_out;
    cuda_out.mu_a.resize(batch_size * input_size);
    cuda_out.var_a.resize(batch_size * input_size);
    cuda_out.jcb.resize(batch_size * input_size);
    cuda_out.block_size = batch_size;
    cuda_out.actual_size = input_size;
    cuda_out.size = batch_size * input_size;
#endif

    BaseHiddenStates cpu_out;
    cpu_out.mu_a.resize(batch_size * input_size);
    cpu_out.var_a.resize(batch_size * input_size);
    cpu_out.jcb.resize(batch_size * input_size);
    cpu_out.block_size = batch_size;
    cpu_out.actual_size = input_size;
    cpu_out.size = batch_size * input_size;

    BaseTempStates temp;
    Remax cpu_remax;
    Remax cuda_remax;

    // Set input and output sizes directly
    cpu_remax.input_size = input_size;
    cpu_remax.output_size = input_size;
    cuda_remax.input_size = input_size;
    cuda_remax.output_size = input_size;

    Sequential cpu_model{cpu_remax};
    Sequential cuda_model{cuda_remax};
    cuda_model.to_device("cuda");
    cpu_model.forward(input_states);

#ifdef USE_CUDA
    cuda_model.forward(cuda_input_states);
    cuda_model.output_to_host();
#else
    cuda_model.forward(input_states);
#endif

    // Get the layers and transfer CUDA data to host
    auto cpu_layer = dynamic_cast<Remax*>(cpu_model.layers[0].get());
    auto cuda_layer = dynamic_cast<RemaxCuda*>(cuda_model.layers[0].get());
    cuda_layer->data_to_host();
    float tol = 1e-5f;
    ASSERT_EQ(cpu_layer->mu_m.size(), cuda_layer->mu_m.size());
    for (size_t i = 0; i < cpu_layer->mu_m.size(); ++i) {
        EXPECT_NEAR(cpu_layer->mu_m[i], cuda_layer->mu_m[i], tol);
        EXPECT_NEAR(cpu_layer->var_m[i], cuda_layer->var_m[i], tol);
        EXPECT_NEAR(cpu_layer->jcb_m[i], cuda_layer->jcb_m[i], tol);
        EXPECT_NEAR(cpu_layer->mu_log_m[i], cuda_layer->mu_log_m[i], tol);
        EXPECT_NEAR(cpu_layer->var_log_m[i], cuda_layer->var_log_m[i], tol);
        EXPECT_NEAR(cpu_layer->cov_log_m_mt[i], cuda_layer->cov_log_m_mt[i],
                    tol);
    }
    for (size_t i = 0; i < cpu_layer->mu_mt.size(); ++i) {
        EXPECT_NEAR(cpu_layer->mu_mt[i], cuda_layer->mu_mt[i], tol);
        EXPECT_NEAR(cpu_layer->var_mt[i], cuda_layer->var_mt[i], tol);
        EXPECT_NEAR(cpu_layer->mu_log_mt[i], cuda_layer->mu_log_mt[i], tol);
        EXPECT_NEAR(cpu_layer->var_log_mt[i], cuda_layer->var_log_mt[i], tol);
    }
}

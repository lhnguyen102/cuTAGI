#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/base_output_updater.h"
#include "../../include/common.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/sequential.h"
#include "../../include/tlstm_layer.h"
#include "test_utils.h"
#ifdef USE_CUDA
#include "../../include/base_layer_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA
TEST(TLSTMCuda, ForwardBackward_CPUvsCUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    int seq_len = 4;
    int batch_size = 4;
    int ni = 1;
    int no = 8;

    // CPU model
    Sequential cpu_model(TLSTM(ni, no, false, seq_len),
                         TLSTM(no, no, true, seq_len), Linear(no, 1));
    cpu_model.set_threads(1);

    // CUDA model with same weights
    Sequential cuda_model(TLSTM(ni, no, false, seq_len),
                          TLSTM(no, no, true, seq_len), Linear(no, 1));
    // Copy weights from CPU model to CUDA model before to_device
    for (size_t i = 0; i < cpu_model.layers.size(); i++) {
        cuda_model.layers[i]->mu_w = cpu_model.layers[i]->mu_w;
        cuda_model.layers[i]->var_w = cpu_model.layers[i]->var_w;
        cuda_model.layers[i]->mu_b = cpu_model.layers[i]->mu_b;
        cuda_model.layers[i]->var_b = cpu_model.layers[i]->var_b;
    }
    cuda_model.to_device("cuda");

    // Input/output data
    std::default_random_engine gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> x_batch(batch_size * seq_len * ni);
    for (auto &v : x_batch) v = dist(gen);
    std::vector<float> var_x;
    std::vector<int> shapes = {batch_size, seq_len, ni};

    std::vector<float> y_batch(batch_size);
    for (auto &v : y_batch) v = dist(gen);
    std::vector<float> var_obs(batch_size, 1.0f);

    // Compare forward outputs
    cpu_model.forward(x_batch, var_x, shapes);
    cuda_model.forward(x_batch, var_x, shapes);

    cuda_model.output_to_host();
    float tol = 1e-4f;
    ASSERT_EQ(cpu_model.output_z_buffer->mu_a.size(),
              cuda_model.output_z_buffer->mu_a.size());
    for (size_t i = 0; i < cpu_model.output_z_buffer->mu_a.size(); i++) {
        EXPECT_NEAR(cpu_model.output_z_buffer->mu_a[i],
                    cuda_model.output_z_buffer->mu_a[i], tol)
            << "Forward mu_a mismatch at " << i;
        EXPECT_NEAR(cpu_model.output_z_buffer->var_a[i],
                    cuda_model.output_z_buffer->var_a[i], tol)
            << "Forward var_a mismatch at " << i;
    }

    // Compare backward deltas
    OutputUpdater cpu_updater("cpu");
    OutputUpdater cuda_updater("cuda");
    cpu_updater.update(*cpu_model.output_z_buffer, y_batch, var_obs,
                       *cpu_model.input_delta_z_buffer);
    cuda_updater.update(*cuda_model.output_z_buffer, y_batch, var_obs,
                        *cuda_model.input_delta_z_buffer);

    cpu_model.backward();
    cuda_model.backward();

    for (size_t l = 0; l < cpu_model.layers.size(); l++) {
        auto &cpu_layer = *cpu_model.layers[l];
        auto &cuda_layer = *cuda_model.layers[l];

        auto *cuda_base =
            dynamic_cast<BaseLayerCuda *>(cuda_model.layers[l].get());
        if (cuda_base) cuda_base->delta_params_to_host();

        for (size_t i = 0; i < cpu_layer.delta_mu_w.size(); i++) {
            EXPECT_NEAR(cpu_layer.delta_mu_w[i], cuda_layer.delta_mu_w[i], tol)
                << "Layer " << l << " delta_mu_w mismatch at " << i;
            EXPECT_NEAR(cpu_layer.delta_var_w[i], cuda_layer.delta_var_w[i],
                        tol)
                << "Layer " << l << " delta_var_w mismatch at " << i;
        }
        for (size_t i = 0; i < cpu_layer.delta_mu_b.size(); i++) {
            EXPECT_NEAR(cpu_layer.delta_mu_b[i], cuda_layer.delta_mu_b[i], tol)
                << "Layer " << l << " delta_mu_b mismatch at " << i;
            EXPECT_NEAR(cpu_layer.delta_var_b[i], cuda_layer.delta_var_b[i],
                        tol)
                << "Layer " << l << " delta_var_b mismatch at " << i;
        }
    }
}
#endif

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/positional_encoding.h"

#ifdef USE_CUDA
#include "../../include/data_struct_cuda.cuh"
#include "../../include/positional_encoding_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

TEST(PositionalEncodingCuda, ForwardMatchesCPU) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    int batch_size = 2, seq_len = 8, embed_dim = 5;
    int n = batch_size * seq_len * embed_dim;
    std::vector<float> mu(n), var(n);
    std::default_random_engine gen(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        mu[i] = nd(gen);
        var[i] = 0.1f + 0.1f * std::abs(nd(gen));
    }

    PositionalEncoding cpu(embed_dim, /*max_seq_len=*/64);
    BaseHiddenStates cpu_in, cpu_out;
    cpu_in.mu_a = mu;
    cpu_in.var_a = var;
    cpu_in.jcb.assign(n, 1.0f);
    cpu_in.size = n;
    cpu_in.block_size = batch_size;
    cpu_in.actual_size = embed_dim;
    cpu_in.seq_len = seq_len;
    cpu_out.mu_a.assign(n, 0.0f);
    cpu_out.var_a.assign(n, 0.0f);
    cpu_out.jcb.assign(n, 0.0f);
    cpu_out.size = n;
    BaseTempStates temp;
    cpu.forward(cpu_in, cpu_out, temp);

    auto cuda_layer = cpu.to_cuda(0);
    HiddenStateCuda cu_in(n, batch_size), cu_out(n, batch_size);
    cu_in.mu_a = mu;
    cu_in.var_a = var;
    cu_in.jcb.assign(n, 1.0f);
    cu_in.block_size = batch_size;
    cu_in.actual_size = embed_dim;
    cu_in.seq_len = seq_len;
    cu_in.to_device();
    cuda_layer->forward(cu_in, cu_out, temp);
    cu_out.to_host();

    EXPECT_EQ(cu_out.seq_len, seq_len);
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(cpu_out.mu_a[i], cu_out.mu_a[i], 1e-5f) << i;
        EXPECT_NEAR(cpu_out.var_a[i], cu_out.var_a[i], 1e-5f) << i;
        EXPECT_NEAR(cpu_out.jcb[i], cu_out.jcb[i], 1e-5f) << i;
    }
}

#endif  // USE_CUDA

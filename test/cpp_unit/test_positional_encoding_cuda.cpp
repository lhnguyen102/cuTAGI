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

namespace {
struct PEInput {
    std::vector<float> mu, var;
    int batch_size, seq_len, embed_dim;
};

PEInput make_input(int B, int T, int D, unsigned seed) {
    PEInput in;
    in.batch_size = B;
    in.seq_len = T;
    in.embed_dim = D;
    int n = B * T * D;
    in.mu.resize(n);
    in.var.resize(n);
    std::default_random_engine gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        in.mu[i] = nd(gen);
        in.var[i] = 0.1f + 0.1f * std::abs(nd(gen));
    }
    return in;
}
}  // namespace

TEST(PositionalEncodingCuda, ForwardMatchesCPU) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    PEInput in = make_input(/*B=*/2, /*T=*/8, /*D=*/16, /*seed=*/42);
    int n = in.batch_size * in.seq_len * in.embed_dim;

    PositionalEncoding cpu(in.embed_dim, /*max_seq_len=*/64);

    BaseHiddenStates cpu_in;
    cpu_in.mu_a = in.mu;
    cpu_in.var_a = in.var;
    cpu_in.jcb.assign(n, 1.0f);
    cpu_in.size = n;
    cpu_in.block_size = in.batch_size;
    cpu_in.actual_size = in.embed_dim;
    cpu_in.seq_len = in.seq_len;

    BaseHiddenStates cpu_out;
    cpu_out.mu_a.assign(n, 0.0f);
    cpu_out.var_a.assign(n, 0.0f);
    cpu_out.jcb.assign(n, 0.0f);
    cpu_out.size = n;
    BaseTempStates cpu_temp;

    cpu.forward(cpu_in, cpu_out, cpu_temp);

    // CUDA layer
    auto cuda_layer = cpu.to_cuda(0);
    HiddenStateCuda cu_in(n, in.batch_size);
    cu_in.mu_a = in.mu;
    cu_in.var_a = in.var;
    cu_in.jcb.assign(n, 1.0f);
    cu_in.block_size = in.batch_size;
    cu_in.actual_size = in.embed_dim;
    cu_in.seq_len = in.seq_len;
    cu_in.to_device();

    HiddenStateCuda cu_out(n, in.batch_size);
    cu_out.block_size = in.batch_size;
    cu_out.actual_size = in.embed_dim;
    cu_out.seq_len = in.seq_len;
    BaseTempStates cu_temp;

    cuda_layer->forward(cu_in, cu_out, cu_temp);
    cu_out.to_host();

    constexpr float TOL = 1e-5f;
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(cpu_out.mu_a[i], cu_out.mu_a[i], TOL)
            << "mu_a mismatch at index " << i;
        EXPECT_NEAR(cpu_out.var_a[i], cu_out.var_a[i], TOL)
            << "var_a mismatch at index " << i;
        EXPECT_NEAR(cpu_out.jcb[i], cu_out.jcb[i], TOL)
            << "jcb mismatch at index " << i;
    }
}

#endif  // USE_CUDA

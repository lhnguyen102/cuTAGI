#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/rmsnorm_layer.h"

#ifdef USE_CUDA
#include "../../include/data_struct_cuda.cuh"
#include "../../include/rmsnorm_layer_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

namespace {
void compare_vec(const std::vector<float> &a, const std::vector<float> &b,
                 const char *name) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_NEAR(a[i], b[i], 1e-4f) << name << " mismatch at index " << i;
    }
}
}  // namespace

TEST(RMSNormCuda, ForwardBackwardMatchesCPU) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    int batch_size = 4, seq_len = 3, ni = 8;
    int n = batch_size * seq_len * ni;
    std::vector<float> mu_a(n), var_a(n), delta_mu(n), delta_var(n);
    std::default_random_engine gen(777);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        mu_a[i] = nd(gen);
        var_a[i] = 0.1f + 0.1f * std::abs(nd(gen));
        delta_mu[i] = 0.05f * nd(gen);
        delta_var[i] = 0.05f + 0.05f * std::abs(nd(gen));
    }

    // Per-feature mu_w != 1 and distinct var_w catch a missing mu_w factor
    // and row/column index mix-ups.
    RMSNorm cpu({ni}, /*eps=*/1e-6f, /*gain_w=*/1.5f);
    cpu.set_threads(1);
    for (int i = 0; i < ni; i++) {
        cpu.mu_w[i] = 1.0f + 0.5f * nd(gen);
        cpu.var_w[i] = 0.05f + 0.05f * std::abs(nd(gen));
    }

    BaseHiddenStates h_in, h_out;
    h_in.mu_a = mu_a;
    h_in.var_a = var_a;
    h_in.jcb.assign(n, 1.0f);
    h_in.size = n;
    h_in.block_size = batch_size;
    h_in.actual_size = ni;
    h_in.seq_len = seq_len;
    h_out.mu_a.assign(n, 0.0f);
    h_out.var_a.assign(n, 0.0f);
    h_out.jcb.assign(n, 0.0f);
    h_out.size = n;

    BaseDeltaStates d_in, d_out;
    d_in.delta_mu = delta_mu;
    d_in.delta_var = delta_var;
    d_in.size = n;
    d_in.block_size = batch_size;
    d_in.actual_size = ni;
    d_in.seq_len = seq_len;
    d_out.delta_mu.assign(n, 0.0f);
    d_out.delta_var.assign(n, 0.0f);
    d_out.size = n;

    BaseTempStates temp;
    cpu.forward(h_in, h_out, temp);
    cpu.backward(d_in, d_out, temp);

    auto cuda_layer = cpu.to_cuda(0);
    auto *cuda_rms = dynamic_cast<RMSNormCuda *>(cuda_layer.get());
    ASSERT_NE(cuda_rms, nullptr);

    HiddenStateCuda cu_in(n, batch_size), cu_out(n, batch_size);
    DeltaStateCuda cu_din(n, batch_size), cu_dout(n, batch_size);
    cu_in.mu_a = mu_a;
    cu_in.var_a = var_a;
    cu_in.jcb.assign(n, 1.0f);
    cu_in.block_size = batch_size;
    cu_in.actual_size = ni;
    cu_in.seq_len = seq_len;
    cu_in.to_device();
    cu_din.delta_mu = delta_mu;
    cu_din.delta_var = delta_var;
    cu_din.block_size = batch_size;
    cu_din.actual_size = ni;
    cu_din.seq_len = seq_len;
    cu_din.to_device();

    cuda_rms->forward(cu_in, cu_out, temp);
    cuda_rms->backward(cu_din, cu_dout, temp);
    cu_out.to_host();
    cu_dout.to_host();
    cuda_rms->delta_params_to_host();

    compare_vec(h_out.mu_a, cu_out.mu_a, "fwd mu_a");
    compare_vec(h_out.var_a, cu_out.var_a, "fwd var_a");
    compare_vec(d_out.delta_mu, cu_dout.delta_mu, "bwd delta_mu");
    compare_vec(d_out.delta_var, cu_dout.delta_var, "bwd delta_var");
    compare_vec(cpu.delta_mu_w, cuda_rms->delta_mu_w, "delta_mu_w");
    compare_vec(cpu.delta_var_w, cuda_rms->delta_var_w, "delta_var_w");
    EXPECT_EQ(d_out.seq_len, seq_len);
    EXPECT_EQ(cu_dout.seq_len, seq_len);
}

TEST(RMSNormCuda, ToHostKeepsConfigAndParams) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    RMSNorm cpu({6}, /*eps=*/1e-3f, /*gain_w=*/2.0f);
    cpu.mu_w[2] = 0.7f;
    auto cuda_layer = cpu.to_cuda(0);
    auto host = dynamic_cast<RMSNormCuda *>(cuda_layer.get())->to_host();
    auto *host_rms = dynamic_cast<RMSNorm *>(host.get());
    ASSERT_NE(host_rms, nullptr);

    EXPECT_FLOAT_EQ(host_rms->epsilon, 1e-3f);
    EXPECT_FLOAT_EQ(host_rms->gain_w, 2.0f);
    EXPECT_EQ(host_rms->normalized_shape, std::vector<int>{6});
    EXPECT_EQ(host_rms->mu_w, cpu.mu_w);
    EXPECT_EQ(host_rms->var_w, cpu.var_w);
}

#endif  // USE_CUDA

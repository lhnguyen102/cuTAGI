#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/embedding_cpu.h"

#ifdef USE_CUDA
#include "../../include/data_struct_cuda.cuh"
#include "../../include/embedding_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

namespace {
struct EmbConfig {
    int num_embeddings = 16;
    int embedding_dim = 4;
    int input_size = 5;  // tokens per sample
    int batch_size = 3;
    int padding_idx = -1;
    float scale = 1.0f;
};

void compare_vec(const std::vector<float> &a, const std::vector<float> &b,
                 const char *name) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_NEAR(a[i], b[i], 1e-5f) << name << " mismatch at index " << i;
    }
}

void expect_cuda_matches_cpu(const EmbConfig &cfg, unsigned seed) {
    int input_total = cfg.batch_size * cfg.input_size;
    int output_total = input_total * cfg.embedding_dim;

    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> cat_dist(0, cfg.num_embeddings - 1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> tokens(input_total);
    for (auto &token : tokens) token = static_cast<float>(cat_dist(gen));
    std::vector<float> delta_mu(output_total), delta_var(output_total);
    for (int i = 0; i < output_total; i++) {
        delta_mu[i] = 0.05f * nd(gen);
        delta_var[i] = 0.05f + 0.05f * std::abs(nd(gen));
    }

    Embedding cpu(cfg.num_embeddings, cfg.embedding_dim, cfg.input_size,
                  cfg.scale, cfg.padding_idx);

    BaseHiddenStates h_in, h_out;
    h_in.mu_a = tokens;
    h_in.var_a.assign(input_total, 0.0f);
    h_in.jcb.assign(input_total, 1.0f);
    h_in.size = input_total;
    h_in.block_size = cfg.batch_size;
    h_in.actual_size = cfg.input_size;
    h_out.mu_a.assign(output_total, 0.0f);
    h_out.var_a.assign(output_total, 0.0f);
    h_out.jcb.assign(output_total, 0.0f);
    h_out.size = output_total;

    BaseDeltaStates d_in, d_out;
    d_in.delta_mu = delta_mu;
    d_in.delta_var = delta_var;
    d_in.size = output_total;
    d_in.block_size = cfg.batch_size;
    d_in.actual_size = cfg.embedding_dim;
    d_in.seq_len = cfg.input_size;

    BaseTempStates temp;
    cpu.forward(h_in, h_out, temp);
    cpu.backward(d_in, d_out, temp, /*state_udapte=*/false);

    auto cuda_layer = cpu.to_cuda(0);
    auto *cuda_emb = dynamic_cast<EmbeddingCuda *>(cuda_layer.get());
    ASSERT_NE(cuda_emb, nullptr);

    HiddenStateCuda cu_in(input_total, cfg.batch_size);
    HiddenStateCuda cu_out(output_total, cfg.batch_size);
    DeltaStateCuda cu_din(output_total, cfg.batch_size);
    DeltaStateCuda cu_dout(output_total, cfg.batch_size);
    cu_in.mu_a = tokens;
    cu_in.block_size = cfg.batch_size;
    cu_in.actual_size = cfg.input_size;
    cu_in.to_device();
    cu_din.delta_mu = delta_mu;
    cu_din.delta_var = delta_var;
    cu_din.block_size = cfg.batch_size;
    cu_din.actual_size = cfg.embedding_dim;
    cu_din.seq_len = cfg.input_size;
    cu_din.to_device();

    cuda_emb->forward(cu_in, cu_out, temp);
    cuda_emb->backward(cu_din, cu_dout, temp, /*state_udapte=*/false);
    cu_out.to_host();
    cuda_emb->delta_params_to_host();

    EXPECT_EQ(cu_out.seq_len, h_out.seq_len);
    compare_vec(h_out.mu_a, cu_out.mu_a, "fwd mu_a");
    compare_vec(h_out.var_a, cu_out.var_a, "fwd var_a");
    compare_vec(cpu.delta_mu_w, cuda_emb->delta_mu_w, "delta_mu_w");
    compare_vec(cpu.delta_var_w, cuda_emb->delta_var_w, "delta_var_w");
}
}  // namespace

TEST(EmbeddingCuda, ForwardBackwardMatchesCPU) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_cuda_matches_cpu(EmbConfig(), /*seed=*/101);
}

TEST(EmbeddingCuda, ForwardBackwardMatchesCPUWithPadding) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    cfg.padding_idx = 0;
    expect_cuda_matches_cpu(cfg, /*seed=*/202);
}

// scale != 1 makes var_w != 1, so delta_var_w distinguishes var_w from var_w^2.
TEST(EmbeddingCuda, ForwardBackwardMatchesCPUSmallScale) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    cfg.scale = 0.15f;
    expect_cuda_matches_cpu(cfg, /*seed=*/505);
}

#endif  // USE_CUDA

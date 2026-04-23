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

struct EmbInputs {
    std::vector<float> mu_a;   // category indices stored as floats
    std::vector<float> var_a;  // unused by the layer; kept zero
    std::vector<float> jcb;
    std::vector<float> delta_mu;
    std::vector<float> delta_var;
    int input_total;   // batch_size * input_size
    int output_total;  // batch_size * input_size * embedding_dim
};

EmbInputs make_inputs(const EmbConfig &cfg, unsigned seed) {
    EmbInputs in;
    in.input_total = cfg.batch_size * cfg.input_size;
    in.output_total = in.input_total * cfg.embedding_dim;

    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> cat_dist(0, cfg.num_embeddings - 1);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    in.mu_a.resize(in.input_total);
    in.var_a.assign(in.input_total, 0.0f);
    in.jcb.assign(in.input_total, 1.0f);
    for (int i = 0; i < in.input_total; i++) {
        in.mu_a[i] = static_cast<float>(cat_dist(gen));
    }

    in.delta_mu.resize(in.output_total);
    in.delta_var.resize(in.output_total);
    for (int i = 0; i < in.output_total; i++) {
        in.delta_mu[i] = 0.05f * nd(gen);
        in.delta_var[i] = 0.05f + 0.05f * std::abs(nd(gen));
    }
    return in;
}

void compare_vec(const std::vector<float> &a, const std::vector<float> &b,
                 const char *name, float tol = 1e-5f) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_NEAR(a[i], b[i], tol) << name << " mismatch at index " << i;
    }
}

void run_compare(const EmbConfig &cfg, unsigned seed) {
    EmbInputs in = make_inputs(cfg, seed);

    // CPU layer (initializes its own weights; we copy them to CUDA via
    // to_cuda).
    auto cpu =
        std::make_unique<Embedding>(cfg.num_embeddings, cfg.embedding_dim,
                                    cfg.input_size, cfg.scale, cfg.padding_idx);
    cpu->training = true;
    cpu->param_update = true;

    BaseHiddenStates h_in, h_out;
    h_in.mu_a = in.mu_a;
    h_in.var_a = in.var_a;
    h_in.jcb = in.jcb;
    h_in.size = in.input_total;
    h_in.block_size = cfg.batch_size;
    h_in.actual_size = cfg.input_size;
    h_in.seq_len = 1;

    h_out.mu_a.assign(in.output_total, 0.0f);
    h_out.var_a.assign(in.output_total, 0.0f);
    h_out.jcb.assign(in.output_total, 0.0f);
    h_out.size = in.output_total;

    BaseDeltaStates d_in, d_out;  // Embedding::backward ignores d_out.
    d_in.delta_mu = in.delta_mu;
    d_in.delta_var = in.delta_var;
    d_in.size = in.output_total;
    d_in.block_size = cfg.batch_size;
    d_in.actual_size = cfg.embedding_dim * cfg.input_size;

    BaseTempStates h_temp;
    cpu->forward(h_in, h_out, h_temp);
    cpu->backward(d_in, d_out, h_temp, /*state_udapte=*/false);

    // CUDA layer with the same weights.
    auto cuda_layer = cpu->to_cuda(0);
    auto *cuda_emb = dynamic_cast<EmbeddingCuda *>(cuda_layer.get());
    ASSERT_NE(cuda_emb, nullptr);
    cuda_emb->training = true;
    cuda_emb->param_update = true;

    HiddenStateCuda cu_in(in.input_total, cfg.batch_size);
    HiddenStateCuda cu_out(in.output_total, cfg.batch_size);
    DeltaStateCuda cu_din(in.output_total, cfg.batch_size);
    DeltaStateCuda cu_dout(in.output_total, cfg.batch_size);

    cu_in.mu_a = in.mu_a;
    cu_in.var_a = in.var_a;
    cu_in.jcb = in.jcb;
    cu_in.block_size = cfg.batch_size;
    cu_in.actual_size = cfg.input_size;
    cu_in.seq_len = 1;
    cu_in.to_device();

    cu_out.block_size = cfg.batch_size;
    cu_out.actual_size = cfg.embedding_dim;
    cu_out.seq_len = cfg.input_size;

    cu_din.delta_mu = in.delta_mu;
    cu_din.delta_var = in.delta_var;
    cu_din.block_size = cfg.batch_size;
    cu_din.actual_size = cfg.embedding_dim * cfg.input_size;
    cu_din.to_device();

    BaseTempStates cu_temp;
    cuda_emb->forward(cu_in, cu_out, cu_temp);
    cuda_emb->backward(cu_din, cu_dout, cu_temp, /*state_udapte=*/false);

    cu_out.to_host();
    cuda_emb->delta_params_to_host();

    compare_vec(h_out.mu_a, cu_out.mu_a, "fwd mu_a");
    compare_vec(h_out.var_a, cu_out.var_a, "fwd var_a");
    compare_vec(cpu->delta_mu_w, cuda_emb->delta_mu_w, "delta_mu_w");
    compare_vec(cpu->delta_var_w, cuda_emb->delta_var_w, "delta_var_w");
}
}  // namespace

TEST(EmbeddingCuda, ForwardBackwardMatchesCPU_NoPadding) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    run_compare(cfg, /*seed=*/101);
}

TEST(EmbeddingCuda, ForwardBackwardMatchesCPU_WithPadding) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    cfg.padding_idx = 0;  // category 0 should produce zeros and skip bwd
    run_compare(cfg, /*seed=*/202);
}

// scale != 1 so var_w != 1 — exposes the var_w vs var_w^2 mistake in the
// CUDA delta_var_w computation (mathematically invisible when var_w == 1).
TEST(EmbeddingCuda, ForwardBackwardMatchesCPU_SmallScale) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    cfg.scale = 0.15f;
    run_compare(cfg, /*seed=*/505);
}

#endif  // USE_CUDA

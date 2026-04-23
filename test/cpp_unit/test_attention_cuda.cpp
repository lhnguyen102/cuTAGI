#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/attention.h"

#ifdef USE_CUDA
#include "../../include/attention_cuda.cuh"
#include "../../include/data_struct_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

namespace {
struct AttnConfig {
    int batch_size = 16;
    int seq_len = 128;
    int embed_dim = 128;
    int num_heads = 4;
    bool bias = true;
    std::string pos_emb;  // "none" or "rope"
    bool use_causal_mask;
};

struct AttnInputs {
    std::vector<float> mu_a, var_a, jcb;
    std::vector<float> delta_mu, delta_var;
    int total;  // batch_size * seq_len * embed_dim
};

AttnInputs make_inputs(const AttnConfig &cfg, unsigned seed) {
    AttnInputs in;
    in.total = cfg.batch_size * cfg.seq_len * cfg.embed_dim;
    in.mu_a.resize(in.total);
    in.var_a.resize(in.total);
    in.jcb.assign(in.total, 1.0f);
    in.delta_mu.resize(in.total);
    in.delta_var.resize(in.total);
    std::default_random_engine gen(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int i = 0; i < in.total; i++) {
        in.mu_a[i] = nd(gen);
        in.var_a[i] = 0.1f + 0.1f * std::abs(nd(gen));
        in.delta_mu[i] = 0.05f * nd(gen);
        in.delta_var[i] = 0.05f + 0.05f * std::abs(nd(gen));
    }
    return in;
}

void fill_cpu_states(const AttnInputs &in, const AttnConfig &cfg,
                     BaseHiddenStates &h_in, BaseHiddenStates &h_out,
                     BaseDeltaStates &d_in, BaseDeltaStates &d_out) {
    h_in.mu_a = in.mu_a;
    h_in.var_a = in.var_a;
    h_in.jcb = in.jcb;
    h_in.size = in.total;
    h_in.block_size = cfg.batch_size;
    h_in.actual_size = cfg.embed_dim;
    h_in.seq_len = cfg.seq_len;

    h_out.mu_a.assign(in.total, 0.0f);
    h_out.var_a.assign(in.total, 0.0f);
    h_out.jcb.assign(in.total, 0.0f);
    h_out.size = in.total;

    d_in.delta_mu = in.delta_mu;
    d_in.delta_var = in.delta_var;
    d_in.size = in.total;
    d_in.block_size = cfg.batch_size;
    d_in.actual_size = cfg.embed_dim * cfg.seq_len;

    d_out.delta_mu.assign(in.total, 0.0f);
    d_out.delta_var.assign(in.total, 0.0f);
    d_out.size = in.total;
    d_out.block_size = cfg.batch_size;
    d_out.actual_size = cfg.embed_dim;
    d_out.seq_len = cfg.seq_len;
}

void fill_cuda_states(const AttnInputs &in, const AttnConfig &cfg,
                      HiddenStateCuda &h_in, HiddenStateCuda &h_out,
                      DeltaStateCuda &d_in, DeltaStateCuda &d_out) {
    h_in.mu_a = in.mu_a;
    h_in.var_a = in.var_a;
    h_in.jcb = in.jcb;
    h_in.block_size = cfg.batch_size;
    h_in.actual_size = cfg.embed_dim;
    h_in.seq_len = cfg.seq_len;
    h_in.to_device();

    h_out.block_size = cfg.batch_size;
    h_out.actual_size = cfg.embed_dim;
    h_out.seq_len = cfg.seq_len;

    d_in.delta_mu = in.delta_mu;
    d_in.delta_var = in.delta_var;
    d_in.block_size = cfg.batch_size;
    d_in.actual_size = cfg.embed_dim * cfg.seq_len;
    d_in.to_device();

    d_out.block_size = cfg.batch_size;
    d_out.actual_size = cfg.embed_dim;
    d_out.seq_len = cfg.seq_len;
}

void compare_vec(const std::vector<float> &a, const std::vector<float> &b,
                 const char *name, float tol = 1e-3f) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_NEAR(a[i], b[i], tol) << name << " mismatch at index " << i;
    }
}

void run_mha_compare(const AttnConfig &cfg) {
    AttnInputs in = make_inputs(cfg, /*seed=*/123);

    // CPU layer (initializes its own weights).
    auto cpu = std::make_unique<MultiheadAttention>(
        cfg.embed_dim, cfg.num_heads, /*num_kv_heads=*/cfg.num_heads,
        cfg.seq_len, cfg.bias, /*gain_w=*/1.0f, /*gain_b=*/1.0f,
        /*init_method=*/"Xavier", cfg.pos_emb,
        /*rope_theta=*/10000.0f, /*max_seq_len=*/64, cfg.use_causal_mask);
    cpu->training = true;
    cpu->param_update = true;
    cpu->set_threads(1);

    BaseHiddenStates h_in, h_out;
    BaseDeltaStates d_in, d_out;
    BaseTempStates h_temp;
    fill_cpu_states(in, cfg, h_in, h_out, d_in, d_out);
    cpu->forward(h_in, h_out, h_temp);
    cpu->backward(d_in, d_out, h_temp, /*state_udapte=*/true);

    // CUDA layer with the same weights (to_cuda copies them).
    auto cuda_layer = cpu->to_cuda(0);
    auto *cuda_attn = dynamic_cast<MultiheadAttentionCuda *>(cuda_layer.get());
    ASSERT_NE(cuda_attn, nullptr);
    cuda_attn->training = true;
    cuda_attn->param_update = true;

    int n = in.total;
    HiddenStateCuda cu_in(n, cfg.batch_size), cu_out(n, cfg.batch_size);
    DeltaStateCuda cu_din(n, cfg.batch_size), cu_dout(n, cfg.batch_size);
    BaseTempStates cu_temp;
    fill_cuda_states(in, cfg, cu_in, cu_out, cu_din, cu_dout);
    cuda_attn->forward(cu_in, cu_out, cu_temp);
    cuda_attn->backward(cu_din, cu_dout, cu_temp, /*state_udapte=*/true);

    cu_out.to_host();
    cu_dout.to_host();
    cuda_attn->delta_params_to_host();

    compare_vec(h_out.mu_a, cu_out.mu_a, "fwd mu_a");
    compare_vec(h_out.var_a, cu_out.var_a, "fwd var_a");

    AttentionScores cpu_scores = cpu->get_attention_scores();
    AttentionScores cuda_scores = cuda_attn->get_attention_scores();
    EXPECT_EQ(cpu_scores.batch_size, cuda_scores.batch_size);
    EXPECT_EQ(cpu_scores.num_heads, cuda_scores.num_heads);
    EXPECT_EQ(cpu_scores.timestep, cuda_scores.timestep);
    compare_vec(cpu_scores.mu, cuda_scores.mu, "att_score mu");
    compare_vec(cpu_scores.var, cuda_scores.var, "att_score var");

    compare_vec(d_out.delta_mu, cu_dout.delta_mu, "bwd delta_mu");
    compare_vec(d_out.delta_var, cu_dout.delta_var, "bwd delta_var");
    compare_vec(cpu->delta_mu_w, cuda_attn->delta_mu_w, "delta_mu_w");
    compare_vec(cpu->delta_var_w, cuda_attn->delta_var_w, "delta_var_w");
    if (cfg.bias) {
        compare_vec(cpu->delta_mu_b, cuda_attn->delta_mu_b, "delta_mu_b");
        compare_vec(cpu->delta_var_b, cuda_attn->delta_var_b, "delta_var_b");
    }
}

void run_mha_v2_compare(const AttnConfig &cfg) {
    AttnInputs in = make_inputs(cfg, /*seed=*/321);

    auto cpu = std::make_unique<MultiheadAttentionV2>(
        cfg.embed_dim, cfg.num_heads, /*num_kv_heads=*/cfg.num_heads,
        cfg.seq_len, cfg.bias, /*gain_w=*/1.0f, /*gain_b=*/1.0f,
        /*init_method=*/"Xavier", cfg.pos_emb,
        /*rope_theta=*/10000.0f, /*max_seq_len=*/64, cfg.use_causal_mask);
    cpu->training = true;
    cpu->param_update = true;
    cpu->set_threads(1);

    BaseHiddenStates h_in, h_out;
    BaseDeltaStates d_in, d_out;
    BaseTempStates h_temp;
    fill_cpu_states(in, cfg, h_in, h_out, d_in, d_out);
    cpu->forward(h_in, h_out, h_temp);
    cpu->backward(d_in, d_out, h_temp, /*state_udapte=*/true);

    auto cuda_layer = cpu->to_cuda(0);
    auto *cuda_v2 = dynamic_cast<MultiheadAttentionV2Cuda *>(cuda_layer.get());
    ASSERT_NE(cuda_v2, nullptr);
    cuda_v2->training = true;
    cuda_v2->param_update = true;

    int n = in.total;
    HiddenStateCuda cu_in(n, cfg.batch_size), cu_out(n, cfg.batch_size);
    DeltaStateCuda cu_din(n, cfg.batch_size), cu_dout(n, cfg.batch_size);
    BaseTempStates cu_temp;
    fill_cuda_states(in, cfg, cu_in, cu_out, cu_din, cu_dout);
    cuda_v2->forward(cu_in, cu_out, cu_temp);
    cuda_v2->backward(cu_din, cu_dout, cu_temp, /*state_udapte=*/true);

    cu_out.to_host();
    cu_dout.to_host();

    // Pull device delta params back into the host vectors for comparison.
    cudaMemcpy(cuda_v2->delta_mu_w_q.data(), cuda_v2->d_delta_mu_w_q,
               cuda_v2->num_weights_q * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_v2->delta_var_w_q.data(), cuda_v2->d_delta_var_w_q,
               cuda_v2->num_weights_q * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_v2->delta_mu_w_k.data(), cuda_v2->d_delta_mu_w_k,
               cuda_v2->num_weights_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_v2->delta_var_w_k.data(), cuda_v2->d_delta_var_w_k,
               cuda_v2->num_weights_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_v2->delta_mu_w_v.data(), cuda_v2->d_delta_mu_w_v,
               cuda_v2->num_weights_v * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_v2->delta_var_w_v.data(), cuda_v2->d_delta_var_w_v,
               cuda_v2->num_weights_v * sizeof(float), cudaMemcpyDeviceToHost);
    if (cfg.bias) {
        cudaMemcpy(cuda_v2->delta_mu_b_q.data(), cuda_v2->d_delta_mu_b_q,
                   cuda_v2->num_biases_q * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cuda_v2->delta_var_b_q.data(), cuda_v2->d_delta_var_b_q,
                   cuda_v2->num_biases_q * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cuda_v2->delta_mu_b_k.data(), cuda_v2->d_delta_mu_b_k,
                   cuda_v2->num_biases_k * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cuda_v2->delta_var_b_k.data(), cuda_v2->d_delta_var_b_k,
                   cuda_v2->num_biases_k * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cuda_v2->delta_mu_b_v.data(), cuda_v2->d_delta_mu_b_v,
                   cuda_v2->num_biases_v * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(cuda_v2->delta_var_b_v.data(), cuda_v2->d_delta_var_b_v,
                   cuda_v2->num_biases_v * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    compare_vec(h_out.mu_a, cu_out.mu_a, "v2 fwd mu_a");
    compare_vec(h_out.var_a, cu_out.var_a, "v2 fwd var_a");

    AttentionScores cpu_scores = cpu->get_attention_scores();
    AttentionScores cuda_scores = cuda_v2->get_attention_scores();
    EXPECT_EQ(cpu_scores.batch_size, cuda_scores.batch_size);
    EXPECT_EQ(cpu_scores.num_heads, cuda_scores.num_heads);
    EXPECT_EQ(cpu_scores.timestep, cuda_scores.timestep);
    compare_vec(cpu_scores.mu, cuda_scores.mu, "v2 att_score mu");
    compare_vec(cpu_scores.var, cuda_scores.var, "v2 att_score var");

    compare_vec(d_out.delta_mu, cu_dout.delta_mu, "v2 bwd delta_mu");
    compare_vec(d_out.delta_var, cu_dout.delta_var, "v2 bwd delta_var");
    compare_vec(cpu->delta_mu_w_q, cuda_v2->delta_mu_w_q, "v2 delta_mu_w_q");
    compare_vec(cpu->delta_var_w_q, cuda_v2->delta_var_w_q, "v2 delta_var_w_q");
    compare_vec(cpu->delta_mu_w_k, cuda_v2->delta_mu_w_k, "v2 delta_mu_w_k");
    compare_vec(cpu->delta_var_w_k, cuda_v2->delta_var_w_k, "v2 delta_var_w_k");
    compare_vec(cpu->delta_mu_w_v, cuda_v2->delta_mu_w_v, "v2 delta_mu_w_v");
    compare_vec(cpu->delta_var_w_v, cuda_v2->delta_var_w_v, "v2 delta_var_w_v");
    if (cfg.bias) {
        compare_vec(cpu->delta_mu_b_q, cuda_v2->delta_mu_b_q,
                    "v2 delta_mu_b_q");
        compare_vec(cpu->delta_var_b_q, cuda_v2->delta_var_b_q,
                    "v2 delta_var_b_q");
        compare_vec(cpu->delta_mu_b_k, cuda_v2->delta_mu_b_k,
                    "v2 delta_mu_b_k");
        compare_vec(cpu->delta_var_b_k, cuda_v2->delta_var_b_k,
                    "v2 delta_var_b_k");
        compare_vec(cpu->delta_mu_b_v, cuda_v2->delta_mu_b_v,
                    "v2 delta_mu_b_v");
        compare_vec(cpu->delta_var_b_v, cuda_v2->delta_var_b_v,
                    "v2 delta_var_b_v");
    }
}
}  // namespace

TEST(MultiheadAttentionCuda, NoRope_NoMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "none";
    cfg.use_causal_mask = false;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, NoRope_CausalMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "none";
    cfg.use_causal_mask = true;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, Rope_NoMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "rope";
    cfg.use_causal_mask = false;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, Rope_CausalMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "rope";
    cfg.use_causal_mask = true;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionV2Cuda, NoRope_NoMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "none";
    cfg.use_causal_mask = false;
    run_mha_v2_compare(cfg);
}

TEST(MultiheadAttentionV2Cuda, NoRope_CausalMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "none";
    cfg.use_causal_mask = true;
    run_mha_v2_compare(cfg);
}

TEST(MultiheadAttentionV2Cuda, Rope_NoMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "rope";
    cfg.use_causal_mask = false;
    run_mha_v2_compare(cfg);
}

TEST(MultiheadAttentionV2Cuda, Rope_CausalMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.pos_emb = "rope";
    cfg.use_causal_mask = true;
    run_mha_v2_compare(cfg);
}

#endif  // USE_CUDA

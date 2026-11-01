#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "../../include/attention.h"
#include "test_utils.h"

#ifdef USE_CUDA
#include "../../include/attention_cuda.cuh"
#include "../../include/data_struct_cuda.cuh"
#endif

extern bool g_gpu_enabled;

namespace {
struct AttnConfig {
    int batch_size = 16;
    int seq_len = 128;
    int embed_dim = 128;
    int num_heads = 4;
    bool bias = true;
    bool use_rope;
    bool use_causal_mask;
    bool center_score_delta = false;
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

std::unique_ptr<MultiheadAttention> make_cpu_layer(const AttnConfig &cfg) {
    auto cpu = std::make_unique<MultiheadAttention>(
        cfg.embed_dim, cfg.num_heads, /*num_kv_heads=*/cfg.num_heads,
        cfg.seq_len, cfg.bias, /*gain_w=*/1.0f, /*gain_b=*/1.0f,
        /*init_method=*/"Xavier", cfg.use_rope,
        /*rope_theta=*/10000.0f, /*max_seq_len=*/cfg.seq_len,
        cfg.use_causal_mask, cfg.center_score_delta);
    cpu->training = true;
    cpu->param_update = true;
    cpu->set_threads(1);
    return cpu;
}

}  // namespace

TEST(MultiheadAttention, CenterScoreDeltaZeroesAttentionWeightedRowSum) {
    AttnConfig cfg;
    cfg.batch_size = 2;
    cfg.seq_len = 16;
    cfg.embed_dim = 16;
    cfg.use_rope = true;
    cfg.use_causal_mask = true;
    cfg.center_score_delta = true;
    AttnInputs in = make_inputs(cfg, /*seed=*/7);
    auto cpu = make_cpu_layer(cfg);

    BaseHiddenStates h_in, h_out;
    BaseDeltaStates d_in, d_out;
    BaseTempStates h_temp;
    fill_cpu_states(in, cfg, h_in, h_out, d_in, d_out);
    cpu->forward(h_in, h_out, h_temp);
    cpu->backward(d_in, d_out, h_temp, /*state_udapte=*/true);

    const auto &att = cpu->attn_states.mu_att_score;
    const auto &delta = cpu->attn_delta_states.delta_mu_att_score;
    int timestep = cfg.seq_len;
    int num_rows = cfg.batch_size * cfg.num_heads * timestep;
    for (int row = 0; row < num_rows; row++) {
        float weighted_sum = 0.0f, max_abs_delta = 0.0f;
        for (int j = 0; j < timestep; j++) {
            weighted_sum += att[row * timestep + j] * delta[row * timestep + j];
            max_abs_delta =
                std::max(max_abs_delta, std::abs(delta[row * timestep + j]));
        }
        EXPECT_LT(std::abs(weighted_sum), 1e-3f * max_abs_delta)
            << "row " << row;
    }
}

#ifdef USE_CUDA

namespace {

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
    auto cpu = make_cpu_layer(cfg);

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
    compare_vec(cpu_scores.mu_qk, cuda_scores.mu_qk, "pre-Remax mu");
    compare_vec(cpu_scores.var_qk, cuda_scores.var_qk, "pre-Remax var");
    compare_vec(cpu_scores.mu_att, cuda_scores.mu_att, "post-Remax mu");
    compare_vec(cpu_scores.var_att, cuda_scores.var_att, "post-Remax var");

    compare_vec(d_out.delta_mu, cu_dout.delta_mu, "bwd delta_mu");
    compare_vec(d_out.delta_var, cu_dout.delta_var, "bwd delta_var");
    compare_vec(cpu->delta_mu_w, cuda_attn->delta_mu_w, "delta_mu_w");
    compare_vec(cpu->delta_var_w, cuda_attn->delta_var_w, "delta_var_w");
    if (cfg.bias) {
        compare_vec(cpu->delta_mu_b, cuda_attn->delta_mu_b, "delta_mu_b");
        compare_vec(cpu->delta_var_b, cuda_attn->delta_var_b, "delta_var_b");
    }
}

}  // namespace

TEST(MultiheadAttentionCuda, NoRope_NoMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.use_rope = false;
    cfg.use_causal_mask = false;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, NoRope_CausalMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.use_rope = false;
    cfg.use_causal_mask = true;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, Rope_NoMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.use_rope = true;
    cfg.use_causal_mask = false;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, Rope_CausalMask) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.use_rope = true;
    cfg.use_causal_mask = true;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, Rope_CausalMask_CenterScoreDelta) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.use_rope = true;
    cfg.use_causal_mask = true;
    cfg.center_score_delta = true;
    run_mha_compare(cfg);
}

TEST(MultiheadAttentionCuda, ToHostKeepsCenterScoreDelta) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    AttnConfig cfg;
    cfg.use_rope = true;
    cfg.use_causal_mask = true;
    cfg.center_score_delta = true;
    auto cuda_layer = make_cpu_layer(cfg)->to_cuda(0);
    auto host = dynamic_cast<BaseLayerCuda *>(cuda_layer.get())->to_host();
    auto *attn = dynamic_cast<MultiheadAttention *>(host.get());
    ASSERT_NE(attn, nullptr);
    EXPECT_TRUE(attn->center_score_delta);
}

namespace {

std::vector<float> device_vec(const float *d_src, size_t n) {
    std::vector<float> h_dst(n);
    cudaMemcpy(h_dst.data(), d_src, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h_dst;
}

constexpr int SCORE_STAGE_HEADS = 8;

struct ScoreStageCase {
    int batch_size, seq_len, head_dim;
    bool use_rope, use_causal_mask, center_score_delta;

    std::string str() const {
        return "B=" + std::to_string(batch_size) +
               " T=" + std::to_string(seq_len) +
               " head_dim=" + std::to_string(head_dim) +
               " rope=" + std::to_string(use_rope) +
               " mask=" + std::to_string(use_causal_mask) +
               " center=" + std::to_string(center_score_delta);
    }
};

std::vector<ScoreStageCase> score_stage_cases() {
    std::vector<ScoreStageCase> cases;
    for (int batch_size : {1, 16})
        for (int seq_len : {1, 64, 65})
            for (int head_dim : {96, 64, 48, 32})
                for (bool use_rope : {false, true})
                    for (bool use_causal_mask : {false, true})
                        for (bool center_score_delta : {false, true})
                            cases.push_back({batch_size, seq_len, head_dim,
                                             use_rope, use_causal_mask,
                                             center_score_delta});
    return cases;
}

// Host copies of what the CUDA score stages read and wrote during one
// forward + backward of MultiheadAttentionCuda.
struct ScoreStageIO {
    std::vector<float> mu_q, var_q, mu_k, var_k;
    std::vector<float> mu_qk, var_qk;
    std::vector<float> mu_att, mu_v, delta_mu, delta_var;
    std::vector<float> delta_mu_s, delta_var_s;
};

ScoreStageIO read_score_stage_io(const MultiheadAttentionCuda &layer,
                                 const ScoreStageCase &c) {
    size_t comp_size =
        (size_t)c.batch_size * SCORE_STAGE_HEADS * c.seq_len * c.head_dim;
    size_t qk_size =
        (size_t)c.batch_size * SCORE_STAGE_HEADS * c.seq_len * c.seq_len;
    const AttentionStatesCuda &s = layer.attn_states;
    const AttentionDeltaStatesCuda &d = layer.attn_delta_states;
    ScoreStageIO io;
    io.mu_q = device_vec(c.use_rope ? s.d_mu_q_rope : s.d_mu_q, comp_size);
    io.var_q = device_vec(c.use_rope ? s.d_var_q_rope : s.d_var_q, comp_size);
    io.mu_k = device_vec(c.use_rope ? s.d_mu_k_rope : s.d_mu_k, comp_size);
    io.var_k = device_vec(c.use_rope ? s.d_var_k_rope : s.d_var_k, comp_size);
    io.mu_qk = device_vec(s.d_mu_qk, qk_size);
    io.var_qk = device_vec(s.d_var_qk, qk_size);
    io.mu_att = device_vec(s.d_mu_att_score, qk_size);
    io.mu_v = device_vec(s.d_mu_v, comp_size);
    io.delta_mu = device_vec(d.d_delta_mu_buffer, comp_size);
    io.delta_var = device_vec(d.d_delta_var_buffer, comp_size);
    io.delta_mu_s = device_vec(d.d_delta_mu_att_score, qk_size);
    io.delta_var_s = device_vec(d.d_delta_var_att_score, qk_size);
    return io;
}

struct ScoreStageRun {
    ScoreStageCase c;
    AttnConfig cfg;
    AttnInputs in;
    std::unique_ptr<BaseLayer> layer;
    MultiheadAttentionCuda *attn = nullptr;
    std::unique_ptr<HiddenStateCuda> cu_in, cu_out;
    std::unique_ptr<DeltaStateCuda> cu_din, cu_dout;
    BaseTempStates cu_temp;

    explicit ScoreStageRun(const ScoreStageCase &c) : c(c) {
        cfg.batch_size = c.batch_size;
        cfg.seq_len = c.seq_len;
        cfg.embed_dim = SCORE_STAGE_HEADS * c.head_dim;
        cfg.num_heads = SCORE_STAGE_HEADS;
        cfg.use_rope = c.use_rope;
        cfg.use_causal_mask = c.use_causal_mask;
        cfg.center_score_delta = c.center_score_delta;
        in = make_inputs(cfg, /*seed=*/321);
        layer = make_cpu_layer(cfg)->to_cuda(0);
        attn = dynamic_cast<MultiheadAttentionCuda *>(layer.get());
        attn->training = true;
        attn->param_update = true;
        cu_in = std::make_unique<HiddenStateCuda>(in.total, cfg.batch_size);
        cu_out = std::make_unique<HiddenStateCuda>(in.total, cfg.batch_size);
        cu_din = std::make_unique<DeltaStateCuda>(in.total, cfg.batch_size);
        cu_dout = std::make_unique<DeltaStateCuda>(in.total, cfg.batch_size);
        fill_cuda_states(in, cfg, *cu_in, *cu_out, *cu_din, *cu_dout);
    }

    ScoreStageIO forward_backward() {
        attn->forward(*cu_in, *cu_out, cu_temp);
        attn->backward(*cu_din, *cu_dout, cu_temp, /*state_udapte=*/true);
        return read_score_stage_io(*attn, c);
    }
};

}  // namespace

TEST(MultiheadAttentionCuda, ScoresMatchCPUStageFunction) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const ScoreStageCase &c : score_stage_cases()) {
        SCOPED_TRACE(c.str());
        ScoreStageRun run(c);
        ScoreStageIO io = run.forward_backward();
        std::vector<float> mu_qk(io.mu_qk.size()), var_qk(io.var_qk.size());
        query_key(io.mu_q, io.var_q, io.mu_k, io.var_k, c.batch_size,
                  SCORE_STAGE_HEADS, c.seq_len, c.head_dim, mu_qk, var_qk);
        expect_close(mu_qk, io.mu_qk, "mu_qk");
        expect_close(var_qk, io.var_qk, "var_qk");
    }
}

TEST(MultiheadAttentionCuda, DeltaScoresMatchCPUStageFunction) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const ScoreStageCase &c : score_stage_cases()) {
        SCOPED_TRACE(c.str());
        ScoreStageRun run(c);
        ScoreStageIO io = run.forward_backward();
        std::vector<float> delta_mu_s(io.delta_mu_s.size());
        std::vector<float> delta_var_s(io.delta_var_s.size());
        mha_delta_score(io.mu_v, io.delta_mu, io.delta_var, c.batch_size,
                        SCORE_STAGE_HEADS, c.seq_len, c.head_dim, delta_mu_s,
                        delta_var_s);
        if (c.center_score_delta) {
            center_delta_score(io.mu_att, c.batch_size, SCORE_STAGE_HEADS,
                               c.seq_len, delta_mu_s);
        }
        expect_close(delta_mu_s, io.delta_mu_s, "delta_mu_s");
        expect_close(delta_var_s, io.delta_var_s, "delta_var_s");
    }
}

TEST(MultiheadAttentionCuda, ScoreStagesAreDeterministic) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const ScoreStageCase &c : score_stage_cases()) {
        SCOPED_TRACE(c.str());
        ScoreStageRun run(c);
        ScoreStageIO first = run.forward_backward();
        ScoreStageIO second = run.forward_backward();
        expect_bit_equal(first.mu_qk, second.mu_qk, "mu_qk");
        expect_bit_equal(first.var_qk, second.var_qk, "var_qk");
        expect_bit_equal(first.delta_mu_s, second.delta_mu_s, "delta_mu_s");
        expect_bit_equal(first.delta_var_s, second.delta_var_s, "delta_var_s");
    }
}

#endif  // USE_CUDA

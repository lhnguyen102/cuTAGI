#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <random>
#include <set>
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
    int tokens() const { return batch_size * input_size; }
};

enum class TokenPattern { Random, Repeated, Distinct };

struct EmbCase {
    EmbConfig cfg;
    std::vector<float> tokens, delta_mu, delta_var;

    EmbCase(const EmbConfig &cfg, TokenPattern pattern, unsigned seed)
        : cfg(cfg) {
        int input_total = cfg.tokens();
        int output_total = input_total * cfg.embedding_dim;
        std::default_random_engine gen(seed);
        std::uniform_int_distribution<int> cat_dist(0, cfg.num_embeddings - 1);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        tokens.resize(input_total);
        for (int i = 0; i < input_total; i++) {
            switch (pattern) {
                case TokenPattern::Random:
                    tokens[i] = static_cast<float>(cat_dist(gen));
                    break;
                case TokenPattern::Repeated:
                    tokens[i] = 3.0f;
                    break;
                case TokenPattern::Distinct:
                    tokens[i] = static_cast<float>(i % cfg.num_embeddings);
                    break;
            }
        }
        delta_mu.resize(output_total);
        delta_var.resize(output_total);
        for (int i = 0; i < output_total; i++) {
            delta_mu[i] = 0.05f * nd(gen);
            delta_var[i] = 0.05f + 0.05f * std::abs(nd(gen));
        }
    }

    std::vector<int> touched_rows() const {
        std::set<int> rows;
        for (float t : tokens) {
            int row = static_cast<int>(t);
            if (row != cfg.padding_idx) rows.insert(row);
        }
        return {rows.begin(), rows.end()};
    }

    std::vector<int> untouched_rows() const {
        std::vector<bool> touched(cfg.num_embeddings, false);
        for (int row : touched_rows()) touched[row] = true;
        std::vector<int> rows;
        for (int row = 0; row < cfg.num_embeddings; row++) {
            if (!touched[row]) rows.push_back(row);
        }
        return rows;
    }
};

// CPU layer run forward + backward on the case, and the CUDA copy of it
// with device states ready for the same forward + backward.
struct EmbFixture {
    const EmbCase &emb_case;
    Embedding cpu;
    BaseHiddenStates h_in, h_out;
    std::unique_ptr<BaseLayer> cuda_layer;
    EmbeddingCuda *cuda = nullptr;
    HiddenStateCuda cu_in, cu_out;
    DeltaStateCuda cu_din, cu_dout;
    BaseTempStates temp;

    explicit EmbFixture(const EmbCase &emb_case)
        : emb_case(emb_case),
          cpu(emb_case.cfg.num_embeddings, emb_case.cfg.embedding_dim,
              emb_case.cfg.input_size, emb_case.cfg.scale,
              emb_case.cfg.padding_idx),
          cu_in(emb_case.cfg.tokens(), emb_case.cfg.batch_size),
          cu_out(emb_case.cfg.tokens() * emb_case.cfg.embedding_dim,
                 emb_case.cfg.batch_size),
          cu_din(emb_case.cfg.tokens() * emb_case.cfg.embedding_dim,
                 emb_case.cfg.batch_size),
          cu_dout(emb_case.cfg.tokens() * emb_case.cfg.embedding_dim,
                  emb_case.cfg.batch_size) {
        int input_total = emb_case.cfg.tokens();
        int output_total = input_total * emb_case.cfg.embedding_dim;

        h_in.mu_a = emb_case.tokens;
        h_in.var_a.assign(input_total, 0.0f);
        h_in.jcb.assign(input_total, 1.0f);
        h_in.size = input_total;
        h_in.block_size = emb_case.cfg.batch_size;
        h_in.actual_size = emb_case.cfg.input_size;
        h_out.mu_a.assign(output_total, 0.0f);
        h_out.var_a.assign(output_total, 0.0f);
        h_out.jcb.assign(output_total, 0.0f);
        h_out.size = output_total;

        BaseDeltaStates d_in, d_out;
        d_in.delta_mu = emb_case.delta_mu;
        d_in.delta_var = emb_case.delta_var;
        d_in.size = output_total;
        d_in.block_size = emb_case.cfg.batch_size;
        d_in.actual_size = emb_case.cfg.embedding_dim;
        d_in.seq_len = emb_case.cfg.input_size;
        cpu.forward(h_in, h_out, temp);
        cpu.backward(d_in, d_out, temp, /*state_udapte=*/false);

        cuda_layer = cpu.to_cuda(0);
        cuda = dynamic_cast<EmbeddingCuda *>(cuda_layer.get());
        cu_in.mu_a = emb_case.tokens;
        cu_in.block_size = emb_case.cfg.batch_size;
        cu_in.actual_size = emb_case.cfg.input_size;
        cu_in.to_device();
        cu_din.delta_mu = emb_case.delta_mu;
        cu_din.delta_var = emb_case.delta_var;
        cu_din.block_size = emb_case.cfg.batch_size;
        cu_din.actual_size = emb_case.cfg.embedding_dim;
        cu_din.seq_len = emb_case.cfg.input_size;
        cu_din.to_device();
    }

    void forward() { cuda->forward(cu_in, cu_out, temp); }
    void backward() {
        cuda->backward(cu_din, cu_dout, temp, /*state_udapte=*/false);
    }
};

void expect_rows_bit_equal(const std::vector<float> &a,
                           const std::vector<float> &b,
                           const std::vector<int> &rows, int dim,
                           const char *name) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (int row : rows) {
        size_t start = static_cast<size_t>(row) * dim;
        EXPECT_EQ(std::memcmp(&a[start], &b[start], dim * sizeof(float)), 0)
            << name << " differs on row " << row;
    }
}

void expect_cuda_matches_cpu(const EmbConfig &cfg, unsigned seed,
                             TokenPattern pattern = TokenPattern::Random) {
    EmbCase emb_case(cfg, pattern, seed);
    EmbFixture fixture(emb_case);
    ASSERT_NE(fixture.cuda, nullptr);
    fixture.forward();
    fixture.backward();
    fixture.cu_out.to_host();
    fixture.cuda->delta_params_to_host();

    EXPECT_EQ(fixture.cu_out.seq_len, fixture.h_out.seq_len);
    ASSERT_EQ(fixture.cu_out.mu_a.size(), fixture.h_out.mu_a.size());
    for (size_t i = 0; i < fixture.h_out.mu_a.size(); i++) {
        EXPECT_NEAR(fixture.h_out.mu_a[i], fixture.cu_out.mu_a[i], 1e-5f)
            << "fwd mu_a " << i;
        EXPECT_NEAR(fixture.h_out.var_a[i], fixture.cu_out.var_a[i], 1e-5f)
            << "fwd var_a " << i;
    }
    std::vector<int> rows = emb_case.touched_rows();
    expect_rows_bit_equal(fixture.cpu.delta_mu_w, fixture.cuda->delta_mu_w,
                          rows, cfg.embedding_dim, "delta_mu_w");
    expect_rows_bit_equal(fixture.cpu.delta_var_w, fixture.cuda->delta_var_w,
                          rows, cfg.embedding_dim, "delta_var_w");

    std::vector<float> first_mu = fixture.cuda->delta_mu_w;
    std::vector<float> first_var = fixture.cuda->delta_var_w;
    fixture.backward();
    fixture.cuda->delta_params_to_host();
    expect_rows_bit_equal(fixture.cuda->delta_mu_w, first_mu, rows,
                          cfg.embedding_dim, "second backward delta_mu_w");
    expect_rows_bit_equal(fixture.cuda->delta_var_w, first_var, rows,
                          cfg.embedding_dim, "second backward delta_var_w");
}

// Step reference: the dense update kernel on the same deltas, and the CPU
// var decay on the same var_w.
void expect_step_touches_rows_only(const EmbConfig &cfg, TokenPattern pattern,
                                   unsigned seed) {
    EmbCase emb_case(cfg, pattern, seed);
    EmbFixture fixture(emb_case);
    ASSERT_NE(fixture.cuda, nullptr);
    std::vector<int> touched = emb_case.touched_rows();
    std::vector<int> untouched = emb_case.untouched_rows();
    fixture.forward();
    fixture.backward();

    fixture.cuda->BaseLayerCuda::update_weights();
    fixture.cuda->params_to_host();
    std::vector<float> dense_mu = fixture.cuda->mu_w;
    std::vector<float> dense_var = fixture.cuda->var_w;

    fixture.cuda->mu_w = fixture.cpu.mu_w;
    fixture.cuda->var_w = fixture.cpu.var_w;
    static_cast<BaseLayerCuda *>(fixture.cuda)->params_to_device();
    fixture.cuda->update_weights();
    fixture.cuda->params_to_host();
    expect_rows_bit_equal(fixture.cuda->mu_w, dense_mu, touched,
                          cfg.embedding_dim, "touched mu_w");
    expect_rows_bit_equal(fixture.cuda->var_w, dense_var, touched,
                          cfg.embedding_dim, "touched var_w");
    expect_rows_bit_equal(fixture.cuda->mu_w, fixture.cpu.mu_w, untouched,
                          cfg.embedding_dim, "untouched mu_w");
    expect_rows_bit_equal(fixture.cuda->var_w, fixture.cpu.var_w, untouched,
                          cfg.embedding_dim, "untouched var_w");

    const float tau = 10.0f;
    fixture.cuda->set_var_decay(tau);
    fixture.cuda->apply_var_decay();
    fixture.cuda->params_to_host();
    fixture.cpu.set_var_decay(tau);
    fixture.cpu.apply_var_decay();
    expect_rows_bit_equal(fixture.cuda->mu_w, fixture.cpu.mu_w, untouched,
                          cfg.embedding_dim, "untouched mu_w after decay");
    expect_rows_bit_equal(fixture.cuda->var_w, fixture.cpu.var_w, untouched,
                          cfg.embedding_dim, "untouched var_w after decay");
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

// Every token hits the same row, so the per-row sum order decides the bits.
TEST(EmbeddingCuda, BackwardIsDeterministicOnRepeatedTokens) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    cfg.batch_size = 16;
    cfg.input_size = 64;
    cfg.embedding_dim = 64;
    EmbCase emb_case(cfg, TokenPattern::Repeated, /*seed=*/303);
    EmbFixture fixture(emb_case);
    ASSERT_NE(fixture.cuda, nullptr);
    fixture.forward();
    std::vector<int> rows = emb_case.touched_rows();
    ASSERT_EQ(rows.size(), 1u);

    fixture.backward();
    fixture.cuda->delta_params_to_host();
    std::vector<float> first_mu = fixture.cuda->delta_mu_w;
    std::vector<float> first_var = fixture.cuda->delta_var_w;
    fixture.backward();
    fixture.cuda->delta_params_to_host();

    expect_rows_bit_equal(fixture.cuda->delta_mu_w, first_mu, rows,
                          cfg.embedding_dim, "delta_mu_w");
    expect_rows_bit_equal(fixture.cuda->delta_var_w, first_var, rows,
                          cfg.embedding_dim, "delta_var_w");
}

TEST(EmbeddingCuda, StepUpdatesTouchedRowsOnlyAndMatchesDense) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_step_touches_rows_only(EmbConfig(), TokenPattern::Random,
                                  /*seed=*/404);
}

TEST(EmbeddingCuda, StepWithPaddingLeavesPaddingRowAlone) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EmbConfig cfg;
    cfg.padding_idx = 3;
    EmbCase emb_case(cfg, TokenPattern::Repeated, /*seed=*/606);
    ASSERT_TRUE(emb_case.touched_rows().empty());
    EmbFixture fixture(emb_case);
    ASSERT_NE(fixture.cuda, nullptr);
    fixture.forward();
    fixture.backward();
    fixture.cuda->update_weights();
    fixture.cuda->params_to_host();
    expect_rows_bit_equal(fixture.cuda->mu_w, fixture.cpu.mu_w,
                          emb_case.untouched_rows(), cfg.embedding_dim, "mu_w");
    expect_rows_bit_equal(fixture.cuda->var_w, fixture.cpu.var_w,
                          emb_case.untouched_rows(), cfg.embedding_dim,
                          "var_w");
}

TEST(EmbeddingCuda, ShapesAndTokenPatternsMatchCPUAndStepDense) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    unsigned seed = 700;
    for (int batch_size : {1, 16}) {
        for (int input_size : {1, 64}) {
            for (auto pattern : {TokenPattern::Repeated, TokenPattern::Distinct,
                                 TokenPattern::Random}) {
                EmbConfig cfg;
                cfg.batch_size = batch_size;
                cfg.input_size = input_size;
                cfg.num_embeddings = 1100;
                cfg.embedding_dim = 48;
                SCOPED_TRACE("B=" + std::to_string(batch_size) +
                             " T=" + std::to_string(input_size) +
                             " pattern=" + std::to_string(int(pattern)));
                expect_cuda_matches_cpu(cfg, seed, pattern);
                expect_step_touches_rows_only(cfg, pattern, seed);
                seed++;
            }
        }
    }
}

#endif  // USE_CUDA

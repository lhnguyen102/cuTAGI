#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "../../include/linear_layer.h"
#include "test_utils.h"

#ifdef USE_CUDA
#include "../../include/data_struct_cuda.cuh"
#include "../../include/linear_layer_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

namespace {
struct ReadoutConfig {
    int batch_size = 2;
    int seq_len = 3;
    int input_size = 768;
    int output_size = 1024;
    int n_obs = 16;
    unsigned seed = 7;
    int tokens() const { return batch_size * seq_len; }
};

struct ReadoutCase {
    ReadoutConfig cfg;
    std::vector<float> mu_a, var_a;
    std::vector<int> idx;
    std::vector<float> delta_mu, delta_var;

    explicit ReadoutCase(const ReadoutConfig &c) : cfg(c) {
        int n_in = cfg.tokens() * cfg.input_size;
        int n_pairs = cfg.tokens() * cfg.n_obs;
        std::default_random_engine gen(cfg.seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::uniform_int_distribution<int> unit(2, cfg.output_size);
        mu_a.resize(n_in);
        var_a.resize(n_in);
        for (int i = 0; i < n_in; i++) {
            mu_a[i] = nd(gen);
            var_a[i] = 0.1f + 0.1f * std::abs(nd(gen));
        }
        idx.resize(n_pairs);
        for (int t = 0; t < cfg.tokens(); t++) {
            idx[t * cfg.n_obs] = 1;
            for (int k = 1; k < cfg.n_obs; k++) {
                int u;
                do {
                    u = unit(gen);
                } while (std::find(idx.begin() + t * cfg.n_obs,
                                   idx.begin() + t * cfg.n_obs + k,
                                   u) != idx.begin() + t * cfg.n_obs + k);
                idx[t * cfg.n_obs + k] = u;
            }
        }
        delta_mu.resize(n_pairs);
        delta_var.resize(n_pairs);
        for (int i = 0; i < n_pairs; i++) {
            delta_mu[i] = 0.5f * nd(gen);
            delta_var[i] = -0.1f * std::abs(nd(gen));
        }
    }

    void dense_delta(std::vector<float> &dmu, std::vector<float> &dvar) const {
        dmu.assign(static_cast<size_t>(cfg.tokens()) * cfg.output_size, 0.0f);
        dvar.assign(dmu.size(), 0.0f);
        for (int t = 0; t < cfg.tokens(); t++) {
            for (int k = 0; k < cfg.n_obs; k++) {
                int p = t * cfg.n_obs + k;
                size_t d =
                    static_cast<size_t>(t) * cfg.output_size + idx[p] - 1;
                dmu[d] = delta_mu[p];
                dvar[d] = delta_var[p];
            }
        }
    }
};

struct ReadoutFixture {
    std::unique_ptr<BaseLayer> owner;
    LinearCuda *layer;
    HiddenStateCuda input, output;
    DeltaStateCuda delta_in, delta_out;
    BaseTempStates temp;

    ReadoutFixture(const ReadoutCase &c)
        : input(static_cast<size_t>(c.cfg.tokens()) * c.cfg.input_size,
                c.cfg.batch_size),
          output(static_cast<size_t>(c.cfg.tokens()) * c.cfg.output_size,
                 c.cfg.batch_size),
          delta_in(static_cast<size_t>(c.cfg.tokens()) * c.cfg.output_size,
                   c.cfg.batch_size),
          delta_out(static_cast<size_t>(c.cfg.tokens()) * c.cfg.input_size,
                    c.cfg.batch_size) {
        Linear cpu(c.cfg.input_size, c.cfg.output_size, true);
        owner = cpu.to_cuda(0);
        layer = dynamic_cast<LinearCuda *>(owner.get());
        // The sparse kernels mirror the sum order of the tiled dense kernels,
        // not of cuBLAS.
        layer->gemm.use_cublas = false;
        input.mu_a = c.mu_a;
        input.var_a = c.var_a;
        input.block_size = c.cfg.batch_size;
        input.seq_len = c.cfg.seq_len;
        input.actual_size = c.cfg.input_size;
        input.to_device();
        delta_in.block_size = c.cfg.batch_size;
        delta_in.seq_len = c.cfg.seq_len;
        delta_out.block_size = c.cfg.batch_size;
        delta_out.seq_len = c.cfg.seq_len;
    }

    void forward() { layer->forward(input, output, temp); }

    void backward(const std::vector<float> &dmu,
                  const std::vector<float> &dvar) {
        std::copy(dmu.begin(), dmu.end(), delta_in.delta_mu.begin());
        std::copy(dvar.begin(), dvar.end(), delta_in.delta_var.begin());
        delta_in.to_device();
        layer->backward(delta_in, delta_out, temp, true);
    }
};

void expect_sparse_forward_matches_dense(const ReadoutConfig &cfg) {
    ReadoutCase c(cfg);
    ReadoutFixture r(c);

    r.forward();
    r.output.to_host();
    ASSERT_EQ(r.output.actual_size, static_cast<size_t>(cfg.output_size));
    std::vector<float> gathered_mu(c.idx.size()), gathered_var(c.idx.size());
    for (int t = 0; t < cfg.tokens(); t++) {
        for (int k = 0; k < cfg.n_obs; k++) {
            int p = t * cfg.n_obs + k;
            size_t d = static_cast<size_t>(t) * cfg.output_size + c.idx[p] - 1;
            gathered_mu[p] = r.output.mu_a[d];
            gathered_var[p] = r.output.var_a[d];
        }
    }

    r.layer->set_output_indices(c.idx, cfg.n_obs);
    r.forward();
    r.output.to_host();
    ASSERT_EQ(r.output.actual_size, static_cast<size_t>(cfg.n_obs));
    std::vector<float> sparse_mu(r.output.mu_a.begin(),
                                 r.output.mu_a.begin() + c.idx.size());
    std::vector<float> sparse_var(r.output.var_a.begin(),
                                  r.output.var_a.begin() + c.idx.size());
    expect_bit_equal(sparse_mu, gathered_mu, "forward mu");
    expect_bit_equal(sparse_var, gathered_var, "forward var");
}

struct Deltas {
    std::vector<float> dz_mu, dz_var, dw_mu, dw_var, db_mu, db_var;
};

Deltas read_deltas(ReadoutFixture &r, size_t num_dz) {
    r.delta_out.to_host();
    r.layer->delta_params_to_host();
    Deltas d;
    d.dz_mu.assign(r.delta_out.delta_mu.begin(),
                   r.delta_out.delta_mu.begin() + num_dz);
    d.dz_var.assign(r.delta_out.delta_var.begin(),
                    r.delta_out.delta_var.begin() + num_dz);
    d.dw_mu = r.layer->delta_mu_w;
    d.dw_var = r.layer->delta_var_w;
    d.db_mu = r.layer->delta_mu_b;
    d.db_var = r.layer->delta_var_b;
    return d;
}

void expect_sparse_backward_matches_dense(const ReadoutConfig &cfg) {
    ReadoutCase c(cfg);
    ReadoutFixture r(c);
    size_t num_dz = static_cast<size_t>(cfg.tokens()) * cfg.input_size;

    std::vector<float> dense_mu, dense_var;
    c.dense_delta(dense_mu, dense_var);
    r.forward();
    r.backward(dense_mu, dense_var);
    Deltas dense = read_deltas(r, num_dz);

    r.layer->set_output_indices(c.idx, cfg.n_obs);
    r.forward();
    r.backward(c.delta_mu, c.delta_var);
    Deltas sparse = read_deltas(r, num_dz);

    expect_bit_equal(sparse.dz_mu, dense.dz_mu, "delta_mu z");
    expect_bit_equal(sparse.dz_var, dense.dz_var, "delta_var z");
    expect_bit_equal(sparse.dw_mu, dense.dw_mu, "delta_mu_w");
    expect_bit_equal(sparse.dw_var, dense.dw_var, "delta_var_w");
    expect_bit_equal(sparse.db_mu, dense.db_mu, "delta_mu_b");
    expect_bit_equal(sparse.db_var, dense.db_var, "delta_var_b");
}

struct Params {
    std::vector<float> mu_w, var_w, mu_b, var_b;
};

Params read_params(ReadoutFixture &r) {
    r.layer->params_to_host();
    return {r.layer->mu_w, r.layer->var_w, r.layer->mu_b, r.layer->var_b};
}

void expect_sparse_step_matches_dense(const ReadoutConfig &cfg) {
    ReadoutCase c(cfg);
    ReadoutFixture r(c);
    Params before = read_params(r);

    std::vector<float> dense_mu, dense_var;
    c.dense_delta(dense_mu, dense_var);
    r.forward();
    r.backward(dense_mu, dense_var);
    r.layer->update_weights();
    r.layer->update_biases();
    Params dense = read_params(r);

    r.layer->mu_w = before.mu_w;
    r.layer->var_w = before.var_w;
    r.layer->mu_b = before.mu_b;
    r.layer->var_b = before.var_b;
    static_cast<BaseLayerCuda *>(r.layer)->params_to_device();
    r.layer->set_output_indices(c.idx, cfg.n_obs);
    r.forward();
    r.backward(c.delta_mu, c.delta_var);
    r.layer->update_weights();
    r.layer->update_biases();
    Params sparse = read_params(r);

    expect_bit_equal(sparse.mu_w, dense.mu_w, "mu_w");
    expect_bit_equal(sparse.var_w, dense.var_w, "var_w");
    expect_bit_equal(sparse.mu_b, dense.mu_b, "mu_b");
    expect_bit_equal(sparse.var_b, dense.var_b, "var_b");

    std::vector<bool> touched(cfg.output_size, false);
    for (int u : c.idx) touched[u - 1] = true;
    int changed_rows = 0, unchanged_rows = 0;
    for (int u = 0; u < cfg.output_size; u++) {
        size_t row = static_cast<size_t>(u) * cfg.input_size;
        bool same = std::memcmp(&sparse.mu_w[row], &before.mu_w[row],
                                cfg.input_size * sizeof(float)) == 0 &&
                    std::memcmp(&sparse.var_w[row], &before.var_w[row],
                                cfg.input_size * sizeof(float)) == 0 &&
                    sparse.mu_b[u] == before.mu_b[u] &&
                    sparse.var_b[u] == before.var_b[u];
        if (touched[u]) {
            changed_rows += !same;
        } else {
            ASSERT_TRUE(same) << "untouched row " << u << " changed";
            unchanged_rows++;
        }
    }
    EXPECT_GT(changed_rows, 0);
    EXPECT_GT(unchanged_rows, 0);
}
}  // namespace

TEST(SparseReadout, ForwardMatchesDenseGatheredAtPathUnits) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_sparse_forward_matches_dense(ReadoutConfig());
}

// 50267 % 4 != 0 selects the dense v3 kernels, the readout of fineweb_gpt2.
TEST(SparseReadout, ForwardMatchesDenseOnVocabWidth) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    ReadoutConfig cfg;
    cfg.batch_size = 16;
    cfg.seq_len = 64;
    cfg.output_size = 50267;
    expect_sparse_forward_matches_dense(cfg);
}

TEST(SparseReadout, SingleTokenMatchesDense) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    ReadoutConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = 1;
    expect_sparse_forward_matches_dense(cfg);
    expect_sparse_backward_matches_dense(cfg);
    expect_sparse_step_matches_dense(cfg);
}

TEST(SparseReadout, BackwardMatchesDenseWithZeroDeltaOutsidePathUnits) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_sparse_backward_matches_dense(ReadoutConfig());
}

TEST(SparseReadout, BackwardMatchesDenseOnVocabWidth) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    ReadoutConfig cfg;
    cfg.batch_size = 16;
    cfg.seq_len = 64;
    cfg.output_size = 50267;
    expect_sparse_backward_matches_dense(cfg);
}

TEST(SparseReadout, StepUpdatesTouchedRowsOnlyAndMatchesDense) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_sparse_step_matches_dense(ReadoutConfig());
}

TEST(SparseReadout, StepMatchesDenseOnVocabWidth) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    ReadoutConfig cfg;
    cfg.batch_size = 16;
    cfg.seq_len = 64;
    cfg.output_size = 50267;
    expect_sparse_step_matches_dense(cfg);
}

TEST(SparseReadout, IndicesAreConsumedByOneForward) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    ReadoutConfig cfg;
    cfg.output_size = 50267;
    ReadoutCase c(cfg);
    ReadoutFixture r(c);
    r.forward();
    r.output.to_host();
    std::vector<float> dense_mu = r.output.mu_a;
    std::vector<float> dense_var = r.output.var_a;

    r.layer->set_output_indices(c.idx, cfg.n_obs);
    r.forward();
    r.forward();
    r.output.to_host();
    ASSERT_EQ(r.output.actual_size, static_cast<size_t>(cfg.output_size));
    expect_bit_equal(r.output.mu_a, dense_mu, "second forward mu");
    expect_bit_equal(r.output.var_a, dense_var, "second forward var");
}

TEST(SparseReadout, BackwardIsDeterministic) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    ReadoutConfig cfg;
    cfg.batch_size = 16;
    cfg.seq_len = 64;
    ReadoutCase c(cfg);
    ReadoutFixture r(c);
    size_t num_dz = static_cast<size_t>(cfg.tokens()) * cfg.input_size;

    r.layer->set_output_indices(c.idx, cfg.n_obs);
    r.forward();
    r.backward(c.delta_mu, c.delta_var);
    Deltas first = read_deltas(r, num_dz);
    r.backward(c.delta_mu, c.delta_var);
    Deltas second = read_deltas(r, num_dz);

    expect_bit_equal(second.dz_mu, first.dz_mu, "delta_mu z");
    expect_bit_equal(second.dz_var, first.dz_var, "delta_var z");
    expect_bit_equal(second.dw_mu, first.dw_mu, "delta_mu_w");
    expect_bit_equal(second.dw_var, first.dw_var, "delta_var_w");
    expect_bit_equal(second.db_mu, first.db_mu, "delta_mu_b");
    expect_bit_equal(second.db_var, first.db_var, "delta_var_b");
}

#endif  // USE_CUDA

TEST(SparseReadout, HostLinearRejectsOutputIndices) {
    Linear cpu(8, 16, true);
    std::vector<int> idx = {1, 2};
    EXPECT_THROW(cpu.set_output_indices(idx, 2), std::runtime_error);
}

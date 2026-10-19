#include <gtest/gtest.h>

#include <algorithm>
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
struct LinearShape {
    int batch_size = 2;
    int seq_len = 32;
    int input_size = 768;
    int output_size = 1024;
    bool bias = true;
    int tokens() const { return batch_size * seq_len; }
};

struct LinearCase {
    LinearShape shape;
    Linear cpu;
    std::vector<float> mu_a, var_a, delta_mu, delta_var;

    explicit LinearCase(const LinearShape &s)
        : shape(s), cpu(s.input_size, s.output_size, s.bias) {
        size_t n_in = static_cast<size_t>(s.tokens()) * s.input_size;
        size_t n_out = static_cast<size_t>(s.tokens()) * s.output_size;
        std::default_random_engine gen(11);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        mu_a.resize(n_in);
        var_a.resize(n_in);
        for (size_t i = 0; i < n_in; i++) {
            mu_a[i] = nd(gen);
            var_a[i] = 0.1f + 0.1f * std::abs(nd(gen));
        }
        delta_mu.resize(n_out);
        delta_var.resize(n_out);
        for (size_t i = 0; i < n_out; i++) {
            delta_mu[i] = 0.05f * nd(gen);
            delta_var[i] = -0.01f * std::abs(nd(gen));
        }
    }
};

struct Outputs {
    std::vector<float> mu_z, var_z, dz_mu, dz_var, dw_mu, dw_var, db_mu, db_var;
};

struct LinearFixture {
    const LinearShape shape;
    std::unique_ptr<BaseLayer> owner;
    LinearCuda *layer;
    HiddenStateCuda in, out;
    DeltaStateCuda din, dout;
    BaseTempStates temp;

    explicit LinearFixture(LinearCase &c)
        : shape(c.shape),
          owner(c.cpu.to_cuda(0)),
          layer(dynamic_cast<LinearCuda *>(owner.get())),
          in(c.mu_a.size(), shape.batch_size),
          out(c.delta_mu.size(), shape.batch_size),
          din(c.delta_mu.size(), shape.batch_size),
          dout(c.mu_a.size(), shape.batch_size) {
        in.mu_a = c.mu_a;
        in.var_a = c.var_a;
        in.jcb.assign(c.mu_a.size(), 1.0f);
        in.block_size = shape.batch_size;
        in.seq_len = shape.seq_len;
        in.actual_size = shape.input_size;
        in.to_device();
        din.delta_mu = c.delta_mu;
        din.delta_var = c.delta_var;
        din.block_size = shape.batch_size;
        din.seq_len = shape.seq_len;
        din.actual_size = shape.output_size;
        din.to_device();
        dout.block_size = shape.batch_size;
        dout.seq_len = shape.seq_len;
    }

    Outputs run(bool use_cublas, bool use_tf32) {
        layer->gemm.use_cublas = use_cublas;
        layer->set_tf32(use_tf32);
        layer->forward(in, out, temp);
        layer->backward(din, dout, temp, true);
        out.to_host();
        dout.to_host();
        layer->delta_params_to_host();
        return {out.mu_a,          out.var_a,         dout.delta_mu,
                dout.delta_var,    layer->delta_mu_w, layer->delta_var_w,
                layer->delta_mu_b, layer->delta_var_b};
    }
};

// The abs floor scales with the largest reference value: a sum whose terms
// cancel keeps the rounding of its partial sums, not of its small result.
void expect_close_scaled(const std::vector<float> &a,
                         const std::vector<float> &b, const char *name,
                         float rel) {
    float scale = 0.0f;
    for (float v : b) scale = std::max(scale, std::abs(v));
    expect_close(a, b, name, rel, rel * scale);
}

// delta_z sums over output_size terms, ten times the longest other sum at the
// vocabulary width, so it gets its own tolerance.
void expect_outputs_close(const Outputs &a, const Outputs &b, float rel,
                          float rel_dz) {
    expect_close_scaled(a.mu_z, b.mu_z, "fwd mu_z", rel);
    expect_close_scaled(a.var_z, b.var_z, "fwd var_z", rel);
    expect_close_scaled(a.dz_mu, b.dz_mu, "delta_mu z", rel_dz);
    expect_close_scaled(a.dz_var, b.dz_var, "delta_var z", rel_dz);
    expect_close_scaled(a.dw_mu, b.dw_mu, "delta_mu_w", rel);
    expect_close_scaled(a.dw_var, b.dw_var, "delta_var_w", rel);
    expect_close_scaled(a.db_mu, b.db_mu, "delta_mu_b", rel);
    expect_close_scaled(a.db_var, b.db_var, "delta_var_b", rel);
}

void expect_outputs_bit_equal(const Outputs &a, const Outputs &b) {
    expect_bit_equal(a.mu_z, b.mu_z, "fwd mu_z");
    expect_bit_equal(a.var_z, b.var_z, "fwd var_z");
    expect_bit_equal(a.dz_mu, b.dz_mu, "delta_mu z");
    expect_bit_equal(a.dz_var, b.dz_var, "delta_var z");
    expect_bit_equal(a.dw_mu, b.dw_mu, "delta_mu_w");
    expect_bit_equal(a.dw_var, b.dw_var, "delta_var_w");
    expect_bit_equal(a.db_mu, b.db_mu, "delta_mu_b");
    expect_bit_equal(a.db_var, b.db_var, "delta_var_b");
}

void expect_cublas_matches_kernels(const LinearShape &shape) {
    LinearCase c(shape);
    LinearFixture r(c);
    ASSERT_NE(r.layer, nullptr);
    Outputs kernels = r.run(false, false);
    Outputs cublas = r.run(true, false);
    float rel_dz = shape.output_size > 1024 ? 1e-4f : 1e-5f;
    expect_outputs_close(cublas, kernels, 1e-5f, rel_dz);
}

Outputs run_cpu(LinearCase &c) {
    const LinearShape &s = c.shape;
    size_t n_in = c.mu_a.size(), n_out = c.delta_mu.size();
    BaseHiddenStates h_in, h_out;
    h_in.mu_a = c.mu_a;
    h_in.var_a = c.var_a;
    h_in.jcb.assign(n_in, 1.0f);
    h_in.size = n_in;
    h_in.block_size = s.batch_size;
    h_in.seq_len = s.seq_len;
    h_in.actual_size = s.input_size;
    h_out.mu_a.assign(n_out, 0.0f);
    h_out.var_a.assign(n_out, 0.0f);
    h_out.jcb.assign(n_out, 0.0f);
    h_out.size = n_out;

    BaseDeltaStates d_in, d_out;
    d_in.delta_mu = c.delta_mu;
    d_in.delta_var = c.delta_var;
    d_in.size = n_out;
    d_in.block_size = s.batch_size;
    d_in.seq_len = s.seq_len;
    d_in.actual_size = s.output_size;
    d_out.delta_mu.assign(n_in, 0.0f);
    d_out.delta_var.assign(n_in, 0.0f);
    d_out.size = n_in;

    BaseTempStates temp;
    c.cpu.set_threads(1);
    c.cpu.forward(h_in, h_out, temp);
    c.cpu.backward(d_in, d_out, temp, true);
    return {h_out.mu_a,       h_out.var_a,      d_out.delta_mu,
            d_out.delta_var,  c.cpu.delta_mu_w, c.cpu.delta_var_w,
            c.cpu.delta_mu_b, c.cpu.delta_var_b};
}

const std::vector<LinearShape> kShapes = {
    {1, 1, 768, 1024},         {2, 32, 768, 1024},  {16, 64, 768, 1024},
    {1, 1, 768, 50267},        {2, 32, 768, 50267}, {16, 64, 768, 50267},
    {1, 1, 1, 1024},           {2, 32, 1, 1024},    {16, 64, 1, 1024},
    {2, 32, 768, 1024, false},
};
}  // namespace

TEST(LinearCuda, CublasMatchesKernels) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const auto &shape : kShapes) {
        SCOPED_TRACE("tokens=" + std::to_string(shape.tokens()) +
                     " in=" + std::to_string(shape.input_size) +
                     " out=" + std::to_string(shape.output_size) +
                     " bias=" + std::to_string(shape.bias));
        expect_cublas_matches_kernels(shape);
    }
}

TEST(LinearCuda, CublasMatchesHostLinear) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    LinearCase c(LinearShape{2, 32, 768, 1024});
    LinearFixture r(c);
    ASSERT_NE(r.layer, nullptr);
    Outputs cublas = r.run(true, false);
    expect_outputs_close(cublas, run_cpu(c), 1e-5f, 1e-5f);
}

TEST(LinearCuda, Tf32MatchesKernelsAt1e3) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const auto &shape : kShapes) {
        SCOPED_TRACE("tokens=" + std::to_string(shape.tokens()) +
                     " in=" + std::to_string(shape.input_size) +
                     " out=" + std::to_string(shape.output_size));
        LinearCase c(shape);
        LinearFixture r(c);
        Outputs kernels = r.run(false, false);
        Outputs tf32 = r.run(true, true);
        expect_outputs_close(tf32, kernels, 1e-3f, 1e-3f);
    }
}

TEST(LinearCuda, Tf32OffRestoresFp32Bits) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    LinearCase c(LinearShape{16, 64, 768, 1024});
    LinearFixture r(c);
    Outputs fp32 = r.run(true, false);
    Outputs tf32 = r.run(true, true);
    Outputs fp32_again = r.run(true, false);
    expect_outputs_bit_equal(fp32_again, fp32);
    EXPECT_NE(tf32.mu_z, fp32.mu_z) << "TF32 flag had no effect";
}

TEST(LinearCuda, CublasIsDeterministic) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    LinearCase c(LinearShape{16, 64, 768, 50267});
    LinearFixture r(c);
    Outputs first = r.run(true, false);
    Outputs second = r.run(true, false);
    expect_outputs_bit_equal(second, first);
}

#endif  // USE_CUDA

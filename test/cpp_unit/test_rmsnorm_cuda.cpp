#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../include/rmsnorm_layer.h"

#ifdef USE_CUDA
#include "../../include/data_struct_cuda.cuh"
#include "../../include/rmsnorm_layer_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

namespace {
// The CUDA reductions sum in a different order than the CPU loop. delta_mu_w
// sums thousands of alternating-sign terms, so its rounding error is absolute
// (up to 5e-7 at 4096 rows), hence the floor on top of rel 1e-5.
void expect_close(const std::vector<float> &a, const std::vector<float> &b,
                  const char *name) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        float tol = 1e-5f * std::max(std::abs(a[i]), std::abs(b[i])) + 1e-6f;
        ASSERT_NEAR(a[i], b[i], tol) << name << " mismatch at index " << i;
    }
}

void expect_bit_equal(const std::vector<float> &a, const std::vector<float> &b,
                      const char *name) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        ASSERT_EQ(std::memcmp(&a[i], &b[i], sizeof(float)), 0)
            << name << " differs at " << i << ": " << a[i] << " vs " << b[i];
    }
}

struct Shape {
    int batch_size, seq_len, ni;
};

// rows {1, 1024, 4096} x width {768, 1024, 50}: one row, the fineweb batch,
// a larger batch; the two model widths and an odd width.
std::vector<Shape> training_shapes() {
    std::vector<Shape> shapes;
    for (int rows : {1, 1024, 4096}) {
        for (int ni : {768, 1024, 50}) {
            int seq_len = rows == 1 ? 1 : 64;
            shapes.push_back({rows / seq_len, seq_len, ni});
        }
    }
    return shapes;
}

std::string shape_trace(const Shape &s) {
    return "batch_size=" + std::to_string(s.batch_size) +
           " seq_len=" + std::to_string(s.seq_len) +
           " ni=" + std::to_string(s.ni);
}

struct RMSNormCase {
    int batch_size, seq_len, ni;
    std::vector<float> mu_a, var_a, delta_mu, delta_var;
    RMSNorm cpu;

    RMSNormCase(int batch_size, int seq_len, int ni)
        : batch_size(batch_size),
          seq_len(seq_len),
          ni(ni),
          cpu({ni}, /*eps=*/1e-6f, /*gain_w=*/1.5f) {
        int n = size();
        mu_a.resize(n);
        var_a.resize(n);
        delta_mu.resize(n);
        delta_var.resize(n);
        std::default_random_engine gen(777);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < n; i++) {
            mu_a[i] = nd(gen);
            var_a[i] = 0.1f + 0.1f * std::abs(nd(gen));
            delta_mu[i] = 0.05f * nd(gen);
            delta_var[i] = 0.05f + 0.05f * std::abs(nd(gen));
        }
        // Per-feature mu_w != 1 and distinct var_w catch a missing mu_w
        // factor and row/column index mix-ups.
        cpu.set_threads(1);
        for (int i = 0; i < ni; i++) {
            cpu.mu_w[i] = 1.0f + 0.5f * nd(gen);
            cpu.var_w[i] = 0.05f + 0.05f * std::abs(nd(gen));
        }
    }

    int size() const { return batch_size * seq_len * ni; }

    void run_cpu(BaseHiddenStates &h_out, BaseDeltaStates &d_out) {
        int n = size();
        BaseHiddenStates h_in;
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

        BaseDeltaStates d_in;
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
    }
};

struct RMSNormFixture {
    std::unique_ptr<BaseLayer> layer;
    RMSNormCuda *rms;
    HiddenStateCuda in, out;
    DeltaStateCuda din, dout;

    explicit RMSNormFixture(RMSNormCase &c)
        : layer(c.cpu.to_cuda(0)),
          rms(dynamic_cast<RMSNormCuda *>(layer.get())),
          in(c.size(), c.batch_size),
          out(c.size(), c.batch_size),
          din(c.size(), c.batch_size),
          dout(c.size(), c.batch_size) {
        in.mu_a = c.mu_a;
        in.var_a = c.var_a;
        in.jcb.assign(c.size(), 1.0f);
        in.block_size = c.batch_size;
        in.actual_size = c.ni;
        in.seq_len = c.seq_len;
        in.to_device();
        din.delta_mu = c.delta_mu;
        din.delta_var = c.delta_var;
        din.block_size = c.batch_size;
        din.actual_size = c.ni;
        din.seq_len = c.seq_len;
        din.to_device();
    }

    void forward_backward() {
        BaseTempStates temp;
        rms->forward(in, out, temp);
        rms->backward(din, dout, temp);
        out.to_host();
        dout.to_host();
        rms->delta_params_to_host();
    }
};

void expect_cuda_matches_cpu(const Shape &shape) {
    RMSNormCase c(shape.batch_size, shape.seq_len, shape.ni);
    BaseHiddenStates h_out;
    BaseDeltaStates d_out;
    c.run_cpu(h_out, d_out);

    RMSNormFixture r(c);
    ASSERT_NE(r.rms, nullptr);
    r.forward_backward();

    expect_close(h_out.mu_a, r.out.mu_a, "fwd mu_a");
    expect_close(h_out.var_a, r.out.var_a, "fwd var_a");
    expect_close(d_out.delta_mu, r.dout.delta_mu, "bwd delta_mu");
    expect_close(d_out.delta_var, r.dout.delta_var, "bwd delta_var");
    expect_close(c.cpu.delta_mu_w, r.rms->delta_mu_w, "delta_mu_w");
    expect_close(c.cpu.delta_var_w, r.rms->delta_var_w, "delta_var_w");
    EXPECT_EQ(d_out.seq_len, shape.seq_len);
    EXPECT_EQ(r.dout.seq_len, shape.seq_len);
}

void expect_two_runs_bit_equal(const Shape &shape) {
    RMSNormCase c(shape.batch_size, shape.seq_len, shape.ni);
    RMSNormFixture r(c);
    ASSERT_NE(r.rms, nullptr);

    r.forward_backward();
    std::vector<float> mu_a = r.out.mu_a, var_a = r.out.var_a;
    std::vector<float> delta_mu = r.dout.delta_mu;
    std::vector<float> delta_var = r.dout.delta_var;
    std::vector<float> delta_mu_w = r.rms->delta_mu_w;
    std::vector<float> delta_var_w = r.rms->delta_var_w;
    r.forward_backward();

    expect_bit_equal(r.out.mu_a, mu_a, "fwd mu_a");
    expect_bit_equal(r.out.var_a, var_a, "fwd var_a");
    expect_bit_equal(r.dout.delta_mu, delta_mu, "bwd delta_mu");
    expect_bit_equal(r.dout.delta_var, delta_var, "bwd delta_var");
    expect_bit_equal(r.rms->delta_mu_w, delta_mu_w, "delta_mu_w");
    expect_bit_equal(r.rms->delta_var_w, delta_var_w, "delta_var_w");
}
}  // namespace

TEST(RMSNormCuda, ForwardBackwardMatchesCPU) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_cuda_matches_cpu({4, 3, 8});
}

TEST(RMSNormCuda, ForwardBackwardMatchesCPUAtTrainingShapes) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const Shape &shape : training_shapes()) {
        SCOPED_TRACE(shape_trace(shape));
        expect_cuda_matches_cpu(shape);
    }
}

TEST(RMSNormCuda, ForwardBackwardIsDeterministic) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (const Shape &shape : training_shapes()) {
        SCOPED_TRACE(shape_trace(shape));
        expect_two_runs_bit_equal(shape);
    }
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

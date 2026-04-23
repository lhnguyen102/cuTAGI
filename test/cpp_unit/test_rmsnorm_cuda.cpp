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
struct RMSConfig {
    int batch_size = 4;
    int seq_len = 3;
    int feature_size = 8;
};

struct RMSInputs {
    std::vector<float> mu_a, var_a, jcb;
    std::vector<float> delta_mu, delta_var;
    int total;
};

RMSInputs make_inputs(const RMSConfig &cfg, unsigned seed) {
    RMSInputs in;
    in.total = cfg.batch_size * cfg.seq_len * cfg.feature_size;
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

void compare_vec(const std::vector<float> &a, const std::vector<float> &b,
                 const char *name, float tol = 1e-4f) {
    ASSERT_EQ(a.size(), b.size()) << name;
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_NEAR(a[i], b[i], tol) << name << " mismatch at index " << i;
    }
}
}  // namespace

TEST(RMSNormCuda, ForwardBackwardMatchesCPU) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    RMSConfig cfg;
    RMSInputs in = make_inputs(cfg, /*seed=*/777);

    std::vector<int> normalized_shape = {cfg.feature_size};

    // CPU layer. Use gain_w != 1 so var_w isn't a "round" value, then
    // overwrite mu_w/var_w with random per-element values so:
    //   - mu_w != 1 (catches "forgot * mu_w" and mu_w vs mu_w^2 mistakes)
    //   - per-element var_w differs (catches index-mixing like var_w[row]
    //     instead of var_w[col])
    auto cpu = std::make_unique<RMSNorm>(normalized_shape, /*eps=*/1e-6f,
                                         /*gain_w=*/1.5f);
    cpu->training = true;
    cpu->param_update = true;
    cpu->set_threads(1);

    std::default_random_engine wgen(/*seed=*/909);
    std::normal_distribution<float> w_dist(0.0f, 0.5f);
    for (size_t i = 0; i < cpu->mu_w.size(); i++) {
        cpu->mu_w[i] = 1.0f + w_dist(wgen);
        cpu->var_w[i] = 0.05f + 0.05f * std::abs(w_dist(wgen));
    }

    BaseHiddenStates h_in, h_out;
    h_in.mu_a = in.mu_a;
    h_in.var_a = in.var_a;
    h_in.jcb = in.jcb;
    h_in.size = in.total;
    h_in.block_size = cfg.batch_size;
    h_in.actual_size = cfg.feature_size;
    h_in.seq_len = cfg.seq_len;

    h_out.mu_a.assign(in.total, 0.0f);
    h_out.var_a.assign(in.total, 0.0f);
    h_out.jcb.assign(in.total, 0.0f);
    h_out.size = in.total;

    BaseDeltaStates d_in, d_out;
    d_in.delta_mu = in.delta_mu;
    d_in.delta_var = in.delta_var;
    d_in.size = in.total;
    d_in.block_size = cfg.batch_size;
    d_in.actual_size = cfg.feature_size * cfg.seq_len;

    d_out.delta_mu.assign(in.total, 0.0f);
    d_out.delta_var.assign(in.total, 0.0f);
    d_out.size = in.total;
    d_out.block_size = cfg.batch_size;
    d_out.actual_size = cfg.feature_size;
    d_out.seq_len = cfg.seq_len;

    BaseTempStates h_temp;
    cpu->forward(h_in, h_out, h_temp);
    cpu->backward(d_in, d_out, h_temp, /*state_udapte=*/true);

    // CUDA layer with the same weights (to_cuda copies them).
    auto cuda_layer = cpu->to_cuda(0);
    auto *cuda_rms = dynamic_cast<RMSNormCuda *>(cuda_layer.get());
    ASSERT_NE(cuda_rms, nullptr);
    cuda_rms->training = true;
    cuda_rms->param_update = true;

    int n = in.total;
    HiddenStateCuda cu_in(n, cfg.batch_size), cu_out(n, cfg.batch_size);
    DeltaStateCuda cu_din(n, cfg.batch_size), cu_dout(n, cfg.batch_size);

    cu_in.mu_a = in.mu_a;
    cu_in.var_a = in.var_a;
    cu_in.jcb = in.jcb;
    cu_in.block_size = cfg.batch_size;
    cu_in.actual_size = cfg.feature_size;
    cu_in.seq_len = cfg.seq_len;
    cu_in.to_device();

    cu_out.block_size = cfg.batch_size;
    cu_out.actual_size = cfg.feature_size;
    cu_out.seq_len = cfg.seq_len;

    cu_din.delta_mu = in.delta_mu;
    cu_din.delta_var = in.delta_var;
    cu_din.block_size = cfg.batch_size;
    cu_din.actual_size = cfg.feature_size * cfg.seq_len;
    cu_din.to_device();

    cu_dout.block_size = cfg.batch_size;
    cu_dout.actual_size = cfg.feature_size;
    cu_dout.seq_len = cfg.seq_len;

    BaseTempStates cu_temp;
    cuda_rms->forward(cu_in, cu_out, cu_temp);
    cuda_rms->backward(cu_din, cu_dout, cu_temp, /*state_udapte=*/true);

    cu_out.to_host();
    cu_dout.to_host();
    cuda_rms->delta_params_to_host();

    compare_vec(h_out.mu_a, cu_out.mu_a, "fwd mu_a");
    compare_vec(h_out.var_a, cu_out.var_a, "fwd var_a");
    compare_vec(d_out.delta_mu, cu_dout.delta_mu, "bwd delta_mu");
    compare_vec(d_out.delta_var, cu_dout.delta_var, "bwd delta_var");
    compare_vec(cpu->delta_mu_w, cuda_rms->delta_mu_w, "delta_mu_w");
    compare_vec(cpu->delta_var_w, cuda_rms->delta_var_w, "delta_var_w");
}

#endif  // USE_CUDA

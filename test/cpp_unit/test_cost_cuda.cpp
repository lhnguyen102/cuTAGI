#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/cost.h"

#ifdef USE_CUDA
#include "../../include/cost_cuda.cuh"
#endif

extern bool g_gpu_enabled;

#ifdef USE_CUDA

namespace {
void expect_cuda_matches_cpu(int rows, int n_classes, bool calibrated) {
    HRCSoftmax hs = class_to_obs(n_classes);
    std::default_random_engine gen(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> mz(static_cast<size_t>(rows) * hs.len);
    std::vector<float> Sz(mz.size());
    for (size_t i = 0; i < mz.size(); i++) {
        mz[i] = 2.0f * nd(gen);
        Sz[i] = 0.1f + 0.5f * std::abs(nd(gen));
    }
    std::vector<float> level_scale, level_offset;
    if (calibrated) {
        for (int k = 0; k < hs.n_obs; k++) {
            level_scale.push_back(0.5f + 0.1f * k);
            level_offset.push_back(-0.3f + 0.05f * k);
        }
    }

    std::vector<float> P_cuda(static_cast<size_t>(rows) * n_classes);
    obs_to_class_cuda(mz.data(), Sz.data(), rows, hs, n_classes, level_scale,
                      level_offset, P_cuda.data());

    std::vector<float> row_mz(hs.len), row_Sz(hs.len);
    for (int r = 0; r < rows; r++) {
        std::copy(mz.begin() + static_cast<size_t>(r) * hs.len,
                  mz.begin() + static_cast<size_t>(r + 1) * hs.len,
                  row_mz.begin());
        std::copy(Sz.begin() + static_cast<size_t>(r) * hs.len,
                  Sz.begin() + static_cast<size_t>(r + 1) * hs.len,
                  row_Sz.begin());
        auto P_cpu = obs_to_class(row_mz, row_Sz, hs, n_classes, level_scale,
                                  level_offset);
        // Nodes near saturation make |P_z - 1| an ulp-level quantity on
        // both devices, so tiny class probabilities only agree absolutely.
        for (int c = 0; c < n_classes; c++) {
            float got = P_cuda[static_cast<size_t>(r) * n_classes + c];
            ASSERT_NEAR(got, P_cpu[c], 1e-6f * P_cpu[c] + 2e-7f)
                << "row " << r << " class " << c;
        }
    }
}
}  // namespace

TEST(CostCuda, ObsToClassMatchesCpu) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    for (int rows : {1, 64, 256}) {
        for (int n_classes : {10, 50257}) {
            expect_cuda_matches_cpu(rows, n_classes, false);
            expect_cuda_matches_cpu(rows, n_classes, true);
        }
    }
}

#endif

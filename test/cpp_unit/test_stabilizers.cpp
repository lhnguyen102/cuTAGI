#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../../include/base_output_updater.h"
#include "../../include/linear_layer.h"
#include "../../include/sequential.h"

extern bool g_gpu_enabled;

namespace {

const int INPUT_SIZE = 4;
const int OUTPUT_SIZE = 3;
const int BATCH_SIZE = 8;

std::vector<ParameterTuple> host_parameters(Sequential &model) {
#ifdef USE_CUDA
    if (model.device == "cuda") model.params_to_host();
#endif
    return model.parameters();
}

// Targets far from the prior predictions so the raw update exceeds the clip.
// A huge var_obs makes the Kalman update vanish, leaving only the var decay.
void train_one_step(Sequential &model, float var_obs = 0.01f) {
    std::vector<float> x(BATCH_SIZE * INPUT_SIZE);
    std::vector<float> y(BATCH_SIZE * OUTPUT_SIZE);
    for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (i % 7) - 0.3f;
    for (size_t i = 0; i < y.size(); i++) y[i] = (i % 2 == 0) ? 50.0f : -50.0f;
    std::vector<float> var_x(x.size(), 0.0f);
    std::vector<float> var_y(y.size(), var_obs);

    OutputUpdater output_updater(model.device);
    model.forward(x, var_x, {BATCH_SIZE, INPUT_SIZE});
    output_updater.update(*model.output_z_buffer, y, var_y,
                          *model.input_delta_z_buffer);
    model.backward();
    model.step();
}

float max_step_over_cap(const std::vector<ParameterTuple> &before,
                        const std::vector<ParameterTuple> &after,
                        float cap_factor) {
    float max_ratio = 0.0f;
    for (size_t l = 0; l < before.size(); l++) {
        const auto &mu_w0 = std::get<0>(before[l]);
        const auto &var_w0 = std::get<1>(before[l]);
        const auto &mu_b0 = std::get<2>(before[l]);
        const auto &var_b0 = std::get<3>(before[l]);
        const auto &mu_w1 = std::get<0>(after[l]);
        const auto &mu_b1 = std::get<2>(after[l]);
        for (size_t i = 0; i < mu_w0.size(); i++) {
            float bound = std::sqrt(var_w0[i]) / cap_factor;
            max_ratio =
                std::max(max_ratio, std::abs(mu_w1[i] - mu_w0[i]) / bound);
        }
        for (size_t i = 0; i < mu_b0.size(); i++) {
            float bound = std::sqrt(var_b0[i]) / cap_factor;
            max_ratio =
                std::max(max_ratio, std::abs(mu_b1[i] - mu_b0[i]) / bound);
        }
    }
    return max_ratio;
}

float max_step_over_cap_after_one_step(const std::string &device,
                                       float cap_factor, bool set_cap) {
    Sequential model(Linear(INPUT_SIZE, OUTPUT_SIZE));
    if (device == "cuda") model.to_device("cuda");
    if (set_cap) model.set_cap_factor(cap_factor);
    auto before = host_parameters(model);
    train_one_step(model);
    auto after = host_parameters(model);
    return max_step_over_cap(before, after, cap_factor);
}

const float VAR_OBS_NO_UPDATE = 1e20f;

std::vector<float> var_w_of(Sequential &model, size_t layer_idx) {
    return std::get<1>(host_parameters(model)[layer_idx]);
}

void expect_var_w_ratio(const std::vector<float> &var_w,
                        const std::vector<float> &var_w0, float ratio,
                        const std::string &context) {
    ASSERT_EQ(var_w.size(), var_w0.size());
    for (size_t i = 0; i < var_w.size(); i++) {
        EXPECT_NEAR(var_w[i] / var_w0[i], ratio, 1e-5f)
            << context << " i=" << i;
    }
}

// Two Linear layers; layer 0 is hidden and layer 1 is the output layer.
std::vector<std::vector<float>> hidden_var_w_trace(const std::string &device,
                                                   int num_steps, float tau,
                                                   float min_gain) {
    Sequential model(Linear(INPUT_SIZE, 5), Linear(5, OUTPUT_SIZE));
    if (device == "cuda") model.to_device("cuda");
    model.set_var_decay(tau, min_gain);
    std::vector<std::vector<float>> trace = {var_w_of(model, 0)};
    auto output_var_w0 = var_w_of(model, 1);
    for (int n = 1; n <= num_steps; n++) {
        train_one_step(model, VAR_OBS_NO_UPDATE);
        trace.push_back(var_w_of(model, 0));
        expect_var_w_ratio(var_w_of(model, 1), output_var_w0, 1.0f,
                           "output layer n=" + std::to_string(n));
    }
    return trace;
}

}  // namespace

TEST(Stabilizers, DefaultUpdateExceedsTightTrustRegion) {
    EXPECT_GT(max_step_over_cap_after_one_step("cpu", 50.0f, false), 1.0f);
}

TEST(Stabilizers, SetCapFactorClipsEveryParamStepCpu) {
    EXPECT_LE(max_step_over_cap_after_one_step("cpu", 50.0f, true), 1.0001f);
}

#ifdef USE_CUDA
TEST(Stabilizers, SetCapFactorClipsEveryParamStepCuda) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EXPECT_LE(max_step_over_cap_after_one_step("cuda", 50.0f, true), 1.0001f);
}
#endif

TEST(Stabilizers, VarDecayFollowsScheduleUntilMinGainThenFreezes) {
    const float tau = 10.0f, min_gain = 0.5f;
    auto trace = hidden_var_w_trace("cpu", 15, tau, min_gain);
    for (int n = 1; n < (int)trace.size(); n++) {
        float expected = std::max(1.0f / (1.0f + n / tau), min_gain);
        expect_var_w_ratio(trace[n], trace[0], expected,
                           "hidden layer n=" + std::to_string(n));
    }
}

#ifdef USE_CUDA
TEST(Stabilizers, VarDecayCudaMatchesCpu) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    auto cpu_trace = hidden_var_w_trace("cpu", 15, 10.0f, 0.5f);
    auto cuda_trace = hidden_var_w_trace("cuda", 15, 10.0f, 0.5f);
    for (size_t n = 1; n < cpu_trace.size(); n++) {
        expect_var_w_ratio(cuda_trace[n], cpu_trace[n], 1.0f,
                           "n=" + std::to_string(n));
    }
}
#endif

TEST(Stabilizers, LoadKeepsVarDecayCounter) {
    const float tau = 10.0f;
    const std::string path = testing::TempDir() + "stabilizers_var_decay.bin";
    Sequential model(Linear(INPUT_SIZE, 5), Linear(5, OUTPUT_SIZE));
    model.set_var_decay(tau);
    for (int n = 0; n < 3; n++) train_one_step(model, VAR_OBS_NO_UPDATE);
    model.save(path);

    Sequential resumed(Linear(INPUT_SIZE, 5), Linear(5, OUTPUT_SIZE));
    resumed.set_var_decay(tau);
    resumed.load(path);

    train_one_step(model, VAR_OBS_NO_UPDATE);
    train_one_step(resumed, VAR_OBS_NO_UPDATE);
    expect_var_w_ratio(var_w_of(resumed, 0), var_w_of(model, 0), 1.0f,
                       "resumed vs continuing");
}

TEST(Stabilizers, ZeroSettingsLeaveStepUnchanged) {
    Sequential untouched(Linear(INPUT_SIZE, 5), Linear(5, OUTPUT_SIZE));
    Sequential disabled(Linear(INPUT_SIZE, 5), Linear(5, OUTPUT_SIZE));
    disabled.params_from(untouched);
    disabled.set_cap_factor(0.0f);
    disabled.set_var_decay(0.0f);
    train_one_step(untouched);
    train_one_step(disabled);
    auto expected = host_parameters(untouched);
    auto actual = host_parameters(disabled);
    for (size_t l = 0; l < expected.size(); l++) {
        EXPECT_EQ(std::get<0>(actual[l]), std::get<0>(expected[l]));
        EXPECT_EQ(std::get<1>(actual[l]), std::get<1>(expected[l]));
        EXPECT_EQ(std::get<2>(actual[l]), std::get<2>(expected[l]));
        EXPECT_EQ(std::get<3>(actual[l]), std::get<3>(expected[l]));
    }
}

#include <gtest/gtest.h>

#include "../../include/common.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/lstm_layer.h"
#include "../../include/sequential.h"
#include "../../include/slinear_layer.h"
#include "../../include/slstm_layer.h"
#include "test_utils.h"
#ifdef USE_CUDA
#include "../../include/lstm_layer_cuda.cuh"
#endif

extern bool g_gpu_enabled;

class SineSignalTest : public ::testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SineSignalTest, LSTMTest_CPU) {
    int input_seq_len = 4;
    Sequential model(LSTM(1, 8, input_seq_len), LSTM(8, 8, input_seq_len),
                     Linear(8 * input_seq_len, 1));
    model.set_threads(2);
    float avg_error;
    float log_lik;
    float mse_threshold = 0.5f;
    float log_lik_threshold = -3.0f;
    sin_signal_lstm_test_runner(model, input_seq_len, avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}

#ifdef USE_CUDA
TEST_F(SineSignalTest, LSTMTestUserOutputUpdater_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    int input_seq_len = 4;
    Sequential model(LSTM(1, 8, input_seq_len), LSTM(8, 8, input_seq_len),
                     Linear(8 * input_seq_len, 1));
    model.to_device("cuda");
    float avg_error;
    float log_lik;
    float mse_threshold = 0.5f;
    float log_lik_threshold = -3.0f;
    sin_signal_lstm_user_output_updater_test_runner(model, input_seq_len,
                                                    avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}
#endif

TEST_F(SineSignalTest, SmootherTest_CPU) {
    int input_seq_len = 24;
    int num_features = 1;
    Sequential model(SLSTM(num_features + input_seq_len - 1, 8, 1),
                     SLSTM(8, 8, 1), SLinear(8, 1));
    model.set_threads(2);
    float avg_error;
    float log_lik;
    float mse_threshold = 0.5f;
    float log_lik_threshold = -3.0f;
    sin_signal_smoother_test_runner(model, input_seq_len, num_features,
                                    avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}

#ifdef USE_CUDA
TEST_F(SineSignalTest, LSTMTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    int input_seq_len = 4;
    Sequential model(LSTM(1, 8, input_seq_len), LSTM(8, 8, input_seq_len),
                     Linear(8 * input_seq_len, 1));
    model.to_device("cuda");

    float avg_error;
    float log_lik;
    float mse_threshold = 0.5f;
    float log_lik_threshold = -3.0f;
    sin_signal_lstm_test_runner(model, input_seq_len, avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}
#endif

TEST(LSTMStateTest, GetAndSetStatesCPU) {
    LSTM layer(3, 2, 1);
    std::vector<float> mu_h{0.1f, 0.2f};
    std::vector<float> var_h{0.3f, 0.4f};
    std::vector<float> mu_c{0.5f, 0.6f};
    std::vector<float> var_c{0.7f, 0.8f};

    layer.set_LSTM_states(mu_h, var_h, mu_c, var_c);

    auto states = layer.get_LSTM_states();
    EXPECT_EQ(std::get<0>(states), mu_h);
    EXPECT_EQ(std::get<1>(states), var_h);
    EXPECT_EQ(std::get<2>(states), mu_c);
    EXPECT_EQ(std::get<3>(states), var_c);

    EXPECT_EQ(layer.lstm_states.mu_h_prev, mu_h);
    EXPECT_EQ(layer.lstm_states.var_h_prev, var_h);
    EXPECT_EQ(layer.lstm_states.mu_c_prev, mu_c);
    EXPECT_EQ(layer.lstm_states.var_c_prev, var_c);
}

#ifdef USE_CUDA
TEST(LSTMStateTest, GetAndSetStatesCUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";

    LSTM layer(3, 2, 1);
    auto cuda_layer = layer.to_cuda();
    auto* lstm_cuda = dynamic_cast<LSTMCuda*>(cuda_layer.get());
    ASSERT_NE(lstm_cuda, nullptr);

    constexpr int batch_size = 1;
    size_t num_states =
        lstm_cuda->output_size * lstm_cuda->seq_len * batch_size;
    size_t num_inputs = lstm_cuda->input_size * lstm_cuda->seq_len * batch_size;
    lstm_cuda->lstm_state.set_num_states(num_states, num_inputs,
                                         lstm_cuda->device_idx);

    std::vector<float> mu_h{0.1f, 0.2f};
    std::vector<float> var_h{0.3f, 0.4f};
    std::vector<float> mu_c{0.5f, 0.6f};
    std::vector<float> var_c{0.7f, 0.8f};

    lstm_cuda->d_set_LSTM_states(mu_h, var_h, mu_c, var_c);

    std::vector<float> mu_h_out, var_h_out, mu_c_out, var_c_out;
    lstm_cuda->d_get_LSTM_states(mu_h_out, var_h_out, mu_c_out, var_c_out);

    EXPECT_EQ(mu_h_out, mu_h);
    EXPECT_EQ(var_h_out, var_h);
    EXPECT_EQ(mu_c_out, mu_c);
    EXPECT_EQ(var_c_out, var_c);

    lstm_cuda->lstm_state.to_host();
    EXPECT_EQ(lstm_cuda->lstm_state.mu_h_prior, mu_h);
    EXPECT_EQ(lstm_cuda->lstm_state.var_h_prior, var_h);
    EXPECT_EQ(lstm_cuda->lstm_state.mu_c_prior, mu_c);
    EXPECT_EQ(lstm_cuda->lstm_state.var_c_prior, var_c);
}
#endif

TEST(LSTMStateTest, SLSTMGetSmoothedStatesAtTimeStep) {
    Sequential model(SLSTM(2, 3, 1));
    ASSERT_FALSE(model.layers.empty());

    auto* slstm = dynamic_cast<SLSTM*>(model.layers.front().get());
    ASSERT_NE(slstm, nullptr);

    constexpr int num_timesteps = 4;
    slstm->smooth_states.set_num_states(slstm->output_size, num_timesteps);

    auto fill_sequence = [](std::vector<float>& target, float offset) {
        for (size_t idx = 0; idx < target.size(); ++idx) {
            target[idx] = offset + static_cast<float>(idx);
        }
    };

    fill_sequence(slstm->smooth_states.mu_h_smooths, 0.0f);
    fill_sequence(slstm->smooth_states.var_h_smooths, 100.0f);
    fill_sequence(slstm->smooth_states.mu_c_smooths, 200.0f);
    fill_sequence(slstm->smooth_states.var_c_smooths, 300.0f);

    constexpr int time_step = 2;
    const size_t num_states = slstm->smooth_states.num_states;
    const size_t start = static_cast<size_t>(time_step) * num_states;
    const size_t end = start + num_states;

    std::vector<float> expected_mu_h(
        slstm->smooth_states.mu_h_smooths.begin() + start,
        slstm->smooth_states.mu_h_smooths.begin() + end);
    std::vector<float> expected_var_h(
        slstm->smooth_states.var_h_smooths.begin() + start,
        slstm->smooth_states.var_h_smooths.begin() + end);
    std::vector<float> expected_mu_c(
        slstm->smooth_states.mu_c_smooths.begin() + start,
        slstm->smooth_states.mu_c_smooths.begin() + end);
    std::vector<float> expected_var_c(
        slstm->smooth_states.var_c_smooths.begin() + start,
        slstm->smooth_states.var_c_smooths.begin() + end);

    auto states_map = model.get_lstm_states(time_step);
    auto it = states_map.find(0);
    ASSERT_NE(it, states_map.end());

    const auto& states = it->second;
    EXPECT_EQ(std::get<0>(states), expected_mu_h);
    EXPECT_EQ(std::get<1>(states), expected_var_h);
    EXPECT_EQ(std::get<2>(states), expected_mu_c);
    EXPECT_EQ(std::get<3>(states), expected_var_c);
}

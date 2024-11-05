#include <gtest/gtest.h>

#include "../../include/common.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/lstm_layer.h"
#include "../../include/sequential.h"
#include "../../include/slinear_layer.h"
#include "../../include/slstm_layer.h"
#include "test_utils.h"

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

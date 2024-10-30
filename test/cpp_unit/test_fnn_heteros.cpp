#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "../../include/activation.h"
#include "../../include/common.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/sequential.h"
#include "test_utils.h"

class SinSignaTest : public ::testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SinSignaTest, HeterosNoiseTest_CPU) {
    Sequential model(Linear(13, 16), ReLU(), Linear(16, 2), EvenExp());
    model.set_threads(2);

    float avg_error;
    float log_lik;
    float mse_threshold = 8.0f;
    float log_lik_threshold = -3.0f;
    heteros_test_runner(model, avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}

#ifdef USE_CUDA
TEST_F(SinSignaTest, HeterosNoiseTest_CUDA) {
    Sequential model(Linear(13, 16), ReLU(), Linear(16, 2), EvenExp());
    model.to_device("cuda");

    float avg_error;
    float log_lik;
    float mse_threshold = 8.0f;
    float log_lik_threshold = -3.0f;
    heteros_test_runner(model, avg_error, log_lik);
    EXPECT_LT(avg_error, mse_threshold) << "MSE is higher than threshold";
    EXPECT_GT(log_lik, log_lik_threshold)
        << "Log likelihood is lower than threshold";
}
#endif
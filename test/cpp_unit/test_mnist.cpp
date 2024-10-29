
#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/conv2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"
#include "test_utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class MnistTest : public ::testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(MnistTest, FNNModelTest_CPU) {
    Sequential model(Linear(784, 32), ReLU(), Linear(32, 32), ReLU(),
                     Linear(32, 11));
    float avg_error;
    float threshold = 0.5;  // Heuristic threshold
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MixtureReLUModelTest_CPU) {
    Sequential model(Linear(784, 32), MixtureReLU(), Linear(32, 32),
                     MixtureReLU(), Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormFNNTest_CPU) {
    Sequential model(Linear(784, 32), BatchNorm2d(32), ReLU(), Linear(32, 32),
                     BatchNorm2d(1024), ReLU(), Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormFNNTest_CPU) {
    Sequential model(Linear(784, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, CNNTest_CPU) {
    Sequential model(Conv2d(1, 8, 4, true, 1, 1, 1, 28, 28), ReLU(),
                     AvgPool2d(3, 2), Conv2d(8, 8, 5), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.set_threads(2);
    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormCNNTest_CPU) {
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(8),
                     ReLU(), AvgPool2d(3, 2), Conv2d(8, 8, 5, false),
                     BatchNorm2d(8), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.set_threads(2);
    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold), "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormCNNTest_CPU) {
    Sequential model(
        Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28),
        LayerNorm(std::vector<int>({8, 27, 27})), ReLU(), AvgPool2d(3, 2),
        Conv2d(8, 8, 5, false), LayerNorm(std::vector<int>({8, 9, 9})), ReLU(),
        AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.set_threads(2);
    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold), "Error rate is higher than threshold";
}

#ifdef USE_CUDA
TEST_F(MnistTest, FNNModelTest_CUDA) {
    Sequential model(Linear(784, 32), ReLU(), Linear(32, 32), ReLU(),
                     Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MixtureReLUModelTest_CUDA) {
    Sequential model(Linear(784, 32), MixtureReLU(), Linear(32, 32),
                     MixtureReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormFNNTest_CUDA) {
    Sequential model(Linear(784, 32), BatchNorm2d(32), ReLU(), Linear(32, 32),
                     BatchNorm2d(1024), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormFNNTest_CUDA) {
    Sequential model(Linear(784, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, CNNTest_CUDA) {
    Sequential model(Conv2d(1, 8, 4, true, 1, 1, 1, 28, 28), ReLU(),
                     AvgPool2d(3, 2), Conv2d(8, 8, 5), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormCNNTest_CUDA) {
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(8),
                     ReLU(), AvgPool2d(3, 2), Conv2d(8, 8, 5, false),
                     BatchNorm2d(8), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold), "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormCNNTest_CUDA) {
    Sequential model(
        Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28),
        LayerNorm(std::vector<int>({8, 27, 27})), ReLU(), AvgPool2d(3, 2),
        Conv2d(8, 8, 5, false), LayerNorm(std::vector<int>({8, 9, 9})), ReLU(),
        AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold), "Error rate is higher than threshold";
}
#endif

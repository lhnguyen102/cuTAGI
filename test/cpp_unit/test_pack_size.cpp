
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
#include "../../include/batchnorm_layer.h"
#include "../../include/conv2d_layer.h"
#include "../../include/convtranspose2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/layernorm_layer.h"
#include "../../include/linear_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

extern bool g_gpu_enabled;

void pack_size_runner() {
    // Model
    Sequential model(Linear(784 + 1, 6000), ReLU(), Linear(6000, 784));
    // model.set_threads(2);
    model.to_device("cuda");

    // Updater
    OutputUpdater output_updater(model.device);

    int batch_size = 16;
    std::vector<float> var_obs(batch_size * 784, 1.0f);

    for (int i = 0; i < 1; i++) {
        // generate random input data
        std::vector<float> x_batch(batch_size * 785, 0.0f);
        for (int j = 0; j < batch_size * 785; j++) {
            x_batch[j] = static_cast<float>(rand()) / RAND_MAX;
        }
        // Generate a target data
        std::vector<float> y_batch(batch_size * 784, 0.0f);
        for (int j = 0; j < batch_size * 784; j++) {
            y_batch[j] = static_cast<float>(rand()) / RAND_MAX;
        }

        // Forward pass
        model.forward(x_batch);
        output_updater.update(*model.output_z_buffer, y_batch, var_obs,
                              *model.input_delta_z_buffer);

        // Backward pass
        model.backward();
        model.step();
    }
}

class PackSizeTest : public ::testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};

#ifdef USE_CUDA
TEST_F(PackSizeTest, PackSizeFNNTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    pack_size_runner();
    EXPECT_TRUE(true);
}
#endif
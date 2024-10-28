
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

// Include necessary headers for your Sequential, Linear, ReLU, and other
// dependencies #include "path/to/your/model/definitions"

void fnn_mnist_test(Sequential& model, float threshold,
                    float& avg_error_output) {
    //////////////////////////////////////////////////////////////////////
    // Data preprocessing
    //////////////////////////////////////////////////////////////////////
    std::vector<std::string> x_train_paths = {
        "./data/mnist/train-images-idx3-ubyte"};
    std::vector<std::string> y_train_paths = {
        "./data/mnist/train-labels-idx1-ubyte"};
    std::vector<std::string> x_test_paths = {
        "./data/mnist/t10k-images-idx3-ubyte"};
    std::vector<std::string> y_test_paths = {
        "./data/mnist/t10k-labels-idx1-ubyte"};

    std::string data_name = "mnist";
    std::vector<float> mu = {0.1309};
    std::vector<float> sigma = {2.0f};
    int num_train_data = 60000;
    int num_test_data = 10000;
    int num_classes = 10;
    int width = 28;
    int height = 28;
    int channel = 1;
    int n_x = width * height;
    int n_y = 11;

    auto train_db = get_images_v2(data_name, x_train_paths, y_train_paths, mu,
                                  sigma, num_train_data, num_classes, width,
                                  height, channel, true);
    auto test_db =
        get_images_v2(data_name, x_test_paths, y_test_paths, mu, sigma,
                      num_test_data, num_classes, width, height, channel, true);

    //////////////////////////////////////////////////////////////////////
    // Output Updater
    //////////////////////////////////////////////////////////////////////
    OutputUpdater output_updater(model.device);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = 42;
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 16;
    float sigma_obs = 1.0;
    int iters = train_db.num_data / batch_size;
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> var_obs(batch_size * train_db.output_len,
                               pow(sigma_obs, 2));
    std::vector<float> y_batch(batch_size * train_db.output_len, 0.0f);
    std::vector<int> batch_idx(batch_size);
    std::vector<int> idx_ud_batch(train_db.output_len * batch_size, 0);
    std::vector<int> label_batch(batch_size, 0);
    std::vector<float> mu_a_output(batch_size * n_y, 0);
    std::vector<float> var_a_output(batch_size * n_y, 0);
    auto data_idx = create_range(train_db.num_data);

    int mt_idx = 0;
    std::vector<int> error_rate(train_db.num_data, 0);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;

    std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
    for (int i = 0; i < 200; i++) {
        get_batch_images_labels(train_db, data_idx, batch_size, i, x_batch,
                                y_batch, idx_ud_batch, label_batch);

        model.forward(x_batch);
        output_updater.update_using_indices(*model.output_z_buffer, y_batch,
                                            var_obs, idx_ud_batch,
                                            *model.input_delta_z_buffer);
        model.backward();
        model.step();

        if (model.device == "cuda") {
            model.output_to_host();
        }

        for (int j = 0; j < batch_size * n_y; j++) {
            mu_a_output[j] = model.output_z_buffer->mu_a[j];
            var_a_output[j] = model.output_z_buffer->var_a[j];
        }

        std::tie(error_rate_batch, prob_class_batch) = get_error(
            mu_a_output, var_a_output, label_batch, num_classes, batch_size);
        mt_idx = i * batch_size;
        update_vector(error_rate, error_rate_batch, mt_idx, 1);
    }

    int curr_idx = mt_idx + batch_size;
    avg_error_output = compute_average_error_rate(error_rate, curr_idx, 100);
}

class MnistTest : public ::testing::Test {
   protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(MnistTest, FNNModelTest) {
    Sequential model(Linear(784, 16), ReLU(), Linear(16, 16), ReLU(),
                     Linear(16, 11));
    float avg_error;
    float threshold = 0.5;  // Set desired threshold for average error
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold);
}

TEST_F(MnistTest, MixtureReLUModelTest) {
    Sequential model(Linear(784, 16), MixtureReLU(), Linear(16, 16),
                     MixtureReLU(), Linear(16, 11));
    float avg_error;
    float threshold = 0.5;  // Set desired threshold for average error
    fnn_mnist_test(model, threshold, avg_error);
    EXPECT_LT(avg_error, threshold);
}

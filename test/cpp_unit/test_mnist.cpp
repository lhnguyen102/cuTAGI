
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
#include "../../include/max_pooling_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

extern bool g_gpu_enabled;

void mnist_test_runner(Sequential &model, float &avg_error_output) {
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

    ////////////////////////////////////////////////////////////////////////////
    // Training
    ////////////////////////////////////////////////////////////////////////////
    OutputUpdater output_updater(model.device);

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
    for (int i = 0; i < 100; i++) {
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

std::vector<float> autoencoder_test_runner(Sequential &encoder,
                                           Sequential &decoder)
/*
 */
{
    //////////////////////////////////////////////////////////////////////
    // Data preprocessing
    //////////////////////////////////////////////////////////////////////
    std::vector<std::string> x_train_paths, y_train_paths, x_test_paths,
        y_test_paths;
    std::string x_train_path = "./data/mnist/train-images-idx3-ubyte";
    std::string y_train_path = "./data/mnist/train-labels-idx1-ubyte";
    std::string x_test_path = "./data/mnist/t10k-images-idx3-ubyte";
    std::string y_test_path = "./data/mnist/t10k-labels-idx1-ubyte";
    x_train_paths.push_back(x_train_path);
    y_train_paths.push_back(y_train_path);
    x_test_paths.push_back(x_test_path);
    y_test_paths.push_back(y_test_path);

    std::string data_name = "mnist";
    std::vector<float> mu = {0.1309};
    std::vector<float> sigma = {1.0f};
    int num_train_data = 60000;
    int num_test_data = 10000;
    int num_classes = 10;
    int width = 28;
    int height = 28;
    int channel = 1;
    int n_x = width * height;
    int n_y = n_x;
    auto train_db = get_images_v2(data_name, x_train_paths, y_train_paths, mu,
                                  sigma, num_train_data, num_classes, width,
                                  height, channel, true);

    auto test_db =
        get_images_v2(data_name, x_test_paths, y_test_paths, mu, sigma,
                      num_test_data, num_classes, width, height, channel, true);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    OutputUpdater output_updater(encoder.device);

    unsigned seed = 42;
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 16;
    float sigma_obs = 20.0;
    int iters = train_db.num_data / batch_size;
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> var_obs(batch_size * n_y, pow(sigma_obs, 2));
    std::vector<int> batch_idx(batch_size);
    std::vector<int> label_batch(batch_size, 0);
    std::vector<float> mu_a_output(batch_size * n_y, 0);
    std::vector<float> var_a_output(batch_size * n_y, 0);

    auto data_idx = create_range(train_db.num_data);

    for (int i = 0; i < 10; i++) {
        // Load input data for encoder and output data for decoder
        get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
        get_batch_data(train_db.images, batch_idx, n_x, x_batch);
        get_batch_data(train_db.labels, batch_idx, 1, label_batch);

        // Forward pass
        encoder.forward(x_batch);
        decoder.forward(*encoder.output_z_buffer);

        // Extract output
        if (decoder.device == "cuda") {
            decoder.output_to_host();
        }

        for (int j = 0; j < batch_size * n_y; j++) {
            mu_a_output[j] = decoder.output_z_buffer->mu_a[j];
            var_a_output[j] = decoder.output_z_buffer->var_a[j];
        }

        // Output layer's update i.e., loss function
        output_updater.update(*decoder.output_z_buffer, x_batch, var_obs,
                              *decoder.input_delta_z_buffer);

        // Backward pass
        decoder.backward();
        decoder.step();

        encoder.input_delta_z_buffer->copy_from(*decoder.output_delta_z_buffer,
                                                n_y * batch_size);

        encoder.backward();
        encoder.step();
    }
    return mu_a_output;
}

class MnistTest : public ::testing::Test {
   protected:
    void SetUp() override {
        const std::string x_train_path = "./data/mnist/train-images-idx3-ubyte";
        const std::string y_train_path = "./data/mnist/train-labels-idx1-ubyte";
        const std::string x_test_path = "./data/mnist/t10k-images-idx3-ubyte";
        const std::string y_test_path = "./data/mnist/t10k-labels-idx1-ubyte";

        if (!file_exists(x_train_path) || !file_exists(y_train_path) ||
            !file_exists(x_test_path) || !file_exists(y_test_path)) {
            std::cout
                << "One or more MNIST data files are missing. Downloading..."
                << std::endl;
            if (!download_mnist_data()) {
                std::cerr << "Failed to download MNIST data files."
                          << std::endl;
            }
        }
    }

    void TearDown() override {}

    bool file_exists(const std::string &path) {
        std::ifstream file(path);
        return file.good();
    }

    bool download_mnist_data() {
        if (system("mkdir -p ./data/mnist") != 0) {
            std::cerr << "Failed to create directory ./data/mnist" << std::endl;
            return false;
        }

        std::string base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/";
        if (!download_and_extract("train-images-idx3-ubyte", base_url) ||
            !download_and_extract("train-labels-idx1-ubyte", base_url) ||
            !download_and_extract("t10k-images-idx3-ubyte", base_url) ||
            !download_and_extract("t10k-labels-idx1-ubyte", base_url)) {
            return false;
        }

        std::cout << "MNIST data files downloaded successfully." << std::endl;
        return true;
    }

    bool download_and_extract(const std::string &filename,
                              const std::string &base_url) {
        // Download file with curl, handle failure with --fail option
        std::string command = "curl --fail -o ./data/mnist/" + filename +
                              ".gz " + base_url + filename + ".gz";
        if (system(command.c_str()) != 0) {
            std::cerr << "Failed to download " << filename << ".gz"
                      << std::endl;
            return false;
        }

        // Verify if downloaded file is a valid gzip file
        command = "file ./data/mnist/" + filename +
                  ".gz | grep 'gzip compressed data'";
        if (system(command.c_str()) != 0) {
            std::cerr << filename
                      << ".gz is not in gzip format or file check failed."
                      << std::endl;
            return false;
        }

        // Extract gzip file
        command = "gunzip -f ./data/mnist/" + filename + ".gz";
        if (system(command.c_str()) != 0) {
            std::cerr << "Failed to extract " << filename << ".gz" << std::endl;
            return false;
        }
        return true;
    }
};

TEST_F(MnistTest, BatchnormWithoutBiases_CPU) {
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28),
                     BatchNorm2d(8, 1e-5, 0, false), ReLU(), AvgPool2d(3, 2),
                     Conv2d(8, 8, 5, false), BatchNorm2d(8, 1e-5, 0, false),
                     ReLU(), AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(),
                     Linear(32, 11));
    model.set_threads(4);

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, FNNModelTest_CPU) {
    Sequential model(Linear(784, 32), ReLU(), Linear(32, 32), ReLU(),
                     Linear(32, 11));
    float avg_error;
    float threshold = 0.5;  // Heuristic threshold
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MixtureReLUModelTest_CPU) {
    Sequential model(Linear(784, 32), MixtureReLU(), Linear(32, 32),
                     MixtureReLU(), Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormFNNTest_CPU) {
    Sequential model(Linear(784, 32), BatchNorm2d(32), ReLU(), Linear(32, 32),
                     BatchNorm2d(32), ReLU(), Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormFNNTest_CPU) {
    Sequential model(Linear(784, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, CNNTest_CPU) {
    Sequential model(Conv2d(1, 8, 4, true, 1, 1, 1, 28, 28), ReLU(),
                     AvgPool2d(3, 2), Conv2d(8, 8, 5), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.set_threads(4);

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MaxPoolingTest_CPU) {
    Sequential model(Conv2d(1, 8, 4, true, 1, 1, 1, 28, 28), ReLU(),
                     MaxPool2d(3, 2), Conv2d(8, 8, 5), ReLU(), MaxPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.set_threads(4);

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormCNNTest_CPU) {
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(8),
                     ReLU(), AvgPool2d(3, 2), Conv2d(8, 8, 5, false),
                     BatchNorm2d(8), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.set_threads(4);

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
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
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MismatchSizeDetection) {
    Sequential model(Linear(784, 32), ReLU(), Linear(300, 32), ReLU(),
                     Linear(32, 11));
    float avg_error;
    float threshold = 0.5;
    EXPECT_THROW({ mnist_test_runner(model, avg_error); }, std::runtime_error);
}

#ifdef USE_CUDA
TEST_F(MnistTest, BatchnormWithoutBiases_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28),
                     BatchNorm2d(8, 1e-5, 0, false), ReLU(), AvgPool2d(3, 2),
                     Conv2d(8, 8, 5, false), BatchNorm2d(8, 1e-5, 0, false),
                     ReLU(), AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(),
                     Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayernormWithoutBiases_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28),
                     LayerNorm(std::vector<int>({8, 27, 27}), 1e-5, false),
                     ReLU(), AvgPool2d(3, 2), Conv2d(8, 8, 5, false),
                     LayerNorm(std::vector<int>({8, 9, 9}), 1e-5, false),
                     ReLU(), AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(),
                     Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, FNNModelTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Linear(784, 32), ReLU(), Linear(32, 32), ReLU(),
                     Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MixtureReLUModelTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Linear(784, 32), MixtureReLU(), Linear(32, 32),
                     MixtureReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormFNNTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Linear(784, 32), BatchNorm2d(32), ReLU(), Linear(32, 32),
                     BatchNorm2d(1024), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormFNNTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Linear(784, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 32), LayerNorm(std::vector<int>({32})), ReLU(),
                     Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, CNNTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Conv2d(1, 8, 4, true, 1, 1, 1, 28, 28), ReLU(),
                     AvgPool2d(3, 2), Conv2d(8, 8, 5), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, MaxPoolingTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Conv2d(1, 8, 4, true, 1, 1, 1, 28, 28), ReLU(),
                     MaxPool2d(3, 2), Conv2d(8, 8, 5), ReLU(), MaxPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold) << "Error rate is higher than threshold";
}

TEST_F(MnistTest, BatchNormCNNTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(8),
                     ReLU(), AvgPool2d(3, 2), Conv2d(8, 8, 5, false),
                     BatchNorm2d(8), ReLU(), AvgPool2d(3, 2),
                     Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold), "Error rate is higher than threshold";
}

TEST_F(MnistTest, LayerNormCNNTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential model(
        Conv2d(1, 8, 4, false, 1, 1, 1, 28, 28),
        LayerNorm(std::vector<int>({8, 27, 27})), ReLU(), AvgPool2d(3, 2),
        Conv2d(8, 8, 5, false), LayerNorm(std::vector<int>({8, 9, 9})), ReLU(),
        AvgPool2d(3, 2), Linear(8 * 4 * 4, 32), ReLU(), Linear(32, 11));
    model.to_device("cuda");

    float avg_error;
    float threshold = 0.5;
    mnist_test_runner(model, avg_error);
    EXPECT_LT(avg_error, threshold), "Error rate is higher than threshold";
}

TEST_F(MnistTest, AutoencoderTest_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    Sequential encoder(Conv2d(1, 16, 3, true, 1, 1, 1, 28, 28), BatchNorm2d(16),
                       ReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(16, 32, 3, true, 1, 1, 1), BatchNorm2d(32),
                       ReLU(), AvgPool2d(3, 2, 1, 2), Linear(32 * 7 * 7, 100),
                       ReLU(), Linear(100, 10));
    encoder.to_device("cuda");

    Sequential decoder(Linear(10, 32 * 7 * 7), ReLU(),
                       ConvTranspose2d(32, 32, 3, true, 2, 1, 2, 7, 7), ReLU(),
                       ConvTranspose2d(32, 16, 3, true, 2, 1, 2), ReLU(),
                       ConvTranspose2d(16, 1, 3, true, 1, 1, 1));
    decoder.input_state_update = true;
    decoder.to_device("cuda");

    auto decoder_output = autoencoder_test_runner(encoder, decoder);
    for (auto &val : decoder_output) {
        EXPECT_FALSE(std::isnan(val));
    }
}
#endif
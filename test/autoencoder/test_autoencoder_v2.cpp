///////////////////////////////////////////////////////////////////////////////
// File:         test_autoencoder_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 15, 2024
// Updated:      March 15, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_autoencoder_v2.h"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/conv2d_layer.h"
#include "../../include/convtranspose2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/debugger.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

#ifdef USE_CUDA
void debug_autoencoder()
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

    // Specify network properties for the decoder
    const std::vector<int> LAYERS_D = {1, 1, 21, 21, 21};
    const std::vector<int> NODES_D = {2, 392, 0, 0, 784};
    const std::vector<int> KERNELS_D = {1, 3, 3, 3, 1};
    const std::vector<int> STRIDES_D = {0, 2, 2, 1, 0};
    const std::vector<int> WIDTHS_D = {0, 7, 0, 0, 0};
    const std::vector<int> HEIGHTS_D = {0, 7, 0, 0, 0};
    const std::vector<int> FILTERS_D = {1, 8, 8, 4, 1};
    const std::vector<int> PADS_D = {0, 1, 1, 1, 0};
    const std::vector<int> PAD_TYPES_D = {0, 2, 2, 1, 0};
    const std::vector<int> ACTIVATIONS_D = {0, 4, 4, 4, 0};

    const int BATCH_SIZE = 2;
    const int SIGMA_V = 8;
    const int SIGMA_V_MIN = 2;
    const float DECAT_FACTOR_SIGMA_V = 0.95;
    const int NUM_CLASSES = 10;
    const std::vector<float> MU = {0.1309};
    const std::vector<float> SIGMA = {1.0};
    const std::string INIT_METHOD = "He";

    // Decoder from the previous version
    std::string device = "cuda";
    Network net_prop_d;
    net_prop_d.layers = LAYERS_D;
    net_prop_d.nodes = NODES_D;
    net_prop_d.kernels = KERNELS_D;
    net_prop_d.strides = STRIDES_D;
    net_prop_d.widths = WIDTHS_D;
    net_prop_d.heights = HEIGHTS_D;
    net_prop_d.filters = FILTERS_D;
    net_prop_d.pads = PADS_D;
    net_prop_d.pad_types = PAD_TYPES_D;
    net_prop_d.activations = ACTIVATIONS_D;
    net_prop_d.batch_size = BATCH_SIZE;
    net_prop_d.sigma_v = SIGMA_V;
    net_prop_d.sigma_v_min = SIGMA_V_MIN;
    net_prop_d.decay_factor_sigma_v = DECAT_FACTOR_SIGMA_V;
    net_prop_d.init_method = INIT_METHOD;

    net_prop_d.device = device;

    TagiNetwork net_d(net_prop_d);
    net_d.prop.last_backward_layer = 0;

    std::string param_path = "test/autoencoder/saved_param/";
    std::string model_name = "t_cnn";
    std::string test_name = "mnist";
    // save_net_param(test_name, model_name, param_path, net_d.theta);
    load_net_param(test_name, model_name, param_path, net_d.theta);
    net_d.theta_gpu.copy_host_to_device();

    // Decoder from newest version
    Sequential decoder(Linear(2, 8 * 7 * 7), ReLU(),
                       ConvTranspose2d(8, 8, 3, true, 2, 1, 2, 7, 7), ReLU(),
                       ConvTranspose2d(8, 4, 3, true, 2, 1, 2), ReLU(),
                       ConvTranspose2d(4, 1, 3, true, 1, 1, 1));
    decoder.input_state_update = true;
    decoder.set_threads(8);

    // VALIDATOR
    std::string param_prefix = param_path + test_name + "_" + model_name;
    decoder.preinit_layer();
    CrossValidator validator(decoder, &net_d, param_prefix);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = 1;
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 2;
    float sigma_obs = 8.0;
    int iters = train_db.num_data / batch_size;
    std::cout << "num_iter: " << iters << "\n";
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> var_obs(batch_size * n_y, pow(sigma_obs, 2));
    std::vector<int> batch_idx(batch_size);
    std::vector<int> label_batch(batch_size, 0);
    std::vector<int> idx_ud_batch = {};

    auto data_idx = create_range(train_db.num_data);

    for (int e = 0; e < n_epochs; e++) {
        if (e > 0) {
            // Shuffle data
            std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
        }
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
        std::cout << "Training...\n";
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 1; i++) {
            // Load input data for encoder and output data for decoder
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(train_db.images, batch_idx, n_x, x_batch);
            get_batch_data(train_db.labels, batch_idx, 1, label_batch);

            std::vector<float> latent_var = {1, 2, 3, 4};

            validator.validate_forward(latent_var);
            validator.validate_backward(x_batch, var_obs, idx_ud_batch);
        }

        // Report computational time
        std::cout << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto run_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
        std::cout << " Time left     : ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60 << "mins\n ";
    }
}
#endif

void cnn_autoencoder()
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

    ////////////////////////////////////////////////////////////////////////////
    // Model
    ////////////////////////////////////////////////////////////////////////////
    Sequential encoder(Conv2d(1, 16, 3, false, 1, 1, 1, 28, 28),
                       BatchNorm2d(16), ReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(16, 32, 3, false, 1, 1, 1), BatchNorm2d(32),
                       ReLU(), AvgPool2d(3, 2, 1, 2), Linear(32 * 7 * 7, 100),
                       ReLU(), Linear(100, 10));

    encoder.set_threads(8);
    // encoder.to_device("cuda");

    Sequential decoder(Linear(10, 32 * 7 * 7), ReLU(),
                       ConvTranspose2d(32, 32, 3, true, 2, 1, 2, 7, 7), ReLU(),
                       ConvTranspose2d(32, 16, 3, true, 2, 1, 2), ReLU(),
                       ConvTranspose2d(16, 1, 3, true, 1, 1, 1));
    decoder.input_state_update = true;
    decoder.set_threads(8);
    // decoder.to_device("cuda");

    OutputUpdater output_updater(encoder.device);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed =
        1;  // std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 20;
    float sigma_obs = 20.0;
    int iters = train_db.num_data / batch_size;
    std::cout << "num_iter: " << iters << "\n";
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> var_obs(batch_size * n_y, pow(sigma_obs, 2));
    std::vector<int> batch_idx(batch_size);
    std::vector<int> label_batch(batch_size, 0);

    auto data_idx = create_range(train_db.num_data);

    for (int e = 0; e < n_epochs; e++) {
        if (e > 0) {
            // Shuffle data
            std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
        }
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
        std::cout << "Training...\n";
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            // Load input data for encoder and output data for decoder
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(train_db.images, batch_idx, n_x, x_batch);
            get_batch_data(train_db.labels, batch_idx, 1, label_batch);

            // Forward pass
            encoder.forward(x_batch);
            decoder.forward(*encoder.output_z_buffer);

            // Output layer's update i.e., loss function
            output_updater.update(*decoder.output_z_buffer, x_batch, var_obs,
                                  *decoder.input_delta_z_buffer);

            // Backward pass
            decoder.backward();
            decoder.step();

            encoder.input_delta_z_buffer->copy_from(
                *decoder.output_delta_z_buffer, n_y * batch_size);

            encoder.backward();
            encoder.step();
            // std::cout << " Iters #" << i + 1 << "\n";
        }

        // Report computational time
        std::cout << std::endl;
        auto end = std::chrono::steady_clock::now();
        auto run_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        std::cout << " Time per epoch: " << run_time * 1e-9 << " sec\n";
        std::cout << " Time left     : ";
        std::cout << std::fixed;
        std::cout << std::setprecision(3);
        std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60 << "mins\n ";
    }

    ////////////////////////////////////////////////////////////////////////////
    // Testing
    ////////////////////////////////////////////////////////////////////////////
    std::cout << "################\n";
    std::cout << "Testing...\n";
    std::vector<float> mu_a_output(100 * n_y, 0);
    std::vector<float> var_a_output(100 * n_y, 0);
    int start_idx;
    for (int i = 0; i < 5; i++) {
        start_idx = batch_size * i;
        // Load input data for encoder and output data for decoder
        get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
        get_batch_data(test_db.images, batch_idx, n_x, x_batch);
        get_batch_data(test_db.labels, batch_idx, 1, label_batch);

        // Forward pass
        encoder.forward(x_batch);
        decoder.forward(*encoder.output_z_buffer);

        // Extract output
        if (decoder.device == "cuda") {
            decoder.output_to_host();
        }

        for (int j = start_idx; j < 100 * n_y; j++) {
            mu_a_output[j] = decoder.output_z_buffer->mu_a[j];
            var_a_output[j] = decoder.output_z_buffer->var_a[j];
        }
    }

    // Save generated images
    std::string suffix = "test";
    std::string saved_inference_path = "./saved_results/";
    save_generated_images(saved_inference_path, mu_a_output, suffix);
}

int test_autoecoder_v2()
/*
 */
{
    // debug_autoencoder();
    cnn_autoencoder();
    return 0;
}
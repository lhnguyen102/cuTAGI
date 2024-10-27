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
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

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
    Sequential encoder(Conv2d(1, 16, 3, true, 1, 1, 1, 28, 28), BatchNorm2d(16),
                       ReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(16, 32, 3, true, 1, 1, 1), BatchNorm2d(32),
                       ReLU(), AvgPool2d(3, 2, 1, 2), Linear(32 * 7 * 7, 100),
                       ReLU(), Linear(100, 10));

    // encoder.set_threads(8);
    encoder.to_device("cuda");

    Sequential decoder(Linear(10, 32 * 7 * 7), ReLU(),
                       ConvTranspose2d(32, 32, 3, true, 2, 1, 2, 7, 7), ReLU(),
                       ConvTranspose2d(32, 16, 3, true, 2, 1, 2), ReLU(),
                       ConvTranspose2d(16, 1, 3, true, 1, 1, 1));
    decoder.input_state_update = true;
    // decoder.set_threads(8);
    decoder.to_device("cuda");

    OutputUpdater output_updater(encoder.device);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed =
        1;  // std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 16;
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
        for (int i = 0; i < 1; i++) {
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

    // ////////////////////////////////////////////////////////////////////////////
    // // Testing
    // ////////////////////////////////////////////////////////////////////////////
    // std::cout << "################\n";
    // std::cout << "Testing...\n";
    // std::vector<float> mu_a_output(100 * n_y, 0);
    // std::vector<float> var_a_output(100 * n_y, 0);
    // int start_idx;
    // for (int i = 0; i < 5; i++) {
    //     start_idx = batch_size * i;
    //     // Load input data for encoder and output data for decoder
    //     get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
    //     get_batch_data(test_db.images, batch_idx, n_x, x_batch);
    //     get_batch_data(test_db.labels, batch_idx, 1, label_batch);

    //     // Forward pass
    //     encoder.forward(x_batch);
    //     decoder.forward(*encoder.output_z_buffer);

    //     // Extract output
    //     if (decoder.device == "cuda") {
    //         decoder.output_to_host();
    //     }

    //     for (int j = start_idx; j < 100 * n_y; j++) {
    //         mu_a_output[j] = decoder.output_z_buffer->mu_a[j];
    //         var_a_output[j] = decoder.output_z_buffer->var_a[j];
    //     }
    // }

    // // Save generated images
    // std::string suffix = "test";
    // std::string saved_inference_path = "./saved_results/";
    // save_generated_images(saved_inference_path, mu_a_output, suffix);
}

int test_autoecoder_v2()
/*
 */
{
    // debug_autoencoder();
    cnn_autoencoder();
    return 0;
}
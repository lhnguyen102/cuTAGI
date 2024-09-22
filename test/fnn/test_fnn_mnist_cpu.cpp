///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_mnist_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      November 25, 2023
// Updated:      February 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_mnist_cpu.h"

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
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
// #include "../../include/debugger.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

void fnn_mnist() {
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
    // TAGI network
    //////////////////////////////////////////////////////////////////////
    // Sequential model(Linear(784, 4 * 4096), ReLU(), Linear(4 * 4096, 4 *
    // 4096),
    //                  ReLU(), Linear(4 * 4096, 11));

    // Sequential model(Linear(784, 1024), BatchNorm2d(1024), ReLU(),
    //                  Linear(1024, 1024), BatchNorm2d(1024), ReLU(),
    //                  Linear(1024, 11));

    // Sequential model(Linear(784, 100), LayerNorm(std::vector<int>({100})),
    //                  ReLU(), Linear(100, 100),
    //                  LayerNorm(std::vector<int>({100})), ReLU(),
    //                  Linear(100, 11));

    // Sequential model(Conv2d(1, 16, 4, true, 1, 1, 1, 28, 28), ReLU(),
    //                  AvgPool2d(3, 2), Conv2d(16, 32, 5), ReLU(),
    //                  AvgPool2d(3, 2), Linear(32 * 4 * 4, 100), ReLU(),
    //                  Linear(100, 11));

    Sequential model(Conv2d(1, 32, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(32),
                     ReLU(), AvgPool2d(3, 2), Conv2d(32, 64, 5, false),
                     BatchNorm2d(64), ReLU(), AvgPool2d(3, 2),
                     Linear(64 * 4 * 4, 128), ReLU(), Linear(128, 11));

    // Sequential model(Conv2d(1, 16, 4, false, 1, 1, 1, 28, 28),
    //                  LayerNorm(std::vector<int>({16, 27, 27})), ReLU(),
    //                  AvgPool2d(3, 2), Conv2d(16, 32, 5, false),
    //                  LayerNorm(std::vector<int>({32, 9, 9})), ReLU(),
    //                  AvgPool2d(3, 2), Linear(32 * 4 * 4, 100), ReLU(),
    //                  Linear(100, 11));

    // model.set_threads(1);
    model.to_device("cuda");
    // model.preinit_layer();
    // model.load("test_model/test_model.bin");

    // // CPU Model
    // Sequential cpu_model(Linear(784, 100),
    // LayerNorm(std::vector<int>({100})),
    //                      ReLU(), Linear(100, 100),
    //                      LayerNorm(std::vector<int>({100})), ReLU(),
    //                      Linear(100, 11));
    // Sequential cpu_model(Conv2d(1, 16, 4, false, 1, 1, 1, 28, 28),
    //                      LayerNorm(std::vector<int>({16, 27, 27})), ReLU(),
    //                      AvgPool2d(3, 2), Conv2d(16, 32, 5, false),
    //                      LayerNorm(std::vector<int>({32, 9, 9})), ReLU(),
    //                      AvgPool2d(3, 2), Linear(32 * 4 * 4, 100), ReLU(),
    //                      Linear(100, 11));

    // Sequential cpu_model(Linear(784, 100),
    // LayerNorm(std::vector<int>({100})),
    //                      ReLU(), Linear(100, 100),
    //                      LayerNorm(std::vector<int>({100})), ReLU(),
    //                      Linear(100, 11));

    // Sequential cpu_model(
    //     Conv2d(1, 16, 4, false, 1, 1, 1, 28, 28), BatchNorm2d(16), ReLU(),
    //     AvgPool2d(3, 2), Conv2d(16, 32, 5, false), BatchNorm2d(32), ReLU(),
    //     AvgPool2d(3, 2), Linear(32 * 4 * 4, 100), ReLU(), Linear(100, 11));

    // // DEBUGGER
    // cpu_model.preinit_layer();
    // cpu_model.params_from(model);
    // // cpu_model.set_threads(8);
    // ModelDebugger model_debugger(model, cpu_model);

    //////////////////////////////////////////////////////////////////////
    // Output Updater
    //////////////////////////////////////////////////////////////////////
    OutputUpdater output_updater(model.device);
    // OutputUpdater cpu_output_updater(cpu_model.device);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 16;
    float sigma_obs = 2.0;
    int iters = train_db.num_data / batch_size;
    std::cout << "num_iter: " << iters << "\n";
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

    // Error rate for training
    int mt_idx = 0;
    std::vector<int> error_rate(train_db.num_data, 0);
    std::vector<int> error_rate_batch;
    std::vector<float> prob_class_batch;
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
            // Load data
            get_batch_images_labels(train_db, data_idx, batch_size, i, x_batch,
                                    y_batch, idx_ud_batch, label_batch);

            // // Forward pass
            model.forward(x_batch);
            // model.save("test_model/test_model.bin");
            // model.load("test_model/test_model.bin");

            // model_debugger.debug_forward(x_batch);
            // model_debugger.debug_backward(y_batch, var_obs, idx_ud_batch);

            // Output layer
            output_updater.update_using_indices(*model.output_z_buffer, y_batch,
                                                var_obs, idx_ud_batch,
                                                *model.input_delta_z_buffer);
            // cpu_output_updater.update_using_indices(
            //     *cpu_model.output_z_buffer, y_batch, var_obs,
            // idx_ud_batch,
            //     *cpu_model.input_delta_z_buffer);

            // Backward pass
            model.backward();
            model.step();

            // // cpu_model.backward();
            // // cpu_model.step();

            // Extract output
            if (model.device == "cuda") {
                model.output_to_host();
            }
            // model.delta_z_to_host();

            for (int j = 0; j < batch_size * n_y; j++) {
                mu_a_output[j] = model.output_z_buffer->mu_a[j];
                var_a_output[j] = model.output_z_buffer->var_a[j];
            }
            std::tie(error_rate_batch, prob_class_batch) =
                get_error(mu_a_output, var_a_output, label_batch, num_classes,
                          batch_size);

            mt_idx = i * batch_size;
            update_vector(error_rate, error_rate_batch, mt_idx, 1);

            if (i % 100 == 0 && i != 0) {
                int curr_idx = mt_idx + batch_size;
                auto avg_error =
                    compute_average_error_rate(error_rate, curr_idx, 100);

                std::cout << "\tError rate for last 100 observation: ";
                std::cout << std::fixed;
                std::cout << std::setprecision(3);
                std::cout << avg_error << "\n";
            }
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
        std::cout << (run_time * 1e-9) * (n_epochs - e - 1) / 60 << " mins\n";
    }

    // //////////////////////////////////////////////////////////////////////
    // // Testing
    // //////////////////////////////////////////////////////////////////////
    // int test_batch_size = 96;
    // int test_iters = test_db.num_data / test_batch_size;
    // std::vector<float> test_x_batch(test_batch_size * n_x, 0.0f);
    // std::vector<float> test_var_obs(test_batch_size * test_db.output_len,
    //                                 pow(sigma_obs, 2));
    // std::vector<int> test_batch_idx(test_batch_size);
    // auto test_data_idx = create_range(test_db.num_data);
    // for (int i = 0; i < 2; i++) {
    //     // Load data
    //     get_batch_images_labels(test_db, test_data_idx, test_batch_size, i,
    //                             x_batch, y_batch, idx_ud_batch, label_batch);

    //     // // Forward pass
    //     model.forward(x_batch);
    // }
}

int test_fnn_mnist() {
    fnn_mnist();
    return 0;
}
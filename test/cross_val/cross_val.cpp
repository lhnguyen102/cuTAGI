///////////////////////////////////////////////////////////////////////////////
// File:         cross_val.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 28, 2024
// Updated:      February 28, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "cross_val.h"

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
#include "../../include/debugger.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"
#include "../../include/struct_var.h"
#include "../../include/utils.h"

// Specify network properties
const std::vector<int> LAYERS = {2, 2, 5, 4, 2, 5, 4, 1, 1};
const std::vector<int> NODES = {784, 0, 0, 0, 0, 0, 0, 100, 11};
const std::vector<int> KERNELS = {4, 1, 3, 5, 1, 3, 1, 1, 1};
const std::vector<int> STRIDES = {1, 0, 2, 1, 0, 2, 0, 0, 0};
const std::vector<int> WIDTHS = {28, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> HEIGHTS = {28, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> FILTERS = {1, 4, 4, 4, 8, 8, 8, 1, 1};
const std::vector<int> PADS = {1, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> PAD_TYPES = {1, 0, 0, 0, 0, 0, 0, 0, 0};
const std::vector<int> ACTIVATIONS = {0, 0, 4, 0, 0, 4, 0, 4, 12};

// // FC LAYER
// const std::vector<int> LAYERS = {1, 1, 1, 1};
// const std::vector<int> NODES = {784, 100, 100, 11};
// const std::vector<int> ACTIVATIONS = {0, 4, 4, 12};

// // Conv
// const std::vector<int> LAYERS = {2, 2, 4, 2, 4, 1, 1};
// const std::vector<int> NODES = {784, 0, 0, 0, 0, 100, 11};
// const std::vector<int> KERNELS = {4, 3, 5, 3, 1, 1, 1};
// const std::vector<int> STRIDES = {1, 2, 1, 2, 0, 0, 0};
// const std::vector<int> WIDTHS = {28, 0, 0, 0, 0, 0, 0};
// const std::vector<int> HEIGHTS = {28, 0, 0, 0, 0, 0, 0};
// const std::vector<int> FILTERS = {1, 16, 16, 32, 32, 1, 1};
// const std::vector<int> PADS = {1, 0, 0, 0, 0, 0, 0};
// const std::vector<int> PAD_TYPES = {1, 0, 0, 0, 0, 0, 0};
// const std::vector<int> ACTIVATIONS = {0, 4, 0, 4, 0, 4, 12};

const int BATCH_SIZE = 2;
const int SIGMA_V = 1;
const int NUM_CLASSES = 10;
const std::vector<float> MU = {0.1309};
const std::vector<float> SIGMA = {1.0};

#ifdef USE_CUDA
void cross_val_mnist() {
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
    // Sequential model(Linear(784, 100), ReLU(), Linear(100, 100), ReLU(),
    //                  Linear(100, 11));

    // Sequential model(Linear(784, 100), BatchNorm2d(), ReLU(), Linear(100,
    // 100),
    //                  BatchNorm2d(), ReLU(), Linear(100, 11));

    // Sequential model(Linear(784, 100), LayerNorm(std::vector<int>({100})),
    //                  ReLU(), Linear(100, 100),
    //                  LayerNorm(std::vector<int>({100})), ReLU(),
    //                  Linear(100, 11));

    // Sequential model(Conv2d(1, 16, 4, true, 1, 1, 1, 28, 28), ReLU(),
    //                  AvgPool2d(3, 2), Conv2d(16, 32, 5), ReLU(),
    //                  AvgPool2d(3, 2), Linear(32 * 4 * 4, 100), ReLU(),
    //                  Linear(100, 11));

    // Sequential model(Conv2d(1, 16, 4, 1, 1, 1, 28, 28), BatchNorm2d(),
    // ReLU(),
    //                  AvgPool2d(3, 2), Conv2d(16, 32, 5), BatchNorm2d(),
    //                  ReLU(), AvgPool2d(3, 2), Linear(32 * 4 * 4, 100),
    //                  ReLU(), Linear(100, 11));

    Sequential model(
        Conv2d(1, 4, 4, false, 1, 1, 1, 28, 28),
        LayerNorm(std::vector<int>({4, 27, 27})), ReLU(), AvgPool2d(3, 2),
        Conv2d(4, 8, 5, false), LayerNorm(std::vector<int>({8, 9, 9})), ReLU(),
        AvgPool2d(3, 2), Linear(8 * 4 * 4, 100), ReLU(), Linear(100, 11));

    // model.set_threads(8);
    // model.to_device("cuda");

    // Ref Model from older version
    Network net;

    net.layers = LAYERS;
    net.nodes = NODES;
    net.kernels = KERNELS;
    net.strides = STRIDES;
    net.widths = WIDTHS;
    net.heights = HEIGHTS;
    net.filters = FILTERS;
    net.pads = PADS;
    net.pad_types = PAD_TYPES;
    net.activations = ACTIVATIONS;
    net.batch_size = BATCH_SIZE;
    net.sigma_v = SIGMA_V;

    std::string device = "cuda";
    net.device = device;
    auto hrs = class_to_obs(NUM_CLASSES);
    net.nye = hrs.n_obs;

    if (net.activations.back() == net.act_names.hr_softmax) {
        net.is_idx_ud = true;
        auto hrs = class_to_obs(NUM_CLASSES);
        net.nye = hrs.n_obs;
    }

    TagiNetwork ref_model(net);

    std::string param_path = "test/cross_val/saved_param/";
    std::string model_name = "layernorm_cnn";
    std::string test_name = "mnist";
    // save_net_param(test_name, model_name, param_path, ref_model.theta);
    load_net_param(test_name, model_name, param_path, ref_model.theta);
    ref_model.theta_gpu.copy_host_to_device();

    // VALIDATOR
    std::string param_prefix = param_path + test_name + "_" + model_name;
    model.preinit_layer();
    CrossValidator validator(model, &ref_model, param_prefix);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed =
        1;  // std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = BATCH_SIZE;
    float sigma_obs = 1.0;
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
        for (int i = 0; i < 2; i++) {
            // Load data
            get_batch_images_labels(train_db, data_idx, batch_size, i, x_batch,
                                    y_batch, idx_ud_batch, label_batch);

            // // Forward pass
            // std::vector<float> Sx_f_batch;
            // std::vector<float> Sx_batch(x_batch.size(), 0);
            // ref_model.prop.ra_mt = 0.9;
            // ref_model.feed_forward(x_batch, Sx_batch, Sx_f_batch);
            validator.validate_forward(x_batch);
            validator.validate_backward(y_batch, var_obs, idx_ud_batch);
        }
    }
}
#endif
int cross_val_with_old_version() {
#ifdef USE_CUDA
    cross_val_mnist();
#endif
    return 0;
}
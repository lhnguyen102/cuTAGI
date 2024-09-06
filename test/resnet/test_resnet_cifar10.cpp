#include "test_resnet_cifar10.h"

#include <iostream>

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/conv2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
// #include "../../include/debugger.h"
#include "../../include/layer_block.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/resnet_block.h"
#include "../../include/sequential.h"

// Function to convert normalized RGB to grayscale
float rgb_to_gray(float red, float green, float blue) {
    return 0.299 * red + 0.587 * green + 0.114 * blue;
}

// Function to map grayscale to ASCII
char gray_to_ascii(float gray) {
    const std::string chars = "@%#*+=-:. ";
    return chars[int(gray * (chars.size() - 1))];
}

void visualize_image(std::vector<float> &data)
/**/
{
    const int width = 32;
    const int height = 32;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = 3072 + y * width + x;
            float r = data[idx];
            float g = data[idx + 1024];
            float b = data[idx + 2048];
            float gray = rgb_to_gray(r, g, b);
            std::cout << gray_to_ascii(gray);
        }
        std::cout << '\n';
    }
}

LayerBlock create_layer_block(int in_channels, int out_channels, int stride = 1,
                              int padding_type = 1) {
    return LayerBlock(
        Conv2d(in_channels, out_channels, 3, false, stride, 1, padding_type),
        BatchNorm2d(out_channels), ReLU(),
        Conv2d(out_channels, out_channels, 3, false, 1, 1),
        BatchNorm2d(out_channels));
}

void resnet_cifar10()
/**/
{
    ////////////////////////////////////////////////////////////////////////////
    // Data preprocessing
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> y_train_paths, y_test_paths;
    std::vector<std::string> x_train_paths = {
        "./data/cifar/cifar-10-batches-c/data_batch_1.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_2.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_3.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_4.bin",
        "./data/cifar/cifar-10-batches-c/data_batch_5.bin"};
    std::vector<std::string> x_test_paths = {
        "./data/cifar/cifar-10-batches-c/test_batch.bin"};

    std::string data_name = "cifar";
    std::vector<float> mu = {0.4914, 0.4822, 0.4465};
    std::vector<float> sigma = {0.2023, 0.1994, 0.2010};
    int num_train_data = 50000;
    int num_test_data = 10000;
    int num_classes = 10;
    int width = 32;
    int height = 32;
    int channel = 3;
    int n_x = width * height * channel;
    int n_y = 11;
    auto train_db = get_images_v2(data_name, x_train_paths, y_train_paths, mu,
                                  sigma, num_train_data, num_classes, width,
                                  height, channel, true);

    auto test_db =
        get_images_v2(data_name, x_test_paths, y_test_paths, mu, sigma,
                      num_test_data, num_classes, width, height, channel, true);

    // visualize_image(train_db.images);

    ////////////////////////////////////////////////////////////////////////////
    // Model
    ////////////////////////////////////////////////////////////////////////////
    auto block_1 = create_layer_block(64, 64);
    auto block_2 = create_layer_block(64, 64);
    auto block_3 = create_layer_block(64, 128, 2, 2);
    auto block_4 = create_layer_block(128, 128);
    auto block_5 = create_layer_block(128, 256, 2, 2);
    auto block_6 = create_layer_block(256, 256);
    auto block_7 = create_layer_block(256, 512, 2, 2);
    auto block_8 = create_layer_block(512, 512);

    ResNetBlock resnet_block_1(block_1);
    ResNetBlock resnet_block_2(block_2);

    ResNetBlock resnet_block_3(
        block_3, LayerBlock(Conv2d(64, 128, 2, false, 2), BatchNorm2d(128)));
    ResNetBlock resnet_block_4(block_4);

    ResNetBlock resnet_block_5(
        block_5, LayerBlock(Conv2d(128, 256, 2, false, 2), BatchNorm2d(256)));
    ResNetBlock resnet_block_6(block_6);

    ResNetBlock resnet_block_7(
        block_7, LayerBlock(Conv2d(256, 512, 2, false, 2), BatchNorm2d(512)));
    ResNetBlock resnet_block_8(block_8);

    Sequential model(
        // Input block
        Conv2d(3, 64, 3, false, 1, 1, 1, 32, 32), BatchNorm2d(64), ReLU(),

        // Residual blocks
        resnet_block_1, ReLU(), resnet_block_2, ReLU(), resnet_block_3, ReLU(),
        resnet_block_4, ReLU(), resnet_block_5, ReLU(), resnet_block_6, ReLU(),
        resnet_block_7, ReLU(), resnet_block_8, ReLU(),

        // Output block
        AvgPool2d(4), Linear(512, 11));

    // auto block_1 = create_layer_block(8, 8);
    // auto block_2 = create_layer_block(8, 8);
    // auto block_3 = create_layer_block(8, 16, 2, 2);
    // auto block_4 = create_layer_block(16, 16);
    // auto block_5 = create_layer_block(16, 32, 2, 2);
    // auto block_6 = create_layer_block(32, 32);
    // auto block_7 = create_layer_block(32, 64, 2, 2);
    // auto block_8 = create_layer_block(64, 64);

    // ResNetBlock resnet_block_1(block_1);
    // ResNetBlock resnet_block_2(block_2);

    // ResNetBlock resnet_block_3(
    //     block_3, LayerBlock(Conv2d(8, 16, 2, false, 2), BatchNorm2d(16)));
    // ResNetBlock resnet_block_4(block_4);

    // ResNetBlock resnet_block_5(
    //     block_5, LayerBlock(Conv2d(16, 32, 2, false, 2), BatchNorm2d(32)));
    // ResNetBlock resnet_block_6(block_6);

    // ResNetBlock resnet_block_7(
    //     block_7, LayerBlock(Conv2d(32, 64, 2, false, 2), BatchNorm2d(64)));
    // ResNetBlock resnet_block_8(block_8);

    // Sequential model(
    //     // Input block
    //     Conv2d(3, 8, 3, false, 1, 1, 1, 32, 32), BatchNorm2d(8), ReLU(),

    //     // Residual blocks
    //     resnet_block_1, ReLU(), resnet_block_2, ReLU(), resnet_block_3,
    //     ReLU(), resnet_block_4, ReLU(), resnet_block_5, ReLU(),
    //     resnet_block_6, ReLU(), resnet_block_7, ReLU(), resnet_block_8,
    //     ReLU(),

    //     // Output block
    //     AvgPool2d(4), Linear(64, 11));

    // Sequential model(Conv2d(3, 32, 5, false, 1, 2, 1, 32, 32),
    // BatchNorm2d(32),
    //                  ReLU(), AvgPool2d(3, 2, 1, 2),
    //                  Conv2d(32, 32, 5, false, 1, 2, 1), BatchNorm2d(32),
    //                  ReLU(), AvgPool2d(3, 2, 1, 2), Conv2d(32, 64, 5, false,
    //                  1, 2, 1), BatchNorm2d(64), ReLU(), AvgPool2d(3, 2, 1,
    //                  2), Linear(64 * 4 * 4, 100), ReLU(), Linear(100, 11));

    // Sequential model(
    //     Conv2d(3, 32, 5, true, 1, 2, 1, 32, 32), ReLU(), AvgPool2d(3, 2, 1,
    //     2), Conv2d(32, 32, 5, true, 1, 2, 1), ReLU(), AvgPool2d(3, 2, 1, 2),
    //     Conv2d(32, 64, 5, true, 1, 2, 1), ReLU(), AvgPool2d(3, 2, 1, 2),
    //     Linear(64 * 4 * 4, 128), ReLU(), Linear(128, 11));

    // model.set_threads(1);
    // model.preinit_layer();
    // model.save("test_model/resnet.bin");
    model.to_device("cuda");

    // model.load("test_model/resnet.bin");

    OutputUpdater output_updater(model.device);

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int n_epochs = 20;
    int batch_size = 128;
    float sigma_obs = 1;
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
        if (e >= 0) {
            // Shuffle data
            std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
        }
        std::cout << "################\n";
        std::cout << "Epoch #" << e + 1 << "/" << n_epochs << "\n";
        std::cout << "Training...\n";
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            mt_idx = i * batch_size;
            // Load data
            get_batch_images_labels(train_db, data_idx, batch_size, mt_idx,
                                    x_batch, y_batch, idx_ud_batch,
                                    label_batch);

            // // Forward pass
            model.forward(x_batch);

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

            update_vector(error_rate, error_rate_batch, mt_idx, 1);

            // // Backward pass
            // Output layer
            output_updater.update_using_indices(*model.output_z_buffer, y_batch,
                                                var_obs, idx_ud_batch,
                                                *model.input_delta_z_buffer);
            model.backward();
            model.step();

            if (i % 100 == 0 && i != 0) {
                int curr_idx = mt_idx + batch_size;
                auto avg_error =
                    compute_average_error_rate(error_rate, curr_idx, mt_idx);

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
    ////////////////////////////////////////////////////////////////////////
    // Testing
    ////////////////////////////////////////////////////////////////////////
    model.eval();
    std::cout << "Testing...\n";
    int test_batch_size = 128;
    int test_iters = test_db.num_data / test_batch_size;
    std::vector<float> test_x_batch(test_batch_size * n_x, 0.0f);
    std::vector<float> test_y_batch(test_batch_size * test_db.output_len, 0.0f);
    std::vector<int> test_batch_idx(test_batch_size);
    std::vector<int> test_idx_ud_batch(test_db.output_len * test_batch_size, 0);
    std::vector<int> test_label_batch(test_batch_size, 0);
    std::vector<float> test_mu_a_output(test_batch_size * n_y, 0);
    std::vector<float> test_var_a_output(test_batch_size * n_y, 0);
    std::vector<int> test_error_rate(test_db.num_data, 0);
    std::vector<int> test_error_rate_batch, test_pred_label_batch;
    std::vector<float> test_prob_class_batch;
    auto test_data_idx = create_range(test_db.num_data);

    for (int i = 0; i < test_iters; i++) {
        int mt_idx = i * test_batch_size;
        // Load test data
        get_batch_images_labels(test_db, test_data_idx, test_batch_size, mt_idx,
                                test_x_batch, test_y_batch, test_idx_ud_batch,
                                test_label_batch);

        // Forward pass
        model.forward(test_x_batch);

        // Extract output
        if (model.device == "cuda") {
            model.output_to_host();
        }

        for (int j = 0; j < test_batch_size * n_y; j++) {
            test_mu_a_output[j] = model.output_z_buffer->mu_a[j];
            test_var_a_output[j] = model.output_z_buffer->var_a[j];
        }

        std::tie(test_error_rate_batch, test_prob_class_batch,
                 test_pred_label_batch) =
            get_error_v2(test_mu_a_output, test_var_a_output, test_label_batch,
                         num_classes, test_batch_size);

        update_vector(test_error_rate, test_error_rate_batch, mt_idx, 1);
    }

    // Compute average error rate
    auto avg_error = compute_average_error_rate(
        test_error_rate, test_db.num_data, test_iters * test_batch_size);
    std::cout << "Average error rate on test set: " << std::fixed
              << std::setprecision(3) << avg_error << "\n";
}

int test_resnet_cifar10()
/**/
{
    resnet_cifar10();
    return 0;
}
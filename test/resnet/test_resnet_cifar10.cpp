#include "test_resnet_cifar10.h"

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/conv2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/debugger.h"
#include "../../include/layer_block.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/resnet_block.h"
#include "../../include/sequential.h"

void resnet_cifar10()
/**/
{
    ////////////////////////////////////////////////////////////////////////////
    // Data preprocessing
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> x_train_paths, , x_test_paths;
    std::string x_train_path = "./data/cifar/train-images-idx3-ubyte";
    std::string y_train_path = "./data/cifar/train-labels-idx1-ubyte";
    std::string x_test_path = "./data/cifar/t10k-images-idx3-ubyte";
    std::string y_test_path = "./data/cifar/t10k-labels-idx1-ubyte";
    std::vector<std::string> x_train_paths = {
        "./data/cifar/data_batch_1.bin", "./data/cifar/data_batch_2.bin",
        "./data/cifar/data_batch_3.bin", "./data/cifar/data_batch_4.bin",
        "./data/cifar/data_batch_5.bin"};
    std::vector<std::string> x_train_paths = {"./data/cifar/test_batch.bin"}

    std::string data_name = "cifar";
    std::vector<float> mu;
    std::vector<float> sigma;
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

    ////////////////////////////////////////////////////////////////////////////
    // Model
    ////////////////////////////////////////////////////////////////////////////
    LayerBlock block_1(Conv2d(64, 64, 3, false, 1), BatchNorm2d(64),
                       Conv2d(64, 64, 3, false, 1), BatchNorm2d(64));
    LayerBlock block_2(Conv2d(128, 128, 3, false, 1), BatchNorm2d(128),
                       Conv2d(128, 128, 3, false, 1), BatchNorm2d(128));

    Sequential model(
        Conv2d(3, 32, 5, true, 1, 2, 1, 32, 32), ReLU(), AvgPool2d(3, 2, 1, 2),
        Conv2d(32, 32, 5, true, 2, 1), ReLU(), AvgPool2d(3, 2, 1, 2),
        Conv2d(32, 64, 5, true, 2, 1), ReLU(), AvgPool2d(3, 2, 1, 2),
        Linear(64 * 4 * 4, 100), ReLU(), Linear(100, 11));

    // model.set_threads(8);
    model.to_device("cuda");

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int n_epochs = 1;
    int batch_size = 4;
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
        for (int i = 0; i < 2; i++) {
            // Load data
            get_batch_images_labels(train_db, data_idx, batch_size, i, x_batch,
                                    y_batch, idx_ud_batch, label_batch);

            // // Forward pass
            model.forward(x_batch);

            // Output layer
            output_updater.update_using_indices(*model.output_z_buffer, y_batch,
                                                var_obs, idx_ud_batch,
                                                *model.input_delta_z_buffer);
            // Backward pass
            model.backward();
            model.step();

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
}

int test_resnet_cifar10()
/**/
{
    resnet_cifar10();
    return 0;
}
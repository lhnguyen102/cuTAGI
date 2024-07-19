#include "test_resnet_1d_toy.h"

#include "../../include/activation.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/layer_block.h"
#include "../../include/linear_layer.h"
#include "../../include/resnet_block.h"
#include "../../include/sequential.h"

void resnet_1d_toy()
/*
 */
{
    //////////////////////////////////////////////////////////////////////
    // Data preprocessing
    //////////////////////////////////////////////////////////////////////
    std::string x_train_dir, y_train_dir, x_test_dir, y_test_dir;
    std::vector<std::string> x_train_path, y_train_path, x_test_path,
        y_test_path;
    std::string data_path = "/test/data/1D";
    x_train_dir = data_path + "/x_train.csv";
    y_train_dir = data_path + "/y_train.csv";
    x_test_dir = data_path + "/x_test.csv";
    y_test_dir = data_path + "/y_test.csv";
    x_train_path.push_back(x_train_dir);
    y_train_path.push_back(y_train_dir);
    x_test_path.push_back(x_test_dir);
    y_test_path.push_back(y_test_dir);

    int n_x = 1;
    int n_y = 1;
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    auto train_db = get_dataloader(x_train_path, y_train_path, mu_x, sigma_x,
                                   mu_y, sigma_y, 20, n_x, n_y, true);
    auto test_db = get_dataloader(x_test_path, y_test_path, train_db.mu_x,
                                  train_db.sigma_x, train_db.mu_y,
                                  train_db.sigma_y, 100, n_x, n_y, true);

    // Model
    LayerBlock block_2(Linear(50, 50), ReLU(), Linear(50, 50), ReLU());
    LayerBlock block_1(Linear(50, 50), ReLU(), Linear(50, 50), ReLU());

    // NOTE: block_1
    ResNetBlock resnet_1(block_1, Linear(50, 50));
    ResNetBlock resnet_2(block_2);

    Linear linear(1, 50);

    Sequential model(Linear(1, 50), ReLU(), resnet_1, Linear(50, 1));

    model.set_threads(1);
    // model.to_device("cuda");

    //////////////////////////////////////////////////////////////////////
    // Training
    //////////////////////////////////////////////////////////////////////
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine seed_e(seed);
    int batch_size = 10;
    float sigma_obs = 0.06;
    int iters = test_db.num_data / batch_size;
    std::vector<float> var_obs(batch_size * n_y, pow(sigma_obs, 2));
    std::vector<float> x_batch(batch_size * n_x, 0.0f);
    std::vector<float> y_batch(batch_size * n_y, 0.0f);
    std::vector<int> batch_idx(batch_size);
    auto data_idx = create_range(train_db.num_data);
    for (int e = 0; e < 1; e++) {
        // if (e > 0) {
        //     // Shuffle data
        //     std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
        // }
        for (int i = 0; i < 1; i++) {
            // Load data
            get_batch_idx(data_idx, i * batch_size, batch_size, batch_idx);
            get_batch_data(test_db.x, batch_idx, n_x, x_batch);
            get_batch_data(test_db.y, batch_idx, n_y, y_batch);

            // Forward pass
            model.forward(x_batch);

            // Output layer
            update_output_delta_z(*model.output_z_buffer, y_batch, var_obs,
                                  model.input_delta_z_buffer->delta_mu,
                                  model.input_delta_z_buffer->delta_var);

            // Backward pass
            model.backward();
            model.step();
        }
    }
}

int test_resnet_1d_toy() {
    resnet_1d_toy();
    return 0;
}
///////////////////////////////////////////////////////////////////////////////
// File:         test_fnn_cpu_v2.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 25, 2023
// Updated:      November 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_fnn_cpu_v2.h"

void forward_fnn_v2()
/*
 */
{
    // Data preprocessing
    std::string x_train_dir, y_train_dir, x_test_dir, y_test_dir;
    std::vector<std::string> x_train_path, y_train_path, x_test_path,
        y_test_path;
    SavePath path;
    path.curr_path = get_current_dir();
    std::string data_path = path.curr_path + "/test/data/" + "Boston_housing";
    x_train_dir = data_path + "/x_train.csv";
    y_train_dir = data_path + "/y_train.csv";
    x_test_dir = data_path + "/x_test.csv";
    y_test_dir = data_path + "/y_test.csv";
    x_train_path.push_back(x_train_dir);
    y_train_path.push_back(y_train_dir);
    x_test_path.push_back(x_test_dir);
    y_test_path.push_back(y_test_dir);

    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    auto train_db = get_dataloader(x_train_path, y_train_path, mu_x, sigma_x,
                                   mu_y, sigma_y, 455, 13, 1, true);
    auto test_db = get_dataloader(x_test_path, y_test_path, mu_x, sigma_x, mu_y,
                                  sigma_y, 51, 13, 1, true);

    // TAGI network
    LayerStack model;
    model.add_layer(std::make_unique<FullyConnectedLayer>(13, 10));
    model.add_layer(std::make_unique<Relu>());
    model.add_layer(std::make_unique<FullyConnectedLayer>(10, 5));
    model.add_layer(std::make_unique<Relu>());
    model.add_layer(std::make_unique<FullyConnectedLayer>(5, 1));

    // Forward pass
    HiddenStates input_states(26, 2);
    model.forward(input_states.mu_z, input_states.var_z);

    // Backward pass
    std::vector<float> obs = {1, 2};
    std::vector<float> var_obs = {1, 1};

    update_output_delta_z(model.output_z_buffer, obs, var_obs,
                          model.input_delta_z_buffer.delta_mu,
                          model.input_delta_z_buffer.delta_var);

    model.backward();

    int check = 1;
}

int test_fnn_cpu_v2() {
    forward_fnn_v2();
    return 0;
}
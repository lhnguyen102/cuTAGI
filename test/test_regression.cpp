///////////////////////////////////////////////////////////////////////////////
// File:         test_regression.cpp
// Description:  Auxiliar independent script to perform regression
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_regression.h"

void regression_train(TagiNetworkCPU &net, Dataloader &db, int epochs) {
    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(net.prop.batch_size * net.prop.n_x, 0);
    std::vector<float> Sx_batch(net.prop.batch_size * net.prop.n_x,
                                pow(net.prop.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                               pow(net.prop.sigma_v, 2));
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);

    // *TODO: Is there any better way?
    if (net.prop.is_full_cov) {
        float var_x = pow(net.prop.sigma_x, 2);
        auto Sx_f = initialize_upper_triu(var_x, net.prop.n_x);
        Sx_f_batch = repmat_vector(Sx_f, net.prop.batch_size);
    }

    for (int e = 0; e < epochs; e++) {
        std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                                   pow(net.prop.sigma_v, 2));

        for (int i = 0; i < n_iter; i++) {
            // Load data
            get_batch_idx(data_idx, i * net.prop.batch_size,
                          net.prop.batch_size, batch_idx);
            get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);
            get_batch_data(db.y, batch_idx, net.prop.n_y, y_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Feed backward for hidden states
            net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

            // Feed backward for parameters
            net.param_feed_backward();
        }
    }
}

void forward_pass(TagiNetworkCPU &net, Dataloader &db) {
    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(net.prop.batch_size * net.prop.n_x, 0);
    std::vector<float> Sx_batch(net.prop.batch_size * net.prop.n_x,
                                pow(net.prop.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<int> batch_idx(net.prop.batch_size);

    for (int i = 0; i < n_iter; i++) {
        // Load data
        get_batch_idx(data_idx, i * net.prop.batch_size, net.prop.batch_size,
                      batch_idx);
        get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);

        // Feed forward
        net.feed_forward(x_batch, Sx_batch, Sx_f_batch);
    }
}

///////////////////////////////////////////////////////////////////////////////
// File:         test_lstm.cpp
// Description:  Auxiliar independent script to perform lstm
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      March 16, 2023
// Updated:      March 16, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_lstm.h"

void train_time_series(TagiNetworkCPU &net, Dataloader &db, int n_epochs) {
    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    int n_input_ts = net.prop.n_x * net.prop.input_seq_len;
    int n_output_ts = net.prop.n_y;
    int n_input_seq_batch = net.prop.batch_size * net.prop.input_seq_len;
    std::vector<int> data_idx = create_range(db.num_data);

    // Initialize the data's variables
    std::vector<float> x_batch(n_input_ts * net.prop.batch_size, 0);
    std::vector<float> Sx_batch(n_input_ts * net.prop.batch_size,
                                pow(net.prop.sigma_x, 2));
    std::vector<float> Sx_f_batch;
    std::vector<float> y_batch(n_output_ts * net.prop.batch_size, 0);
    std::vector<float> V_batch(n_output_ts * net.prop.batch_size,
                               pow(net.prop.sigma_v, 2));
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);

    for (int e = 0; e < n_epochs; e++) {
        std::vector<float> V_batch(net.prop.batch_size * n_output_ts,
                                   pow(net.prop.sigma_v, 2));

        for (int i = 0; i < n_iter; i++) {
            // Load data
            get_batch_idx(data_idx, i * net.prop.batch_size,
                          net.prop.batch_size, batch_idx);
            get_batch_data(db.x, batch_idx, n_input_ts, x_batch);
            get_batch_data(db.y, batch_idx, n_output_ts, y_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Feed backward for hidden states
            net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

            // Feed backward for parameters
            net.param_feed_backward();

            // Save current cell & hidden states for next step
            if (net.prop.batch_size == 1 && net.prop.input_seq_len == 1) {
                save_prev_states_cpu(net.prop, net.state);
            }
        }
    }

    // Retrieve homocesdastic noise distribution's parameters
    if (net.prop.noise_type.compare("homosce") == 0) {
        get_homosce_noise_param(net.state.noise_state.ma_v2b_prior,
                                net.state.noise_state.Sa_v2b_prior,
                                net.prop.mu_v2b, net.prop.sigma_v2b);
    }
}
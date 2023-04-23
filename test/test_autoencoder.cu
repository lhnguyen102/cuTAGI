///////////////////////////////////////////////////////////////////////////////
// File:         test_autoencoder.cu
// Description:  Auxiliar independent script to perform image generation
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      April 12, 2023
// Updated:      April 13, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_autoencoder.cuh"

void train_autoencoder(TagiNetwork &net_e, TagiNetwork &net_d, ImageData &imdb,
                       int n_classes) {
    // Batch size check
    if (net_e.prop.batch_size != net_d.prop.batch_size) {
        throw std::invalid_argument(
            " Batch size is not equal - Task - Autoencoder");
    }

    // Compute number of data
    int n_iter = imdb.num_data / net_d.prop.batch_size;

    // Input and output layer
    std::vector<float> x_batch, Sx_batch, y_batch, y_batch_e, V_batch_e,
        delta_mz, delta_Sz;
    std::vector<int> data_idx = create_range(imdb.num_data);
    std::vector<int> test_data_idx = create_range(imdb.num_data);
    std::vector<int> batch_idx(net_d.prop.batch_size);
    std::vector<int> idx_ud_batch(net_d.prop.nye * net_d.prop.batch_size, 0);
    std::vector<int> idx_ud_batch_e(net_e.prop.nye * net_e.prop.batch_size, 0);
    std::vector<int> label_batch(net_d.prop.batch_size, 0);

    x_batch.resize(net_e.prop.batch_size * net_e.prop.n_x, 0);
    Sx_batch.resize(net_e.prop.batch_size * net_e.prop.n_x,
                    powf(net_e.prop.sigma_x, 2));
    y_batch.resize(net_d.prop.batch_size * net_d.prop.n_y, 0);
    y_batch_e.resize(net_e.prop.batch_size * net_e.prop.n_y, 0);
    V_batch_e.resize(net_e.prop.batch_size * net_e.prop.n_y, 0);

    std::vector<float> Sx_f_batch;

    std::vector<float> V_batch(net_d.prop.batch_size * net_d.prop.n_y,
                               powf(net_d.prop.sigma_v, 2));

    for (int i = 0; i < n_iter; i++) {
        ;
        // TODO: Make a cleaner way to handle both cases
        if (i == 0) {
            net_e.prop.ra_mt = 0.0f;
            net_d.prop.ra_mt = 0.0f;
        } else {
            net_e.prop.ra_mt = 0.9f;
            net_d.prop.ra_mt = 0.9f;
        }

        // Load input data for encoder and output data for decoder
        get_batch_idx(data_idx, i * net_d.prop.batch_size,
                      net_e.prop.batch_size, batch_idx);
        get_batch_data(imdb.images, batch_idx, net_e.prop.n_x, x_batch);
        get_batch_data(imdb.labels, batch_idx, 1, label_batch);

        // Feed forward for encoder
        net_e.feed_forward(x_batch, Sx_batch, Sx_f_batch);

        // Get all output's states of encoder
        net_e.get_all_network_outputs();

        // Feed forward for decoder
        net_d.connected_feed_forward(net_e.ma, net_e.Sa, net_e.mz, net_e.Sz,
                                     net_e.J);

        // Feed backward for decoder
        net_d.state_feed_backward(x_batch, V_batch, idx_ud_batch);
        net_d.param_feed_backward();

        // Feed backward for encoder
        std::tie(delta_mz, delta_Sz) = net_d.get_state_delta_mean_var();
        net_e.state_feed_backward(delta_mz, delta_Sz, idx_ud_batch);
        net_e.param_feed_backward();
    }
}

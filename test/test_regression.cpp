#include "test_regression.h"

/**
 * @brief Perform a linear regression on the train data.
 *
 * @param[out] net the network to train with specified architecture and
 * parameters
 * @param[in] db the database to train the network on
 */
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
        if (e > 0) {
            // Decay observation noise
            decay_obs_noise(net.prop.sigma_v, net.prop.decay_factor_sigma_v,
                            net.prop.sigma_v_min);
        }

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

/**
 * @brief Perform a linear regression on the test data.
 *
 * @param[out] net the network to test with specified architecture and
 * parameters
 * @param[in] db the database to test the network on
 */
void regression_test(TagiNetworkCPU &net, Dataloader &db) {
    // Number of data points
    int n_iter = db.num_data / net.prop.batch_size;
    int derivative_layer = 0;

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

    // Output results
    std::vector<float> ma_batch_out(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> Sa_batch_out(net.prop.batch_size * net.prop.n_y, 0);
    std::vector<float> ma_out(db.num_data * net.prop.n_y, 0);
    std::vector<float> Sa_out(db.num_data * net.prop.n_y, 0);

    // Derivative results for the input layers
    std::vector<float> mdy_batch_in, Sdy_batch_in, mdy_in, Sdy_in;
    if (net.prop.collect_derivative) {
        mdy_batch_in.resize(net.prop.batch_size * net.prop.n_x, 0);
        Sdy_batch_in.resize(net.prop.batch_size * net.prop.n_x, 0);
        mdy_in.resize(db.num_data * net.prop.n_x, 0);
        Sdy_in.resize(db.num_data * net.prop.n_x, 0);
    }

    int mt_idx = 0;

    // Prediction
    for (int i = 0; i < n_iter; i++) {
        mt_idx = i * net.prop.batch_size * net.prop.n_y;

        // Load data
        get_batch_idx(data_idx, i * net.prop.batch_size, net.prop.batch_size,
                      batch_idx);
        get_batch_data(db.x, batch_idx, net.prop.n_x, x_batch);
        get_batch_data(db.y, batch_idx, net.prop.n_y, y_batch);

        // Feed forward
        net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

        // Derivatives
        if (net.prop.collect_derivative) {
            compute_network_derivatives_cpu(net.prop, net.theta, net.state,
                                            derivative_layer);
            get_input_derv_states(net.state.derv_state.md_layer,
                                  net.state.derv_state.Sd_layer, mdy_batch_in,
                                  Sdy_batch_in);
            update_vector(mdy_in, mdy_batch_in, mt_idx, net.prop.n_x);
            update_vector(Sdy_in, Sdy_batch_in, mt_idx, net.prop.n_x);
        }

        // Get hidden states for output layers
        output_hidden_states(net.state, net.prop, ma_batch_out, Sa_batch_out);

        // Update the final hidden state vector for last layer
        update_vector(ma_out, ma_batch_out, mt_idx, net.prop.n_y);
        update_vector(Sa_out, Sa_batch_out, mt_idx, net.prop.n_y);
    }
    // Denormalize data
    std::vector<float> sy_norm(db.y.size(), 0);
    std::vector<float> my(sy_norm.size(), 0);
    std::vector<float> sy(sy_norm.size(), 0);
    std::vector<float> y_test(sy_norm.size(), 0);

    // Compute log-likelihood
    for (int k = 0; k < db.y.size(); k++) {
        sy_norm[k] = pow(Sa_out[k] + pow(net.prop.sigma_v, 2), 0.5);
    }
    denormalize_mean(ma_out, db.mu_y, db.sigma_y, net.prop.n_y, my);
    denormalize_mean(db.y, db.mu_y, db.sigma_y, net.prop.n_y, y_test);
    denormalize_std(sy_norm, db.mu_y, db.sigma_y, net.prop.n_y, sy);

    // Compute metrics
    auto mse = mean_squared_error(my, y_test);
    auto log_lik = avg_univar_log_lik(my, y_test, sy);
}

#include "test_classification.cuh"

void test_classification(TagiNetwork &net, ImageData &imdb,
                         ImageData &test_imdb, int n_epochs, int n_classes) {
    /*Classification task

    Args:
        Net: Tagi network
        imdb: Image database
        n_epochs: Number of epochs
        n_classes: Number of classes of image data
        path: Directory stored the final results
        debug: Debugging mode allows saving inference data
     */
    {
        // Seed
        unsigned seed = 1;
        std::default_random_engine seed_e(seed);

        // Compute number of data points
        int n_iter = imdb.num_data / net.prop.batch_size;
        int test_n_iter = test_imdb.num_data / net.prop.batch_size;
        std::vector<int> data_idx = create_range(imdb.num_data);
        std::vector<int> test_data_idx = create_range(test_imdb.num_data);

        // Input and output layer
        auto hrs = class_to_obs(n_classes);
        std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
        std::vector<int> batch_idx(net.prop.batch_size);
        std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);
        std::vector<int> label_batch(net.prop.batch_size, 0);

        x_batch.resize(net.prop.batch_size * net.prop.n_x, 0);
        Sx_batch.resize(net.prop.batch_size * net.prop.n_x,
                        powf(net.prop.sigma_x, 2));
        y_batch.resize(net.prop.batch_size * hrs.n_obs, 0);
        V_batch.resize(net.prop.batch_size * hrs.n_obs,
                       powf(net.prop.sigma_v, 2));

        // *TODO: Is there any better way?
        std::vector<float> Sx_f_batch;
        if (net.prop.is_full_cov) {
            float var_x = powf(net.prop.sigma_x, 2);
            auto Sx_f = initialize_upper_triu(var_x, net.prop.n_x);
            Sx_f_batch = repmat_vector(Sx_f, net.prop.batch_size);
        }

        // Error rate for training
        std::vector<int> error_rate(imdb.num_data, 0);
        std::vector<float> prob_class(imdb.num_data * n_classes);
        std::vector<int> error_rate_batch;
        std::vector<float> prob_class_batch;

        // Error rate for testing
        std::vector<float> test_epoch_error_rate(n_epochs, 0);
        std::vector<int> test_error_rate(test_imdb.num_data, 0);
        std::vector<float> prob_class_test(test_imdb.num_data * n_classes);
        std::vector<float> test_epoch_prob_class(test_imdb.num_data *
                                                 n_classes * n_epochs);
        int mt_idx = 0;

        for (int e = 0; e < n_epochs; e++) {
            /* TRAINING */
            if (e > 0) {
                // Shufle data
                std::shuffle(data_idx.begin(), data_idx.end(), seed_e);
            }
            // Timer
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < n_iter; i++) {
                // TODO: Make a cleaner way to handle both cases
                if (i == 0 && e == 0) {
                    net.prop.ra_mt = 0.0f;
                } else {
                    net.prop.ra_mt = 0.9f;
                }

                // Load data
                get_batch_idx(data_idx, i * net.prop.batch_size,
                              net.prop.batch_size, batch_idx);
                get_batch_data(imdb.images, batch_idx, net.prop.n_x, x_batch);
                get_batch_data(imdb.obs_label, batch_idx, hrs.n_obs, y_batch);
                get_batch_data(imdb.obs_idx, batch_idx, hrs.n_obs,
                               idx_ud_batch);
                get_batch_data(imdb.labels, batch_idx, 1, label_batch);

                // Feed forward
                net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

                // Feed backward for hidden states
                net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

                // Feed backward for parameters
                net.param_feed_backward();

                // Compute error rate
                net.get_network_outputs();
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(net.ma, net.Sa, label_batch, hrs, n_classes,
                              net.prop.batch_size);
                mt_idx = i * net.prop.batch_size;
                update_vector(error_rate, error_rate_batch, mt_idx, 1);
            }

            /* TESTING */
            for (int i = 0; i < test_n_iter; i++) {
                // TODO: set = 0.9 when i > 0 or disable mean and variance in
                // feed forward
                net.prop.ra_mt = 0.0f;

                // Load data
                get_batch_idx(test_data_idx, i, net.prop.batch_size, batch_idx);
                get_batch_data(test_imdb.images, batch_idx, net.prop.n_x,
                               x_batch);
                get_batch_data(test_imdb.obs_label, batch_idx, hrs.n_obs,
                               y_batch);
                get_batch_data(test_imdb.obs_idx, batch_idx, hrs.n_obs,
                               idx_ud_batch);
                get_batch_data(test_imdb.labels, batch_idx, 1, label_batch);

                // Feed forward
                net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

                // Compute error rate
                net.get_network_outputs();
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(net.ma, net.Sa, label_batch, hrs, n_classes,
                              net.prop.batch_size);
                mt_idx = i * net.prop.batch_size;
                update_vector(test_error_rate, error_rate_batch, mt_idx, 1);
            }

            auto test_avg_error = compute_average_error_rate(
                test_error_rate, test_imdb.num_data, test_imdb.num_data);
            test_epoch_error_rate[e] = test_avg_error;
        }
        net.theta_gpu.copy_device_to_host();
    }
}
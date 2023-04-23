///////////////////////////////////////////////////////////////////////////////
// File:         test_classification.cu
// Description:  Auxiliar independent script to perform classification
// Authors:      Miquel Florensa, Luong-Ha Nguyen & James-A. Goulet
// Created:      March 21, 2023
// Updated:      March 21, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_classification.cuh"

void train_classification(TagiNetwork &net, ImageData &imdb, int n_classes) {
    int n_epochs = 1;

    // Compute number of data points
    int n_iter = imdb.num_data / net.prop.batch_size;
    std::vector<int> data_idx = create_range(imdb.num_data);
    // Input and output layer
    net.prop.nye = imdb.output_len;
    std::vector<float> x_batch, Sx_batch, y_batch, V_batch;
    std::vector<int> batch_idx(net.prop.batch_size);
    std::vector<int> idx_ud_batch(net.prop.nye * net.prop.batch_size, 0);
    std::vector<int> label_batch(net.prop.batch_size, 0);

    x_batch.resize(net.prop.batch_size * net.prop.n_x, 0);
    Sx_batch.resize(net.prop.batch_size * net.prop.n_x,
                    powf(net.prop.sigma_x, 2));
    y_batch.resize(net.prop.batch_size * imdb.output_len, 0);
    V_batch.resize(net.prop.batch_size * imdb.output_len,
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

    int mt_idx = 0;

    for (int e = 0; e < n_epochs; e++) {
        std::vector<float> V_batch(net.prop.batch_size * net.prop.n_y,
                                   powf(net.prop.sigma_v, 2));
        // Timer
        for (int i = 0; i < n_iter; i++) {
            // TODO: Make a cleaner way to handle both cases
            if (i == 0 && e == 0) {
                net.prop.ra_mt = 0.0f;
            } else {
                net.prop.ra_mt = 0.9f;
            }

            // Load data
            get_batch_images_labels(imdb, data_idx, net.prop.batch_size, i,
                                    x_batch, y_batch, idx_ud_batch,
                                    label_batch);

            // Feed forward
            net.feed_forward(x_batch, Sx_batch, Sx_f_batch);

            // Feed backward for hidden states
            net.state_feed_backward(y_batch, V_batch, idx_ud_batch);

            // Feed backward for parameters
            net.param_feed_backward();

            // Compute error rate
            net.get_network_outputs();
            if (net.prop.activations.back() == net.prop.act_names.hr_softmax) {
                std::tie(error_rate_batch, prob_class_batch) =
                    get_error(net.ma, net.Sa, label_batch, n_classes,
                              net.prop.batch_size);
            } else {
                error_rate_batch = get_class_error(
                    net.ma, label_batch, n_classes, net.prop.batch_size);
            }
            // std::tie(error_rate_batch, prob_class_batch) = get_error(
            //     net.ma, net.Sa, label_batch, n_classes, net.prop.batch_size);
            mt_idx = i * net.prop.batch_size;
            update_vector(error_rate, error_rate_batch, mt_idx, 1);
        }
    }
}
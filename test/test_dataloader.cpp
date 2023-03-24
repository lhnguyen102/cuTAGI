///////////////////////////////////////////////////////////////////////////////
// File:         test_dataloader.cpp
// Description:  Auxiliar data loader file for testing
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      March 22, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "test_dataloader.h"

Dataloader train_data(std::string problem, TagiNetworkCPU &net,
                      std::string data_path, bool normalize) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_train_data;
    x_dir = data_path + "/x_train.csv";
    y_dir = data_path + "/y_train.csv";

    if (problem == "Boston_housing") {
        num_train_data = 455;
    } else if (problem == "1D") {
        num_train_data = 20;
    } else if (problem == "1D_full_cov") {
        num_train_data = 500;
    } else if (problem == "1D_derivatives") {
        num_train_data = 100;
    } else {
        num_train_data = 400;
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    // Initialize the mu and sigma
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;

    // Train data
    return get_dataloader(x_path, y_path, mu_x, sigma_x, mu_y, sigma_y,
                          num_train_data, net.prop.n_x, net.prop.n_y,
                          normalize);
}

Dataloader test_data(std::string problem, TagiNetworkCPU &net,
                     std::string data_path, Dataloader &train_db,
                     bool normalize) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_test_data;
    x_dir = data_path + "/x_test.csv";
    y_dir = data_path + "/y_test.csv";

    if (problem == "Boston_housing") {
        num_test_data = 51;
    } else if (problem == "1D") {
        num_test_data = 100;
    } else if (problem == "1D_full_cov") {
        num_test_data = 100;
    } else if (problem == "1D_derivatives") {
        num_test_data = 200;
    } else {
        num_test_data = 200;
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    // Test data
    return get_dataloader(x_path, y_path, train_db.mu_x, train_db.sigma_x,
                          train_db.mu_y, train_db.sigma_y, num_test_data,
                          net.prop.n_x, net.prop.n_y, normalize);
}

Dataloader test_time_series_datloader(Network &net, std::string mode,
                                      int num_features, std::string data_path,
                                      std::vector<int> output_col,
                                      bool data_norm) {
    int num;
    std::vector<std::string> data_file;
    if (mode.compare("train") == 0) {
        num = 924;
        data_file = {data_path + "/x_train.csv"};
        data_file.push_back(data_path + "/y_train.csv");
    } else {
        num = 232;
        data_file = {data_path + "/x_test.csv"};
        data_file.push_back(data_path + "/y_test.csv");
    }

    Dataloader db;
    int num_outputs = output_col.size();
    std::vector<float> x(num_features * num, 0), cat_x;

    // Load input data from csv file that contains the input & output data.
    // NOTE: csv file need to have a header for each columns
    for (int i = 0; i < data_file.size(); i++) {
        read_csv(data_file[i], x, num_features, true);
        cat_x.insert(cat_x.end(), x.begin(), x.end());
    };

    // Compute sample mean and std for dataset
    std::vector<float> mu_x(num_features, 0);
    std::vector<float> sigma_x(num_features, 1);
    std::vector<float> mu_y(num_outputs, 0);
    std::vector<float> sigma_y(num_outputs, 1);
    if (data_norm) {
        compute_mean_std(cat_x, mu_x, sigma_x, num_features);
        normalize_data(cat_x, mu_x, sigma_x, num_features);
        for (int i = 0; i < num_outputs; i++) {
            mu_y[i] = mu_x[output_col[i]];
            sigma_y[i] = sigma_x[output_col[i]];
        }
    }

    // Create rolling windows
    int num_samples =
        (cat_x.size() / num_features - net.input_seq_len - net.output_seq_len) /
            net.seq_stride +
        1;
    std::vector<float> input_data(net.input_seq_len * num_features *
                                  num_samples);
    std::vector<float> output_data(net.output_seq_len * num_outputs *
                                   num_samples);
    create_rolling_windows(cat_x, output_col, net.input_seq_len,
                           net.output_seq_len, num_features, net.seq_stride,
                           input_data, output_data);

    // Set data to output variable
    db.x = input_data;
    db.mu_x = mu_x;
    db.sigma_x = sigma_x;
    db.nx = num_features * net.input_seq_len;

    db.y = output_data;
    db.mu_y = mu_y;
    db.sigma_y = sigma_y;
    db.ny = num_outputs * net.output_seq_len;
    db.num_data = num_samples;

    return db;
}

ImageData image_dataloader(std::string data_name, std::string data_path, 
                           std::string mode, std::vector<float> mu, 
                           std::vector<float> sigma, 
                           int num_classes, Network &net_prop) {

    // Directory of the data
    std::string x_dir, y_dir;

    // Number of data
    int num;

    if (mode.compare("train") == 0) {
        x_dir = data_path + "/train-images-subset-idx3-ubyte";
        y_dir = data_path + "/train-labels-subset-idx1-ubyte";
        num = 600;
    } else {
        x_dir = data_path + "/test-images-subset-idx3-ubyte";
        y_dir = data_path + "/test-labels-subset-idx1-ubyte";
        num = 100;
    }

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    return get_images(data_name, x_path, y_path, mu, sigma, num, 
                      num_classes, net_prop);
}
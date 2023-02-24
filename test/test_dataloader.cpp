///////////////////////////////////////////////////////////////////////////////
// File:         test_dataloader.cpp
// Description:  Auxiliar data loader file for testing
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
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

    if (problem == "Boston_housing") {
        x_dir = data_path + "/x_train.csv";
        y_dir = data_path + "/y_train.csv";
        num_train_data = 455;
    } else if (problem == "1D") {
        num_train_data = 20;
        x_dir = data_path + "/x_train.csv";
        y_dir = data_path + "/y_train.csv";
    } else {
        num_train_data = 400;
        x_dir = data_path + "/x_train.csv";
        y_dir = data_path + "/y_train.csv";
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

    if (problem == "Boston_housing") {
        x_dir = data_path + "/x_test.csv";
        y_dir = data_path + "/y_test.csv";
        num_test_data = 51;
    } else if (problem == "1D") {
        num_test_data = 100;
        x_dir = data_path + "/x_test.csv";
        y_dir = data_path + "/y_test.csv";
    } else {
        num_test_data = 200;
        x_dir = data_path + "/x_test.csv";
        y_dir = data_path + "/y_test.csv";
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

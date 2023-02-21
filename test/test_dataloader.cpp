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

/**
 * @brief Compare two csv files.
 *
 * @param[in] file1 the first file to compare
 * @param[in] file2 the second file to compare
 */
bool compare_csv_files(const std::string &file1, const std::string &file2) {
    std::ifstream f1(file1);
    std::ifstream f2(file2);

    if (!f1.is_open() || !f2.is_open()) {
        std::cout << "Error opening one of the files." << std::endl;
        return false;
    }

    std::string line1, line2;
    int lineNumber = 1;

    while (std::getline(f1, line1) && std::getline(f2, line2)) {
        if (line1 != line2) {
            std::cout << "Files differ at line " << lineNumber << std::endl;
            std::cout << "File 1: " << line1 << std::endl;
            std::cout << "File 2: " << line2 << std::endl;
            return false;
        }
        lineNumber++;
    }

    if (std::getline(f1, line1) || std::getline(f2, line2)) {
        std::cout << "Files have different number of lines." << std::endl;
        return false;
    }

    f1.close();
    f2.close();

    return true;
}

/**
 * @brief Train the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 */
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

    // Train data

    // Initialize the mu and sigma
    std::vector<float> mu_x, sigma_x, mu_y, sigma_y;
    return get_dataloader(x_path, y_path, mu_x, sigma_x, mu_y, sigma_y,
                          num_train_data, net.prop.n_x, net.prop.n_y,
                          normalize);
}

/**
 * @brief Test the data
 *
 * @param problem contains a string of the problem name
 * @param net contains the network
 * @param data_path contains the path to the data
 * @param train_db contains the training data
 */
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

ImageData image_train_data(std::string problem, std::string data_path,
                           std::vector<float> &mu, std::vector<float> &sigma,
                           int w, int h, int d, HrSoftmax &hrs) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_train_data;

    x_dir = data_path + "/x_train.csv";
    y_dir = data_path + "/y_train.csv";
    num_train_data = 60000;

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    return get_images(problem, x_path, y_path, mu, sigma, w, h, d, hrs,
                      num_train_data);
}

ImageData image_train_data(std::string data_path, std::vector<float> mu,
                           std::vector<float> sigma, int w, int h, int d,
                           HrSoftmax &hrs) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_train_data;

    x_dir = data_path + "/x_train";
    y_dir = data_path + "/y_train";
    num_train_data = 60000;

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    return get_images("mnist", x_path, y_path, mu, sigma, w, h, d, hrs,
                      num_train_data);
}

ImageData image_test_data(std::string data_path, std::vector<float> mu,
                          std::vector<float> sigma, int w, int h, int d,
                          HrSoftmax &hrs) {
    // Directory of the data
    std::string x_dir, y_dir;

    // Num of data in each set
    int num_test_data;

    x_dir = data_path + "/x_test";
    y_dir = data_path + "/y_test";
    num_test_data = 10000;

    std::vector<std::string> x_path;
    std::vector<std::string> y_path;
    x_path.push_back(x_dir);
    y_path.push_back(y_dir);

    return get_images("mnist", x_path, y_path, mu, sigma, w, h, d, hrs,
                      num_test_data);
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
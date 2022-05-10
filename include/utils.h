///////////////////////////////////////////////////////////////////////////////
// File:         utils.h
// Description:  Header file for utils
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 10, 2022
// Updated:      May 09, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <string>
#include <vector>

#include "common.h"
#include "data_transfer.cuh"
#include "indices.h"
#include "net_prop.h"
#include "struct_var.h"

template <typename T>
void read_csv(std::string filename, std::vector<T> &v) {
    // Create input filestream
    std::ifstream myFile(filename);

    // Check if the file can open
    if (!myFile.is_open()) {
        throw std::runtime_error("Could not open the file - utils.h");
    }

    // Initialization
    std::string line;
    T d;

    // Set column counter
    int row_counter = 0;
    while (std::getline(myFile, line)) {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Get the data
        if (ss.good()) {
            ss >> d;
            v[row_counter] = d;
            row_counter++;
        }
    }

    // Check output size
    if (v.size() != row_counter) {
        throw std::runtime_error("There is missing data - utils.h");
    }
}

template <typename T>
void write_csv(std::string filename, T &v) {
    // Create file name
    std::ofstream file(filename);

    // Save data to created file
    for (int i = 0; i < v.size(); i++) {
        file << v[i] << "\n";
    }

    // Close the file
    file.close();
}

void save_error_rate(std::string &res_path, std::vector<float> &error_rate,
                     std::string &suffix);

void save_generated_images(std::string &res_path, std::vector<float> &imgs,
                           std::string &suffix);

void save_hidden_states(std::string &res_path, NetState &state);

void save_delta_param(std::string &res_path, DeltaParamGPU &d_param);

void save_inference_results(std::string &res_path, DeltaStateGPU &d_state_gpu,
                            Param &theta);

void save_idx(std::string &idx_path, IndexOut &idx);

void save_param(std::string &param_path, Param &theta);

void load_net_param(std::string &model_name, std::string &net_name,
                    std::string &path, Param &theta);

void save_net_param(std::string &model_name, std::string &net_name,
                    std::string path, Param &theta);

void save_net_prop(std::string &param_path, std::string &idx_path, Param &theta,
                   IndexOut &idx);

void save_autoencoder_net_prop(Param &theta_e, Param &theta_d, IndexOut &idx_e,
                               IndexOut &idx_d, std::string &debug_path);

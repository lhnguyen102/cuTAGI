///////////////////////////////////////////////////////////////////////////////
// File:         test_utils.h
// Description:  Header file for the utils functions for unitest
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      April 30, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <stdio.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include "../include/common.h"
#include "../include/cost.h"
#include "../include/data_transfer_cpu.h"
#include "../include/dataloader.h"
#include "../include/indices.h"
#include "../include/net_init.h"
#include "../include/net_prop.h"
#include "../include/struct_var.h"

/**
 * @brief Initialize the weights and bias
 *
 * @param weights weights
 * @param weights_sc weights standard deviation
 * @param bias bias
 * @param bias_sc bias standard deviation
 * @param net neural network
 */
template <typename Net>
void add_weights_and_bias(std::vector<std::vector<float> *> &weights,
                          std::vector<std::vector<float> *> &weights_sc,
                          std::vector<std::vector<float> *> &bias,
                          std::vector<std::vector<float> *> &bias_sc,
                          Net &net) {
    weights.push_back(&net.theta.mw);
    weights.push_back(&net.theta.Sw);

    weights_sc.push_back(&net.theta.mw_sc);
    weights_sc.push_back(&net.theta.Sw_sc);

    bias.push_back(&net.theta.mb);
    bias.push_back(&net.theta.Sb);

    bias_sc.push_back(&net.theta.mb_sc);
    bias_sc.push_back(&net.theta.Sb_sc);
}

/**
 * @brief Initialize the forward states
 *
 * @param forward_states forward states
 * @param net neural network
 */
template <typename Net>
void add_forward_states(std::vector<std::vector<float> *> &forward_states,
                        Net &net) {
    forward_states.push_back(&net.state.mz);
    forward_states.push_back(&net.state.Sz);
    forward_states.push_back(&net.state.ma);
    forward_states.push_back(&net.state.Sa);
    forward_states.push_back(&net.state.J);
}

/**
 * @brief Initialize the backward states
 *
 * @param backward_states backward states
 * @param net neural network
 */
template <typename Net>
void add_backward_states(std::vector<std::vector<float>> &backward_states,
                         std::string &backward_states_header, Net &net,
                         int layers) {
    for (int i = 0; i < layers - 2; i++) {
        backward_states_header +=
            "mean_" + std::to_string(i) + ",sigma_" + std::to_string(i) + ",";
        backward_states.push_back(std::get<0>(net.get_inovation_mean_var(i)));
        backward_states.push_back(std::get<1>(net.get_inovation_mean_var(i)));
    }
}

/**
 * @brief Indicate if a directory exists
 *
 * @param path directory path
 */
inline bool directory_exists(const std::string &path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    } else if (info.st_mode & S_IFDIR) {
        return true;
    } else {
        return false;
    }
}

/**
 * @brief Create a directory if it does not exist
 *
 * @param path directory path
 */
inline bool create_directory_if_not_exists(const std::string &path) {
    if (directory_exists(path)) {
        return true;
    } else {
        int result = mkdir(path.c_str(), 0777);
        if (result == 0) {
            std::cout << "Directory created at " << path << std::endl;
            return true;
        } else {
            std::cerr << "Error creating directory at " << path << std::endl;
            return false;
        }
    }
}

/**
 * @brief Chek if two floats are almost equal
 *
 * @param test test float
 * @param ref reference float
 *
 * @return true if the floats are almost equal, false otherwise
 */
inline bool is_almost_equal(float test, float ref) {
#ifdef __APPLE__
    const float tolerance = 2e-2;
    float threshold = tolerance * std::max(std::abs(test), std::abs(ref));
#else
    const float tolerance = 1e-4;
    float epsilon = std::numeric_limits<float>::epsilon();
    float threshold;
    if (std::abs(ref) >= epsilon) {
        threshold = epsilon;
    } else {
        threshold = tolerance * std::max(std::abs(test), std::abs(ref));
    }
#endif

    return std::abs(test - ref) <= threshold;
}

/**
 * @brief Compare two vectors of vectors
 *
 * @param ref_vector reference vector of vectors
 * @param test_vector test vector of vectors
 *
 * @return true if the vectors are equal, false otherwise
 */
template <typename T>
bool compare_vectors(const std::vector<std::vector<T> *> &ref_vector,
                     const std::vector<std::vector<T> *> &test_vector,
                     std::string data, std::string vector_names) {
    if (ref_vector.size() != test_vector.size()) {
        std::cout << "Different number of vectors in " << vector_names
                  << " for " << data << " data" << std::endl;
        std::cout << "ref_vector.size() = " << ref_vector.size() << std::endl;
        std::cout << "test_vector.size() = " << test_vector.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < ref_vector.size(); i++) {
        for (size_t j = 0; j < ref_vector[i]->size(); j++) {
            float test = (*test_vector[i])[j];
            float ref = (*ref_vector[i])[j];
            bool passed = is_almost_equal(test, ref);
            if (!passed) {
                std::cout << "Different values in " << vector_names << " for "
                          << data << " data" << std::endl;
                std::cout << "ref_vector[" << i << "][" << j
                          << "] = " << std::setprecision(9) << ref << std::endl;
                std::cout << "test_vector[" << i << "][" << j
                          << "] = " << std::setprecision(9) << test
                          << std::endl;
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Read a vector of vectors from a CSV file
 *
 * @param[in] filename name of the file to read from
 * @param[out] vector vector of vectors where data is stored
 */
template <typename T>
void read_vector_from_csv(std::string filename,
                          std::vector<std::vector<T> *> &vector) {
    // Clear existing data in vectors
    for (auto &vec : vector) {
        if (vec != nullptr) {
            vec->clear();
        }
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading.");
    }

    std::string line;
    std::getline(file, line);  // Skip header
    int row_idx = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;  // Skip empty lines
        }
        if (line.back() != ',') {
            line.push_back(',');
        }
        std::stringstream line_ss(line);
        std::string cell;
        int col_idx = 0;
        while (std::getline(line_ss, cell, ',')) {
            if (col_idx >= vector.size()) {
                break;  // Skip extra columns
            }
            if (cell.empty()) {
                ++col_idx;  // Skip missing cell
            } else {
                T value;
                std::stringstream cell_ss(cell);
                cell_ss >> value;
                vector[col_idx]->push_back(value);
                ++col_idx;
            }
        }
        ++row_idx;
    }
    file.close();
}

/**
 * @brief Write a vector of vectors to a CSV file
 *
 * @param[in] filename name of the file to write to
 * @param[in] header header of the CSV file
 * @param[out] vector vector of vectors where data is stored
 */
template <typename T>
void write_vector_to_csv(std::string filename, std::string header,
                         std::vector<std::vector<T> *> &vector) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file for reading.");
    }

    file << header << std::endl;

    // Maximum number of rows in the vectors
    int rows = 0;
    for (const auto &col : vector) {
        rows = std::max(rows, (int)col->size());
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < vector.size(); j++) {
            if (i < vector[j]->size()) {
                file << std::fixed;
                file << std::setprecision(15);
                file << (*vector[j])[i];
            }
            if (j < vector.size() - 1) {
                file << ",";
            }
        }
        file << std::endl;
    }

    file.close();
}

/**
 * @brief Class to store the paths to the data files
 */
class TestSavingPaths {
   public:
    /**
     * @brief Construct a new Test Saving Paths object
     *
     * @param curr_path current path
     * @param arch architecture
     * @param data data
     * @param date date
     * @param add_encoder add encoder
     * @param add_decoder add decoder
     */
    TestSavingPaths(std::string curr_path, std::string arch, std::string data,
                    std::string date, bool add_encoder = false,
                    bool add_decoder = false) {
        std::string encoder_suffix = add_encoder ? "encoder_" : "";
        std::string decoder_suffix = add_decoder ? "decoder_" : "";
        std::string data_dir = curr_path + "/test/" + arch + "/data/" + date +
                               "_" + encoder_suffix + decoder_suffix;
        std::string path_sufix = "_" + arch + "_" + data + ".csv";

        init_param_path_w = data_dir + "init_param_weights_w" + path_sufix;
        init_param_path_w_sc =
            data_dir + "init_param_weights_w_sc" + path_sufix;
        init_param_path_b = data_dir + "init_param_bias_b" + path_sufix;
        init_param_path_b_sc = data_dir + "init_param_bias_b_sc" + path_sufix;
        opt_param_path_w = data_dir + "opt_param_weights_w" + path_sufix;
        opt_param_path_w_sc = data_dir + "opt_param_weights_w_sc" + path_sufix;
        opt_param_path_b = data_dir + "opt_param_bias_b" + path_sufix;
        opt_param_path_b_sc = data_dir + "opt_param_bias_b_sc" + path_sufix;
        forward_states_path = data_dir + "forward_hidden_states" + path_sufix;
        backward_states_path = data_dir + "backward_hidden_states" + path_sufix;
        input_derivative_path = data_dir + "input_derivative" + path_sufix;
    }

    std::string init_param_path_w;
    std::string init_param_path_w_sc;
    std::string init_param_path_b;
    std::string init_param_path_b_sc;
    std::string opt_param_path_w;
    std::string opt_param_path_w_sc;
    std::string opt_param_path_b;
    std::string opt_param_path_b_sc;
    std::string forward_states_path;
    std::string backward_states_path;
    std::string input_derivative_path;
};

/**
 * @brief Class to store the parameters and states of the network
 */
class TestParamAndStates {
   public:
    /**
     * @brief Construct a new Test Param And States object
     *
     * @param net network
     */
    template <typename Net>
    TestParamAndStates(Net &net) {
        add_weights_and_bias(weights, weights_sc, bias, bias_sc, net);
    }

    std::vector<std::vector<float> *> weights;
    std::vector<std::vector<float> *> weights_sc;
    std::vector<std::vector<float> *> bias;
    std::vector<std::vector<float> *> bias_sc;
    std::vector<std::vector<float> *> forward_states;
    std::vector<std::vector<float> *> backward_states;
    std::vector<std::vector<float> *> input_derivatives;
    std::string backward_states_header;

    /**
     * @brief Write the parameters to a CSV file
     * test_saving_paths paths to the data files
     * init true if the parameters are initialized, false if they are optimized
     */
    void write_params(TestSavingPaths test_saving_paths, bool init) {
        if (init) {
            write_vector_to_csv(test_saving_paths.init_param_path_w, "mw,Sw",
                                weights);
            write_vector_to_csv(test_saving_paths.init_param_path_w_sc,
                                "mw_sc,Sw_sc", weights_sc);
            write_vector_to_csv(test_saving_paths.init_param_path_b, "mb,Sb",
                                bias);
            write_vector_to_csv(test_saving_paths.init_param_path_b_sc,
                                "mb_sc,Sb_sc", bias_sc);
        } else {
            write_vector_to_csv(test_saving_paths.opt_param_path_w, "mw,Sw",
                                weights);
            write_vector_to_csv(test_saving_paths.opt_param_path_w_sc,
                                "mw_sc,Sw_sc", weights_sc);
            write_vector_to_csv(test_saving_paths.opt_param_path_b, "mb,Sb",
                                bias);
            write_vector_to_csv(test_saving_paths.opt_param_path_b_sc,
                                "mb_sc,Sb_sc", bias_sc);
        }
    }

    /**
     * @brief Read the states from a CSV file
     * test_saving_paths paths to the data files
     * init true if the states are initialized, false if they are optimized
     */
    void read_params(TestSavingPaths test_saving_paths, bool init) {
        if (init) {
            read_vector_from_csv(test_saving_paths.init_param_path_w, weights);
            read_vector_from_csv(test_saving_paths.init_param_path_w_sc,
                                 weights_sc);
            read_vector_from_csv(test_saving_paths.init_param_path_b, bias);
            read_vector_from_csv(test_saving_paths.init_param_path_b_sc,
                                 bias_sc);
        } else {
            read_vector_from_csv(test_saving_paths.opt_param_path_w, weights);
            read_vector_from_csv(test_saving_paths.opt_param_path_w_sc,
                                 weights_sc);
            read_vector_from_csv(test_saving_paths.opt_param_path_b, bias);
            read_vector_from_csv(test_saving_paths.opt_param_path_b_sc,
                                 bias_sc);
        }
    }
};

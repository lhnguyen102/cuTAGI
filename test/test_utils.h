///////////////////////////////////////////////////////////////////////////////
// File:         test_utils.h
// Description:  Header file for the utils functions for unitest
// Authors:      Florensa, Miquel & Luong-Ha Nguyen & James-A. Goulet
// Created:      February 20, 2023
// Updated:      February 20, 2023
// Contact:      miquelflorensa11@gmail.com & luongha.nguyen@gmail.com &
//               james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet.
// Some rights reserved.
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
#include <random>
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
            // We can't do the comparison directly because when the values are
            // written and read from a file, they are converted to strings and
            // back, which can cause some precision loss. So we convert them to
            // strings and compare the strings.
            std::stringstream aa;
            aa << (*ref_vector[i])[j];
            std::string a = aa.str();
            std::stringstream bb;
            bb << (*test_vector[i])[j];
            std::string b = bb.str();

            if (a != b) {
                std::cout << "Different values in " << vector_names << " for "
                          << data << " data" << std::endl;
                std::cout << "ref_vector[" << i << "][" << j << "] = " << a
                          << std::endl;
                std::cout << "test_vector[" << i << "][" << j << "] = " << b
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
    T value;
    int col_idx = 0;
    while (std::getline(file, line)) {
        std::stringstream line_ss(line);
        col_idx = 0;
        while (line_ss >> value) {
            if (col_idx >= vector.size()) {
                // The file has more columns than the input vector
                break;
            }
            vector[col_idx]->push_back(value);
            col_idx++;
            if (line_ss.peek() == ',') {
                line_ss.ignore();
            }
        }
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
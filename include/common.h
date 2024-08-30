///////////////////////////////////////////////////////////////////////////////
// File:         common.h
// Description:  Header file for common.h
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      April 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

template <typename T>
void print_matrix(std::vector<T> &M, int w, int h)
/*
 * Print a matrix.
 *
 * Args:
 *    M: Matrix to be printed
 *    w: Number of colunms
 *    h: Number of rows*/
{
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            std::cout << std::right << std::setw(7) << M[i * w + j];
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print_matrix_1Dvec(std::vector<T> &M)
/*
 * Print a matrix.
 *
 * Args:
 *    M: Matrix to be printed
 */
{
    for (int i = 0; i < M.size(); i++) {
        std::cout << M[i] << "  ";
    }
    std::cout << std::endl;
    std::cout << "---------" << std::endl;
}

template <typename T>
std::vector<T> cumsum(std::vector<T> &v)
/*
 * Cummulative sumation of a vector.
 *
 * Args:
 *    v: Vector
 *
 * Returns:
 *    cs: Cummulative sum of the vector v
 **/
{
    std::vector<T> cs(v.size());
    T tmp = 0;
    for (int i = 0; i < v.size(); i++) {
        tmp += v[i];
        cs[i] = tmp;
    }

    return cs;
}

template <typename T>
std::vector<T> multiply_vector_by_scalar(std::vector<T> &v, T a)
/*Multiply a vector by a scalar
 *
 * Args:
 *    v: A vector
 *    a: A scalar
 *
 * Returns:
 *    mv: Multiplied vector
 *    */
{
    std::vector<T> mv(v.size());
    for (int i = 0; i < v.size(); i++) {
        mv[i] = v[i] * a;
    }
}

template <typename T>
void push_back_with_idx(T &v, T &m, int idx)
/*
 * Put a vector into the main vector given its index.
 *
 * Args:
 *    v: Main vector
 *    m: A vector
 *    idx: Index of the m in v
 **/
{
    for (int i = 0; i < m.size(); i++) {
        v[idx + i] = m[i];
    }
}

template <typename T>
void update_vector(std::vector<T> &v, std::vector<T> &new_values, int idx,
                   int w)
/*Save new value to vector.

Args:
    v: Vector of interest
    new_values: Values to be stored in the vector v
    idx: Indices of new value in vector v
*/
{
    int N = std::min(new_values.size(), v.size() - idx) / w;
    // if (v.size() - idx < new_values.size()) {
    //     throw std::invalid_argument(
    //         "Vector capacity is insufficient - task.cu");
    // }
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < w; c++) {
            v[idx + i * w + c] = new_values[w * i + c];
        }
    }
}

template <typename T>
void read_csv(std::string &filename, std::vector<T> &v, int num_col,
              bool skip_header) {
    // Create input filestream
    std::ifstream myFile(filename);

    // Check if the file can open
    if (!myFile.is_open()) {
        throw std::runtime_error("Could not open the file - utils.h");
    }

    // Initialization
    std::string line, col;
    T d;

    // Set counter
    int count = -1;
    int line_count = 0;
    int num_data = v.size();
    while (std::getline(myFile, line)) {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        if (line_count > 0 || !skip_header) {
            // Get the data
            if (ss.good()) {
                while (std::getline(ss, col, ',')) {
                    std::stringstream ss_col(col);
                    if (ss_col.good()) {
                        count++;
                        ss_col >> d;
                        v[count] = d;
                    }
                }
            }
        }
        line_count++;
        if (count + 1 >= num_data) {
            break;
        }
    }

    // Total number of data
    int us_num_data = v.size() / num_col;
    int ac_num_row = (count + 1) / num_col;

    // Check output size
    if (num_data != count + 1) {
        std::cout << "\nUser-specified number of data: " << us_num_data << "x"
                  << num_col << "\n";
        std::cout << "Actual number of data        : " << ac_num_row << "x"
                  << num_col << "\n";
        std::cout << std::endl;
        throw std::runtime_error("There is missing data - common.h");
    }

    // Close file
    myFile.close();
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

template <typename T>
std::vector<T> repmat_vector(std::vector<T> &v, int num_copies) {
    std::vector<T> new_v(v.size() * num_copies, 0);
    for (int i = 0; i < num_copies; i++) {
        for (int j = 0; j < v.size(); j++) {
            new_v[i * v.size() + j] = v[j];
        }
    }
    return new_v;
};

std::string get_current_dir();

int sum(std::vector<int> &v);

std::vector<int> transpose_matrix(std::vector<int> &M, int w, int h);

void create_directory(std::string &path);

void decay_obs_noise(float &sigma_v, float &decay_factor, float &sigma_v_min);

void get_multithread_indices(int i, int n_batch, int rem_batch, int &start_idx,
                             int &end_idx);

//////////////////////////////////////////////////////////////////////
/// OUTPUT HIDDEN STATES
//////////////////////////////////////////////////////////////////////
void get_output_hidden_states_cpu(std::vector<float> &z, int z_pos,
                                  std::vector<float> &z_mu);

void get_output_hidden_states_ni_cpu(std::vector<float> &z, int ny, int z_pos,
                                     std::vector<float> &z_mu);

void get_noise_hidden_states_cpu(std::vector<float> &z, int ny, int z_pos,
                                 std::vector<float> &z_v2);

void get_output_states(std::vector<float> &ma, std::vector<float> Sa,
                       std::vector<float> &ma_output,
                       std::vector<float> &Sa_output, int idx);

void get_input_derv_states(std::vector<float> &md, std::vector<float> &Sd,
                           std::vector<float> &md_output,
                           std::vector<float> &Sd_output);

std::vector<float> initialize_upper_triu(float &Sx, int n);

void get_1st_column_data(std::vector<float> &dataset, int seq_len,
                         int num_outputs, std::vector<float> &sub_dataset);

//////////////////////////////////////////////////////////////////////
/// NOISE INFERENCE
//////////////////////////////////////////////////////////////////////
void set_homosce_noise_param(std::vector<float> &mu_v2b,
                             std::vector<float> &sigma_v2b,
                             std::vector<float> &ma_v2b_prior,
                             std::vector<float> &Sa_v2b_prior);

void get_homosce_noise_param(std::vector<float> &ma_v2b_prior,
                             std::vector<float> &Sa_v2b_prior,
                             std::vector<float> &mu_v2b,
                             std::vector<float> &sigma_v2b);

//////////////////////////////////////////////////////////////////////
/// DISTRIBUTION
//////////////////////////////////////////////////////////////////////
float normcdf_cpu(float x);
float normpdf_cpu(float x, float mu, float sigma);

///////////////////////////////////////////////////////
// INDEX
///////////////////////////////////////////////////////
int get_sub_layer_idx(std::vector<int> &layer, int curr_layer, int layer_label);
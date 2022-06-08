///////////////////////////////////////////////////////////////////////////////
// File:         common.cpp
// Description:  Common function used for computing indices for TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      June 04, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/common.h"

std::string get_current_dir() {
    char buff[FILENAME_MAX];  // create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    std::string current_working_dir(buff);

    return current_working_dir;
}

int sum(std::vector<int> &v)
/*
 * Compute summation of a vector.
 *
 * Args:
 *    v: Vector
 *
 * Returns:
 *    s: Summation of the vector v
 **/
{
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }

    return s;
}

std::vector<int> transpose_matrix(std::vector<int> &M, int w, int h)
/*
 * Transpose a matrix.
 *
 * Args:
 *    M: Matrix
 *
 * Returns:
 *    w: Number of columns
 *    h: Number of rows
 **/
{
    std::vector<int> tM(M.size());
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            tM[r + c * h] = M[c + r * w];
        }
    }
    return tM;
}

void create_directory(std::string &path) {
    /* Check if the directory exists if not create the folder
     */
    struct stat st = {0};
    const char *res_path_c = path.c_str();
#if defined(__linux__) || defined(__APPLE__)
    if (stat(res_path_c, &st) == -1) {
        mkdir(res_path_c, 0700);
    }
#endif

#ifdef _WIN32
    if (stat(res_path_c, &st) == -1) {
        _mdir(res_path_c);
    }
#endif
}

void decay_obs_noise(float &sigma_v, float &decay_factor, float &sigma_v_min)
/* Decrease the value of observation noise after each epoch

Args:
    sigma_v: Observation noise
    decay_factor: Decreasing percentage (default value: 0.99)
    sigma_v_min: Minimum value of observation noise (default value: 0.3)

*/
{
    sigma_v = decay_factor * sigma_v;
    if (sigma_v < sigma_v_min) {
        sigma_v = sigma_v_min;
    }
}

void get_output_states(std::vector<float> &ma, std::vector<float> Sa,
                       std::vector<float> &ma_output,
                       std::vector<float> &Sa_output, int idx)
/*Get output's distrinution

Args:
    ma: Mean of activation units of the entire network
    ma: Variance of activation units of the entire network
    ma_output: mean of activation units of the output layer
    Sa_output: Variance of activation units of the output layer
    idx: Starting index of the output layer
*/
{
    for (int i = 0; i < ma_output.size(); i++) {
        ma_output[i] = ma[idx + i];
        Sa_output[i] = Sa[idx + i];
    }
}

std::vector<float> initialize_upper_triu(float &Sx, int n)
/* Initialize the covariance matrix where only the elements of the triangle
upper matrix are stored in a vector.

Args:
    Sx: Initial value of the diagonal term of the covariance matrix
    n: Size of the covatiance matrix

Returns:
    Sx_tu: Vector of the triangle upper matrix
*/
{
    std::vector<float> Sx_tu;
    int tu;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row <= col) {
                tu = n * row + col;
                if (row == col) {
                    Sx_tu.push_back(Sx);
                } else {
                    Sx_tu.push_back(0.0f);
                }
            }
        }
    }
    return Sx_tu;
}

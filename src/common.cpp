///////////////////////////////////////////////////////////////////////////////
// File:         common.cpp
// Description:  Common function used for computing indices for TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      May 10, 2022
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
#ifdef __linux__
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

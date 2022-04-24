///////////////////////////////////////////////////////////////////////////////
// File:         common.cpp
// Description:  Common function used for computing indices for TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      February 06, 2022
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

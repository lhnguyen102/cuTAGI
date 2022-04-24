///////////////////////////////////////////////////////////////////////////////
// File:         common.h
// Description:  Header file for common.h
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      Apirl 10, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#ifdef WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

std::string get_current_dir();

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

int sum(std::vector<int> &v);

std::vector<int> transpose_matrix(std::vector<int> &M, int w, int h);

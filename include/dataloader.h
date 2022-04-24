///////////////////////////////////////////////////////////////////////////////
// File:         dataloader.h
// Description:  Header file for dataloader
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 06, 2022
// Updated:      April 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <assert.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "cost.h"
#include "struct_var.h"

std::vector<int> create_range(int N);

void get_batch_idx(std::vector<int> &idx, int iter, int B,
                   std::vector<int> &batch_idx);

template <typename T>
void get_batch_data(std::vector<T> &data, std::vector<int> &batch_idx, int w,
                    std::vector<T> &batch_data)
/*
 * Get batch of data.
 *
 * Args:
 *    data: vector of data
 *    batch_idx: Batch of indices
 *    w: Number of covariates associated with each data point
 *
 * Returns:
 *    batch_data: Batch of data
 *    */
{
    for (int r = 0; r < batch_idx.size(); r++) {
        for (int c = 0; c < w; c++) {
            batch_data[r * w + c] = data[batch_idx[r] * w + c];
        }
    }
};

ImageData get_images(std::string data_name,
                     std::vector<std::string> &image_file,
                     std::vector<std::string> &label_file,
                     std::vector<float> &mu, std::vector<float> &sigma, int w,
                     int h, int d, HrSoftmax &hrs, int num);

void normalize_images(std::vector<float> &imgs, std::vector<float> &mu,
                      std::vector<float> &sigma, int w, int h, int d, int num);

void compute_mean_std_each_channel(std::vector<float> &imgs,
                                   std::vector<float> &mu,
                                   std::vector<float> &sigma, int w, int h,
                                   int d, int num);

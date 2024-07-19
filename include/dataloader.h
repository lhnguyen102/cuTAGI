///////////////////////////////////////////////////////////////////////////////
// File:         dataloader.h
// Description:  Header file for dataloader
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 06, 2022
// Updated:      April 02, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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
#include "data_struct.h"
#include "utils.h"

struct Dataloader {
    /* Regression-like database's format
    Args:
        x: Input data
        y: Output data
        mu_x: Sample mean of input data
        sigma_x: Sample standard deviation of input data
        mu_y: Sample mean of output data
        sigma_y: Sample standard deviation of output data
        num_data: Total number of data
        nx: Number of input features
        ny: Number of output features
     */
    std::vector<float> x, y, mu_x, sigma_x, mu_y, sigma_y;
    int num_data, nx, ny;
};

struct ImageData {
    /* Image database's format
       Args:
        images: Images stored as a giant vector
        obs_label: Converted observation from the labels
        obs_idx: Observation indices assigned to each label
        labels: Raw labels

    NOTE*: In the context of TAGI, we uses the hierarchical softmax for the
    classification problem. So, we convert the raw lable for the label fictive
    observation.
    */
    std::vector<float> images;
    std::vector<float> obs_label;
    std::vector<int> obs_idx;
    std::vector<int> labels;
    int num_data, image_len, output_len;
};

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

void get_batch_images_labels(ImageData &imdb, std::vector<int> &data_idx,
                             int batch_size, int iter,
                             std::vector<float> &x_batch,
                             std::vector<float> &y_batch,
                             std::vector<int> &idx_ud_batch,
                             std::vector<int> &label_batch);

void get_batch_images(ImageData &imdb, std::vector<int> &data_idx,
                      int batch_size, int iter, std::vector<float> &x_batch,
                      std::vector<int> &label_batch);

void labels_to_hrs(std::vector<int> &labels, HRCSoftmax &hrs,
                   std::vector<float> &obs, std::vector<int> &obs_idx);

std::vector<float> label_to_one_hot(std::vector<int> &labels, int n_classes);

std::vector<float> load_mnist_images(std::string image_file, int num);

std::vector<int> load_mnist_labels(std::string label_file, int num);

std::tuple<std::vector<float>, std::vector<int>> load_cifar_images(
    std::string image_file, int num);

ImageData get_images_v2(std::string data_name,
                        std::vector<std::string> &image_file,
                        std::vector<std::string> &label_file,
                        std::vector<float> &mu, std::vector<float> &sigma,
                        int num, int num_classes, int width, int height,
                        int channel, bool is_hr_softmax);

Dataloader get_dataloader(std::vector<std::string> &input_file,
                          std::vector<std::string> &output_file,
                          std::vector<float> mu_x, std::vector<float> sigma_x,
                          std::vector<float> mu_y, std::vector<float> sigma_y,
                          int num, int nx, int ny, bool data_norm);

void normalize_images(std::vector<float> &imgs, std::vector<float> &mu,
                      std::vector<float> &sigma, int w, int h, int d, int num);

void normalize_data(std::vector<float> &x, std::vector<float> &mu,
                    std::vector<float> &sigma, int w);

void denormalize_mean(std::vector<float> &norm_my, std::vector<float> &mu,
                      std::vector<float> &sigma, int w, std::vector<float> &my);

void denormalize_std(std::vector<float> &norm_sy, std::vector<float> &mu,
                     std::vector<float> &sigma, int w, std::vector<float> &sy);

void compute_mean_std(std::vector<float> &x, std::vector<float> &mu,
                      std::vector<float> &sigma, int w);

void compute_mean_std_each_channel(std::vector<float> &imgs,
                                   std::vector<float> &mu,
                                   std::vector<float> &sigma, int w, int h,
                                   int d, int num);

void create_rolling_windows(std::vector<float> &data,
                            std::vector<int> &output_col, int num_input_ts,
                            int num_output_ts, int num_features, int stride,
                            std::vector<float> &input_data,
                            std::vector<float> &output_data);

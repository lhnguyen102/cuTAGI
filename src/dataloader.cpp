///////////////////////////////////////////////////////////////////////////////
// File:         dataloader.cpp
// Description:  Load different batches of data to network
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      February 06, 2022
// Updated:      July 24, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "../include/dataloader.h"

std::vector<int> create_range(int N)
/*
 * Create a range vector.
 *
 * Args:
 *    N: Maximum number
 *
 * Returns:
 *    v: Vector contains number from 0 to N-1 with a step size of 1
 * */
{
    std::vector<int> v(N);
    for (int i = 0; i < N; i++) {
        v[i] = i;
    }
    return v;
}

void get_batch_idx(std::vector<int> &idx, int iter, int B,
                   std::vector<int> &batch_idx)
/*
 * Get batch indices.
 *
 * Args:
 *    idx: Vector of the entire indices
 *    iter: ith iteration
 *    B: Batch size
 *
 * Returns:
 *    batch_idx: Batch of indices
 *    */
{
    for (int i = 0; i < B; i++) {
        batch_idx[i] = idx[iter + i];
    }
}

//////////////////////////////////////
// MNIST
//////////////////////////////////////
int to_int(char *p)
/*
 * Convert hexadeciamal to integer*/
{
    return ((p[0] & 255) << 24) | ((p[1] & 255) << 16) | ((p[2] & 255) << 8) |
           ((p[3] & 255) << 0);
}

std::vector<float> load_mnist_images(std::string image_file, int num)
/*
 * Load the mnist images.
 *
 * Args:
 *    image_file: Name of the file
 *    num: Number of images to be loaded
 *
 * Returns:
 *    n_images: Images
 *    */
{
    // Load file
    std::ifstream data_file(image_file.c_str(),
                            std::ios::in | std::ios::binary);
    if (data_file.fail()) {
        throw std::invalid_argument("Image files do not exist.");
    }

    // Check the magic number (see http://yann.lecun.com/exdb/mnist/)
    char p[4];
    data_file.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 2051);

    // Load the size of images and total number of image
    int m_size;
    int m_rows;
    int m_cols;
    float pixel_j;
    data_file.read(p, 4);
    m_size = to_int(p);
    if (num > 0 && num < m_size) m_size = num;
    data_file.read(p, 4);
    m_rows = to_int(p);
    data_file.read(p, 4);
    m_cols = to_int(p);

    // Load the images
    std::vector<float> n_images(m_size * m_cols * m_rows);
    char *q = new char[m_rows * m_cols];
    for (int i = 0; i < m_size; i++) {
        data_file.read(q, m_rows * m_cols);
        for (int j = 0; j < m_rows * m_cols; j++) {
            if (q[j] < 0) {
                pixel_j = (256 + q[j]) / 255.0f;
            } else {
                pixel_j = q[j] / 255.0f;
            }
            n_images[m_rows * m_cols * i + j] = pixel_j;
        }
    }
    delete[] q;
    data_file.close();

    return n_images;
}

std::vector<int> load_mnist_labels(std::string label_file, int num)
/*
 * Load the mnist labels.
 *
 * Args:
 *    label_file: Name of the file
 *    num: Number of labels to be loaded
 *
 * Returns:
 *    n_labels: labels
 *
 *NOTE: The label order correspond to the image order.
 *    */

{
    // Load the file name
    std::ifstream data_file(label_file.c_str(),
                            std::ios::in | std::ios::binary);

    if (data_file.fail()) {
        throw std::invalid_argument(
            "Label files do not exist - dataloader.cpp");
    }

    // Check the magic number
    char p[4];
    data_file.read(p, 4);
    int magic_number = to_int(p);
    assert(magic_number == 2049);

    // Get the size
    int m_size;
    data_file.read(p, 4);
    m_size = to_int(p);
    if (num > 0 && num < m_size) m_size = num;

    // Load the labels
    std::vector<int> n_labels;
    for (int i = 0; i < m_size; i++) {
        data_file.read(p, 1);
        int label = p[0];
        n_labels.push_back(label);
    }
    return n_labels;
}

void labels_to_hrs(std::vector<int> labels, HrSoftmax &hrs,
                   std::vector<float> &obs, std::vector<int> &obs_idx)
/*
 * Convert labels to hierarchical softmax.
 * */
{
    for (int i = 0; i < labels.size(); i++) {
        int label = labels[i];
        for (int j = 0; j < hrs.n_obs; j++) {
            obs[i * hrs.n_obs + j] = hrs.obs[label * hrs.n_obs + j];
            obs_idx[i * hrs.n_obs + j] = hrs.idx[label * hrs.n_obs + j];
        }
    }
}

//////////////////////////////////////
// CIFAR
//////////////////////////////////////
std::tuple<std::vector<float>, std::vector<int>> load_cifar_images(
    std::string image_file, int num) {
    std::ifstream data_file(image_file.c_str(),
                            std::ios::in | std::ios::binary);
    if (data_file.fail()) {
        throw std::invalid_argument("Image files do not exist.");
    }

    int n_size = 10000;
    int n_rows = 32;
    int n_cols = 32;
    int n_channels = 3;
    if (num != 0 && num < n_size) n_size = num;

    std::vector<float> n_images(n_size * n_cols * n_rows * n_channels, 0);
    std::vector<int> n_labels(n_size, 0);

    char *q = new char[n_rows * n_cols * n_channels + 1];
    float pixel_j;
    for (int i = 0; i < n_size; i++) {
        // Read data fro file
        data_file.read(q, n_rows * n_cols * n_channels + 1);

        // Load labels. See https://www.cs.toronto.edu/~kriz/cifar.html for
        // further details.
        n_labels[i] = q[0];
        for (int j = 1; j < n_rows * n_cols * n_channels + 1; j++) {
            if (q[j] < 0) {
                pixel_j = 256.0f + q[j];
            } else {
                pixel_j = q[j];
            }
            n_images[n_rows * n_cols * n_channels * i + (j - 1)] =
                pixel_j / 255.0f;
        }
    }
    delete[] q;
    data_file.close();

    return {n_images, n_labels};
}

ImageData get_images(std::string data_name,
                     std::vector<std::string> &image_file,
                     std::vector<std::string> &label_file,
                     std::vector<float> &mu, std::vector<float> &sigma, int w,
                     int h, int d, HrSoftmax &hrs, int num)
/*Load image dataset

 Args:
    data_name: Name of dataset e.g. mnist, cifar
    image_file: Directory path to image file
    label_file: Directory path to label file
    num: Number of images files
    HrSoftmax: Hierarchical softmax for classification

Returns:
    ImageData: Image database
*/
{
    std::vector<float> imgs;
    std::vector<int> labels;
    if (data_name == "mnist") {
        // Load images
        for (int i = 0; i < image_file.size(); i++) {
            auto img_i = load_mnist_images(image_file[i], num);
            imgs.insert(imgs.end(), img_i.begin(), img_i.end());
        }

        // Load labels
        for (int i = 0; i < label_file.size(); i++) {
            auto label_i = load_mnist_labels(label_file[i], num);
            labels.insert(labels.end(), label_i.begin(), label_i.end());
        }

    } else if (data_name == "cifar") {
        // Load images and labels
        std::vector<float> img_i;
        std::vector<int> label_i;
        for (int i = 0; i < image_file.size(); i++) {
            std::tie(img_i, label_i) = load_cifar_images(image_file[i], num);
            imgs.insert(imgs.end(), img_i.begin(), img_i.end());
            labels.insert(labels.end(), label_i.begin(), label_i.end());
        }

    } else {
        throw std::invalid_argument("Dataset does not exist - dataloader.cpp");
    }

    // Convert label to hierarchical softmax
    std::vector<float> obs(hrs.n_obs * num);
    std::vector<int> obs_idx(hrs.n_obs * num);
    labels_to_hrs(labels, hrs, obs, obs_idx);

    // Normalization
    if (mu.size() > 0 && sigma.size() > 0) {
        normalize_images(imgs, mu, sigma, w, h, d, num);
    } else {
        mu.resize(d);
        sigma.resize(d);
        compute_mean_std_each_channel(imgs, mu, sigma, w, h, d, num);
    }

    return {imgs, obs, obs_idx, labels, num};
}

Dataloader get_dataloader(std::vector<std::string> &input_file,
                          std::vector<std::string> &output_file,
                          std::vector<float> mu_x, std::vector<float> sigma_x,
                          std::vector<float> mu_y, std::vector<float> sigma_y,
                          int num, int nx, int ny, bool data_norm)
/* Get dataloader for input and output data.

Args:
    input_file: Input data file (*.csv)
    output_file: Output data file (*.csv)
    mu_x: Sample mean of input data
    sigma_x: Sample standard deviation of input data
    mu_y: Sample mean of output data
    sigma_y: Sample standard deviation of output data
    num: Total number of data
    nx: Number of input features
    ny: Number of output features

Returns:
    dataset: Dataloader
 */
{
    Dataloader db;
    std::vector<float> x(nx * num, 0), y(ny * num, 0);

    // Load input data
    for (int i = 0; i < input_file.size(); i++) {
        read_csv(input_file[i], x, nx, true);
        db.x.insert(db.x.end(), x.begin(), x.end());
    };

    for (int i = 0; i < output_file.size(); i++) {
        read_csv(output_file[i], y, ny, true);
        db.y.insert(db.y.end(), y.begin(), y.end());
    };

    // Compute sample mean and std for dataset
    mu_x.resize(nx, 0);
    sigma_x.resize(nx, 1);
    mu_y.resize(ny, 0);
    sigma_y.resize(ny, 1);
    if (data_norm) {
        compute_mean_std(db.x, mu_x, sigma_x, nx);
        compute_mean_std(db.y, mu_y, sigma_y, ny);
        // Normalize dataset
        normalize_data(db.x, mu_x, sigma_x, nx);
        normalize_data(db.y, mu_y, sigma_y, ny);
    }

    // Set data to output variable
    db.mu_x = mu_x;
    db.sigma_x = sigma_x;
    db.mu_y = mu_y;
    db.sigma_y = sigma_y;
    db.nx = nx;
    db.ny = ny;
    int actual_num = db.x.size() / nx;
    db.num_data = num;

    return db;
}

void normalize_images(std::vector<float> &imgs, std::vector<float> &mu,
                      std::vector<float> &sigma, int w, int h, int d, int num) {
    /*
     * Normalize the images between -1 and 1.
     *
     * Args:
     *    imgs: Image dataset
     *    mu: Mean of each channel
     *    sigmu_x: Sample mean of input data
        sigma_x: Sample standard deviation of input datama: Standard deviation
     for each channel
     *    w: Width of images
     *    h: Height of images
     *    d: Depth of images
     *    num: Number of images
     *    */

    for (int i = 0; i < num; i++) {
        for (int j = 0; j < d; j++) {
            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    imgs[i * (w * h * d) + j * (h * w) + r * w + c] =
                        (imgs[i * (w * h * d) + j * (h * w) + r * w + c] -
                         mu[j]) /
                        sigma[j];
                }
            }
        }
    }
}

void normalize_data(std::vector<float> &x, std::vector<float> &mu,
                    std::vector<float> &sigma, int w)
/* Normalize data in the range [-1, 1].

Args:
    x: Dataset
    mu: Mean of data
    sigma: Standard deviation of data
    w: Number of columns of x
*/
{
    int h = x.size() / w;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            x[row * w + col] = (x[row * w + col] - mu[col]) / sigma[col];
        }
    }
}

void denormalize_mean(std::vector<float> &norm_my, std::vector<float> &mu,
                      std::vector<float> &sigma, int w, std::vector<float> &my)
/* Transfer mean value from normalized to original spaces.

Args:
    norm_my: Normalized mean values
    mu: Mean of data
    sigma: Standard deviation of data
    w: Number of columns of x
    my: Mean value in origial space
*/
{
    int h = norm_my.size() / w;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            my[row * w + col] = (norm_my[row * w + col] * sigma[col]) + mu[col];
        }
    }
}

void denormalize_std(std::vector<float> &norm_sy, std::vector<float> &mu,
                     std::vector<float> &sigma, int w, std::vector<float> &sy)
/* Transfer standard deviation value from normalized to original spaces.

Args:
    norm_Sy: Normalized variance values
    mu: Mean of data
    sigma: Standard deviation of data
    w: Number of columns of x
    Sy: Variance value in origial space
*/
{
    int h = norm_sy.size() / w;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            sy[row * w + col] = norm_sy[row * w + col] * sigma[col];
        }
    }
}

void compute_mean_std(std::vector<float> &x, std::vector<float> &mu,
                      std::vector<float> &sigma, int w)
/* Compute stat mean and standard deviation of data.

Args:
    x: Dataset
    mu: Mean of data
    sigma: Standard deviation of data
    w: Number of columns of x
 */
{
    int h = x.size() / w;

    // Compute mean
    for (int col = 0; col < w; col++) {
        float sum = 0;
        for (int row = 0; row < h; row++) {
            sum += x[row * w + col];
        }
        mu[col] = sum / h;
    }

    // Compute standard deviation
    for (int col = 0; col < w; col++) {
        float sq_sum = 0;
        for (int row = 0; row < h; row++) {
            sq_sum += pow(x[row * w + col] - mu[col], 2);
        }
        sigma[col] = pow(sq_sum / h, 0.5);
    }
}

void compute_mean_std_each_channel(std::vector<float> &imgs,
                                   std::vector<float> &mu,
                                   std::vector<float> &sigma, int w, int h,
                                   int d, int num)
/*Compute mean and standard deviation for each channel of images.

Args:
  imgs: Images
  w: Width of image
  h: Height of image
  d: Depth of image
  num: Number of images

Returns:
  mu: Mean for each image channel
  sigma: Standard deviation of each image channel

*/
{
    // Compute mean value
    for (int j = 0; j < d; j++) {
        float tmp = 0.0f;
        for (int i = 0; i < num; i++) {
            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    tmp += imgs[i * (w * h * d) + j * (w * h) + r * w + c];
                }
            }
        }
        mu[j] = tmp / (num * w * h);
    }

    // Standard deviation value
    for (int j = 0; j < d; j++) {
        float tmp = 0.0f;
        for (int i = 0; i < num; i++) {
            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    tmp += pow(
                        imgs[i * (w * h * d) + j * (w * h) + r * w + c] - mu[j],
                        2.0);
                }
            }
        }
        sigma[j] = pow(tmp / (num * w * h - 1), 0.5);
    }
}

///////////////////////////////////////////////////////////////////////////////
// File:         test_mha_cpu.cpp
// Description:  Unittest for multi-head self-attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 07, 2023
// Updated:      May 08, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_mha_cpu.h"

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

std::vector<float> create_observation_matrix(int h_a, int w_a, int w_b)
/*Create observation matrix*/
{
    int num_rows = h_a * w_b;
    int num_cols = h_a * w_a * w_b;
    int num_elements = num_rows * num_cols;
    std::vector<float> matrix_obs(num_elements, 0);
    int idx;
    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < w_a; k++) {
            idx = i * num_cols + k + i * w_a;
            matrix_obs[idx] = 1;
        }
    }
    return matrix_obs;
}

std::vector<float> compute_prod_mean(std::vector<float> &m_a,
                                     std::vector<float> &m_b, int h_a, int w_a,
                                     int w_b)
/* Get product vector of two random variables*/
{
    int num_cols = h_a * w_a * w_b;
    std::vector<float> prod(num_cols, 0);
    for (int i = 0; i < h_a; i++) {
        for (int j = 0; j < w_b; j++) {
            for (int k = 0; k < w_a; k++) {
                prod[i * w_a * w_b + j * w_a + k] =
                    m_a[i * w_a + k] * m_b[j + k * w_b];
            }
        }
    }
    return prod;
}

std::vector<float> forward_mean(std::vector<float> &F, std::vector<float> &mu_x,
                                int h_F, int w_F, int w_mu) {
    std::vector<float> mu_pred(h_F * w_mu, 0);
    for (int row = 0; row < h_F; row++) {
        for (int col = 0; col < w_mu; col++) {
            for (int k = 0; k < w_F; k++) {
                mu_pred[row * w_mu + col] +=
                    F[row * w_F + k] * mu_x[col + k * w_mu];
            }
        }
    }
    return mu_pred;
}

std::vector<float> compute_prod_var(std::vector<float> &m_a,
                                    std::vector<float> &m_b,
                                    std::vector<float> &v_a,
                                    std::vector<float> &v_b, int h_a, int w_a,
                                    int w_b)
/*Get product vector of two random variables*/
{
    int num_cols = h_a * w_a * w_b;
    int idx_a, idx_b, idx_prod;
    std::vector<float> prod(num_cols, 0);
    for (int i = 0; i < h_a; i++) {
        for (int j = 0; j < w_b; j++) {
            for (int k = 0; k < w_a; k++) {
                idx_a = i * w_a + k;
                idx_b = j + k * w_b;
                idx_prod = i * w_a * w_b + j * w_a + k;
                prod[idx_prod] = v_a[idx_a] * v_b[idx_b] +
                                 v_a[idx_a] * powf(m_b[idx_b], 2) +
                                 v_b[idx_b] * powf(m_a[idx_a], 2);
            }
        }
    }
    return prod;
}

float random_float(float min, float max, int seed) {
    static std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

std::vector<float> generate_random_matrix(int height, int width, int min_value,
                                          int max_value, int seed) {
    std::vector<float> matrix(width * height, 0);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            matrix[row * width + col] =
                random_float(min_value, max_value, seed);
        }
    }
    return matrix;
}

std::vector<float> generate_positive_random_matrix(int height, int width,
                                                   int min_value, int max_value,
                                                   int seed) {
    std::vector<float> matrix(width * height, 0);
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            matrix[row * width + col] =
                std::abs(random_float(min_value, max_value, seed));
        }
    }
    return matrix;
}

std::vector<float> transpose_matrix(int height, int width,
                                    std::vector<float> &matrix) {
    std::vector<float> matrix_transpose(width * height, 0);
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < height; col++) {
            matrix_transpose[row * height + col] = matrix[col + row * width];
        }
    }
    return matrix_transpose;
}

bool test_query_key() {
    int seed = 1;
    int batch_size = 1;
    int num_heads = 1;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate data
    auto mu_query =
        generate_random_matrix(timestep, head_size, min_value, max_value, seed);
    auto var_query = generate_positive_random_matrix(
        timestep, head_size, min_value, max_value, seed);
    auto mu_key =
        generate_random_matrix(timestep, head_size, min_value, max_value, seed);
    auto var_key = generate_positive_random_matrix(timestep, head_size,
                                                   min_value, max_value, seed);

    auto mu_key_tsp = transpose_matrix(timestep, head_size, mu_key);
    auto var_key_tsp = transpose_matrix(timestep, head_size, var_key);

    // Observation matrix and product variables
    auto F = create_observation_matrix(timestep, head_size, timestep);
    auto prod_mu =
        compute_prod_mean(mu_query, mu_key_tsp, timestep, head_size, timestep);

    auto prod_var =
        compute_prod_var(mu_query, mu_key_tsp, var_query, var_key_tsp, timestep,
                         head_size, timestep);

    return true;
}
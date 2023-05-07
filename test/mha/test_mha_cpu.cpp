///////////////////////////////////////////////////////////////////////////////
// File:         test_mha_cpu.cpp
// Description:  Unittest for multi-head self-attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 07, 2023
// Updated:      May 07, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_mha_cpu.h"

std::vector<float> create_observation_matrix(int w_a, int h_a, int w_b)
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
                                     std::vector<float> &m_b, int w_a, int h_a,
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

std::vector<float> compute_prod_var(std::vector<float> &m_a,
                                    std::vector<float> &m_b,
                                    std::vector<float> &v_a,
                                    std::vector<float> &v_b, int w_a, int h_a,
                                    int w_b)
/* Get product vector of two random variables*/
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
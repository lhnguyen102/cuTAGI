///////////////////////////////////////////////////////////////////////////////
// File:         test_mha_cpu.cpp
// Description:  Unittest for multi-head self-attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 07, 2023
// Updated:      May 14, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "test_mha_cpu.h"

void print_float_matrix(std::vector<float> &M, int w, int h, int precision = 2)
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
            std::cout << std::right << std::setw(10) << std::fixed
                      << std::setprecision(precision) << M[i * w + j];
        }
        std::cout << std::endl;
    }
}

std::vector<float> get_vec_2d(std::vector<float> &vec_4d, int pos, int len) {
    std::vector<float> vec(len, 0);
    for (int i = 0; i < len; i++) {
        vec[i] = vec_4d[i + pos];
    }
    return vec;
}

void merge_vec(std::vector<float> &vec_2d, int pos, int len,
               std::vector<float> &vec_4d) {
    for (int i = 0; i < len; i++) {
        vec_4d[i + pos] = vec_2d[i];
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

std::vector<float> transpose_matrix(int height, int width,
                                    std::vector<float> &matrix) {
    std::vector<float> matrix_transpose(width * height, 0);
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < height; col++) {
            matrix_transpose[row * height + col] = matrix[col * width + row];
        }
    }
    return matrix_transpose;
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

std::vector<float> compute_prod_mean_qk_batch(std::vector<float> &mu_key,
                                              std::vector<float> &mu_query,
                                              int batch_size, int num_heads,
                                              int timestep, int head_size) {
    std::vector<float> mu_prod(
        batch_size * num_heads * timestep * timestep * head_size, 0);
    std::vector<float> mu_key_2d, mu_key_tsp_2d, mu_query_2d, tmp;
    int pos, pos_prod;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            pos =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_prod = i * num_heads * timestep * timestep * head_size +
                       j * timestep * timestep * head_size;
            mu_key_2d = get_vec_2d(mu_key, pos, timestep * head_size);
            mu_key_tsp_2d = transpose_matrix(timestep, head_size, mu_key_2d);
            mu_query_2d = get_vec_2d(mu_query, pos, timestep * head_size);

            tmp = compute_prod_mean(mu_query_2d, mu_key_tsp_2d, timestep,
                                    head_size, timestep);
            merge_vec(tmp, pos_prod, timestep * timestep * head_size, mu_prod);
        }
    }

    return mu_prod;
}

std::vector<float> compute_prod_var_qk_batch(std::vector<float> &mu_key,
                                             std::vector<float> &mu_query,
                                             std::vector<float> &var_key,
                                             std::vector<float> &var_query,
                                             int batch_size, int num_heads,
                                             int timestep, int head_size) {
    std::vector<float> var_prod(
        batch_size * num_heads * timestep * timestep * head_size, 0);
    std::vector<float> mu_key_2d, mu_key_tsp_2d, var_key_2d, var_key_tsp_2d,
        mu_query_2d, var_query_2d, tmp;
    int pos, pos_prod;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            pos =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_prod = i * num_heads * timestep * timestep * head_size +
                       j * timestep * timestep * head_size;

            mu_key_2d = get_vec_2d(mu_key, pos, timestep * head_size);
            mu_key_tsp_2d = transpose_matrix(timestep, head_size, mu_key_2d);

            var_key_2d = get_vec_2d(var_key, pos, timestep * head_size);
            var_key_tsp_2d = transpose_matrix(timestep, head_size, var_key_2d);

            mu_query_2d = get_vec_2d(mu_query, pos, timestep * head_size);
            var_query_2d = get_vec_2d(var_query, pos, timestep * head_size);

            tmp =
                compute_prod_var(mu_query_2d, mu_key_tsp_2d, var_query_2d,
                                 var_key_tsp_2d, timestep, head_size, timestep);

            merge_vec(tmp, pos_prod, timestep * timestep * head_size, var_prod);
        }
    }

    return var_prod;
}

std::vector<float> forward_mean_qk(std::vector<float> &F,
                                   std::vector<float> &mu_x, int h_F, int w_F) {
    std::vector<float> mu_pred(h_F, 0);
    for (int row = 0; row < h_F; row++) {
        for (int k = 0; k < w_F; k++) {
            mu_pred[row] += F[row * w_F + k] * mu_x[k];
        }
    }
    return mu_pred;
}

std::vector<float> forward_var_qk(std::vector<float> &F,
                                  std::vector<float> &var_x, int h_F, int w_F) {
    std::vector<float> var_pred(h_F, 0);
    for (int row = 0; row < h_F; row++) {
        for (int k = 0; k < w_F; k++) {
            var_pred[row] += F[row * w_F + k] * var_x[k] * F[row * w_F + k];
        }
    }
    return var_pred;
}

std::vector<float> forward_mean_qk_batch(std::vector<float> &F,
                                         std::vector<float> &mu_prod, int h_F,
                                         int w_F, int batch_size,
                                         int num_heads) {
    std::vector<float> mu_pred(batch_size * num_heads * h_F, 0);
    std::vector<float> tmp, mu_prod_2d;
    int pos, pos_pred;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            pos = i * num_heads * w_F + j * w_F;
            mu_prod_2d = get_vec_2d(mu_prod, pos, w_F);
            tmp = forward_mean_qk(F, mu_prod_2d, h_F, w_F);
            pos_pred = i * num_heads * h_F + j * h_F;
            merge_vec(tmp, pos_pred, h_F, mu_pred);
        }
    }
    return mu_pred;
}

std::vector<float> forward_var_qk_batch(std::vector<float> &F,
                                        std::vector<float> &var_prod, int h_F,
                                        int w_F, int batch_size,
                                        int num_heads) {
    std::vector<float> var_pred(batch_size * num_heads * h_F, 0);
    std::vector<float> tmp, var_prod_2d;
    int pos, pos_pred;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            pos = i * num_heads * w_F + j * w_F;
            var_prod_2d = get_vec_2d(var_prod, pos, w_F);
            tmp = forward_var_qk(F, var_prod_2d, h_F, w_F);
            pos_pred = i * num_heads * h_F + j * h_F;
            merge_vec(tmp, pos_pred, h_F, var_pred);
        }
    }
    return var_pred;
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

std::vector<float> create_random_key_query(int batch_size, int num_heads,
                                           int timestep, int head_size,
                                           float min_value, float max_value,
                                           int seed, bool pos_value = false) {
    std::vector<float> vec_qk(batch_size * num_heads * timestep * head_size, 0);
    std::vector<float> vec;
    int pos;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            if (pos_value) {
                vec = generate_positive_random_matrix(
                    timestep, head_size, min_value, max_value, seed);
            } else {
                vec = generate_random_matrix(timestep, head_size, min_value,
                                             max_value, seed);
            }
            pos =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            merge_vec(vec, pos, timestep * head_size, vec_qk);
        }
    }
    return vec_qk;
}

bool test_query_key() {
    int seed = 1;
    int batch_size = 1;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 4;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Observation matrix and product variables
    auto F = create_observation_matrix(timestep, head_size, timestep);
    // std::cout << "Observation matrix"
    //           << "\n"
    //           << std::endl;

    // print_float_matrix(F, timestep * timestep * head_size, timestep *
    // timestep,
    //                    0);

    // Generate data
    auto mu_query = create_random_key_query(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);
    auto var_query =
        create_random_key_query(batch_size, num_heads, timestep, head_size,
                                min_value, max_value, seed, true);

    auto mu_key = create_random_key_query(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);
    auto var_key =
        create_random_key_query(batch_size, num_heads, timestep, head_size,
                                min_value, max_value, seed, true);

    auto mu_prod = compute_prod_mean_qk_batch(mu_key, mu_query, batch_size,
                                              num_heads, timestep, head_size);
    auto var_prod =
        compute_prod_var_qk_batch(mu_key, mu_query, var_key, var_query,
                                  batch_size, num_heads, timestep, head_size);

    auto mu_pred = forward_mean_qk_batch(F, mu_prod, timestep * timestep,
                                         timestep * head_size * timestep,
                                         batch_size, num_heads);
    auto var_pred = forward_var_qk_batch(F, var_prod, timestep * timestep,
                                         timestep * head_size * timestep,
                                         batch_size, num_heads);

    // Self-attention function
    std::vector<float> mu_pred_mha(timestep * timestep * num_heads * batch_size,
                                   0);
    std::vector<float> var_pred_mha(
        timestep * timestep * num_heads * batch_size, 0);
    query_key(mu_query, var_query, mu_key, var_key, 0, batch_size, num_heads,
              timestep, head_size, mu_pred_mha, var_pred_mha);

    // Test
    for (int i = 0; i < mu_pred_mha.size(); i++) {
        if (mu_pred_mha[i] != mu_pred[i] || var_pred_mha[i] != var_pred[i]) {
            std::cout << "Test FAIL"
                      << "\n"
                      << std::endl;
            return false;
        }
    }
    std::cout << "Test PASS" << std::endl;

    return true;
}
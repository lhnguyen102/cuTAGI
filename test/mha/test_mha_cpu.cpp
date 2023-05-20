///////////////////////////////////////////////////////////////////////////////
// File:         test_mha_cpu.cpp
// Description:  Unittest for multi-head self-attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 07, 2023
// Updated:      May 20, 2023
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

std::vector<float> create_mask_matrix(int timestep) {
    std::vector<float> vec(timestep * timestep, 0);
    for (int i = 0; i < timestep; i++) {
        for (int j = 0; j < timestep; j++) {
            if (j <= i) {
                vec[i * timestep + j] = 1;
            }
        }
    }
    return vec;
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

std::vector<std::vector<float>> compute_prod_mask_qk_mean_var(
    std::vector<float> &mask, std::vector<float> &mu_qk,
    std::vector<float> &var_qk, int batch_size, int num_heads, int timestep)
/* Compute mean var between mask and the product of query and key
 */
{
    // Initlization
    int prod_size = timestep * timestep * timestep;
    std::vector<float> mu_prod(batch_size * num_heads * prod_size, 0);
    std::vector<float> var_prod(batch_size * num_heads * prod_size, 0);
    int pos, pos_prod;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // Get 2d matrix position
            pos = i * num_heads * timestep * timestep + j * timestep * timestep;
            pos_prod = i * num_heads * prod_size + j * prod_size;

            // Get 2d matrix for mask and the product of query and key
            auto mu_qk_2d = get_vec_2d(mu_qk, pos, timestep * timestep);
            auto var_qk_2d = get_vec_2d(var_qk, pos, timestep * timestep);

            // Compute the product between mask and the product of query and key
            auto mu_prod_2d =
                compute_prod_mean(mask, mu_qk_2d, timestep, timestep, timestep);
            auto var_prod_2d = compute_prod_var(mask, mu_qk_2d, mask, var_qk_2d,
                                                timestep, timestep, timestep);

            // Merge to the main vector
            merge_vec(mu_prod_2d, pos_prod, prod_size, mu_prod);
            merge_vec(var_prod_2d, pos_prod, prod_size, var_prod);
        }
    }
    return {mu_prod, var_prod};
}

std::vector<std::vector<float>> compute_prod_var_qk_batch(
    std::vector<float> &mu_key, std::vector<float> &mu_query,
    std::vector<float> &var_key, std::vector<float> &var_query, int batch_size,
    int num_heads, int timestep, int head_size)
/*Compute mean and var for the prodcut of query and key*/
{
    // Initialization
    std::vector<float> mu_prod(
        batch_size * num_heads * timestep * timestep * head_size, 0);
    std::vector<float> var_prod(
        batch_size * num_heads * timestep * timestep * head_size, 0);
    std::vector<float> mu_key_2d, mu_key_tsp_2d, var_key_2d, var_key_tsp_2d,
        mu_query_2d, var_query_2d;
    int pos, pos_prod;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // Get 2d matrix position
            pos =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_prod = i * num_heads * timestep * timestep * head_size +
                       j * timestep * timestep * head_size;

            // Get 2d matrix for key and query
            mu_key_2d = get_vec_2d(mu_key, pos, timestep * head_size);
            mu_key_tsp_2d = transpose_matrix(timestep, head_size, mu_key_2d);

            var_key_2d = get_vec_2d(var_key, pos, timestep * head_size);
            var_key_tsp_2d = transpose_matrix(timestep, head_size, var_key_2d);

            mu_query_2d = get_vec_2d(mu_query, pos, timestep * head_size);
            var_query_2d = get_vec_2d(var_query, pos, timestep * head_size);

            // Compute product
            auto mu_prod_2d = compute_prod_mean(mu_query_2d, mu_key_tsp_2d,
                                                timestep, head_size, timestep);
            auto var_prod_2d =
                compute_prod_var(mu_query_2d, mu_key_tsp_2d, var_query_2d,
                                 var_key_tsp_2d, timestep, head_size, timestep);

            // Merge to main vector
            merge_vec(mu_prod_2d, pos_prod, timestep * timestep * head_size,
                      mu_prod);
            merge_vec(var_prod_2d, pos_prod, timestep * timestep * head_size,
                      var_prod);
        }
    }

    return {mu_prod, var_prod};
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

std::vector<std::vector<float>> forward_mean_var_qk_batch(
    std::vector<float> &F, std::vector<float> &mu_prod,
    std::vector<float> &var_prod, int h_F, int w_F, int batch_size,
    int num_heads)
/*Perform the TAGI feed forward*/
{
    std::vector<float> mu_pred(batch_size * num_heads * h_F, 0);
    std::vector<float> var_pred(batch_size * num_heads * h_F, 0);
    std::vector<float> mu_prod_2d, var_prod_2d;
    int pos, pos_pred;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // 2D matrix pos
            pos = i * num_heads * w_F + j * w_F;
            pos_pred = i * num_heads * h_F + j * h_F;

            // Get 2d mean and variance product
            mu_prod_2d = get_vec_2d(mu_prod, pos, w_F);
            var_prod_2d = get_vec_2d(var_prod, pos, w_F);

            // TAGI forward
            auto mu_fwd = forward_mean_qk(F, mu_prod_2d, h_F, w_F);
            auto var_fwd = forward_var_qk(F, var_prod_2d, h_F, w_F);

            // Merge to main vector
            merge_vec(mu_fwd, pos_pred, h_F, mu_pred);
            merge_vec(var_fwd, pos_pred, h_F, var_pred);
        }
    }
    return {mu_pred, var_pred};
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

std::vector<float> create_random_mha_matrix(int batch_size, int num_heads,
                                            int height, int width,
                                            float min_value, float max_value,
                                            int seed, bool pos_value = false) {
    std::vector<float> vec_qk(batch_size * num_heads * height * width, 0);
    std::vector<float> vec;
    int pos;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            if (pos_value) {
                vec = generate_positive_random_matrix(height, width, min_value,
                                                      max_value, seed);
            } else {
                vec = generate_random_matrix(height, width, min_value,
                                             max_value, seed);
            }
            pos = i * num_heads * height * width + j * height * width;
            merge_vec(vec, pos, height * width, vec_qk);
        }
    }
    return vec_qk;
}

bool is_vector_equal(std::string &test_name, std::vector<float> &ref,
                     std::vector<float> &test) {
    if (ref.size() != test.size()) {
        std::cout << test_name << " FAIL"
                  << "\n"
                  << std::endl;
        return false;
    }
    for (int i = 0; i < ref.size(); i++) {
        if (std::abs(ref[i] - test[i]) > 1e-6) {
            std::cout << test_name << " FAIL"
                      << "\n"
                      << std::endl;
            return false;
        }
    }
    std::cout << test_name << " PASS" << std::endl;
    return true;
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

    // Generate data
    auto mu_query = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);
    auto var_query =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    auto mu_key = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);
    auto var_key =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    auto dist_prod =
        compute_prod_var_qk_batch(mu_key, mu_query, var_key, var_query,
                                  batch_size, num_heads, timestep, head_size);
    auto mu_prod = dist_prod[0];
    auto var_prod = dist_prod[1];

    // TAGI forward
    auto dist_pred = forward_mean_var_qk_batch(
        F, mu_prod, var_prod, timestep * timestep,
        timestep * head_size * timestep, batch_size, num_heads);
    auto mu_pred = dist_pred[0];
    auto var_pred = dist_pred[1];

    // Self-attention function
    std::vector<float> mu_pred_mha(timestep * timestep * num_heads * batch_size,
                                   0);
    std::vector<float> var_pred_mha(
        timestep * timestep * num_heads * batch_size, 0);
    query_key(mu_query, var_query, mu_key, var_key, 0, batch_size, num_heads,
              timestep, head_size, mu_pred_mha, var_pred_mha);

    // Test
    std::string mean_test_name = "QK mean";
    std::string var_test_name = "QK var";
    bool is_mean_passed = is_vector_equal(mean_test_name, mu_pred, mu_pred_mha);
    bool is_var_passed = is_vector_equal(var_test_name, var_pred, var_pred_mha);

    return true;
}

bool test_mask_query_key() {
    int seed = 1;
    int batch_size = 1;
    int num_heads = 1;
    int timestep = 3;
    int head_size = 4;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Observation matrix and product variables
    auto F = create_observation_matrix(timestep, timestep, timestep);

    // Generate mask matrix
    auto mask = create_mask_matrix(timestep);

    // Generate mu and var qk
    auto mu_qk = create_random_mha_matrix(batch_size, num_heads, timestep,
                                          timestep, min_value, max_value, seed);
    auto var_qk =
        create_random_mha_matrix(batch_size, num_heads, timestep, timestep,
                                 min_value, max_value, seed, true);

    auto dist_prod_mqk = compute_prod_mask_qk_mean_var(
        mask, mu_qk, var_qk, batch_size, num_heads, timestep);
    auto mu_prod_mqk = dist_prod_mqk[0];
    auto var_prod_mqk = dist_prod_mqk[1];

    // Mask x query-key
    auto dist_mqk = forward_mean_var_qk_batch(
        F, mu_prod_mqk, var_prod_mqk, timestep * timestep,
        timestep * timestep * timestep, batch_size, num_heads);
    auto mu_mqk = dist_mqk[0];
    auto var_mqk = dist_mqk[1];

    // Self-attention funciton
    std::vector<float> mu_mqk_mha(batch_size * num_heads * timestep * timestep,
                                  0);
    std::vector<float> var_mqk_mha(batch_size * num_heads * timestep * timestep,
                                   0);
    mask_query_key(mu_qk, var_qk, batch_size, num_heads, timestep, 1.0,
                   mu_mqk_mha, var_mqk_mha);

    // Test
    std::string mean_test_name = "Masked QK mean";
    std::string var_test_name = "Masked QK var";
    bool is_mean_passed = is_vector_equal(mean_test_name, mu_mqk, mu_mqk_mha);
    bool is_var_passed = is_vector_equal(var_test_name, var_mqk, var_mqk_mha);

    return true;
}

int test_mha() {
    // auto is_qk_passed = test_query_key();
    auto is_passed = test_mask_query_key();
    return 0;
}
///////////////////////////////////////////////////////////////////////////////
// File:         test_mha_cpu.cpp
// Description:  Unittest for multi-head self-attention
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 07, 2023
// Updated:      May 27, 2023
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
                vec[i * timestep + j] = 1.0f;
            }
        }
    }
    return vec;
}

std::vector<float> get_vec_2d(std::vector<float> &vec_4d, int pos, int len) {
    std::vector<float> vec(len, 0.0f);
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
    std::vector<float> matrix_obs(num_elements, 0.0f);
    int idx;
    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < w_a; k++) {
            idx = i * num_cols + k + i * w_a;
            matrix_obs[idx] = 1.0f;
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
    std::vector<float> prod(num_cols, 0.0f);
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
/* Compute mean var between mask and the product of query and key*/
{
    // Initlization
    int prod_size = timestep * timestep * timestep;
    std::vector<float> mu_prod(batch_size * num_heads * prod_size, 0.0f);
    std::vector<float> var_prod(batch_size * num_heads * prod_size, 0.0f);
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

std::vector<std::vector<float>> compute_batch_prod_var_qk(
    std::vector<float> &mu_key, std::vector<float> &mu_query,
    std::vector<float> &var_key, std::vector<float> &var_query, int batch_size,
    int num_heads, int timestep, int head_size)
/*Compute mean and var for the prodcut of query and key*/
{
    // Initialization
    std::vector<float> mu_prod(
        batch_size * num_heads * timestep * timestep * head_size, 0.0f);
    std::vector<float> var_prod(
        batch_size * num_heads * timestep * timestep * head_size, 0.0f);
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
std::vector<std::vector<float>> compute_batch_prod_mean_var_mat_mul(
    std::vector<float> &mu_a, std::vector<float> &var_a,
    std::vector<float> &mu_b, std::vector<float> &var_b, int B, int C, int h_a,
    int w_a, int h_b, int w_b)
/*Compute the product of the matrix multiplication operations a x b*/
{
    // Initialization
    std::vector<float> mu_prod(B * C * h_a * w_a * w_b, 0.0f);
    std::vector<float> var_prod(B * C * h_a * w_a * w_b, 0.0f);
    int pos_a, pos_b, pos_prod;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < C; j++) {
            // Get 2d matrix position
            pos_a = i * C * h_a * w_a + j * h_a * w_a;
            pos_b = i * C * h_b * w_b + j * h_b * w_b;
            pos_prod = i * C * h_a * w_a * w_b + j * h_a * w_a * w_b;

            // Get 2d matrix for a and b
            auto mu_a_2d = get_vec_2d(mu_a, pos_a, h_a * w_a);
            auto var_a_2d = get_vec_2d(var_a, pos_a, h_a * w_a);
            auto mu_b_2d = get_vec_2d(mu_b, pos_b, h_b * w_b);
            auto var_b_2d = get_vec_2d(var_b, pos_b, h_b * w_b);

            // Compute the product
            auto mu_prod_2d =
                compute_prod_mean(mu_a_2d, mu_b_2d, h_a, w_a, w_b);
            auto var_prod_2d = compute_prod_var(mu_a_2d, mu_b_2d, var_a_2d,
                                                var_b_2d, h_a, w_a, w_b);

            // Merge to main vector
            merge_vec(mu_prod_2d, pos_prod, h_a * w_a * w_b, mu_prod);
            merge_vec(var_prod_2d, pos_prod, h_a * w_a * w_b, var_prod);
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

std::vector<std::vector<float>> forward_batch_mean_var_mat_mul(
    std::vector<float> &F, std::vector<float> &mu_prod,
    std::vector<float> &var_prod, int h_F, int w_F, int B, int C)
/*Perform the TAGI feed forward*/
{
    std::vector<float> mu_pred(B * C * h_F, 0.0f);
    std::vector<float> var_pred(B * C * h_F, 0.0f);
    std::vector<float> mu_prod_2d, var_prod_2d;
    int pos, pos_pred;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < C; j++) {
            // 2D matrix pos
            pos = i * C * w_F + j * w_F;
            pos_pred = i * C * h_F + j * h_F;

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
    std::vector<float> matrix(width * height, 0.0f);
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

std::vector<std::vector<float>> compute_score_update(
    std::vector<float> &mu_v, std::vector<float> &var_s,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int timestep,
    int head_size)
/*Compute the update values for score distribution. NOTE: delta_mu and
delta_var represent the update quanties of the observation or previous layer.
The output are the update quanties for the previous layer.
For example, the transition equation is written as
    s = z + v, where v ~N(0, sigma)
The previous layer will use the ouput of the current function following
    mu_z_post = m_z_prior + cov(z, s) * delta_mu_s
    var_z_post = var_z_prior + cov(z, s) * delta_var * cov(s, z)
*/
{
    // Initialization
    std::vector<float> delta_mu_s(timestep * timestep, 0.0f);
    std::vector<float> delta_var_s(timestep * timestep, 0.0f);
    float sum_mu, sum_var;
    int idx_v, idx_obs, idx_s;
    for (int i = 0; i < timestep; i++) {
        for (int j = 0; j < timestep; j++) {
            sum_mu = 0.0f;
            sum_var = 0.0f;
            for (int m = 0; m < head_size; m++) {
                idx_v = j * head_size + m;
                idx_obs = i * head_size + m;
                sum_mu += mu_v[idx_v] * delta_mu[idx_obs];
                sum_var += mu_v[idx_v] * delta_var[idx_obs] * mu_v[idx_v];
            }
            idx_s = i * timestep + j;
            delta_mu_s[idx_s] = sum_mu / var_s[idx_s];
            delta_var_s[idx_s] = sum_var / powf(var_s[idx_s], 2);
        }
    }
    return {delta_mu_s, delta_var_s};
}

std::vector<std::vector<float>> compute_batch_score_update(
    std::vector<float> &mu_v, std::vector<float> &var_s,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int batch_size,
    int num_heads, int timestep, int head_size)
/*Compute updates in batch for score distribution*/
{
    std::vector<float> delta_mu_s(batch_size * num_heads * timestep * timestep,
                                  0.0f);
    std::vector<float> delta_var_s(batch_size * num_heads * timestep * timestep,
                                   0.0f);
    int pos_s, pos_v;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // Matrix position
            pos_v =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_s =
                i * num_heads * timestep * timestep + j * timestep * timestep;

            // Get 2d array vectors
            auto mu_v_2d = get_vec_2d(mu_v, pos_v, timestep * head_size);
            auto var_s_2d = get_vec_2d(var_s, pos_s, timestep * timestep);
            auto delta_mu_2d =
                get_vec_2d(delta_mu, pos_v, timestep * head_size);
            auto delta_var_2d =
                get_vec_2d(delta_var, pos_v, timestep * head_size);

            // Get update values
            auto update_val =
                compute_score_update(mu_v_2d, var_s_2d, delta_mu_2d,
                                     delta_var_2d, timestep, head_size);

            // Merge vectors
            merge_vec(update_val[0], pos_s, timestep * timestep, delta_mu_s);
            merge_vec(update_val[1], pos_s, timestep * timestep, delta_var_s);
        }
    }
    return {delta_mu_s, delta_var_s};
}

std::vector<std::vector<float>> compute_value_update(
    std::vector<float> &mu_s, std::vector<float> &var_v,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int timestep,
    int head_size)
/*Compute the update values for score distribution*/
{
    // Initialization
    std::vector<float> delta_mu_v(timestep * head_size, 0.0f);
    std::vector<float> delta_var_v(timestep * head_size, 0.0f);
    float sum_mu, sum_var;
    int idx_v, idx_obs, idx_s;
    for (int i = 0; i < timestep; i++) {
        for (int j = 0; j < head_size; j++) {
            sum_mu = 0.0f;
            sum_var = 0.0f;
            for (int m = 0; m < timestep; m++) {
                idx_s = m * timestep + i;
                idx_obs = m * head_size + j;
                sum_mu += mu_s[idx_s] * delta_mu[idx_obs];
                sum_var += mu_s[idx_s] * delta_var[idx_obs] * mu_s[idx_s];
            }
            idx_v = i * head_size + j;
            delta_mu_v[idx_v] = sum_mu / var_v[idx_v];
            delta_var_v[idx_v] = sum_var / powf(var_v[idx_v], 2);
        }
    }
    return {delta_mu_v, delta_var_v};
}

std::vector<std::vector<float>> compute_batch_value_update(
    std::vector<float> &mu_s, std::vector<float> &var_v,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int batch_size,
    int num_heads, int timestep, int head_size)
/*Compute updates in batch for score distribution*/
{
    std::vector<float> delta_mu_v(batch_size * num_heads * timestep * head_size,
                                  0.0f);
    std::vector<float> delta_var_v(
        batch_size * num_heads * timestep * head_size, 0.0f);
    int pos_s, pos_v;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // Matrix position
            pos_v =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_s =
                i * num_heads * timestep * timestep + j * timestep * timestep;

            // Get 2d array vectors
            auto var_v_2d = get_vec_2d(var_v, pos_v, timestep * head_size);
            auto mu_s_2d = get_vec_2d(mu_s, pos_s, timestep * timestep);
            auto delta_mu_2d =
                get_vec_2d(delta_mu, pos_v, timestep * head_size);
            auto delta_var_2d =
                get_vec_2d(delta_var, pos_v, timestep * head_size);

            // Get update values
            auto update_val =
                compute_value_update(mu_s_2d, var_v_2d, delta_mu_2d,
                                     delta_var_2d, timestep, head_size);

            // Merge vectors
            merge_vec(update_val[0], pos_v, timestep * head_size, delta_mu_v);
            merge_vec(update_val[1], pos_v, timestep * head_size, delta_var_v);
        }
    }
    return {delta_mu_v, delta_var_v};
}

std::vector<std::vector<float>> compute_query_update(
    std::vector<float> &mu_k, std::vector<float> &var_q,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int num_heads,
    int timestep, int head_size)
/*Compute the update values for query distribution*/
{
    // Initialization
    int idx_q, idx_k, idx_s, block_row, block_col;
    float sum_mu, sum_var;
    std::vector<float> delta_mu_q(timestep * head_size, 0.0f);
    std::vector<float> delta_var_q(timestep * head_size, 0.0f);
    for (int i = 0; i < head_size; i++) {
        for (int j = 0; j < timestep; j++) {
            sum_mu = 0.0f;
            sum_var = 0.0f;
            block_col = (j * head_size + i);
            for (int k = 0; k < timestep; k++) {
                block_row = (j * timestep + k);
                if (block_row > block_col) {
                    idx_k = k * head_size + i;
                    idx_s = j * timestep + k;
                    sum_mu += mu_k[idx_k] * delta_mu[idx_s];
                    sum_var += mu_k[idx_k] * delta_var[idx_s] * mu_k[idx_k];
                }
            }
            idx_q = i + j * head_size;
            delta_mu_q[idx_q] = sum_mu / powf(num_heads, 0.5);
            delta_var_q[idx_q] = sum_var / num_heads;
        }
    }
    return {delta_mu_q, delta_var_q};
}

std::vector<std::vector<float>> compute_batch_query_update(
    std::vector<float> &mu_k, std::vector<float> &var_q,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int batch_size,
    int num_heads, int timestep, int head_size)
/*Compute updates in batch for score distribution*/
{
    std::vector<float> delta_mu_q(batch_size * num_heads * timestep * head_size,
                                  0.0f);
    std::vector<float> delta_var_q(
        batch_size * num_heads * timestep * head_size, 0.0f);
    int pos_q, pos_s;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // Matrix position
            pos_q =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_s =
                i * num_heads * timestep * timestep + j * timestep * timestep;

            // Get 2d array vectors
            auto mu_k_2d = get_vec_2d(mu_k, pos_q, timestep * head_size);
            auto var_q_2d = get_vec_2d(var_q, pos_q, timestep * head_size);
            auto delta_mu_2d = get_vec_2d(delta_mu, pos_s, timestep * timestep);
            auto delta_var_2d =
                get_vec_2d(delta_var, pos_s, timestep * timestep);

            // Get update values
            auto update_val = compute_query_update(
                mu_k_2d, var_q_2d, delta_mu_2d, delta_var_2d, num_heads,
                timestep, head_size);

            // Merge vectors
            merge_vec(update_val[0], pos_q, timestep * head_size, delta_mu_q);
            merge_vec(update_val[1], pos_q, timestep * head_size, delta_var_q);
        }
    }
    return {delta_mu_q, delta_var_q};
}

std::vector<std::vector<float>> compute_key_update(
    std::vector<float> &mu_q, std::vector<float> &var_k,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int num_heads,
    int timestep, int head_size)
/*Compute the update values for key distribution*/
{
    // Initialization
    std::vector<float> delta_mu_k(timestep * head_size, 0.0f);
    std::vector<float> delta_var_k(timestep * head_size, 0.0f);
    int idx_delta, idx_k, idx_s, block_row, block_col;
    float sum_mu, sum_var;
    for (int i = 0; i < head_size; i++) {
        for (int j = 0; j < timestep; j++) {
            sum_mu = 0.0f;
            sum_var = 0.0f;
            block_col = (j * head_size + i);
            for (int k = 0; k < timestep; k++) {
                block_row = (j * timestep + k);
                if (block_row > block_col) {
                    idx_k = k * head_size + i;
                    idx_s = j * timestep + k;
                    sum_mu += var_k[idx_k] * delta_mu[idx_s];
                    sum_var += var_k[idx_k] * delta_var[idx_s] * var_k[idx_k];
                }
            }
            idx_delta = j * head_size + i;
            delta_mu_k[idx_delta] =
                sum_mu * mu_q[idx_delta] / powf(num_heads, 0.5);
            delta_var_k[idx_delta] = mu_q[idx_delta] * sum_var *
                                     mu_q[idx_delta] / num_heads /
                                     powf(var_k[idx_delta], 2);
        }
    }
    return {delta_mu_k, delta_var_k};
}

std::vector<std::vector<float>> compute_batch_key_update(
    std::vector<float> &mu_q, std::vector<float> &var_k,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int batch_size,
    int num_heads, int timestep, int head_size)
/*Compute updates in batch for score distribution*/
{
    std::vector<float> delta_mu_k(batch_size * num_heads * timestep * head_size,
                                  0.0f);
    std::vector<float> delta_var_k(
        batch_size * num_heads * timestep * head_size, 0.0f);
    int pos_k, pos_s;
    int ud_size = timestep * timestep;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            // Matrix position
            pos_k =
                i * num_heads * timestep * head_size + j * timestep * head_size;
            pos_s = i * num_heads * ud_size + j * ud_size;

            // Get 2d array vectors
            auto mu_q_2d = get_vec_2d(mu_q, pos_k, timestep * head_size);
            auto var_k_2d = get_vec_2d(var_k, pos_k, timestep * head_size);
            auto delta_mu_2d = get_vec_2d(delta_mu, pos_s, ud_size);
            auto delta_var_2d = get_vec_2d(delta_var, pos_s, ud_size);

            // Get update values
            auto update_val =
                compute_key_update(mu_q_2d, var_k_2d, delta_mu_2d, delta_var_2d,
                                   num_heads, timestep, head_size);

            // Merge vectors
            merge_vec(update_val[0], pos_k, timestep * head_size, delta_mu_k);
            merge_vec(update_val[1], pos_k, timestep * head_size, delta_var_k);
        }
    }
    return {delta_mu_k, delta_var_k};
}

bool is_close(std::string &test_name, std::vector<float> &ref,
              std::vector<float> &test) {
    if (ref.size() != test.size()) {
        std::cout << test_name << " FAIL"
                  << "\n"
                  << std::endl;
        return false;
    }
    for (int i = 0; i < ref.size(); i++) {
        if (std::abs(ref[i] - test[i]) > 1e-5f) {
            std::cout << std::setw(32) << std::left << test_name
                      << "[\033[31;1mFAIL\033[0m]" << std::endl;
            return false;
        }
    }
    std::cout << std::setw(32) << std::left << test_name
              << "[\033[32;1mPASS\033[0m]" << std::endl;

    return true;
}

bool test_query_key() {
    int seed = 1;
    int batch_size = 1;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 4;
    float min_value = 0.0f;
    float max_value = 2.0f;

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
        compute_batch_prod_var_qk(mu_key, mu_query, var_key, var_query,
                                  batch_size, num_heads, timestep, head_size);
    auto mu_prod = dist_prod[0];
    auto var_prod = dist_prod[1];

    // TAGI forward
    auto dist_pred = forward_batch_mean_var_mat_mul(
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
    std::string mean_test_name = "Mean query x key";
    std::string var_test_name = "Var query x key";
    bool is_mean_passed = is_close(mean_test_name, mu_pred, mu_pred_mha);
    bool is_var_passed = is_close(var_test_name, var_pred, var_pred_mha);

    return (is_mean_passed && is_var_passed);
}

bool test_mask_query_key() {
    int seed = 1;
    int batch_size = 3;
    int num_heads = 2;
    int timestep = 4;
    int head_size = 3;
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
    auto dist_mqk = forward_batch_mean_var_mat_mul(
        F, mu_prod_mqk, var_prod_mqk, timestep * timestep,
        timestep * timestep * timestep, batch_size, num_heads);
    auto mu_mqk = dist_mqk[0];
    auto var_mqk = dist_mqk[1];

    // Self-attention function
    std::vector<float> mu_mqk_mha(batch_size * num_heads * timestep * timestep,
                                  0);
    std::vector<float> var_mqk_mha(batch_size * num_heads * timestep * timestep,
                                   0);
    mask_query_key(mu_qk, var_qk, batch_size, num_heads, timestep, 1.0,
                   mu_mqk_mha, var_mqk_mha);

    // Test
    std::string mean_test_name = "Mean mask x QK";
    std::string var_test_name = "Var mask x QK";
    bool is_mean_passed = is_close(mean_test_name, mu_mqk, mu_mqk_mha);
    bool is_var_passed = is_close(var_test_name, var_mqk, var_mqk_mha);

    return (is_mean_passed && is_var_passed);
}

bool test_tagi_4d_matrix_mul() {
    int seed = 1;
    int batch_size = 3;
    int num_heads = 2;
    int timestep = 4;
    int head_size = 3;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Observation matrix and product variables
    auto F = create_observation_matrix(timestep, timestep, head_size);

    // Generate mu and var for value
    auto mu_v = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_v =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    // Generate mu and var for score
    auto mu_score = create_random_mha_matrix(
        batch_size, num_heads, timestep, timestep, min_value, max_value, seed);
    auto var_score =
        create_random_mha_matrix(batch_size, num_heads, timestep, timestep,
                                 min_value, max_value, seed, true);

    // Compute product
    auto dist_prod = compute_batch_prod_mean_var_mat_mul(
        mu_score, var_score, mu_v, var_v, batch_size, num_heads, timestep,
        timestep, timestep, head_size);
    auto mu_prod = dist_prod[0];
    auto var_prod = dist_prod[1];

    // Forward mat mull
    auto dist_fwd = forward_batch_mean_var_mat_mul(
        F, mu_prod, var_prod, timestep * head_size,
        timestep * timestep * head_size, batch_size, num_heads);
    auto mu_fwd = dist_fwd[0];
    auto var_fwd = dist_fwd[1];

    // self-attention function
    std::vector<float> mu_fwd_mha(batch_size * num_heads * timestep * head_size,
                                  0.0f);
    std::vector<float> var_fwd_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);
    tagi_4d_matrix_mul(mu_score, var_score, mu_v, var_v, 0, 0, 0, batch_size,
                       num_heads, timestep, head_size, timestep, mu_fwd_mha,
                       var_fwd_mha);

    // Test
    std::string mean_test_name = "Mean mat mul";
    std::string var_test_name = "Var mat mul";
    bool is_mean_passed = is_close(mean_test_name, mu_fwd, mu_fwd_mha);
    bool is_var_passed = is_close(var_test_name, var_fwd, var_fwd_mha);

    return is_mean_passed && is_var_passed;
}

bool test_input_projection() {
    int seed = 1;
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate mu and var for value
    auto mu_q = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_q =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);
    auto mu_k = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_k =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);
    auto mu_v = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_v =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    // Merge to embeddings
    std::vector<float> mu_embs(
        3 * batch_size * num_heads * timestep * head_size, 0.0f);
    std::vector<float> var_embs(
        3 * batch_size * num_heads * timestep * head_size, 0.0f);
    cat_intput_projection_components(mu_q, var_q, mu_k, var_k, mu_v, var_v, 0,
                                     0, batch_size, num_heads, timestep,
                                     head_size, mu_embs, var_embs);

    // Separate inputs
    std::vector<float> mu_q_sep(batch_size * num_heads * timestep * head_size,
                                0.0f);
    std::vector<float> var_q_sep(batch_size * num_heads * timestep * head_size,
                                 0.0f);
    std::vector<float> mu_k_sep(batch_size * num_heads * timestep * head_size,
                                0.0f);
    std::vector<float> var_k_sep(batch_size * num_heads * timestep * head_size,
                                 0.0f);
    std::vector<float> mu_v_sep(batch_size * num_heads * timestep * head_size,
                                0.0f);
    std::vector<float> var_v_sep(batch_size * num_heads * timestep * head_size,
                                 0.0f);
    separate_input_projection_components(
        mu_embs, var_embs, 0, 0, batch_size, num_heads, timestep, head_size,
        mu_q_sep, var_q_sep, mu_k_sep, var_k_sep, mu_v_sep, var_v_sep);

    // Test
    std::string q_mean_test_name = "Mean query";
    std::string q_var_test_name = "Var query";
    std::string k_mean_test_name = "Mean key";
    std::string k_var_test_name = "Var key";
    std::string v_mean_test_name = "Mean value";
    std::string v_var_test_name = "Var value";
    bool is_q_mean_passed = is_close(q_mean_test_name, mu_q, mu_q_sep);
    bool is_q_var_passed = is_close(q_var_test_name, var_q, var_q_sep);
    bool is_k_mean_passed = is_close(k_mean_test_name, mu_k, mu_k_sep);
    bool is_k_var_passed = is_close(k_var_test_name, var_k, var_k_sep);
    bool is_v_mean_passed = is_close(v_mean_test_name, mu_v, mu_v_sep);
    bool is_v_var_passed = is_close(v_var_test_name, var_v, var_v_sep);

    return (is_q_mean_passed && is_q_var_passed && is_k_mean_passed &&
            is_k_var_passed && is_v_mean_passed && is_v_var_passed);
}

bool test_output_projection() {
    int seed = 1;
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate mu and var for value
    auto mu_sv = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);
    auto var_sv =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    // Project forward
    std::vector<float> mu_proj(batch_size * num_heads * timestep * head_size,
                               0.0f);
    std::vector<float> var_proj(batch_size * num_heads * timestep * head_size,
                                0.0f);
    project_output_forward(mu_sv, var_sv, 0, 0, batch_size, num_heads, timestep,
                           head_size, mu_proj, var_proj);

    // Project backward
    std::vector<float> mu_sv_pb(batch_size * num_heads * timestep * head_size,
                                0.0f);
    std::vector<float> var_sv_pb(batch_size * num_heads * timestep * head_size,
                                 0.0f);
    project_output_backward(mu_proj, var_proj, 0, 0, batch_size, num_heads,
                            timestep, head_size, mu_sv_pb, var_sv_pb);

    // Test
    std::string mean_test_name = "Mean output projection";
    std::string var_test_name = "Var output projection";
    bool is_mean_passed = is_close(mean_test_name, mu_sv, mu_sv_pb);
    bool is_var_passed = is_close(var_test_name, var_sv, var_sv_pb);

    return (is_mean_passed && is_var_passed);
}

bool test_score_update() {
    int seed = 1;
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate mu and var for value
    auto mu_v = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_s =
        create_random_mha_matrix(batch_size, num_heads, timestep, timestep,
                                 min_value, max_value, seed, true);

    auto delta_mu = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);

    auto delta_var = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);

    // Compute updating values
    auto delta_s_dist =
        compute_batch_score_update(mu_v, var_s, delta_mu, delta_var, batch_size,
                                   num_heads, timestep, head_size);
    auto delta_mu_s = delta_s_dist[0];
    auto delta_var_s = delta_s_dist[1];

    // Self-attention function
    std::vector<float> delta_mu_s_mha(
        batch_size * num_heads * timestep * timestep, 0.0f);
    std::vector<float> delta_var_s_mha(
        batch_size * num_heads * timestep * timestep, 0.0f);

    mha_delta_score(mu_v, var_s, delta_mu, delta_var, 0, 0, batch_size,
                    num_heads, timestep, head_size, delta_mu_s_mha,
                    delta_var_s_mha);

    // Test
    std::string mean_test_name = "Delta mean score";
    std::string var_test_name = "Delta var score";

    bool is_mean_passed = is_close(mean_test_name, delta_mu_s, delta_mu_s_mha);
    bool is_var_passed = is_close(var_test_name, delta_var_s, delta_var_s_mha);

    return is_mean_passed && is_var_passed;
}

bool test_value_update() {
    int seed = 1;
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate data
    auto mu_s = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         timestep, min_value, max_value, seed);
    auto var_v =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    auto delta_mu = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);

    auto delta_var = create_random_mha_matrix(
        batch_size, num_heads, timestep, head_size, min_value, max_value, seed);

    // Compute updating value
    auto delta_v_dist =
        compute_batch_value_update(mu_s, var_v, delta_mu, delta_var, batch_size,
                                   num_heads, timestep, head_size);
    auto delta_mu_v = delta_v_dist[0];
    auto delta_var_v = delta_v_dist[1];

    // Self-attention function
    std::vector<float> delta_mu_v_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);
    std::vector<float> delta_var_v_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);

    mha_delta_value(mu_s, var_v, delta_mu, delta_var, 0, 0, batch_size,
                    num_heads, timestep, head_size, delta_mu_v_mha,
                    delta_var_v_mha);

    // Test
    std::string mean_test_name = "Delta mean value";
    std::string var_test_name = "Delta var value";

    bool is_mean_passed = is_close(mean_test_name, delta_mu_v, delta_mu_v_mha);
    bool is_var_passed = is_close(var_test_name, delta_var_v, delta_var_v_mha);

    return is_mean_passed && is_var_passed;
}

bool test_query_update() {
    int seed = 1;
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate data
    auto mu_k = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_q =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    auto delta_mu = create_random_mha_matrix(
        batch_size, num_heads, timestep, timestep, min_value, max_value, seed);

    auto delta_var = create_random_mha_matrix(
        batch_size, num_heads, timestep, timestep, min_value, max_value, seed);

    // Compute updating values
    auto delta_q_dist =
        compute_batch_query_update(mu_k, var_q, delta_mu, delta_var, batch_size,
                                   num_heads, timestep, head_size);
    auto delta_mu_q = delta_q_dist[0];
    auto delta_var_q = delta_q_dist[1];

    // Selt-attention function
    std::vector<float> delta_mu_q_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);
    std::vector<float> delta_var_q_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);

    mha_delta_query(var_q, mu_k, delta_mu, delta_var, 0, 0, batch_size,
                    num_heads, timestep, head_size, delta_mu_q_mha,
                    delta_var_q_mha);
    // Test
    std::string mean_test_name = "Delta mean query";
    std::string var_test_name = "Delta var query";

    bool is_mean_passed = is_close(mean_test_name, delta_mu_q, delta_mu_q_mha);
    bool is_var_passed = is_close(var_test_name, delta_var_q, delta_var_q_mha);

    return is_mean_passed && is_var_passed;
}

bool test_key_update() {
    int seed = 1;
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 2;
    float min_value = 0.0f;
    float max_value = 5.0f;

    // Generate data
    auto mu_q = create_random_mha_matrix(batch_size, num_heads, timestep,
                                         head_size, min_value, max_value, seed);
    auto var_k =
        create_random_mha_matrix(batch_size, num_heads, timestep, head_size,
                                 min_value, max_value, seed, true);

    auto delta_mu = create_random_mha_matrix(
        batch_size, num_heads, timestep, timestep, min_value, max_value, seed);

    auto delta_var = create_random_mha_matrix(
        batch_size, num_heads, timestep, timestep, min_value, max_value, seed);

    // Compute updating values
    auto delta_k_dist =
        compute_batch_key_update(mu_q, var_k, delta_mu, delta_var, batch_size,
                                 num_heads, timestep, head_size);
    auto delta_mu_k = delta_k_dist[0];
    auto delta_var_k = delta_k_dist[1];

    // Selt-attention function
    std::vector<float> delta_mu_k_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);
    std::vector<float> delta_var_k_mha(
        batch_size * num_heads * timestep * head_size, 0.0f);

    mha_delta_key(var_k, mu_q, delta_mu, delta_var, 0, 0, batch_size, num_heads,
                  timestep, head_size, delta_mu_k_mha, delta_var_k_mha);

    // Test
    std::string mean_test_name = "Delta mean key";
    std::string var_test_name = "Delta var key";

    bool is_mean_passed = is_close(mean_test_name, delta_mu_k, delta_mu_k_mha);
    bool is_var_passed = is_close(var_test_name, delta_var_k, delta_var_k_mha);

    return is_mean_passed && is_var_passed;
}

int test_mha() {
    std::cout << "=========================================" << std::endl;
    std::cout << std::setw(7) << " " << std::setw(7)
              << "Self-Attention Unit Test" << std::endl;
    std::cout << "=========================================" << std::endl;
    auto is_qk_passed = test_query_key();
    auto is_mk_passed = test_mask_query_key();
    auto is_mat_mul_passed = test_tagi_4d_matrix_mul();
    auto is_input_proj_passed = test_input_projection();
    auto is_output_proj_passed = test_output_projection();
    auto is_delta_score_passed = test_score_update();
    auto is_delta_value_passed = test_value_update();
    auto is_delta_query_passed = test_query_update();
    auto is_delta_key_passed = test_key_update();
    std::cout << "=========================================" << std::endl;
    if (!is_qk_passed || !is_mk_passed || !is_mat_mul_passed ||
        !is_input_proj_passed || !is_output_proj_passed ||
        !is_delta_score_passed || !is_delta_value_passed ||
        !is_delta_query_passed || !is_delta_key_passed) {
        return 1;
    }

    return 0;
}
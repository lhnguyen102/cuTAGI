///////////////////////////////////////////////////////////////////////////////
// File:         test_embedding_cpu.cpp
// Description:  Test embeddings layer
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 30, 2023
// Updated:      September 10, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_emb_cpu.h"

std::vector<int> get_unique_vals(std::vector<int> &vec)
/* get unique values of a given integer vector
 */
{
    std::map<int, int> hash_map;
    for (int i = 0; i < vec.size(); i++) {
        hash_map[vec[i]] = i;
    }

    std::vector<int> unique_vec;
    for (const auto &pair : hash_map) {
        unique_vec.push_back(pair.first);
    }
    return unique_vec;
}

std::vector<float> gen_uniform_rand_float(size_t num_data, float max_val,
                                          float min_val,
                                          unsigned int *seed = nullptr)
/* Generate random float number within interval [min_val, max_val] using uniform
 * distribution
 */
{
    // Mersenne twister PRGN - seed
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Create uniform distribution
    std::uniform_real_distribution<float> dist(min_val, max_val);

    // Initialize integer
    std::vector<float> rand_float(num_data, 0);
    for (size_t i = 0; i < num_data; i++) {
        rand_float[i] = dist(gen);
    }

    return rand_float;
}

std::vector<int> gen_randint(size_t num_data, size_t max_val,
                             unsigned int *seed = nullptr)
/*Generate random integer number
 */
{
    // Mersenne twister PRGN - seed
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Create a uniform distribution
    std::uniform_int_distribution<> dist(0, max_val - 1);

    // Integer vector
    std::vector<int> rand_integers(num_data, -1);
    for (size_t i = 0; i < num_data; i++) {
        rand_integers[i] = dist(gen);
    }
    return rand_integers;
}

bool test__get_embedding_values()
/* Test 'get_embedding_values' function that randomly generate the embedding
 * values
 */
{
    // Input
    int num_classes = 10;
    int emb_size = 3;
    int scale = 1;
    unsigned int seed = 42;

    // Test
    auto weight_dist =
        get_embedding_values(num_classes, emb_size, scale, &seed);
    std::vector<float> mu = std::get<0>(weight_dist);
    std::vector<float> var = std::get<1>(weight_dist);

    // Embedding values have to be different than zeros
    for (int i = 0; i < mu.size(); i++) {
        if (mu[i] == 0 || var[i] == 0) {
            return false;
        }
    }

    return mu.size() == num_classes * emb_size &&
           var.size() == num_classes * emb_size;
}

bool test__initalize_embedding_values()
/* Test 'initialize_embedding_values' funtions that randomly generate the
 * embedding values for a givent number of categorical varaibles
 */
{
    // Input
    int num_cat_var = 2;
    std::vector<size_t> cat_sizes = {10, 5};
    std::vector<size_t> emb_sizes = {3, 4};
    int scale = 1;
    unsigned int seed = 42;

    // Test
    auto results = initialize_embedding_values(cat_sizes, emb_sizes,
                                               num_cat_var, scale, &seed);
    std::vector<float> mu = std::get<0>(results);
    std::vector<float> var = std::get<1>(results);

    // Validation
    int tot_num_weights = 0;
    for (int i = 0; i < num_cat_var; i++) {
        tot_num_weights += cat_sizes[i] * emb_sizes[i];
    }

    // Embedding values have to be different than zeros
    for (int i = 0; i < mu.size(); i++) {
        if (mu[i] == 0 || var[i] == 0) {
            return false;
        }
    }

    return mu.size() == tot_num_weights && var.size() == tot_num_weights;
}

bool test__bag_forward()
/*Test embedding forward pass
 */
{
    // Input
    int num_cat_var = 3;
    std::vector<size_t> cat_sizes = {10, 5, 4};
    std::vector<size_t> emb_sizes = {3, 4, 3};
    std::vector<size_t> num_bags = {3, 1, 2};
    std::vector<size_t> bag_sizes = {4, 4, 3};
    int scale = 1;
    unsigned int seed = 42;
    int batch_size = 2;
    int w_pos_in = 0;
    int z_pos_in = 0;
    int z_pos_out = 0;

    // Generata input mean
    std::vector<int> mu_a_int;
    for (size_t j = 0; j < batch_size; j++) {
        for (size_t i = 0; i < num_cat_var; i++) {
            for (size_t k = 0; k < num_bags[i]; k++) {
                seed += j * batch_size + i * num_cat_var + k;
                auto tmp = gen_randint(bag_sizes[i], cat_sizes[i], &seed);
                mu_a_int.insert(mu_a_int.end(), tmp.begin(), tmp.end());
            }
        }
    }
    std::vector<float> mu_a(mu_a_int.size(), 0);
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        mu_a[i] = static_cast<float>(mu_a_int[i]);
    }

    // Generate embedding values
    auto results = initialize_embedding_values(cat_sizes, emb_sizes,
                                               num_cat_var, scale, &seed);
    std::vector<float> mu_w = std::get<0>(results);
    std::vector<float> var_w = std::get<1>(results);

    // Forward pass
    int tot_state = 0;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_cat_var; i++) {
            tot_state += num_bags[i];
        }
    }
    std::vector<float> mu_z(tot_state, 0);
    std::vector<float> var_z(tot_state, 0);
    bag_forward(mu_a, mu_w, var_w, cat_sizes, emb_sizes, num_bags, bag_sizes,
                num_cat_var, batch_size, w_pos_in, z_pos_in, z_pos_out, mu_z,
                var_z);

    // Checking forward pass. All values in mu_z and var_z must be greater than
    // zero
    for (int i = 0; i < mu_z.size(); i++) {
        if (mu_z[i] == 0 || var_z[i] == 0) {
            return false;
        }
    }
    return true;
}

bool test__bag_param_backward()
/*Test embedding's backward pass
 */
{
    // Input
    int num_cat_var = 3;
    std::vector<size_t> cat_sizes = {10, 5, 4};
    std::vector<size_t> emb_sizes = {3, 4, 3};
    std::vector<size_t> num_bags = {3, 1, 2};
    std::vector<size_t> bag_sizes = {4, 4, 3};
    int scale = 1;
    unsigned int seed = 42;
    int batch_size = 2;
    int w_pos_in = 0;
    int z_pos_in = 0;
    int z_pos_out = 0;

    // Generata input mean
    std::vector<int> mu_a_int;
    for (size_t j = 0; j < batch_size; j++) {
        for (size_t i = 0; i < num_cat_var; i++) {
            for (size_t k = 0; k < num_bags[i]; k++) {
                seed += j * batch_size + i * num_cat_var + k;
                auto tmp = gen_randint(bag_sizes[i], cat_sizes[i], &seed);
                mu_a_int.insert(mu_a_int.end(), tmp.begin(), tmp.end());
            }
        }
    }
    std::vector<float> mu_a(mu_a_int.size(), 0);
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        mu_a[i] = static_cast<float>(mu_a_int[i]);
    }

    // Generate embedding values
    auto results = initialize_embedding_values(cat_sizes, emb_sizes,
                                               num_cat_var, scale, &seed);
    std::vector<float> mu_w = std::get<0>(results);
    std::vector<float> var_w = std::get<1>(results);

    // Output's updating quantities
    int tot_state = 0;
    float max_val = 1.0f;
    float min_val = -1.0f;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_cat_var; i++) {
            tot_state += num_bags[i];
        }
    }
    auto delta_mu = gen_uniform_rand_float(tot_state, max_val, min_val, &seed);
    auto delta_var = gen_uniform_rand_float(tot_state, max_val, min_val, &seed);

    // Backward pass
    std::vector<float> delta_mu_w(mu_w.size(), 0);
    std::vector<float> delta_var_w(var_w.size(), 0);
    bag_param_backward(mu_a, var_w, delta_mu, delta_var, cat_sizes, emb_sizes,
                       num_bags, bag_sizes, num_cat_var, batch_size, z_pos_in,
                       w_pos_in, z_pos_out, delta_mu_w, delta_var_w);

    // Get indices of embedding values to be updated
    std::vector<int> checking_idx;
    for (int j = 0; j < batch_size; j++) {
        int bag_pos = 0;
        int emb_pos = 0;
        for (int i = 0; i < num_cat_var; i++) {
            size_t bag = num_bags[i];
            size_t bag_size = bag_sizes[i];
            size_t emb_size = emb_sizes[i];
            size_t cat_size = cat_sizes[i];
            std::vector<int> tmp(bag_size * bag, 0);
            for (int k = 0; k < bag_size * bag; k++) {
                tmp[k] = mu_a[k + bag_pos];
            }
            auto unique_tmp = get_unique_vals(tmp);

            for (int n = 0; n < unique_tmp.size(); n++) {
                for (int m = 0; m < emb_size; m++) {
                    checking_idx.push_back(unique_tmp[n] * emb_size + m);
                }
            }

            // Update pos
            bag_pos += bag_size * bag;
            emb_pos += cat_size * emb_size;
        }
    }

    // The embedding values correspoding to checking indices must be different
    // than zero
    auto unique_checking_idx = get_unique_vals(checking_idx);
    for (const auto &i : unique_checking_idx) {
        if (delta_mu_w[i] == 0 || delta_var_w[i] == 0) {
            return false;
        }
    }

    // Check if the size of unique_checking_idx is equal to the tot_non_zero
    int tot_non_zero = delta_mu_w.size();
    for (int i = 0; i < delta_mu_w.size(); i++) {
        if (delta_mu_w[i] == 0 || delta_var_w[i] == 0) {
            tot_non_zero -= 1;
        }
    }
    return tot_non_zero == unique_checking_idx.size();
}

void test_log(std::string test_name, bool is_passed) {
    if (!is_passed) {
        std::cout << std::setw(32) << std::left << test_name
                  << "[\033[31;1mFAIL\033[0m]" << std::endl;
    } else {
        std::cout << std::setw(32) << std::left << test_name
                  << "[\033[32;1mPASS\033[0m]" << std::endl;
    }
}

int test_embedding() {
    std::cout << "=========================================" << std::endl;
    std::cout << std::setw(7) << " " << std::setw(7)
              << "Unit Test for Embedding Layers" << std::endl;
    std::cout << "=========================================" << std::endl;
    bool test_emb_val = test__get_embedding_values();
    test_log("get_embedding_values", test_emb_val);

    bool test_init_emb_val = test__initalize_embedding_values();
    test_log("initalize_embedding_values", test_init_emb_val);

    bool test_forward_pass = test__bag_forward();
    test_log("bag_forward", test_forward_pass);

    bool test_backward_pass = test__bag_param_backward();
    test_log("bag_backward", test_backward_pass);

    return 0;
}

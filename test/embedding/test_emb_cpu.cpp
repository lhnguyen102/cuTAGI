///////////////////////////////////////////////////////////////////////////////
// File:         test_embedding_cpu.cpp
// Description:  Test embeddings layer
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 30, 2023
// Updated:      September 06, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_emb_cpu.h"

std::vector<int> gen_randint(size_t num_data, size_t max_val,
                             unsigned int *seed = nullptr)
/*Generate random integer number
 */
{
    // Mersenne twister PRGN - seed
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Create a uniform distribution
    std::uniform_int_distribution<> dist(0, max_val);

    // Integer vector
    std::vector<int> rand_integers(num_data, 0);
    for (size_t i = 0; i < num_data; i++) {
        rand_integers.push_back(dist(gen));
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

    return mu.size() == tot_num_weights && var.size() == tot_num_weights;
}

bool test__bag_forward()
/*Test embedding forward pass
 */
{
    // Input
    int num_cat_var = 2;
    std::vector<size_t> cat_sizes = {10, 5};
    std::vector<size_t> emb_sizes = {3, 4};
    std::vector<size_t> num_bags = {3, 1};
    std::vector<size_t> bag_sizes = {4, 4};
    int scale = 1;
    unsigned int seed = 42;
    int batch_size = 3;
    int w_pos_in = 0;
    int z_pos_in = 0;
    int z_pos_out = 0;

    // Generata input mean
    std::vector<int> mu_a_int;
    for (size_t j = 0; j < batch_size; j++) {
        for (size_t i = 0; i < num_cat_var; i++) {
            for (size_t k = 0; k < num_bags[i]; k++) {
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
            for (size_t k = 0; k < num_bags[i]; k++) {
                tot_state += bag_sizes[i];
            }
        }
    }
    std::vector<float> mu_z(tot_state, 0);
    std::vector<float> var_z(tot_state, 0);
    bag_forward(mu_a, mu_w, var_w, cat_sizes, emb_sizes, num_bags, bag_sizes,
                num_cat_var, batch_size, w_pos_in, z_pos_in, z_pos_out, mu_z,
                var_z);

    // Checking forward pass. All values in mu_z and var_z must be greater than
    // zero
    bool passed = true;
    for (int i = 0; i < mu_z.size(); i++) {
        if (mu_z[i] == 0 || var_z[i] == 0) {
            passed = false;
        }
    }
    return passed;
}

bool test__bag_param_backward()
/*Test embedding's backward pass
 */
{
    // Input
    int num_cat_var = 2;
    std::vector<size_t> cat_sizes = {10, 5};
    std::vector<size_t> emb_sizes = {3, 4};
    std::vector<size_t> num_bags = {3, 1};
    std::vector<size_t> bag_sizes = {4, 4};
    int scale = 1;
    unsigned int seed = 42;
    int batch_size = 3;
    int w_pos_in = 0;
    int z_pos_in = 0;
    int z_pos_out = 0;

    // Generata input mean
    std::vector<int> mu_a_int;
    for (size_t j = 0; j < batch_size; j++) {
        for (size_t i = 0; i < num_cat_var; i++) {
            for (size_t k = 0; k < num_bags[i]; k++) {
                auto tmp = gen_randint(bag_sizes[i], cat_sizes[i], &seed);
                mu_a_int.insert(mu_a_int.end(), tmp.begin(), tmp.end());
            }
        }
    }
    std::vector<float> mu_a(mu_a_int.size(), 0);
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        mu_a[i] = static_cast<float>(mu_a_int[i]);
    }
}

///////////////////////////////////////////////////////////////////////////////
// File:         test_embedding_cpu.cpp
// Description:  Test embeddings layer
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 30, 2023
// Updated:      August 30, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "test_emb_cpu.h"

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
    std::vector<int> cat_sizes = {10, 5};
    std::vector<int> emb_sizes = {3, 4};
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
///////////////////////////////////////////////////////////////////////////////
// File:         embedding_cpu.cpp
// Description:  embeddings layer
//               (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 23, 2023
// Updated:      August 27, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////////

#include "../include/embedding_cpu.h"

std::tuple<std::vector<float>, std::vector<float>> get_embedding_values(
    int num_classes, int emb_size, float scale, unsigned int *seed = nullptr)
/*
 */
{
    // Initialize pointer
    std::vector<float> mu_w(num_classes * emb_size, 0);
    std::vector<float> var_w(num_classes * emb_size, pow(scale, 2));

    // Mersenne twister PRGN - seed
    std::mt19937 gen(seed ? *seed : std::random_device{}());

    // Create normal distribution
    std::normal_distribution<float> norm_dist(0.0f, scale);

    // Get sample for weight
    for (int i = 0; i < num_classes * emb_size; i++) {
        mu_w[i] = norm_dist(gen);
    }

    return {mu_w, var_w};
}

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    std::vector<int> cat_sizes, std::vector<int> emb_sizes, int num_cat_var,
    float scale, unsigned int *seed = nullptr)
/*
 */
{
    // Check dim
    if (cat_sizes.size() != emb_sizes.size() ||
        cat_sizes.size() != num_cat_var) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Mismatch in vector sizes or num_cat_var.");
    }
    // Initialize the embedding vectors
    std::vector<float> mu_emb;
    std::vector<float> var_emb;

    for (int i = 0; i < num_cat_var; i++) {
        auto weight_dist =
            get_embedding_values(cat_sizes[i], emb_sizes[i], scale, seed);

        // Insert the values to the embedding vectors directly using std::get
        mu_emb.insert(mu_emb.end(), std::get<0>(weight_dist).begin(),
                      std::get<0>(weight_dist).end());
        var_emb.insert(var_emb.end(), std::get<1>(weight_dist).begin(),
                       std::get<1>(weight_dist).end());
    }

    return {mu_emb, var_emb};
}

///////////////////////////////////////////////////////////////////////////////
// Embedding Layer
///////////////////////////////////////////////////////////////////////////////
void forward(std::vector<float> &ma, std::vector<float> &mu_w,
             std::vector<float> &var_w, std::vector<int> &cat_sizes,
             std::vector<int> &emb_sizes, int num_cat, int batch_size,
             int w_pos_in, int z_pos_in, int z_pos_out,
             std::vector<float> &mu_z, std::vector<float> &var_z)
/**/
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_cat; j++) {
            int cat_idx = ma[j + i * batch_size + z_pos_in];
            int emb_size = emb_sizes[j];
            for (int k = 0; k < emb_size; k++) {
                mu_z[k + z_pos_out] = mu_w[cat_idx * emb_size + k + w_pos_in];
                var_z[k + z_pos_out] = var_w[cat_idx * emb_size + k + w_pos_in];
            }
        }
    }
}

void param_backward(std::vector<float> &ma, std::vector<float> &delta_mu,
                    std::vector<float> &delta_var, std::vector<int> &cat_sizes,
                    std::vector<int> &emb_sizes, int num_cat, int batch_size,
                    int z_pos_in, int z_pos_out, int w_pos_in,
                    std::vector<float> &delta_mu_w,
                    std::vector<float> &delta_var_w)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_cat; j++) {
            int cat_idx = ma[j + i * batch_size + z_pos_in];
            int emb_size = emb_sizes[j];
            for (int k = 0; k < emb_size; k++) {
                delta_mu_w[cat_idx * emb_size + k + w_pos_in] =
                    delta_mu[k + z_pos_out];

                delta_var_w[cat_idx * emb_size + k + w_pos_in] =
                    delta_var[k + z_pos_out];
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Bag Embedding Layer
///////////////////////////////////////////////////////////////////////////////
void bag_forward(std::vector<float> &ma, std::vector<float> &mu_w,
                 std::vector<float> &var_w, std::vector<int> &cat_sizes,
                 std::vector<int> &emb_sizes, std::vector<int> &num_bags,
                 std::vector<int> &bag_sizes, int num_cat, int batch_size,
                 int w_pos_in, int z_pos_in, int z_pos_out,
                 std::vector<float> &mu_z, std::vector<float> &var_z)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_cat; j++) {
            for (int m = 0; m < num_bags[j]; m++) {
                for (int n = 0; n < bag_sizes[m]; n++) {
                    // TO BE DONE
                    int cat_idx = ma[j + i * batch_size + z_pos_in];
                    int emb_size = emb_sizes[j];
                    for (int k = 0; k < emb_size; k++) {
                        mu_z[k + z_pos_out] =
                            mu_w[cat_idx * emb_size + k + w_pos_in];
                        var_z[k + z_pos_out] =
                            var_w[cat_idx * emb_size + k + w_pos_in];
                    }
                }
            }
        }
    }
}

int calculate_embedding_size(int num_categories)
/*
 */
{
    int emb_size = 1.6 * powf(num_categories, 0.56);

    return std::max(600, emb_size);
}
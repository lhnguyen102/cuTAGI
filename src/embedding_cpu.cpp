#include "../include/embedding_cpu.h"

std::tuple<std::vector<float>, std::vector<float>> get_embedding_values(
    size_t num_classes, size_t emb_size, float scale, unsigned int *seed)
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
    for (size_t i = 0; i < num_classes * emb_size; i++) {
        mu_w[i] = norm_dist(gen);
    }

    return {mu_w, var_w};
}

std::tuple<std::vector<float>, std::vector<float>> initialize_embedding_values(
    std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
    int num_cat_var, float scale, unsigned int *seed)
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
             std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
             std::vector<size_t> &emb_sizes, int num_cat, int batch_size,
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

void param_backward(std::vector<float> &ma, std::vector<float> &var_w,
                    std::vector<float> &delta_mu, std::vector<float> &delta_var,
                    std::vector<size_t> &cat_sizes,
                    std::vector<size_t> &emb_sizes, int num_cat, int batch_size,
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
                    delta_mu[k + z_pos_out] *
                    var_w[cat_idx * emb_size + k + w_pos_in];

                delta_var_w[cat_idx * emb_size + k + w_pos_in] =
                    delta_var[k + z_pos_out] *
                    var_w[cat_idx * emb_size + k + w_pos_in];
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Bag Embedding Layer
///////////////////////////////////////////////////////////////////////////////
struct Offsets {
    std::vector<int> batch_in_offsets;
    std::vector<int> batch_out_offsets;
    std::vector<int> cat_in_offsets;
    std::vector<int> cat_out_offsets;
};

Offsets precompute_offsets(const std::vector<size_t> &num_bags,
                           const std::vector<size_t> &bag_sizes, int num_cat,
                           int batch_size) {
    Offsets offsets;
    offsets.batch_in_offsets.resize(batch_size);
    offsets.batch_out_offsets.resize(batch_size);
    offsets.cat_in_offsets.resize(num_cat);
    offsets.cat_out_offsets.resize(num_cat);

    // Precompute batch offsets
    int batch_size_in_offset = 0;
    int batch_size_out_offset = 0;
    for (int i = 0; i < batch_size; ++i) {
        offsets.batch_in_offsets[i] = batch_size_in_offset;
        offsets.batch_out_offsets[i] = batch_size_out_offset;

        int cat_in_offset = 0;
        int cat_out_offset = 0;

        for (int j = 0; j < num_cat; ++j) {
            size_t bag = num_bags[j];
            size_t bag_size = bag_sizes[j];
            offsets.cat_in_offsets[j] = cat_in_offset;
            offsets.cat_out_offsets[j] = cat_out_offset;

            cat_in_offset += bag_size * bag;
            cat_out_offset += bag;
        }

        batch_size_in_offset += cat_in_offset;
        batch_size_out_offset += cat_out_offset;
    }

    return offsets;
}
void bag_forward(std::vector<float> &mu_a, std::vector<float> &mu_w,
                 std::vector<float> &var_w, std::vector<size_t> &cat_sizes,
                 std::vector<size_t> &emb_sizes, std::vector<size_t> &num_bags,
                 std::vector<size_t> &bag_sizes, int num_cat, int batch_size,
                 int w_pos_in, int z_pos_in, int z_pos_out,
                 std::vector<float> &mu_z, std::vector<float> &var_z)
/* Forward pass for bag embedding layer.

Args:
    mu_a: Input mean stored the categorical variable in float
    mu_w: Embedding mean
    var_w: Embedding variance
    cat_sizes: Number of classes for each categorical variable
    emb_sizes: Embedding size for each categoricals
    num_bags: Number of bags for each categorical variable
    bag_sizes: Number of classes for each bag
    num_cat: Number of categorical variables stored in node vector
    batch_size: Number of observation in batch
    w_pos_in: Start position of the embedding vector
    z_pos_in: Start position of the input hidden state vector i.e., mu_a
    z_pos_out: Start position of the output hidden state vector i.e., mu_z
    mu_z: Mean of output hidden states
    var_z: Variance of output hidden states

Example:
    A two categorical varaibles num_cat = 2 each having 5 and 6 classes. In
addition, assuming emb_sizes = [2, 3], num_bags = [3, 1], and bag_sizes = [4, 4]

    embedding vector
    cat_1 = [
        [0.13, 0.23], -> class 0
        [0.53, 0.23], -> class 1
        [0.32, 0.43], -> class 2
        [0.22, 0.13], -> class 3
        [0.72, 0.18], -> class 4
    ]

    cat_2 = [
        [0.13, 0.23, 0.34], -> class 0
        [0.53, 0.23, 0.53], -> class 1
        [0.32, 0.43, 0.65], -> class 2
        [0.22, 0.13, 0.75], -> class 3
        [0.72, 0.18, 0.87], -> class 4
        [0.52, 0.28, 0.17], -> class 5
    ]
    mu_a = [
        [[0, 1, 3, 4], [2, 3, 1, 4], [4, 3, 2, 1]], # bag 0
        [[1, 8, 9, 6]]                              # bag 1
    ]

    output size (3, 1)
 */
{
    // Compute the offsets for each bags due to we store all data in a single
    // vector
    Offsets offsets =
        precompute_offsets(num_bags, bag_sizes, num_cat, batch_size);
    for (int i = 0; i < batch_size; i++) {
        int batch_size_in_offset = offsets.batch_in_offsets[i];
        int batch_size_out_offset = offsets.batch_out_offsets[i];
        for (int j = 0; j < num_cat; j++) {
            size_t bag = num_bags[j];
            size_t emb_size = emb_sizes[j];
            size_t bag_size = bag_sizes[j];
            int cat_in_offset = offsets.cat_in_offsets[j];
            int cat_out_offset = offsets.cat_out_offsets[j];
            for (int m = 0; m < bag; m++) {
                float sum_mu = 0.0f;
                float sum_var = 0.0f;
                for (int n = 0; n < bag_size; n++) {
                    // Convert categorical index in each bag to integer. TODO:
                    // need to avoid this conversion for computing performance
                    int cat_idx = mu_a[n + m * bag_size + cat_in_offset +
                                       batch_size_in_offset + z_pos_in];

                    int offset = cat_idx * emb_size + w_pos_in;
                    // Sum over all embedding values for each bag
                    for (int k = 0; k < emb_size; k++) {
                        sum_mu += mu_w[offset + k];
                        sum_var += var_w[offset + k];
                    }
                }

                // Average the embedding values. Output size (batch_size,
                // num_cat, bag)
                int z_idx =
                    m + cat_out_offset + batch_size_out_offset + z_pos_out;
                mu_z[z_idx] = sum_mu / bag_size;
                var_z[z_idx] = sum_var / bag_size;
            }
        }
    }
}

void bag_param_backward(
    std::vector<float> &mu_a, std::vector<float> &var_w,
    std::vector<float> &delta_mu, std::vector<float> &delta_var,
    std::vector<size_t> &cat_sizes, std::vector<size_t> &emb_sizes,
    std::vector<size_t> &num_bags, std::vector<size_t> &bag_sizes, int num_cat,
    int batch_size, int z_pos_in, int w_pos_in, int z_pos_out,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/*
Args:
    mu_a: Input mean stored the categorical variable in float
    var_w: Embedding variance
    delta_mu: Updating quantities for mean of the output layer
    delta_var: Updating quantion for variance of the output layer
    cat_sizes: Number of classes for each categorical variable
    emb_sizes: Embedding size for each categoricals
    num_bags: Number of bags for each categorical variable
    bag_sizes: Number of classes for each bag
    num_cat: Number of categorical variables stored in node vector
    batch_size: Number of observation in batch
    w_pos_in: Start position of the embedding vector
    z_pos_in: Start position of the input hidden state vector i.e., mu_a
    z_pos_out: Start postiion of the output hidden state vector i.e., mu_z
    delta_mu_w: Updating quantities for mean of the embeddings
    delta_var_w: Updating quantities for variance of the embeddings

 */
{
    // Compute the offsets for each bags due to we store all data in a single
    // vector
    Offsets offsets =
        precompute_offsets(num_bags, bag_sizes, num_cat, batch_size);
    for (int i = 0; i < batch_size; i++) {
        int batch_size_in_offset = offsets.batch_in_offsets[i];
        int batch_size_out_offset = offsets.batch_out_offsets[i];
        for (int j = 0; j < num_cat; j++) {
            size_t bag = num_bags[j];
            size_t emb_size = emb_sizes[j];
            int cat_in_offset = offsets.cat_in_offsets[j];
            int cat_out_offset = offsets.cat_out_offsets[j];

            for (int m = 0; m < bag; m++) {
                size_t bag_size = bag_sizes[m];
                for (int n = 0; n < bag_size; n++) {
                    // Convert categorical index in each bag to integer. TODO:
                    // need to avoid this conversion for computing performance
                    int cat_idx = mu_a[n + m * bag_size + cat_in_offset +
                                       batch_size_in_offset + z_pos_in];

                    // Index for the updating quantities of the output layer
                    int ino_idx =
                        m + cat_out_offset + batch_size_out_offset + z_pos_out;

                    // Calculate the updating quanties for embeddings inside
                    // each bag
                    for (int k = 0; k < emb_size; k++) {
                        // Index for embedding
                        int w_idx = cat_idx * emb_size + k + w_pos_in;

                        // Updating quantities for embedding. TODO Need to set
                        // all delta values to zero
                        delta_mu_w[w_idx] += delta_mu[ino_idx] * var_w[w_idx];
                        delta_var_w[w_idx] += delta_var[ino_idx] * var_w[w_idx];
                    }
                }
            }
        }
    }
}

int calculate_embedding_size(int num_classes)
/*
Calculate the embedding size based on the number of categories using fast.ai
heuristic.

Args:
    num_classes: Number of classes

Return:
    int: Embedding size

 */
{
    int emb_size = 1.6 * powf(num_classes, 0.56);

    return std::max(600, emb_size);
}
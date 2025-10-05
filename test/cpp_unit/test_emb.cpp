#include <gtest/gtest.h>

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "../../include/data_struct.h"
#include "../../include/embedding_cpu.h"

#ifdef USE_CUDA
#include "../../include/embedding_cuda.cuh"
#endif

extern bool g_gpu_enabled;

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
    std::uniform_real_distribution<float> dist(min_val, max_val);

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
    std::uniform_int_distribution<> dist(0, max_val - 1);

    std::vector<int> rand_integers(num_data, -1);
    for (size_t i = 0; i < num_data; i++) {
        rand_integers[i] = dist(gen);
    }
    return rand_integers;
}

bool test_embedding_class_forward()
/* Test Embedding class forward pass
 */
{
    // Input
    int num_embeddings = 10;
    int embedding_dim = 4;
    int num_inputs = 4;  // Increased to 4 for consistency
    float scale = 1.0f;
    int padding_idx = -1;
    unsigned int seed = 42;
    int batch_size = 2;

    Embedding emb_layer(num_embeddings, embedding_dim, scale, padding_idx);

    std::vector<int> mu_a_int;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_inputs; i++) {
            seed += j * batch_size + i;
            auto tmp = gen_randint(1, num_embeddings, &seed);
            mu_a_int.push_back(tmp[0]);
        }
    }

    // Ensure at least two same categories exist
    if (mu_a_int.size() >= 2) {
        mu_a_int[1] = mu_a_int[0];  // Force duplicate in first batch
    }
    if (mu_a_int.size() >= num_inputs + 2) {
        mu_a_int[num_inputs + 1] =
            mu_a_int[num_inputs];  // Force duplicate in second batch
    }

    BaseHiddenStates input_states;
    input_states.mu_a.resize(mu_a_int.size());
    input_states.var_a.resize(mu_a_int.size());
    input_states.jcb.resize(mu_a_int.size());
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        input_states.mu_a[i] = static_cast<float>(mu_a_int[i]);
        input_states.var_a[i] = 1.0f;
        input_states.jcb[i] = 1.0f;
    }
    input_states.actual_size = num_inputs;
    input_states.block_size = batch_size;

    BaseHiddenStates output_states;
    output_states.mu_a.resize(embedding_dim * num_inputs * batch_size, 0);
    output_states.var_a.resize(embedding_dim * num_inputs * batch_size, 0);
    output_states.jcb.resize(embedding_dim * num_inputs * batch_size, 0);
    BaseTempStates temp_states;

    emb_layer.forward(input_states, output_states, temp_states);

    // Validation: Check that output size is correct
    if (output_states.actual_size != num_inputs * embedding_dim) {
        return false;
    }

    // Validation: Check that all output values are non-zero
    for (size_t i = 0; i < output_states.mu_a.size(); i++) {
        if (output_states.mu_a[i] == 0 || output_states.var_a[i] == 0) {
            return false;
        }
    }

    return true;
}

bool test_embedding_class_backward()
/* Test Embedding class backward pass
 */
{
    // Input
    int num_embeddings = 10;
    int embedding_dim = 4;
    int num_inputs = 4;  // Increased to 4 to ensure duplicates
    float scale = 1.0f;
    int padding_idx = -1;
    unsigned int seed = 42;
    int batch_size = 2;

    Embedding emb_layer(num_embeddings, embedding_dim, scale, padding_idx);
    emb_layer.training = true;
    emb_layer.allocate_param_delta();

    // Generate input indices with guaranteed duplicates to test gradient
    // accumulation
    std::vector<int> mu_a_int;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_inputs; i++) {
            seed += j * batch_size + i;
            auto tmp = gen_randint(1, num_embeddings, &seed);
            mu_a_int.push_back(tmp[0]);
        }
    }

    // Ensure at least two same categories exist by duplicating the first
    // element
    if (mu_a_int.size() >= 2) {
        mu_a_int[1] = mu_a_int[0];
    }
    if (mu_a_int.size() >= num_inputs + 2) {
        mu_a_int[num_inputs + 1] = mu_a_int[num_inputs];
    }

    BaseHiddenStates input_states;
    input_states.mu_a.resize(mu_a_int.size());
    input_states.var_a.resize(mu_a_int.size());
    input_states.jcb.resize(mu_a_int.size());
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        input_states.mu_a[i] = static_cast<float>(mu_a_int[i]);
        input_states.var_a[i] = 1.0f;
        input_states.jcb[i] = 1.0f;
    }
    input_states.actual_size = num_inputs;
    input_states.block_size = batch_size;

    BaseHiddenStates output_states;
    output_states.mu_a.resize(embedding_dim * num_inputs * batch_size, 0);
    output_states.var_a.resize(embedding_dim * num_inputs * batch_size, 0);
    output_states.jcb.resize(embedding_dim * num_inputs * batch_size, 0);
    BaseTempStates temp_states;

    emb_layer.forward(input_states, output_states, temp_states);

    float max_val = 1.0f;
    float min_val = -1.0f;
    auto delta_mu = gen_uniform_rand_float(
        embedding_dim * num_inputs * batch_size, max_val, min_val, &seed);
    auto delta_var = gen_uniform_rand_float(
        embedding_dim * num_inputs * batch_size, max_val, min_val, &seed);

    BaseDeltaStates input_delta_states;
    input_delta_states.delta_mu = delta_mu;
    input_delta_states.delta_var = delta_var;
    input_delta_states.block_size = batch_size;
    input_delta_states.actual_size = num_inputs * embedding_dim;

    BaseDeltaStates output_delta_states;
    output_delta_states.delta_mu.resize(num_inputs * embedding_dim, 0);
    output_delta_states.delta_var.resize(num_inputs * embedding_dim, 0);

    emb_layer.backward(input_delta_states, output_delta_states, temp_states,
                       false);

    // Get indices of embedding values that should be updated
    std::vector<int> checking_idx;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_inputs; i++) {
            int cat_idx = mu_a_int[j * num_inputs + i];
            for (int k = 0; k < embedding_dim; k++) {
                checking_idx.push_back(cat_idx * embedding_dim + k);
            }
        }
    }
    auto unique_checking_idx = get_unique_vals(checking_idx);

    // The embedding values corresponding to checking indices must be different
    // than zero
    for (const auto &i : unique_checking_idx) {
        if (emb_layer.delta_mu_w[i] == 0 || emb_layer.delta_var_w[i] == 0) {
            return false;
        }
    }

    // Check if the size of unique_checking_idx is equal to the tot_non_zero
    int tot_non_zero = 0;
    for (size_t i = 0; i < emb_layer.delta_mu_w.size(); i++) {
        if (emb_layer.delta_mu_w[i] != 0 || emb_layer.delta_var_w[i] != 0) {
            tot_non_zero += 1;
        }
    }

    return tot_non_zero == unique_checking_idx.size();
}

TEST(EmbeddingTest, EmbeddingClassForward) {
    EXPECT_TRUE(test_embedding_class_forward());
}

TEST(EmbeddingTest, EmbeddingClassBackward) {
    EXPECT_TRUE(test_embedding_class_backward());
}

bool test_embedding_with_padding()
/* Test Embedding with padding_idx=0
 */
{
    int num_embeddings = 10;
    int embedding_dim = 4;
    int num_inputs = 3;
    float scale = 1.0f;
    int padding_idx = 0;
    int batch_size = 2;

    Embedding emb_layer(num_embeddings, embedding_dim, scale, padding_idx);

    std::vector<int> mu_a_int = {0, 1, 2, 0, 3, 4};

    BaseHiddenStates input_states(mu_a_int.size() * batch_size, batch_size, 0);
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        input_states.mu_a[i] = static_cast<float>(mu_a_int[i]);
        input_states.var_a[i] = 1.0f;
        input_states.jcb[i] = 1.0f;
    }
    input_states.actual_size = num_inputs;
    input_states.block_size = batch_size;

    BaseHiddenStates output_states(embedding_dim * num_inputs * batch_size,
                                   batch_size, 0);
    BaseTempStates temp_states;

    emb_layer.forward(input_states, output_states, temp_states);

    // Check padding positions (index 0) produce zeros
    for (int k = 0; k < embedding_dim; k++) {
        if (output_states.mu_a[k] != 0.0f || output_states.var_a[k] != 0.0f) {
            return false;
        }
        int batch2_idx = num_inputs * embedding_dim + k;
        if (output_states.mu_a[batch2_idx] != 0.0f ||
            output_states.var_a[batch2_idx] != 0.0f) {
            return false;
        }
    }

    // Check non-padding positions produce non-zeros
    for (int k = 0; k < embedding_dim; k++) {
        int idx1 = embedding_dim + k;
        int idx2 = 2 * embedding_dim + k;
        if (output_states.mu_a[idx1] == 0.0f ||
            output_states.var_a[idx1] == 0.0f) {
            return false;
        }
        if (output_states.mu_a[idx2] == 0.0f ||
            output_states.var_a[idx2] == 0.0f) {
            return false;
        }
    }

    return true;
}

TEST(EmbeddingTest, EmbeddingWithPadding) {
    EXPECT_TRUE(test_embedding_with_padding());
}

#ifdef USE_CUDA
bool test_embedding_class_forward_cuda() {
    if (!g_gpu_enabled) return true;

    int num_embeddings = 10;
    int embedding_dim = 4;
    int num_inputs = 4;
    float scale = 1.0f;
    int padding_idx = -1;
    unsigned int seed = 42;
    int batch_size = 2;

    EmbeddingCuda emb_layer(num_embeddings, embedding_dim, scale, padding_idx);
    emb_layer.init_weight_bias();

    std::vector<int> mu_a_int;
    std::vector<float> var_a_int;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_inputs; i++) {
            seed += j * batch_size + i;
            auto tmp = gen_randint(1, num_embeddings, &seed);
            mu_a_int.push_back(tmp[0]);
            var_a_int.push_back(1.0f);
        }
    }

    if (mu_a_int.size() >= 2) {
        mu_a_int[1] = mu_a_int[0];
        var_a_int[1] = 1.0f;
    }
    if (mu_a_int.size() >= num_inputs + 2) {
        mu_a_int[num_inputs + 1] = mu_a_int[num_inputs];
        var_a_int[num_inputs + 1] = 1.0f;
    }

    std::vector<float> mu_a_host(mu_a_int.size());
    std::vector<float> var_a_host(var_a_int.size());
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        mu_a_host[i] = static_cast<float>(mu_a_int[i]);
        var_a_host[i] = static_cast<float>(var_a_int[i]);
    }

    HiddenStateCuda input_states(num_inputs * batch_size, batch_size, 0);
    input_states.set_input_x(mu_a_host, var_a_host, batch_size);

    HiddenStateCuda output_states(num_inputs * embedding_dim * batch_size,
                                  batch_size, 0);
    output_states.allocate_memory();

    BaseTempStates temp_states;

    emb_layer.forward(input_states, output_states, temp_states);

    if (output_states.actual_size != num_inputs * embedding_dim) {
        return false;
    }

    output_states.to_host();

    for (size_t i = 0; i < output_states.mu_a.size(); i++) {
        if (output_states.mu_a[i] == 0 || output_states.var_a[i] == 0) {
            return false;
        }
    }

    return true;
}

bool test_embedding_class_backward_cuda() {
    if (!g_gpu_enabled) return true;

    int num_embeddings = 10;
    int embedding_dim = 4;
    int num_inputs = 4;
    float scale = 1.0f;
    int padding_idx = -1;
    unsigned int seed = 42;
    int batch_size = 2;

    EmbeddingCuda emb_layer(num_embeddings, embedding_dim, scale, padding_idx);
    emb_layer.training = true;
    emb_layer.init_weight_bias();
    emb_layer.allocate_param_delta();

    std::vector<int> mu_a_int;
    std::vector<float> var_a_int;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_inputs; i++) {
            seed += j * batch_size + i;
            auto tmp = gen_randint(1, num_embeddings, &seed);
            mu_a_int.push_back(tmp[0]);
            var_a_int.push_back(1.0f);
        }
    }

    if (mu_a_int.size() >= 2) {
        mu_a_int[1] = mu_a_int[0];
        var_a_int[1] = 1.0f;
    }
    if (mu_a_int.size() >= num_inputs + 2) {
        mu_a_int[num_inputs + 1] = mu_a_int[num_inputs];
        var_a_int[num_inputs + 1] = 1.0f;
    }

    std::vector<float> mu_a_host(mu_a_int.size());
    std::vector<float> var_a_host(mu_a_int.size());
    for (size_t i = 0; i < mu_a_int.size(); i++) {
        mu_a_host[i] = static_cast<float>(mu_a_int[i]);
        var_a_host[i] = 1.0f;
    }

    HiddenStateCuda input_states(num_inputs * batch_size, batch_size, 0);

    input_states.set_input_x(mu_a_host, var_a_host, batch_size);

    HiddenStateCuda output_states(num_inputs * embedding_dim * batch_size,
                                  batch_size, 0);
    output_states.allocate_memory();

    BaseTempStates temp_states;

    emb_layer.forward(input_states, output_states, temp_states);

    float max_val = 1.0f;
    float min_val = -1.0f;
    auto delta_mu = gen_uniform_rand_float(
        embedding_dim * num_inputs * batch_size, max_val, min_val, &seed);
    auto delta_var = gen_uniform_rand_float(
        embedding_dim * num_inputs * batch_size, max_val, min_val, &seed);

    DeltaStateCuda input_delta_states(num_inputs * embedding_dim * batch_size,
                                      batch_size, 0);
    input_delta_states.delta_mu = delta_mu;
    input_delta_states.delta_var = delta_var;
    input_delta_states.allocate_memory();
    input_delta_states.to_device();

    DeltaStateCuda output_delta_states(num_inputs * batch_size, batch_size, 0);
    output_delta_states.allocate_memory();
    output_delta_states.to_device();

    emb_layer.backward(input_delta_states, output_delta_states, temp_states,
                       false);

    emb_layer.delta_params_to_host();

    std::vector<int> checking_idx;
    for (int j = 0; j < batch_size; j++) {
        for (int i = 0; i < num_inputs; i++) {
            int cat_idx = mu_a_int[j * num_inputs + i];
            for (int k = 0; k < embedding_dim; k++) {
                checking_idx.push_back(cat_idx * embedding_dim + k);
            }
        }
    }
    auto unique_checking_idx = get_unique_vals(checking_idx);

    for (const auto &i : unique_checking_idx) {
        if (emb_layer.delta_mu_w[i] == 0 || emb_layer.delta_var_w[i] == 0) {
            return false;
        }
    }

    int tot_non_zero = 0;
    for (size_t i = 0; i < emb_layer.delta_mu_w.size(); i++) {
        if (emb_layer.delta_mu_w[i] != 0 || emb_layer.delta_var_w[i] != 0) {
            tot_non_zero += 1;
        }
    }

    return tot_non_zero == unique_checking_idx.size();
}

TEST(EmbeddingTest, EmbeddingClassForward_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EXPECT_TRUE(test_embedding_class_forward_cuda());
}

TEST(EmbeddingTest, EmbeddingClassBackward_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    EXPECT_TRUE(test_embedding_class_backward_cuda());
}
#endif

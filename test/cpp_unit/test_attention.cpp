#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "../../include/attention.h"
#include "../../include/data_struct.h"

TEST(AttentionForward, BasicForwardPass) {
    int batch_size = 2;
    int timestep = 4;
    int embed_dim = 8;
    int num_heads = 2;
    int head_dim = embed_dim / num_heads;

    std::vector<float> input_data(batch_size * timestep * embed_dim);
    std::vector<float> input_var(batch_size * timestep * embed_dim);
    std::vector<float> input_jcb(batch_size * timestep * embed_dim, 1.0f);

    std::default_random_engine gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = dist(gen);
        input_var[i] = 0.1f + 0.1f * std::abs(dist(gen));
    }

    BaseHiddenStates input_states;
    input_states.mu_a = input_data;
    input_states.var_a = input_var;
    input_states.jcb = input_jcb;
    input_states.block_size = batch_size;
    input_states.actual_size = embed_dim * timestep;
    input_states.size = batch_size * timestep * embed_dim;

    BaseHiddenStates output_states;
    output_states.mu_a.resize(batch_size * timestep * embed_dim);
    output_states.var_a.resize(batch_size * timestep * embed_dim);
    output_states.jcb.resize(batch_size * timestep * embed_dim);

    BaseTempStates temp_states;

    MultiheadAttention attn_layer(embed_dim, num_heads, num_heads, false);
    attn_layer.training = true;

    attn_layer.forward(input_states, output_states, temp_states);

    EXPECT_EQ(output_states.mu_a.size(), batch_size * timestep * embed_dim);
    EXPECT_EQ(output_states.var_a.size(), batch_size * timestep * embed_dim);
    EXPECT_EQ(output_states.actual_size, embed_dim);

    for (size_t i = 0; i < output_states.mu_a.size(); i++) {
        EXPECT_FALSE(std::isnan(output_states.mu_a[i]))
            << "Output mu contains NaN at index " << i;
        EXPECT_FALSE(std::isnan(output_states.var_a[i]))
            << "Output var contains NaN at index " << i;
    }

    int expected_att_score_size = batch_size * num_heads * timestep * timestep;
    EXPECT_EQ(attn_layer.attn_states.mu_att_score.size(),
              expected_att_score_size);
    EXPECT_EQ(attn_layer.attn_states.var_att_score.size(),
              expected_att_score_size);

    float sum = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads; j++) {
            for (int k = 0; k < timestep; k++) {
                sum = 0.0f;
                for (int l = 0; l < timestep; l++) {
                    int idx = i * num_heads * timestep * timestep +
                              j * timestep * timestep + k * timestep + l;
                    sum += attn_layer.attn_states.mu_att_score[idx];
                    EXPECT_FALSE(
                        std::isnan(attn_layer.attn_states.mu_att_score[idx]))
                        << "Attention score contains NaN at index " << idx;
                }
                EXPECT_NEAR(sum, 1.0f, 1e-4f)
                    << "Attention scores should sum to 1.0 for each query "
                       "position";
            }
        }
    }
}

TEST(AttentionBackward, BasicBackwardPass) {
    int batch_size = 2;
    int timestep = 4;
    int embed_dim = 8;
    int num_heads = 2;

    std::vector<float> input_data(batch_size * timestep * embed_dim);
    std::vector<float> input_var(batch_size * timestep * embed_dim);
    std::vector<float> input_jcb(batch_size * timestep * embed_dim, 1.0f);

    std::default_random_engine gen(456);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = dist(gen);
        input_var[i] = 0.1f + 0.1f * std::abs(dist(gen));
    }

    BaseHiddenStates input_states;
    input_states.mu_a = input_data;
    input_states.var_a = input_var;
    input_states.jcb = input_jcb;
    input_states.block_size = batch_size * timestep;
    input_states.actual_size = embed_dim;
    input_states.size = batch_size * timestep * embed_dim;

    BaseHiddenStates output_states;
    output_states.mu_a.resize(batch_size * timestep * embed_dim);
    output_states.var_a.resize(batch_size * timestep * embed_dim);
    output_states.jcb.resize(batch_size * timestep * embed_dim);

    BaseTempStates temp_states;

    MultiheadAttention attn_layer(embed_dim, num_heads, num_heads, false);
    attn_layer.timestep = timestep;
    attn_layer.training = true;
    attn_layer.param_update = true;

    attn_layer.forward(input_states, output_states, temp_states);

    BaseDeltaStates input_delta_states;
    input_delta_states.delta_mu.resize(batch_size * timestep * embed_dim, 1.0f);
    input_delta_states.delta_var.resize(batch_size * timestep * embed_dim,
                                        0.1f);
    input_delta_states.block_size = batch_size * timestep;
    input_delta_states.actual_size = embed_dim;

    BaseDeltaStates output_delta_states;
    output_delta_states.delta_mu.resize(batch_size * timestep * embed_dim);
    output_delta_states.delta_var.resize(batch_size * timestep * embed_dim);
    output_delta_states.block_size = batch_size * timestep;
    output_delta_states.actual_size = embed_dim;

    attn_layer.backward(input_delta_states, output_delta_states, temp_states,
                        true);

    EXPECT_EQ(output_delta_states.delta_mu.size(),
              batch_size * timestep * embed_dim);
    EXPECT_EQ(output_delta_states.delta_var.size(),
              batch_size * timestep * embed_dim);

    for (size_t i = 0; i < output_delta_states.delta_mu.size(); i++) {
        EXPECT_FALSE(std::isnan(output_delta_states.delta_mu[i]))
            << "Output delta mu contains NaN at index " << i;
        EXPECT_FALSE(std::isnan(output_delta_states.delta_var[i]))
            << "Output delta var contains NaN at index " << i;
    }

    EXPECT_EQ(attn_layer.delta_mu_w.size(), attn_layer.mu_w.size());
    EXPECT_EQ(attn_layer.delta_var_w.size(), attn_layer.var_w.size());

    for (size_t i = 0; i < attn_layer.delta_mu_w.size(); i++) {
        EXPECT_FALSE(std::isnan(attn_layer.delta_mu_w[i]))
            << "Weight gradient mu contains NaN at index " << i;
        EXPECT_FALSE(std::isnan(attn_layer.delta_var_w[i]))
            << "Weight gradient var contains NaN at index " << i;
    }
}

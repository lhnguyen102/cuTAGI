#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "../../include/data_struct.h"
#include "../../include/rmsnorm_layer.h"

TEST(RMSNormForward, BasicForwardPass) {
    int batch_size = 2;
    int embed_dim = 8;

    std::vector<float> input_data(batch_size * embed_dim);
    std::vector<float> input_var(batch_size * embed_dim);
    std::vector<float> input_jcb(batch_size * embed_dim, 1.0f);

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
    input_states.actual_size = embed_dim;
    input_states.size = batch_size * embed_dim;

    BaseHiddenStates output_states;
    output_states.mu_a.resize(batch_size * embed_dim);
    output_states.var_a.resize(batch_size * embed_dim);
    output_states.jcb.resize(batch_size * embed_dim);

    BaseTempStates temp_states;

    RMSNorm rmsnorm_layer({embed_dim}, 1e-6);
    rmsnorm_layer.training = true;

    rmsnorm_layer.forward(input_states, output_states, temp_states);

    EXPECT_EQ(output_states.mu_a.size(), batch_size * embed_dim);
    EXPECT_EQ(output_states.var_a.size(), batch_size * embed_dim);
    EXPECT_EQ(output_states.actual_size, embed_dim);

    for (size_t i = 0; i < output_states.mu_a.size(); i++) {
        EXPECT_FALSE(std::isnan(output_states.mu_a[i]))
            << "Output mu contains NaN at index " << i;
        EXPECT_FALSE(std::isnan(output_states.var_a[i]))
            << "Output var contains NaN at index " << i;
        EXPECT_FALSE(std::isinf(output_states.mu_a[i]))
            << "Output mu contains Inf at index " << i;
        EXPECT_FALSE(std::isinf(output_states.var_a[i]))
            << "Output var contains Inf at index " << i;
    }

    EXPECT_EQ(rmsnorm_layer.rms_ra.size(), batch_size);
    for (size_t i = 0; i < rmsnorm_layer.rms_ra.size(); i++) {
        EXPECT_GT(rmsnorm_layer.rms_ra[i], 0.0f)
            << "RMS statistic should be positive at index " << i;
        EXPECT_FALSE(std::isnan(rmsnorm_layer.rms_ra[i]))
            << "RMS statistic contains NaN at index " << i;
    }
}

TEST(RMSNormBackward, BasicBackwardPass) {
    int batch_size = 2;
    int embed_dim = 8;

    std::vector<float> input_data(batch_size * embed_dim);
    std::vector<float> input_var(batch_size * embed_dim);
    std::vector<float> input_jcb(batch_size * embed_dim, 1.0f);

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
    input_states.block_size = batch_size;
    input_states.actual_size = embed_dim;
    input_states.size = batch_size * embed_dim;

    BaseHiddenStates output_states;
    output_states.mu_a.resize(batch_size * embed_dim);
    output_states.var_a.resize(batch_size * embed_dim);
    output_states.jcb.resize(batch_size * embed_dim);

    BaseTempStates temp_states;

    RMSNorm rmsnorm_layer({embed_dim}, 1e-6);
    rmsnorm_layer.training = true;
    rmsnorm_layer.param_update = true;

    rmsnorm_layer.forward(input_states, output_states, temp_states);

    BaseDeltaStates input_delta_states;
    input_delta_states.delta_mu.resize(batch_size * embed_dim, 1.0f);
    input_delta_states.delta_var.resize(batch_size * embed_dim, 0.1f);
    input_delta_states.block_size = batch_size;
    input_delta_states.actual_size = embed_dim;

    BaseDeltaStates output_delta_states;
    output_delta_states.delta_mu.resize(batch_size * embed_dim);
    output_delta_states.delta_var.resize(batch_size * embed_dim);
    output_delta_states.block_size = batch_size;
    output_delta_states.actual_size = embed_dim;

    rmsnorm_layer.backward(input_delta_states, output_delta_states, temp_states,
                           true);

    EXPECT_EQ(output_delta_states.delta_mu.size(), batch_size * embed_dim);
    EXPECT_EQ(output_delta_states.delta_var.size(), batch_size * embed_dim);

    for (size_t i = 0; i < output_delta_states.delta_mu.size(); i++) {
        EXPECT_FALSE(std::isnan(output_delta_states.delta_mu[i]))
            << "Output delta mu contains NaN at index " << i;
        EXPECT_FALSE(std::isnan(output_delta_states.delta_var[i]))
            << "Output delta var contains NaN at index " << i;
    }

    EXPECT_EQ(rmsnorm_layer.delta_mu_w.size(), embed_dim);
    EXPECT_EQ(rmsnorm_layer.delta_var_w.size(), embed_dim);

    for (size_t i = 0; i < rmsnorm_layer.delta_mu_w.size(); i++) {
        EXPECT_FALSE(std::isnan(rmsnorm_layer.delta_mu_w[i]))
            << "Weight delta mu contains NaN at index " << i;
        EXPECT_FALSE(std::isnan(rmsnorm_layer.delta_var_w[i]))
            << "Weight delta var contains NaN at index " << i;
    }
}

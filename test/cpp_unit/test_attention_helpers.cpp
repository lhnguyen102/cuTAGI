#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "../../include/attention.h"

class AttentionHelpersTest : public ::testing::Test {
   protected:
    const float TOLERANCE = 1e-5f;

    void SetUp() override {
        gen.seed(42);
        dist = std::normal_distribution<float>(0.0f, 1.0f);
    }

    std::default_random_engine gen;
    std::normal_distribution<float> dist;

    void fill_random(std::vector<float>& vec) {
        for (auto& val : vec) {
            val = dist(gen);
        }
    }

    bool vectors_equal(const std::vector<float>& a, const std::vector<float>& b,
                       float tol = 1e-5f) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (std::abs(a[i] - b[i]) > tol) return false;
        }
        return true;
    }
};

TEST_F(AttentionHelpersTest, SeparateInputProjectionComponents) {
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_dim = 4;
    int comp_size = batch_size * num_heads * timestep * head_dim;

    std::vector<float> mu_embs(3 * comp_size, 0.0f);
    std::vector<float> var_embs(3 * comp_size, 0.0f);

    // Set distinct values in each of the three sections (Q, K, V)
    for (int i = 0; i < comp_size; i++) {
        mu_embs[i] = 1.0f;                  // Q section
        mu_embs[i + comp_size] = 2.0f;      // K section
        mu_embs[i + 2 * comp_size] = 3.0f;  // V section
        var_embs[i] = 0.1f;
        var_embs[i + comp_size] = 0.2f;
        var_embs[i + 2 * comp_size] = 0.3f;
    }

    std::vector<float> mu_q(comp_size), var_q(comp_size);
    std::vector<float> mu_k(comp_size), var_k(comp_size);
    std::vector<float> mu_v(comp_size), var_v(comp_size);

    separate_input_projection_components(mu_embs, var_embs, batch_size,
                                         num_heads, timestep, head_dim, mu_q,
                                         var_q, mu_k, var_k, mu_v, var_v);

    // Verify all Q values came from Q section
    for (int i = 0; i < comp_size; i++) {
        EXPECT_FLOAT_EQ(mu_q[i], 1.0f);
        EXPECT_FLOAT_EQ(var_q[i], 0.1f);
    }

    // Verify all K values came from K section
    for (int i = 0; i < comp_size; i++) {
        EXPECT_FLOAT_EQ(mu_k[i], 2.0f);
        EXPECT_FLOAT_EQ(var_k[i], 0.2f);
    }

    // Verify all V values came from V section
    for (int i = 0; i < comp_size; i++) {
        EXPECT_FLOAT_EQ(mu_v[i], 3.0f);
        EXPECT_FLOAT_EQ(var_v[i], 0.3f);
    }
}

TEST_F(AttentionHelpersTest, SeparateAndCatAreInverses) {
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 4;
    int comp_size = batch_size * num_heads * timestep * head_size;

    // Test 1: Start with Q, K, V -> cat -> separate -> should get back Q, K, V
    std::vector<float> mu_q_orig(comp_size), var_q_orig(comp_size);
    std::vector<float> mu_k_orig(comp_size), var_k_orig(comp_size);
    std::vector<float> mu_v_orig(comp_size), var_v_orig(comp_size);

    fill_random(mu_q_orig);
    fill_random(var_q_orig);
    fill_random(mu_k_orig);
    fill_random(var_k_orig);
    fill_random(mu_v_orig);
    fill_random(var_v_orig);

    std::vector<float> mu_embs(3 * comp_size);
    std::vector<float> var_embs(3 * comp_size);

    cat_intput_projection_components(
        mu_q_orig, var_q_orig, mu_k_orig, var_k_orig, mu_v_orig, var_v_orig,
        batch_size, num_heads, timestep, head_size, mu_embs, var_embs);

    std::vector<float> mu_q_reconst(comp_size), var_q_reconst(comp_size);
    std::vector<float> mu_k_reconst(comp_size), var_k_reconst(comp_size);
    std::vector<float> mu_v_reconst(comp_size), var_v_reconst(comp_size);

    separate_input_projection_components(
        mu_embs, var_embs, batch_size, num_heads, timestep, head_size,
        mu_q_reconst, var_q_reconst, mu_k_reconst, var_k_reconst, mu_v_reconst,
        var_v_reconst);

    // Verify cat then separate gives back original Q, K, V
    EXPECT_TRUE(vectors_equal(mu_q_orig, mu_q_reconst));
    EXPECT_TRUE(vectors_equal(var_q_orig, var_q_reconst));
    EXPECT_TRUE(vectors_equal(mu_k_orig, mu_k_reconst));
    EXPECT_TRUE(vectors_equal(var_k_orig, var_k_reconst));
    EXPECT_TRUE(vectors_equal(mu_v_orig, mu_v_reconst));
    EXPECT_TRUE(vectors_equal(var_v_orig, var_v_reconst));

    // Test 2: Start with embeddings -> separate -> cat -> should get back
    // embeddings
    std::vector<float> mu_embs_orig(3 * comp_size);
    std::vector<float> var_embs_orig(3 * comp_size);
    fill_random(mu_embs_orig);
    fill_random(var_embs_orig);

    std::vector<float> mu_q(comp_size), var_q(comp_size);
    std::vector<float> mu_k(comp_size), var_k(comp_size);
    std::vector<float> mu_v(comp_size), var_v(comp_size);

    separate_input_projection_components(
        mu_embs_orig, var_embs_orig, batch_size, num_heads, timestep, head_size,
        mu_q, var_q, mu_k, var_k, mu_v, var_v);

    std::vector<float> mu_embs_reconst(3 * comp_size);
    std::vector<float> var_embs_reconst(3 * comp_size);

    cat_intput_projection_components(mu_q, var_q, mu_k, var_k, mu_v, var_v,
                                     batch_size, num_heads, timestep, head_size,
                                     mu_embs_reconst, var_embs_reconst);

    EXPECT_TRUE(vectors_equal(mu_embs_orig, mu_embs_reconst));
    EXPECT_TRUE(vectors_equal(var_embs_orig, var_embs_reconst));
}

TEST_F(AttentionHelpersTest, QueryKey) {
    int batch_size = 1;
    int num_heads = 1;
    int timestep = 2;
    int head_size = 3;

    std::vector<float> mu_q = {1.0f, 0.0f, 0.0f,   // First query: [1, 0, 0]
                               0.0f, 1.0f, 0.0f};  // Second query: [0, 1, 0]
    std::vector<float> var_q(6, 0.0f);
    std::vector<float> mu_k = {1.0f, 0.0f, 0.0f,   // First key: [1, 0, 0]
                               0.0f, 1.0f, 0.0f};  // Second key: [0, 1, 0]
    std::vector<float> var_k(6, 0.0f);

    std::vector<float> mu_qk(batch_size * num_heads * timestep * timestep);
    std::vector<float> var_qk(batch_size * num_heads * timestep * timestep);

    query_key(mu_q, var_q, mu_k, var_k, batch_size, num_heads, timestep,
              head_size, mu_qk, var_qk);

    EXPECT_NEAR(mu_qk[0], 1.0f, TOLERANCE);  // q0 · k0 = 1
    EXPECT_NEAR(mu_qk[1], 0.0f, TOLERANCE);  // q0 · k1 = 0
    EXPECT_NEAR(mu_qk[2], 0.0f, TOLERANCE);  // q1 · k0 = 0
    EXPECT_NEAR(mu_qk[3], 1.0f, TOLERANCE);  // q1 · k1 = 1

    for (size_t i = 0; i < var_qk.size(); i++) {
        EXPECT_GE(var_qk[i], 0.0f);
    }
}

TEST_F(AttentionHelpersTest, MaskQueryKey) {
    int batch_size = 1;
    int num_heads = 1;
    int timestep = 3;
    int head_size = 2;

    std::vector<float> mu_qk(timestep * timestep, 1.0f);
    std::vector<float> var_qk(timestep * timestep, 0.1f);

    std::vector<float> mu_mqk(timestep * timestep);
    std::vector<float> var_mqk(timestep * timestep);

    mask_query_key(mu_qk, var_qk, batch_size, num_heads, timestep, head_size,
                   mu_mqk, var_mqk);

    EXPECT_NEAR(mu_mqk[0 * timestep + 0], 1.0f, TOLERANCE);
    EXPECT_NEAR(mu_mqk[0 * timestep + 1], 0.0f, TOLERANCE);
    EXPECT_NEAR(mu_mqk[0 * timestep + 2], 0.0f, TOLERANCE);

    EXPECT_NEAR(mu_mqk[1 * timestep + 0], 1.0f, TOLERANCE);
    EXPECT_NEAR(mu_mqk[1 * timestep + 1], 1.0f, TOLERANCE);
    EXPECT_NEAR(mu_mqk[1 * timestep + 2], 0.0f, TOLERANCE);

    EXPECT_NEAR(mu_mqk[2 * timestep + 0], 1.0f, TOLERANCE);
    EXPECT_NEAR(mu_mqk[2 * timestep + 1], 1.0f, TOLERANCE);
    EXPECT_NEAR(mu_mqk[2 * timestep + 2], 1.0f, TOLERANCE);

    EXPECT_NEAR(var_mqk[0 * timestep + 0], 0.1f, TOLERANCE);
    EXPECT_NEAR(var_mqk[0 * timestep + 1], 0.0f, TOLERANCE);
    EXPECT_NEAR(var_mqk[0 * timestep + 2], 0.0f, TOLERANCE);

    EXPECT_NEAR(var_mqk[1 * timestep + 0], 0.1f, TOLERANCE);
    EXPECT_NEAR(var_mqk[1 * timestep + 1], 0.1f, TOLERANCE);
    EXPECT_NEAR(var_mqk[1 * timestep + 2], 0.0f, TOLERANCE);

    EXPECT_NEAR(var_mqk[2 * timestep + 0], 0.1f, TOLERANCE);
    EXPECT_NEAR(var_mqk[2 * timestep + 1], 0.1f, TOLERANCE);
    EXPECT_NEAR(var_mqk[2 * timestep + 2], 0.1f, TOLERANCE);
}

TEST_F(AttentionHelpersTest, Tagi4DMatrixMul) {
    // Test: attention_scores @ values
    // attention_scores: [batch=1, heads=1, timestep=2, timestep=2]
    // values: [batch=1, heads=1, timestep=2, head_dim=3]
    // result: [batch=1, heads=1, timestep=2, head_dim=3]
    int N = 1, C = 1, H = 2, W = 3, D = 2;

    std::vector<float> mu_a = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> var_a(4, 0.0f);

    std::vector<float> mu_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> var_b(6, 0.0f);

    std::vector<float> mu_ab(N * C * H * W);
    std::vector<float> var_ab(N * C * H * W);

    tagi_4d_matrix_mul(mu_a, var_a, mu_b, var_b, N, C, H, W, D, mu_ab, var_ab);

    EXPECT_NEAR(mu_ab[0], 1.0f, TOLERANCE);
    EXPECT_NEAR(mu_ab[1], 2.0f, TOLERANCE);
    EXPECT_NEAR(mu_ab[2], 3.0f, TOLERANCE);
    EXPECT_NEAR(mu_ab[3], 4.0f, TOLERANCE);
    EXPECT_NEAR(mu_ab[4], 5.0f, TOLERANCE);
    EXPECT_NEAR(mu_ab[5], 6.0f, TOLERANCE);

    for (size_t i = 0; i < var_ab.size(); i++) {
        EXPECT_GE(var_ab[i], 0.0f);
    }
}

TEST_F(AttentionHelpersTest, ProjectOutputForwardBackwardAreInverses) {
    int batch_size = 2;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 4;
    int size = batch_size * num_heads * timestep * head_size;

    // Test: forward then backward should give back original
    std::vector<float> mu_orig(size);
    std::vector<float> var_orig(size);
    fill_random(mu_orig);
    fill_random(var_orig);

    std::vector<float> mu_fwd(size);
    std::vector<float> var_fwd(size);

    project_output_forward(mu_orig, var_orig, batch_size, num_heads, timestep,
                           head_size, mu_fwd, var_fwd);

    std::vector<float> mu_reconst(size);
    std::vector<float> var_reconst(size);

    project_output_backward(mu_fwd, var_fwd, batch_size, num_heads, timestep,
                            head_size, mu_reconst, var_reconst);

    // Verify forward then backward gives back original
    EXPECT_TRUE(vectors_equal(mu_orig, mu_reconst));
    EXPECT_TRUE(vectors_equal(var_orig, var_reconst));

    std::vector<float> mu_bwd(size);
    std::vector<float> var_bwd(size);

    project_output_backward(mu_orig, var_orig, batch_size, num_heads, timestep,
                            head_size, mu_bwd, var_bwd);

    std::vector<float> mu_reconst2(size);
    std::vector<float> var_reconst2(size);

    project_output_forward(mu_bwd, var_bwd, batch_size, num_heads, timestep,
                           head_size, mu_reconst2, var_reconst2);

    EXPECT_TRUE(vectors_equal(mu_orig, mu_reconst2));
    EXPECT_TRUE(vectors_equal(var_orig, var_reconst2));
}

TEST_F(AttentionHelpersTest, AttentionValueDelta) {
    // Forward: output = attention_scores @ values
    // Backward: delta_scores, delta_values from delta_output
    int batch_size = 1;
    int num_heads = 1;
    int timestep = 2;
    int head_size = 2;

    std::vector<float> mu_att_scores = {0.5f, 0.5f, 0.5f, 0.5f};
    std::vector<float> var_att_scores = {0.1f, 0.1f, 0.1f, 0.1f};

    std::vector<float> mu_v = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> var_v = {0.1f, 0.1f, 0.1f, 0.1f};

    std::vector<float> delta_mu_out = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> delta_var_out = {0.1f, 0.1f, 0.1f, 0.1f};

    std::vector<float> delta_mu_scores(timestep * timestep);
    std::vector<float> delta_var_scores(timestep * timestep);

    mha_delta_score(mu_v, var_att_scores, delta_mu_out, delta_var_out,
                    batch_size, num_heads, timestep, head_size, delta_mu_scores,
                    delta_var_scores);

    std::vector<float> delta_mu_v(batch_size * num_heads * timestep *
                                  head_size);
    std::vector<float> delta_var_v(batch_size * num_heads * timestep *
                                   head_size);

    mha_delta_value(mu_att_scores, var_v, delta_mu_out, delta_var_out,
                    batch_size, num_heads, timestep, head_size, delta_mu_v,
                    delta_var_v);

    float sum_delta_scores = 0.0f;
    for (auto val : delta_mu_scores) sum_delta_scores += std::abs(val);
    EXPECT_GT(sum_delta_scores, 0.0f);

    float sum_delta_v = 0.0f;
    for (auto val : delta_mu_v) sum_delta_v += std::abs(val);
    EXPECT_GT(sum_delta_v, 0.0f);
}

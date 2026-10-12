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

    // The linear layer writes one row per token: [Q(E) | K(E) | V(E)] with
    // E = num_heads * head_dim.
    int emb_size = num_heads * head_dim;
    int row_size = 3 * emb_size;
    for (int token = 0; token < batch_size * timestep; token++) {
        for (int c = 0; c < emb_size; c++) {
            mu_embs[token * row_size + c] = 1.0f;
            mu_embs[token * row_size + emb_size + c] = 2.0f;
            mu_embs[token * row_size + 2 * emb_size + c] = 3.0f;
            var_embs[token * row_size + c] = 0.1f;
            var_embs[token * row_size + emb_size + c] = 0.2f;
            var_embs[token * row_size + 2 * emb_size + c] = 0.3f;
        }
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

    // Scores are scaled by 1/sqrt(head_size) = 1/sqrt(3).
    float scale = 1.0f / std::sqrt(3.0f);
    EXPECT_NEAR(mu_qk[0], scale, TOLERANCE);  // q0 · k0 = 1
    EXPECT_NEAR(mu_qk[1], 0.0f, TOLERANCE);   // q0 · k1 = 0
    EXPECT_NEAR(mu_qk[2], 0.0f, TOLERANCE);   // q1 · k0 = 0
    EXPECT_NEAR(mu_qk[3], scale, TOLERANCE);  // q1 · k1 = 1

    for (size_t i = 0; i < var_qk.size(); i++) {
        EXPECT_GE(var_qk[i], 0.0f);
    }
}

TEST_F(AttentionHelpersTest, CausalMaskPreRemaxPushesFutureToMaskValue) {
    int timestep = 3;
    std::vector<float> mu_qk(timestep * timestep, 1.0f);
    std::vector<float> var_qk(timestep * timestep, 0.1f);
    std::vector<float> mu_mqk(timestep * timestep);
    std::vector<float> var_mqk(timestep * timestep);

    causal_mask_pre_remax(mu_qk, var_qk, 1, 1, timestep, mu_mqk, var_mqk);

    for (int query = 0; query < timestep; query++) {
        for (int key = 0; key < timestep; key++) {
            int idx = query * timestep + key;
            if (key <= query) {
                EXPECT_FLOAT_EQ(mu_mqk[idx], 1.0f);
                EXPECT_FLOAT_EQ(var_mqk[idx], 0.1f);
            } else {
                EXPECT_FLOAT_EQ(mu_mqk[idx], ATTENTION_MASK_MU);
                EXPECT_FLOAT_EQ(var_mqk[idx], ATTENTION_MASK_VAR);
            }
        }
    }
}

TEST_F(AttentionHelpersTest, CausalMaskPostRemaxZeroesFuture) {
    int timestep = 3;
    std::vector<float> mu_att(timestep * timestep, 0.3f);
    std::vector<float> var_att(timestep * timestep, 0.1f);
    std::vector<float> jcb(timestep * timestep, 0.2f);

    causal_mask_post_remax(mu_att, var_att, jcb, 1, 1, timestep);

    for (int query = 0; query < timestep; query++) {
        for (int key = 0; key < timestep; key++) {
            int idx = query * timestep + key;
            float expected = key <= query ? 1.0f : 0.0f;
            EXPECT_FLOAT_EQ(mu_att[idx], 0.3f * expected);
            EXPECT_FLOAT_EQ(var_att[idx], 0.1f * expected);
            EXPECT_FLOAT_EQ(jcb[idx], 0.2f * expected);
        }
    }
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

    mha_delta_score(mu_v, delta_mu_out, delta_var_out, batch_size, num_heads,
                    timestep, head_size, delta_mu_scores, delta_var_scores);

    std::vector<float> delta_mu_v(batch_size * num_heads * timestep *
                                  head_size);
    std::vector<float> delta_var_v(batch_size * num_heads * timestep *
                                   head_size);

    mha_delta_value(mu_att_scores, delta_mu_out, delta_var_out, batch_size,
                    num_heads, timestep, head_size, delta_mu_v, delta_var_v);

    float sum_delta_scores = 0.0f;
    for (auto val : delta_mu_scores) sum_delta_scores += std::abs(val);
    EXPECT_GT(sum_delta_scores, 0.0f);

    float sum_delta_v = 0.0f;
    for (auto val : delta_mu_v) sum_delta_v += std::abs(val);
    EXPECT_GT(sum_delta_v, 0.0f);
}

TEST_F(AttentionHelpersTest, AttentionQueryKeyDelta) {
    int batch_size = 1;
    int num_heads = 2;
    int timestep = 3;
    int head_size = 4;

    std::vector<float> mu_q(batch_size * num_heads * timestep * head_size);
    std::vector<float> var_q(batch_size * num_heads * timestep * head_size);
    std::vector<float> mu_k(batch_size * num_heads * timestep * head_size);
    std::vector<float> var_k(batch_size * num_heads * timestep * head_size);

    fill_random(mu_q);
    fill_random(var_q);
    fill_random(mu_k);
    fill_random(var_k);

    for (auto& val : var_q) val = std::abs(val) + 0.1f;
    for (auto& val : var_k) val = std::abs(val) + 0.1f;

    std::vector<float> delta_mu_att_score(batch_size * num_heads * timestep *
                                          timestep);
    std::vector<float> delta_var_att_score(batch_size * num_heads * timestep *
                                           timestep);
    std::vector<float> jcb_att_score(batch_size * num_heads * timestep *
                                     timestep);

    fill_random(delta_mu_att_score);
    fill_random(delta_var_att_score);
    fill_random(jcb_att_score);

    for (auto& val : delta_var_att_score) val = std::abs(val);

    std::vector<float> delta_mu_q(batch_size * num_heads * timestep *
                                  head_size);
    std::vector<float> delta_var_q(batch_size * num_heads * timestep *
                                   head_size);

    mha_delta_query(var_q, mu_k, delta_mu_att_score, delta_var_att_score,
                    jcb_att_score, batch_size, num_heads, timestep, head_size,
                    delta_mu_q, delta_var_q);

    float sum_delta_q = 0.0f;
    for (auto val : delta_mu_q) sum_delta_q += std::abs(val);
    EXPECT_GT(sum_delta_q, 0.0f);

    for (size_t i = 0; i < delta_var_q.size(); i++) {
        EXPECT_GE(delta_var_q[i], 0.0f);
    }

    std::vector<float> delta_mu_k(batch_size * num_heads * timestep *
                                  head_size);
    std::vector<float> delta_var_k(batch_size * num_heads * timestep *
                                   head_size);

    mha_delta_key(var_k, mu_q, delta_mu_att_score, delta_var_att_score,
                  jcb_att_score, batch_size, num_heads, timestep, head_size,
                  delta_mu_k, delta_var_k);

    float sum_delta_k = 0.0f;
    for (auto val : delta_mu_k) sum_delta_k += std::abs(val);
    EXPECT_GT(sum_delta_k, 0.0f);

    for (size_t i = 0; i < delta_var_k.size(); i++) {
        EXPECT_GE(delta_var_k[i], 0.0f);
    }
}

TEST_F(AttentionHelpersTest, DeltaQueryKeyScaleBySqrtHeadDim) {
    // One token, one head, head_size 4: the score delta reaches q and k
    // through the 1/sqrt(head_size) = 1/2 factor of the forward pass.
    int head_size = 4;
    std::vector<float> mu_q = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> mu_k = {-1.0f, 0.5f, 2.0f, 1.0f};
    std::vector<float> var_unused(head_size, 0.0f);
    std::vector<float> delta_mu_s = {2.0f};
    std::vector<float> delta_var_s = {4.0f};
    std::vector<float> jcb = {1.0f};

    std::vector<float> delta_mu_q(head_size), delta_var_q(head_size);
    mha_delta_query(var_unused, mu_k, delta_mu_s, delta_var_s, jcb, 1, 1, 1,
                    head_size, delta_mu_q, delta_var_q);
    std::vector<float> delta_mu_k(head_size), delta_var_k(head_size);
    mha_delta_key(var_unused, mu_q, delta_mu_s, delta_var_s, jcb, 1, 1, 1,
                  head_size, delta_mu_k, delta_var_k);

    std::vector<float> expected_mu_q = {-1.0f, 0.5f, 2.0f, 1.0f};
    std::vector<float> expected_var_q = {1.0f, 0.25f, 4.0f, 1.0f};
    std::vector<float> expected_mu_k = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected_var_k = {1.0f, 4.0f, 9.0f, 16.0f};
    EXPECT_TRUE(vectors_equal(delta_mu_q, expected_mu_q));
    EXPECT_TRUE(vectors_equal(delta_var_q, expected_var_q));
    EXPECT_TRUE(vectors_equal(delta_mu_k, expected_mu_k));
    EXPECT_TRUE(vectors_equal(delta_var_k, expected_var_k));
}

TEST_F(AttentionHelpersTest, DeltaQueryKeySumOverScoreRowsAndColumns) {
    // Two tokens, head_size 1, no mask. Score s[p][k] = q[p] * k[k], so
    // delta_q[p] sums row p of the score delta and delta_k[k] sums column k.
    std::vector<float> mu_q = {2.0f, 3.0f};
    std::vector<float> mu_k = {2.0f, 3.0f};
    std::vector<float> var_unused(2, 0.0f);
    std::vector<float> delta_mu_s = {1.0f, 4.0f, 5.0f, 7.0f};
    std::vector<float> delta_var_s = {1.0f, 4.0f, 5.0f, 7.0f};
    std::vector<float> jcb(4, 1.0f);

    std::vector<float> delta_mu_q(2), delta_var_q(2);
    mha_delta_query(var_unused, mu_k, delta_mu_s, delta_var_s, jcb, 1, 1, 2, 1,
                    delta_mu_q, delta_var_q);
    std::vector<float> delta_mu_k(2), delta_var_k(2);
    mha_delta_key(var_unused, mu_q, delta_mu_s, delta_var_s, jcb, 1, 1, 2, 1,
                  delta_mu_k, delta_var_k);

    // delta_q[0] = 2*1 + 3*4, delta_q[1] = 2*5 + 3*7
    EXPECT_TRUE(vectors_equal(delta_mu_q, {14.0f, 31.0f}));
    // var: 4*1 + 9*4, 4*5 + 9*7
    EXPECT_TRUE(vectors_equal(delta_var_q, {40.0f, 83.0f}));
    // delta_k[0] = 2*1 + 3*5, delta_k[1] = 2*4 + 3*7
    EXPECT_TRUE(vectors_equal(delta_mu_k, {17.0f, 29.0f}));
    // var: 4*1 + 9*5, 4*4 + 9*7
    EXPECT_TRUE(vectors_equal(delta_var_k, {49.0f, 79.0f}));
}

TEST(AttentionLayerBackward, ScoreDeltaUsesValues) {
    // One token, one head, head_dim 1: out = att * v with att = 1, so the
    // score delta is v * delta_out and its variance v^2 * delta_var_out.
    MultiheadAttention attn(1, 1, 1, 1, false, 1.0f, 1.0f, "Xavier", false);
    attn.training = true;
    attn.mu_w = {0.5f, -0.7f, 3.0f};
    attn.var_w = {0.01f, 0.01f, 0.01f};

    BaseHiddenStates input_states;
    input_states.mu_a = {2.0f};
    input_states.var_a = {0.1f};
    input_states.jcb = {1.0f};
    input_states.block_size = 1;
    input_states.actual_size = 1;
    BaseHiddenStates output_states;
    output_states.mu_a.resize(1);
    output_states.var_a.resize(1);
    output_states.jcb.resize(1);
    BaseTempStates temp_states;
    attn.forward(input_states, output_states, temp_states);

    BaseDeltaStates input_delta_states;
    input_delta_states.delta_mu = {0.5f};
    input_delta_states.delta_var = {0.2f};
    input_delta_states.block_size = 1;
    BaseDeltaStates output_delta_states;
    output_delta_states.delta_mu.resize(1);
    output_delta_states.delta_var.resize(1);
    attn.backward(input_delta_states, output_delta_states, temp_states);

    float mu_v = attn.attn_states.mu_v[0];
    EXPECT_FLOAT_EQ(mu_v, 6.0f);
    EXPECT_FLOAT_EQ(attn.attn_delta_states.delta_mu_att_score[0], 3.0f);
    EXPECT_FLOAT_EQ(attn.attn_delta_states.delta_var_att_score[0], 7.2f);
}

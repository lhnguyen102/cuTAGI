#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "../../include/base_output_updater.h"
#include "../../include/data_struct.h"
#include "../../include/linear_layer.h"
#include "../../include/positional_encoding.h"
#include "../../include/sequential.h"

namespace {
constexpr int SEQ_LEN = 3;

struct PEStates {
    BaseHiddenStates in, out;
    BaseTempStates temp;
};

PEStates make_zero_input(int embed_dim, int seq_len, float var) {
    int n = seq_len * embed_dim;
    PEStates s;
    s.in.mu_a.assign(n, 0.0f);
    s.in.var_a.assign(n, var);
    s.in.jcb.assign(n, 1.0f);
    s.in.size = n;
    s.in.block_size = 1;
    s.in.actual_size = embed_dim;
    s.in.seq_len = seq_len;
    s.out.mu_a.assign(n, 0.0f);
    s.out.var_a.assign(n, 0.0f);
    s.out.jcb.assign(n, 0.0f);
    s.out.size = n;
    return s;
}

// PE[t, d] = sin(t / 10000^(2*floor(d/2)/D)) for even d, cos(...) for odd d.
// Rows t = 1, 2; row t = 0 is sin(0)=0, cos(0)=1.
void expect_reference(int embed_dim,
                      const std::vector<std::vector<float>> &rows_t1_t2) {
    PositionalEncoding pe(embed_dim, /*max_seq_len=*/8);
    auto s = make_zero_input(embed_dim, SEQ_LEN, 0.3f);
    pe.forward(s.in, s.out, s.temp);

    for (int d = 0; d < embed_dim; d++) {
        EXPECT_NEAR(s.out.mu_a[d], d % 2 == 0 ? 0.0f : 1.0f, 1e-6f) << d;
    }
    for (int t = 1; t < SEQ_LEN; t++) {
        for (int d = 0; d < embed_dim; d++) {
            EXPECT_NEAR(s.out.mu_a[t * embed_dim + d], rows_t1_t2[t - 1][d],
                        1e-5f)
                << "t=" << t << " d=" << d;
        }
    }
    for (int i = 0; i < SEQ_LEN * embed_dim; i++) {
        EXPECT_FLOAT_EQ(s.out.var_a[i], 0.3f);
        EXPECT_FLOAT_EQ(s.out.jcb[i], 1.0f);
    }
    EXPECT_EQ(s.out.seq_len, SEQ_LEN);
    EXPECT_EQ(s.out.actual_size, embed_dim);
}
}  // namespace

TEST(PositionalEncoding, MatchesSinusoidReferenceOddEmbedDim) {
    expect_reference(
        5, {{0.8414710f, 0.5403023f, 0.0251162f, 0.9996845f, 0.0006310f},
            {0.9092974f, -0.4161468f, 0.0502166f, 0.9987384f, 0.0012619f}});
}

TEST(PositionalEncoding, MatchesSinusoidReferenceEvenEmbedDim) {
    expect_reference(6, {{0.8414710f, 0.5403023f, 0.0463992f, 0.9989230f,
                          0.0021544f, 0.9999977f},
                         {0.9092974f, -0.4161468f, 0.0926985f, 0.9956942f,
                          0.0043089f, 0.9999907f}});
}

TEST(PositionalEncoding, ThrowsWhenSeqLenExceedsMax) {
    PositionalEncoding pe(4, /*max_seq_len=*/2);
    auto s = make_zero_input(4, SEQ_LEN, 0.1f);
    EXPECT_THROW(pe.forward(s.in, s.out, s.temp), std::runtime_error);
}

TEST(PositionalEncoding, ThrowsWhenInputSizeDiffersFromEmbedDim) {
    PositionalEncoding pe(4, /*max_seq_len=*/8);
    auto s = make_zero_input(5, SEQ_LEN, 0.1f);
    EXPECT_THROW(pe.forward(s.in, s.out, s.temp), std::runtime_error);
}

// Adding PE and shifting the targets by the same PE must leave the delta
// reaching Linear unchanged, so both models learn identical weights.
TEST(PositionalEncoding, BackwardPassesDeltaUnchanged) {
    int batch_size = 2, input_size = 3, embed_dim = 4;
    int num_tokens = batch_size * SEQ_LEN;
    Sequential plain_model(Linear(input_size, embed_dim));
    Sequential pe_model(Linear(input_size, embed_dim),
                        PositionalEncoding(embed_dim, /*max_seq_len=*/8));
    pe_model.load_state_dict(plain_model.state_dict());

    std::vector<float> x(num_tokens * input_size);
    for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (i % 7) - 0.3f;
    std::vector<float> var_x(x.size(), 0.0f);
    std::vector<float> y(num_tokens * embed_dim, 0.5f);
    std::vector<float> var_y(y.size(), 0.01f);
    std::vector<float> pe = build_sinusoidal_pe(embed_dim, SEQ_LEN);
    std::vector<float> y_shifted(y.size());
    for (size_t i = 0; i < y.size(); i++) {
        y_shifted[i] = y[i] + pe[i % pe.size()];
    }

    std::vector<int> shape = {batch_size, SEQ_LEN, input_size};
    OutputUpdater output_updater("cpu");
    plain_model.forward(x, var_x, shape);
    output_updater.update(*plain_model.output_z_buffer, y, var_y,
                          *plain_model.input_delta_z_buffer);
    plain_model.backward();
    plain_model.step();

    pe_model.forward(x, var_x, shape);
    output_updater.update(*pe_model.output_z_buffer, y_shifted, var_y,
                          *pe_model.input_delta_z_buffer);
    pe_model.backward();
    pe_model.step();

    auto plain_params = plain_model.parameters();
    auto pe_params = pe_model.parameters();
    ASSERT_EQ(plain_params.size(), pe_params.size());
    for (size_t i = 0; i < plain_params.size(); i++) {
        auto &plain_mu = std::get<0>(plain_params[i]);
        auto &pe_mu = std::get<0>(pe_params[i]);
        ASSERT_EQ(plain_mu.size(), pe_mu.size());
        for (size_t j = 0; j < plain_mu.size(); j++) {
            EXPECT_NEAR(plain_mu[j], pe_mu[j], 1e-5f) << i << "," << j;
        }
    }
}

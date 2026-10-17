#include <gtest/gtest.h>

#include <vector>

#include "../../include/activation.h"
#include "../../include/base_output_updater.h"
#include "../../include/layer_block.h"
#include "../../include/linear_layer.h"
#include "../../include/resnet_block.h"
#include "../../include/rmsnorm_layer.h"
#include "../../include/sequential.h"

extern bool g_gpu_enabled;

namespace {

std::vector<float> first_weight_means(Sequential &model) {
    return std::get<0>(model.parameters().front());
}

void train_one_step(Sequential &model, std::vector<float> &x,
                    std::vector<int> shape, std::vector<float> &y) {
    OutputUpdater output_updater(model.device);
    std::vector<float> var_y(y.size(), 0.01f);
    std::vector<float> var_x(x.size(), 0.0f);
    model.forward(x, var_x, shape);
    output_updater.update(*model.output_z_buffer, y, var_y,
                          *model.input_delta_z_buffer);
    model.backward();
    model.step();
}

}  // namespace

TEST(SeqLenLayers, SingleLayerBlockAsFirstLayerUpdatesWeights) {
    Sequential model(LayerBlock(Linear(2, 1)));
    std::vector<float> x = {1.0f, -1.0f, 0.5f, 2.0f};
    std::vector<float> y = {3.0f, -3.0f};

    auto mu_w_before = first_weight_means(model);
    train_one_step(model, x, {2, 2}, y);
    auto mu_w_after = first_weight_means(model);

    EXPECT_NE(mu_w_before, mu_w_after);
}

void expect_selected_delta_for_every_token(const std::string &device) {
    int batch_size = 2, seq_len = 4, output_size = 3;
    int num_tokens = batch_size * seq_len;
    int selected_output = 2;  // 1-based, one per token
    Sequential model(Linear(2, output_size));
    model.to_device(device);
    std::vector<float> x(num_tokens * 2, 1.0f);
    std::vector<float> var_x(x.size(), 0.0f);
    std::vector<float> y(num_tokens, 5.0f);
    std::vector<float> var_y(num_tokens, 0.01f);
    std::vector<int> selected_idx(num_tokens, selected_output);

    model.forward(x, var_x, {batch_size, seq_len, 2});
    OutputUpdater output_updater(model.device);
    output_updater.update_using_indices(*model.output_z_buffer, y, var_y,
                                        selected_idx,
                                        *model.input_delta_z_buffer);
    model.delta_z_to_host();

    auto &delta_mu = model.input_delta_z_buffer->delta_mu;
    for (int t = 0; t < num_tokens; t++) {
        EXPECT_NE(delta_mu[t * output_size + selected_output - 1], 0.0f)
            << device << " token " << t;
    }
}

// These models act on each token independently, so [B, T, n] must train
// exactly like [B*T, n]. Both models must have the same architecture.
void expect_tokens_independent(Sequential &flat_model, Sequential &seq_model,
                               int input_size, int output_size,
                               const std::string &device) {
    int batch_size = 2, seq_len = 4;
    int num_tokens = batch_size * seq_len;
    seq_model.load_state_dict(flat_model.state_dict());
    flat_model.to_device(device);
    seq_model.to_device(device);

    std::vector<float> x(num_tokens * input_size);
    for (size_t i = 0; i < x.size(); i++) x[i] = 0.1f * (i % 7) - 0.3f;
    std::vector<float> y(num_tokens * output_size, 0.5f);

    train_one_step(flat_model, x, {num_tokens, input_size}, y);
    train_one_step(seq_model, x, {batch_size, seq_len, input_size}, y);
    flat_model.params_to_host();
    seq_model.params_to_host();

    auto flat_params = flat_model.parameters();
    auto seq_params = seq_model.parameters();
    ASSERT_EQ(flat_params.size(), seq_params.size());
    for (size_t i = 0; i < flat_params.size(); i++) {
        auto &flat_mu_w = std::get<0>(flat_params[i]);
        auto &seq_mu_w = std::get<0>(seq_params[i]);
        for (size_t j = 0; j < flat_mu_w.size(); j++) {
            EXPECT_NEAR(flat_mu_w[j], seq_mu_w[j], 1e-5)
                << device << " layer " << i << " weight " << j;
        }
    }
}

void expect_resnet_tokens_independent(const std::string &device) {
    int hidden = 4;
    Sequential flat_model(ResNetBlock(
        LayerBlock(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))));
    Sequential seq_model(ResNetBlock(
        LayerBlock(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden))));
    expect_tokens_independent(flat_model, seq_model, hidden, hidden, device);
}

void expect_softmax_tokens_independent(const std::string &device) {
    int input_size = 4, num_classes = 3;
    Sequential flat_model(Linear(input_size, num_classes), ClosedFormSoftmax());
    Sequential seq_model(Linear(input_size, num_classes), ClosedFormSoftmax());
    expect_tokens_independent(flat_model, seq_model, input_size, num_classes,
                              device);
}

TEST(SeqLenLayers, RMSNormTrainsTokensIndependently) {
    int hidden = 4;
    Sequential flat_model(Linear(hidden, hidden), RMSNorm({hidden}),
                          Linear(hidden, hidden));
    Sequential seq_model(Linear(hidden, hidden), RMSNorm({hidden}),
                         Linear(hidden, hidden));
    expect_tokens_independent(flat_model, seq_model, hidden, hidden, "cpu");
}

TEST(SeqLenLayers, ResNetBlockTrainsTokensIndependently) {
    expect_resnet_tokens_independent("cpu");
}

TEST(SeqLenLayers, ResNetBlockTrainsTokensIndependently_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_resnet_tokens_independent("cuda");
}

TEST(SeqLenLayers, ClosedFormSoftmaxTrainsTokensIndependently) {
    expect_softmax_tokens_independent("cpu");
}

TEST(SeqLenLayers, ClosedFormSoftmaxTrainsTokensIndependently_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_softmax_tokens_independent("cuda");
}

TEST(SeqLenLayers, SelectedOutputUpdateWritesDeltaForEveryToken) {
    expect_selected_delta_for_every_token("cpu");
}

TEST(SeqLenLayers, SelectedOutputUpdateWritesDeltaForEveryToken_CUDA) {
    if (!g_gpu_enabled) GTEST_SKIP() << "GPU tests are disabled.";
    expect_selected_delta_for_every_token("cuda");
}

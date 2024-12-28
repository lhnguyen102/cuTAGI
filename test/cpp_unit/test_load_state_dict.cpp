#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <stdexcept>
#include <string>

#include "../../include/activation.h"
#include "../../include/batchnorm_layer.h"
#include "../../include/conv2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/layer_block.h"
#include "../../include/layernorm_layer.h"
#include "../../include/linear_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/resnet_block.h"
#include "../../include/sequential.h"

LayerBlock create_basic_block(int in_channels, int out_channels, int stride = 1,
                              int padding_type = 1) {
    return LayerBlock(
        Conv2d(in_channels, out_channels, 3, false, stride, 1, padding_type),
        BatchNorm2d(out_channels), ReLU(),
        Conv2d(out_channels, out_channels, 3, false, 1, 1),
        BatchNorm2d(out_channels));
}

std::shared_ptr<Sequential> create_resnet_model() {
    // Create your blocks as usual
    auto block_1 = create_basic_block(8, 8);
    auto block_2 = create_basic_block(8, 8);
    auto block_3 = create_basic_block(8, 16, 2, 2);
    auto block_4 = create_basic_block(16, 16);
    auto block_5 = create_basic_block(16, 32, 2, 2);
    auto block_6 = create_basic_block(32, 32);
    auto block_7 = create_basic_block(32, 64, 2, 2);
    auto block_8 = create_basic_block(64, 64);

    ResNetBlock resnet_block_1(block_1);
    ResNetBlock resnet_block_2(block_2);

    ResNetBlock resnet_block_3(
        block_3, LayerBlock(Conv2d(8, 16, 2, false, 2), BatchNorm2d(16)));
    ResNetBlock resnet_block_4(block_4);

    ResNetBlock resnet_block_5(
        block_5, LayerBlock(Conv2d(16, 32, 2, false, 2), BatchNorm2d(32)));
    ResNetBlock resnet_block_6(block_6);

    ResNetBlock resnet_block_7(
        block_7, LayerBlock(Conv2d(32, 64, 2, false, 2), BatchNorm2d(64)));
    ResNetBlock resnet_block_8(block_8);

    // Instead of creating Sequential by value, use make_shared.
    auto model = std::make_shared<Sequential>(
        // Input block
        Conv2d(3, 8, 3, false, 1, 1, 1, 32, 32), BatchNorm2d(8), ReLU(),

        // Residual blocks
        resnet_block_1, ReLU(), resnet_block_2, ReLU(), resnet_block_3, ReLU(),
        resnet_block_4, ReLU(), resnet_block_5, ReLU(), resnet_block_6, ReLU(),
        resnet_block_7, ReLU(), resnet_block_8, ReLU(),

        // Output block
        AvgPool2d(4), Linear(64, 11));

    // Now return the shared pointer
    return model;
}

void load_dict_state() {
    Sequential model_1(Conv2d(3, 32, 5, true, 1, 2, 1, 32, 32), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Conv2d(32, 32, 5, true, 1, 2, 1),
                       MixtureReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(32, 64, 5, true, 1, 2, 1), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Linear(64 * 4 * 4, 100),
                       MixtureReLU(), Linear(100, 11));

    Sequential model_2(Conv2d(3, 32, 5, true, 1, 2, 1, 32, 32), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Conv2d(32, 32, 5, true, 1, 2, 1),
                       MixtureReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(32, 64, 5, true, 1, 2, 1), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Linear(64 * 4 * 4, 100),
                       MixtureReLU(), Linear(100, 11));

    model_1.preinit_layer();
    model_2.preinit_layer();

    auto state_dict_2 = model_2.state_dict();
    model_1.load_state_dict_v2(state_dict_2);

    for (size_t i = 0; i < model_1.layers.size(); ++i) {
        const auto &layer = model_1.layers[i];
        const auto &layer_2 = model_2.layers[i];

        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            ASSERT_EQ(layer->mu_w, layer_2->mu_w);
            ASSERT_EQ(layer->var_w, layer_2->var_w);
            ASSERT_EQ(layer->mu_b, layer_2->mu_b);
            ASSERT_EQ(layer->var_b, layer_2->var_b);
        }
    }
}

void load_resnet_state_dict() {
    auto model_1 = create_resnet_model();
    auto model_2 = create_resnet_model();

    model_1->preinit_layer();
    model_2->preinit_layer();

    auto state_dict_2 = model_2->state_dict();
    model_1->load_state_dict_v2(state_dict_2);

    // for (size_t i = 0; i < model_1.layers.size(); ++i) {
    //     const auto &layer = model_1.layers[i];
    //     const auto &layer_2 = model_2.layers[i];

    //     if (layer->get_layer_type() != LayerType::Activation &&
    //         layer->get_layer_type() != LayerType::Pool2d) {
    //         ASSERT_EQ(layer->mu_w, layer_2->mu_w);
    //         ASSERT_EQ(layer->var_w, layer_2->var_w);
    //         ASSERT_EQ(layer->mu_b, layer_2->mu_b);
    //         ASSERT_EQ(layer->var_b, layer_2->var_b);
    //     }
    // }
}

void parameters() {
    Sequential model_1(Conv2d(3, 32, 5, true, 1, 2, 1, 32, 32), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Conv2d(32, 32, 5, true, 1, 2, 1),
                       MixtureReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(32, 64, 5, true, 1, 2, 1), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Linear(64 * 4 * 4, 100),
                       MixtureReLU(), Linear(100, 11));
    Sequential model_2(Conv2d(3, 32, 5, true, 1, 2, 1, 32, 32), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Conv2d(32, 32, 5, true, 1, 2, 1),
                       MixtureReLU(), AvgPool2d(3, 2, 1, 2),
                       Conv2d(32, 64, 5, true, 1, 2, 1), MixtureReLU(),
                       AvgPool2d(3, 2, 1, 2), Linear(64 * 4 * 4, 100),
                       MixtureReLU(), Linear(100, 11));
    model_1.preinit_layer();
    model_2.preinit_layer();

    auto params_1 = model_1.parameters();
    auto params_2 = model_2.parameters();

    for (size_t i = 0; i < params_1.size(); ++i) {
        auto &param_1 = params_1[i].get();
        auto &param_2 = params_2[i].get();
        for (size_t j = 0; j < param_1.size(); ++j) {
            param_1[j] = param_2[j];
        }
    }

    for (size_t i = 0; i < model_1.layers.size(); ++i) {
        const auto &layer = model_1.layers[i];
        const auto &layer_2 = model_2.layers[i];

        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            ASSERT_EQ(layer->mu_w, layer_2->mu_w);
            ASSERT_EQ(layer->var_w, layer_2->var_w);
            ASSERT_EQ(layer->mu_b, layer_2->mu_b);
            ASSERT_EQ(layer->var_b, layer_2->var_b);
        }
    }
}

TEST(ModelStateTest, LoadDictState) { load_dict_state(); }

TEST(ModelStateTest, Parameters) { parameters(); }

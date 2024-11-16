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
#include "../../include/layernorm_layer.h"
#include "../../include/linear_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

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

    model_1.load_state_dict(model_2.get_state_dict());

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

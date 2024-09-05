#include "test_load_state_dict.h"

#include <chrono>
#include <ctime>
#include <stdexcept>
#include <string>

#include "../../include/activation.h"
#include "../../include/conv2d_layer.h"
#include "../../include/data_struct.h"
#include "../../include/dataloader.h"
#include "../../include/linear_layer.h"
#include "../../include/norm_layer.h"
#include "../../include/pooling_layer.h"
#include "../../include/sequential.h"

void load_dict_state() {
    // Create a toy model
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

    // Check if the state dict is loaded correctly
    for (size_t i = 0; i < model_1.layers.size(); ++i) {
        const auto &layer = model_1.layers[i];
        const auto &layer_2 = model_2.layers[i];

        if (layer->get_layer_type() != LayerType::Activation ||
            layer->get_layer_type() != LayerType::Pool2d) {
            if (layer->mu_w != layer_2->mu_w ||
                layer->var_w != layer_2->var_w ||
                layer->mu_b != layer_2->mu_b ||
                layer->var_b != layer_2->var_b) {
                throw std::runtime_error(
                    "Error in file: " + std::string(__FILE__) +
                    " at line: " + std::to_string(__LINE__) +
                    "Mismatch in layer parameters for layer " +
                    layer->get_layer_name());
            }
        }
    }
    // Print a success message
    std::cout << "State dict loaded successfully" << std::endl;
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

    // create a soft-update  for mu_w, var_w, mu_b, var_b
    for (size_t i = 0; i < params_1.size(); ++i) {
        auto &param_1 = params_1[i].get();
        auto &param_2 = params_2[i].get();
        for (size_t j = 0; j < param_1.size(); ++j) {
            param_1[j] = param_2[j];
        }
    }
    // create a test for the parameters
    for (size_t i = 0; i < model_1.layers.size(); ++i) {
        const auto &layer = model_1.layers[i];
        const auto &layer_2 = model_2.layers[i];

        if (layer->get_layer_type() != LayerType::Activation ||
            layer->get_layer_type() != LayerType::Pool2d) {
            if (layer->mu_w != layer_2->mu_w ||
                layer->var_w != layer_2->var_w ||
                layer->mu_b != layer_2->mu_b ||
                layer->var_b != layer_2->var_b) {
                throw std::runtime_error(
                    "Error in file: " + std::string(__FILE__) +
                    " at line: " + std::to_string(__LINE__) +
                    "Mismatch in layer parameters for layer " +
                    layer->get_layer_name());
            }
        }
    }
    // Print a success message
    std::cout << "Tested successfully" << std::endl;
}

int test_load_state_dict() {
    parameters();
    load_dict_state();
    return 0;
}
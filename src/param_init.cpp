///////////////////////////////////////////////////////////////////////////////
// File:         param_init.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 15, 2023
// Updated:      January 04, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/param_init.h"

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_linear(const std::string &init_method, const float gain_w,
                        const float gain_b, const int input_size,
                        const int output_size)
/**/
{
    int num_weights = input_size * output_size;
    float scale;
    if (init_method.compare("Xavier") == 0 ||
        init_method.compare("xavier") == 0) {
        scale = xavier_init(input_size, output_size);
    } else if (init_method.compare("He") == 0 ||
               init_method.compare("he") == 0) {
        scale = he_init(input_size);
    } else {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Initial parameter method [" +
                                    init_method + "] is not supported.");
    }

    // Initalize weights & biases
    std::vector<float> mu_w, var_w, mu_b, var_b;
    std::tie(mu_w, var_w) = gaussian_param_init(scale, gain_w, num_weights);

    std::tie(mu_b, var_b) = gaussian_param_init(scale, gain_b, output_size);

    return {mu_w, var_w, mu_b, var_b};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_conv2d(const size_t kernel_size, const size_t in_channels,
                        const size_t out_channels,
                        const std::string &init_method, const float gain_w,
                        const float gain_b, int num_weights, int num_biases)
/*
 */
{
    int fan_in = pow(kernel_size, 2) * in_channels;
    int fan_out = pow(kernel_size, 2) * out_channels;

    float scale;
    if (init_method.compare("Xavier") == 0 ||
        init_method.compare("xavier") == 0) {
        scale = xavier_init(fan_in, fan_out);
    } else if (init_method.compare("He") == 0 ||
               init_method.compare("he") == 0) {
        scale = he_init(fan_in);
    } else {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Initial parameter method [" +
                                    init_method + "] is not supported.");
    }

    // Initalize weights & biases
    std::vector<float> mu_w, var_w, mu_b, var_b;
    std::tie(mu_w, var_w) = gaussian_param_init(scale, gain_w, num_weights);

    if (num_biases > 0) {
        std::tie(mu_b, var_b) = gaussian_param_init(scale, gain_b, num_biases);
    }
    return {mu_w, var_w, mu_b, var_b};
}
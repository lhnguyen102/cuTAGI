///////////////////////////////////////////////////////////////////////////////
// File:         param_init.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 15, 2023
// Updated:      March 23, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/param_init.h"

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_linear(const std::string &init_method, const float gain_w,
                        const float gain_b, const int input_size,
                        const int output_size, int num_weights, int num_biases)
/**/
{
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
    if (num_biases > 0) {
        std::tie(mu_b, var_b) = gaussian_param_init(scale, gain_b, num_biases);
    }

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

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_lstm(const std::string &init_method, const float gain_w,
                      const float gain_b, const int input_size,
                      const int output_size, int num_weights, int num_biases)
/**/
{
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

    // Initalize weights & biases for 4 gates
    std::vector<float> mu_w_f, var_w_f, mu_b_f, var_b_f;
    std::vector<float> mu_w_i, var_w_i, mu_b_i, var_b_i;
    std::vector<float> mu_w_c, var_w_c, mu_b_c, var_b_c;
    std::vector<float> mu_w_o, var_w_o, mu_b_o, var_b_o;
    std::vector<float> mu_w, var_w, mu_b, var_b;
    int num_weight_gate = output_size * (input_size + output_size);

    std::tie(mu_w_f, var_w_f) =
        gaussian_param_init(scale, gain_w, num_weight_gate);
    std::tie(mu_w_i, var_w_i) =
        gaussian_param_init(scale, gain_w, num_weight_gate);
    std::tie(mu_w_c, var_w_c) =
        gaussian_param_init(scale, gain_w, num_weight_gate);
    std::tie(mu_w_o, var_w_o) =
        gaussian_param_init(scale, gain_w, num_weight_gate);

    mu_w.insert(mu_w.end(), mu_w_f.begin(), mu_w_f.end());
    mu_w.insert(mu_w.end(), mu_w_i.begin(), mu_w_i.end());
    mu_w.insert(mu_w.end(), mu_w_c.begin(), mu_w_c.end());
    mu_w.insert(mu_w.end(), mu_w_o.begin(), mu_w_o.end());

    var_w.insert(var_w.end(), var_w_f.begin(), var_w_f.end());
    var_w.insert(var_w.end(), var_w_i.begin(), var_w_i.end());
    var_w.insert(var_w.end(), var_w_c.begin(), var_w_c.end());
    var_w.insert(var_w.end(), var_w_o.begin(), var_w_o.end());

    if (num_biases > 0) {
        std::tie(mu_b_f, var_b_f) =
            gaussian_param_init(scale, gain_b, output_size);
        std::tie(mu_b_i, var_b_i) =
            gaussian_param_init(scale, gain_b, output_size);
        std::tie(mu_b_c, var_b_c) =
            gaussian_param_init(scale, gain_b, output_size);
        std::tie(mu_b_o, var_b_o) =
            gaussian_param_init(scale, gain_b, output_size);

        mu_b.insert(mu_b.end(), mu_b_f.begin(), mu_b_f.end());
        mu_b.insert(mu_b.end(), mu_b_i.begin(), mu_b_i.end());
        mu_b.insert(mu_b.end(), mu_b_c.begin(), mu_b_c.end());
        mu_b.insert(mu_b.end(), mu_b_o.begin(), mu_b_o.end());

        var_b.insert(var_b.end(), var_b_f.begin(), var_b_f.end());
        var_b.insert(var_b.end(), var_b_i.begin(), var_b_i.end());
        var_b.insert(var_b.end(), var_b_c.begin(), var_b_c.end());
        var_b.insert(var_b.end(), var_b_o.begin(), var_b_o.end());
    }

    return {mu_w, var_w, mu_b, var_b};
}
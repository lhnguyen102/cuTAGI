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

float he_init(float fan_in)

/* He initialization for neural networks. Further details can be found in
 * Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification. He et al., 2015.
 *
 * Args:
 *    fan_in: Number of input variables
 * Returns:
 *    scale: Standard deviation for weight distribution
 *
 *  */

{
    float scale = pow(1 / fan_in, 0.5);

    return scale;
}

float xavier_init(float fan_in, float fan_out)

/* Xavier initialization for neural networks. Further details can be found in
 *  Understanding the difficulty of training deep feedforward neural networks
 *  - Glorot, X. & Bengio, Y. (2010).
 *
 * Args:
 *    fan_in: Number of input variables
 *    fan_out: Number of output variables
 *
 * Returns:
 *    scale: Standard deviation for weight distribution
 *
 *  */

{
    float scale;
    scale = pow(2 / (fan_in + fan_out), 0.5);

    return scale;
}

std::tuple<std::vector<float>, std::vector<float>> gaussian_param_init(
    float scale, float gain, int N, int seed)
/* Parmeter initialization of TAGI neural networks.
 *
 * Args:
 *    scale: Standard deviation for weight distribution
 *    gain: Mutiplication factor
 *    N: Number of parameters
 *
 * Returns:
 *    m: Mean
 *    S: Variance
 *
 *  */
{
    // Initialize device
    std::random_device rd;

    // Mersenne twister PRNG - seed
    std::mt19937 gen(seed >= 0 ? seed : rd());

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);

    // Weights
    for (int i = 0; i < N; i++) {
        // Variance
        S[i] = pow(gain * scale, 2);

        // Get normal distribution
        std::normal_distribution<float> d(0.0f, scale);

        // Get sample for weights
        m[i] = d(gen);
    }

    return {m, S};
}

std::tuple<std::vector<float>, std::vector<float>> gaussian_param_init_ni(
    float scale, float gain, float noise_gain, int N, int seed)
/* Parmeter initialization of TAGI neural network including the noise's hidden
 * states
 *
 * Args:
 *    scale: Standard deviation for weight distribution
 *    gain: Mutiplication factor
 *    N: Number of parameters
 *
 * Returns:
 *    m: Mean
 *    S: Variance
 *
 *  */
{
    // Initialize device
    std::random_device rd;

    // Mersenne twister PRNG - seed
    std::mt19937 gen(seed >= 0 ? seed : rd());

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);

    // Weights
    for (int i = 0; i < N; i++) {
        // Variance for output and noise's hidden states
        if (i < N / 2) {
            S[i] = gain * pow(scale, 2);
        } else {
            S[i] = noise_gain * pow(scale, 2);
            scale = pow(S[i], 0.5);
            int a = 0;
        }

        // Get normal distribution
        std::normal_distribution<float> d(0.0f, scale);

        // Get sample for weights
        m[i] = d(gen);
    }

    return {m, S};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_linear(const std::string &init_method, const float gain_w,
                        const float gain_b, const int seed,
                        const int input_size, const int output_size,
                        int num_weights, int num_biases)
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
    std::tie(mu_w, var_w) =
        gaussian_param_init(scale, gain_w, num_weights, seed);
    if (num_biases > 0) {
        std::tie(mu_b, var_b) =
            gaussian_param_init(scale, gain_b, num_biases, seed);
    }

    return {mu_w, var_w, mu_b, var_b};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_conv2d(const size_t kernel_size, const size_t in_channels,
                        const size_t out_channels,
                        const std::string &init_method, const float gain_w,
                        const float gain_b, const int seed, int num_weights,
                        int num_biases)
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
    std::tie(mu_w, var_w) =
        gaussian_param_init(scale, gain_w, num_weights, seed);

    if (num_biases > 0) {
        std::tie(mu_b, var_b) =
            gaussian_param_init(scale, gain_b, num_biases, seed);
    }
    return {mu_w, var_w, mu_b, var_b};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_lstm(const std::string &init_method, const float gain_w,
                      const float gain_b, const int seed, const int input_size,
                      const int output_size, int num_weights, int num_biases)
/**/
{
    float scale;
    if (init_method.compare("Xavier") == 0 ||
        init_method.compare("xavier") == 0) {
        scale = xavier_init(input_size + output_size, output_size);
    } else if (init_method.compare("He") == 0 ||
               init_method.compare("he") == 0) {
        scale = he_init(input_size + output_size);
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
        gaussian_param_init(scale, gain_w, num_weight_gate, seed);
    std::tie(mu_w_i, var_w_i) =
        gaussian_param_init(scale, gain_w, num_weight_gate, seed);
    std::tie(mu_w_c, var_w_c) =
        gaussian_param_init(scale, gain_w, num_weight_gate, seed);
    std::tie(mu_w_o, var_w_o) =
        gaussian_param_init(scale, gain_w, num_weight_gate, seed);

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
            gaussian_param_init(scale, gain_b, output_size, seed);
        std::tie(mu_b_i, var_b_i) =
            gaussian_param_init(scale, gain_b, output_size, seed);
        std::tie(mu_b_c, var_b_c) =
            gaussian_param_init(scale, gain_b, output_size, seed);
        std::tie(mu_b_o, var_b_o) =
            gaussian_param_init(scale, gain_b, output_size, seed);

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
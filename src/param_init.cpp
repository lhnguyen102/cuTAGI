#include "../include/param_init.h"

#include "../include/custom_logger.h"

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
    float scale, float gain, int N)
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
    // Get generator
    std::mt19937 &gen = SeedManager::get_instance().get_engine();

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);
    std::normal_distribution<float> dist_mean(0.0f, scale);

    // Weights
    for (int i = 0; i < N; i++) {
        m[i] = dist_mean(gen);
        S[i] = pow(gain * scale, 2);
        // S[i] = pow(gain * m[i],2);
    }

    return {m, S};
}

std::tuple<std::vector<float>, std::vector<float>> uniform_param_init(
    float scale, float gain, int N)
/* Parameter initialization of TAGI neural networks using uniform distribution.
 *
 * Args:
 *    scale: Range for uniform distribution [-scale, scale]
 *    gain: Multiplication factor
 *    N: Number of parameters
 *
 * Returns:
 *    m: Mean
 *    S: Variance
 *
 */
{
    // Get generator
    std::mt19937 &gen = SeedManager::get_instance().get_engine();

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);

    // Uniform distribution bounds
    float lower_bound = -gain * scale;
    float upper_bound = gain * scale;
    std::uniform_real_distribution<float> d(lower_bound, upper_bound);

    // Weights
    for (int i = 0; i < N; i++) {
        m[i] = d(gen);
        S[i] = pow(gain * scale, 2);
    }

    return {m, S};
}

std::tuple<std::vector<float>, std::vector<float>> gaussian_param_init_ni(
    float scale, float gain, float noise_gain, int N)
/* Parmeter initialization of TAGI neural network including the noise's
 * hidden states
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
    // Get generator
    std::mt19937 &gen = SeedManager::get_instance().get_engine();

    // Initialize pointers
    std::vector<float> S(N);
    std::vector<float> m(N);
    std::uniform_real_distribution<float> dist_std(0.01f * gain * scale,
                                                   0.1f * gain * scale);
    std::normal_distribution<float> dist_mean(0.0f, scale);
    std::normal_distribution<float> dist_mean_noise(0.0f, scale);

    // Weights
    for (int i = 0; i < N; i++) {
        if (i < N / 2) {
            m[i] = dist_mean(gen);
            float stdev = gain * scale;
            S[i] = stdev * stdev;
        } else {
            m[i] = dist_mean_noise(gen);
            float stdev = noise_gain * scale;
            S[i] = stdev * stdev;
        }
    }

    return {m, S};
}

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
        LOG(LogLevel::ERROR,
            "Initial parameter method [" + init_method + "] is not supported.");
    }

    // Initalize weights & biases
    std::vector<float> mu_w, var_w, mu_b, var_b;
    std::tie(mu_w, var_w) = gaussian_param_init(scale, gain_w, num_weights);
    if (num_biases > 0) {
        std::tie(mu_b, var_b) = gaussian_param_init(scale, gain_b, num_biases);
        // std::tie(mu_b, var_b) = uniform_param_init(scale, gain_b,
        // num_biases);
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
        LOG(LogLevel::ERROR,
            "Initial parameter method [" + init_method + "] is not supported.");
    }

    // Initalize weights & biases
    std::vector<float> mu_w, var_w, mu_b, var_b;
    std::tie(mu_w, var_w) = gaussian_param_init(scale, gain_w, num_weights);

    if (num_biases > 0) {
        std::tie(mu_b, var_b) = gaussian_param_init(scale, gain_b, num_biases);
        // std::tie(mu_b, var_b) = uniform_param_init(scale, gain_b,
        // num_biases);
    }
    return {mu_w, var_w, mu_b, var_b};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
init_weight_bias_norm(const std::string &init_method, const float gain_w,
                      const float gain_b, const int input_size,
                      const int output_size, int num_weights, int num_biases) {
    std::vector<float> mu_w, var_w, mu_b, var_b;

    float scale = 2.0f / (input_size + output_size);

    mu_w.resize(num_weights, 1.0f);
    var_w.resize(num_weights, scale * gain_w * gain_w);
    if (num_biases > 0) {
        mu_b.resize(num_weights, 0.0f);
        var_b.resize(num_weights, scale * gain_b * gain_b);
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
        scale = xavier_init(input_size + output_size, output_size);
    } else if (init_method.compare("He") == 0 ||
               init_method.compare("he") == 0) {
        scale = he_init(input_size + output_size);
    } else {
        LOG(LogLevel::ERROR,
            "Initial parameter method [" + init_method + "] is not supported.");
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
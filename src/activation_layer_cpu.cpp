///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      October 15, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/activation_layer_cpu.h"

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////
Relu::Relu(){};
Relu::~Relu(){};
void Relu::relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a)
/*
 */
{
    float zero_pad = 0;
    float one_pad = 1;
    float tmp;
    int col;
    for (col = start_chunk; col < end_chunk; col++) {
        tmp = std::max(mu_z[col], zero_pad);
        mu_a[col] = tmp;
        if (tmp == 0) {
            jcb[col] = zero_pad;
            var_a[col] = zero_pad;
        } else {
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

void Relu::relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            int n, unsigned int num_threads,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(&Relu::relu_mean_var, this, std::ref(mu_z),
                        std::ref(var_z), start_chunk, end_chunk, std::ref(mu_a),
                        std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void Relu::forward(HiddenStates &input_states, HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->relu_mean_var(input_states.mu_z, input_states.var_z, start_chunk,
                        end_chunk, output_states.mu_a, output_states.jcb,
                        output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
Sigmoid::Sigmoid(){};
Sigmoid::~Sigmoid(){};
void Sigmoid::sigmoid_mean_var(std::vector<float> &mu_z,
                               std::vector<float> &var_z, int start_chunk,
                               int end_chunk, std::vector<float> &mu_a,
                               std::vector<float> &jcb,
                               std::vector<float> &var_a)
/*
 */
{
    float tmp;
    for (int col = start_chunk; col < end_chunk; col++) {
        tmp = 1 / (1 + expf(-mu_z[col]));
        mu_a[col] = tmp;
        jcb[col] = tmp * (1 - tmp);
        var_a[col] = tmp * (1 - tmp) * var_z[col] * tmp * (1 - tmp);
    }
}

void Sigmoid::sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                  std::vector<float> &var_z, int n,
                                  unsigned int num_threads,
                                  std::vector<float> &mu_a,
                                  std::vector<float> &jcb,
                                  std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(&Sigmoid::sigmoid_mean_var, this, std::ref(mu_z),
                        std::ref(var_z), start_chunk, end_chunk, std::ref(mu_a),
                        std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void Sigmoid::forward(HiddenStates &input_states, HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->sigmoid_mean_var(input_states.mu_z, input_states.var_z, start_chunk,
                           end_chunk, output_states.mu_a, output_states.jcb,
                           output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
Tanh::Tanh() {}
Tanh::~Tanh() {}
void Tanh::tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a)
/*
 */
{
    float tmp = 0;
    for (int col = start_chunk; col < end_chunk; col++) {
        tmp = tanhf(mu_z[col]);
        mu_a[col] = tmp;
        jcb[col] = (1 - tmp * tmp);
        var_a[col] = (1 - tmp * tmp) * var_z[col] * (1 - tmp * tmp);
    }
}

void Tanh::tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            int n, unsigned int num_threads,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(&Tanh::tanh_mean_var, this, std::ref(mu_z),
                        std::ref(var_z), start_chunk, end_chunk, std::ref(mu_a),
                        std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void Tanh::forward(HiddenStates &input_states, HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->tanh_mean_var(input_states.mu_z, input_states.var_z, start_chunk,
                        end_chunk, output_states.mu_a, output_states.jcb,
                        output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
MixtureRelu::MixtureRelu() {}
MixtureRelu::~MixtureRelu() {}

void MixtureRelu::mixture_relu_mean_var(std::vector<float> &mu_z,
                                        std::vector<float> &var_z,
                                        float omega_tol, int start_chunk,
                                        int end_chunk, std::vector<float> &mu_a,
                                        std::vector<float> &jcb,
                                        std::vector<float> &var_a)
/*
 */
{
    float alpha, beta, omega, kappa, mu_z_til, var_z_til;
    for (int i = start_chunk; i < end_chunk; i++) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mu_z[i] / powf(var_z[i], 0.5);
        omega = std::max(1 - normcdf_cpu(alpha), omega_tol);
        beta = normpdf_cpu(alpha, 0.0f, 1.0f) / omega;
        kappa = 1 + alpha * beta - powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[i] + beta * powf(var_z[i], 0.5);
        var_z_til = kappa * var_z[i];

        // Activation distribution
        if (omega * mu_z_til > omega_tol) {
            mu_a[i] = omega * mu_z_til;
            var_a[i] =
                omega * var_z_til + omega * (1 - omega) * powf(var_z_til, 2);
            jcb[i] = powf(omega * kappa, 0.5);
        } else {
            mu_a[i] = omega_tol;
            var_a[i] =
                omega * var_z_til + omega * (1 - omega) * powf(omega_tol, 2);
            jcb[i] = 0.0f;
        }
    }
}

void MixtureRelu::mixture_relu_mean_var_mp(
    std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol, int n,
    unsigned int num_threads, std::vector<float> &mu_a, std::vector<float> &jcb,
    std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(&MixtureRelu::mixture_relu_mean_var, this,
                                 std::ref(mu_z), std::ref(var_z), omega_tol,
                                 start_chunk, end_chunk, std::ref(mu_a),
                                 std::ref(jcb), std::ref(var_a));
    }
    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void MixtureRelu::forward(HiddenStates &input_states,
                          HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->mixture_relu_mean_var(
        input_states.mu_z, input_states.var_z, this->omega_tol, start_chunk,
        end_chunk, output_states.mu_a, output_states.jcb, output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoid::MixtureSigmoid(){};
MixtureSigmoid::~MixtureSigmoid(){};

void MixtureSigmoid::mixture_sigmoid_mean_var(
    std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
    int start_chunk, int end_chunk, std::vector<float> &mu_a,
    std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = start_chunk; i < end_chunk; i++) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mu_z[i]) / powf(var_z[i], 0.5);
        alpha_upper = (1.0f - mu_z[i]) / powf(var_z[i], 0.5);
        cdf_lower = normcdf_cpu(alpha_lower);
        cdf_upper = normcdf_cpu(alpha_upper);
        pdf_lower = normpdf_cpu(alpha_lower, 0.0f, 1.0f);
        pdf_upper = normpdf_cpu(alpha_upper, 0.0f, 1.0f);

        // Truncated distribution's parameters
        omega = std::max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                ((pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega) -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[i] - beta * pow(var_z[i], 0.5);
        var_z_til = kappa * var_z[i];

        // Activation distribution
        mu_a[i] =
            (omega * mu_z_til - cdf_lower + (1 - cdf_upper)) / 2.0f + 0.5f;
        var_a[i] = (omega * var_z_til + omega * powf(mu_z_til - mu_a[i], 2) +
                    cdf_lower * powf(1 + mu_a[i], 2) +
                    (1 - cdf_upper) * powf(1 - mu_a[i], 2)) /
                   4.0f;
        jcb[i] = powf(omega * kappa, 0.5);
    }
}
void MixtureSigmoid::mixture_sigmoid_mean_var_mp(
    std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol, int n,
    unsigned int num_threads, std::vector<float> &mu_a, std::vector<float> &jcb,
    std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            &MixtureSigmoid::mixture_sigmoid_mean_var, this, std::ref(mu_z),
            std::ref(var_z), omega_tol, start_chunk, end_chunk, std::ref(mu_a),
            std::ref(jcb), std::ref(var_a));
    }
    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}
void MixtureSigmoid::forward(HiddenStates &input_states,
                             HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->mixture_sigmoid_mean_var(
        input_states.mu_z, input_states.var_z, this->omega_tol, start_chunk,
        end_chunk, output_states.mu_a, output_states.jcb, output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanh::MixtureTanh(){};
MixtureTanh::~MixtureTanh(){};

void MixtureTanh::mixture_tanh_mean_var(std::vector<float> &mu_z,
                                        std::vector<float> &var_z,
                                        float omega_tol, int start_chunk,
                                        int end_chunk, std::vector<float> &mu_a,
                                        std::vector<float> &jcb,
                                        std::vector<float> &var_a)
/*
 */
{
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = start_chunk; i < end_chunk; i++) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mu_z[i]) / powf(var_z[i], 0.5);
        alpha_upper = (1.0f - mu_z[i]) / powf(var_z[i], 0.5);
        cdf_lower = normcdf_cpu(alpha_lower);
        cdf_upper = normcdf_cpu(alpha_upper);
        pdf_lower = normpdf_cpu(alpha_lower, 0.0f, 1.0f);
        pdf_upper = normpdf_cpu(alpha_upper, 0.0f, 1.0f);

        // Truncated distribution's parameters
        omega = std::max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                ((pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega) -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[i] - beta * pow(var_z[i], 0.5);
        var_z_til = kappa * var_z[i];

        // Activation distribution
        mu_a[i] = omega * mu_z_til - cdf_lower + (1 - cdf_upper);
        var_a[i] = omega * var_z_til + omega * powf(mu_z_til - mu_a[i], 2) +
                   cdf_lower * powf(1 + mu_a[i], 2) +
                   (1 - cdf_upper) * powf(1 - mu_a[i], 2);
        jcb[i] = powf(omega * kappa, 0.5);
    }
}

void MixtureTanh::mixture_tanh_mean_var_mp(
    std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol, int n,
    unsigned int num_threads, std::vector<float> &mu_a, std::vector<float> &jcb,
    std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(&MixtureTanh::mixture_tanh_mean_var, this,
                                 std::ref(mu_z), std::ref(var_z), omega_tol,
                                 start_chunk, end_chunk, std::ref(mu_a),
                                 std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void MixtureTanh::forward(HiddenStates &input_states,
                          HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->mixture_tanh_mean_var(
        input_states.mu_z, input_states.var_z, this->omega_tol, start_chunk,
        end_chunk, output_states.mu_a, output_states.jcb, output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
Softplus::Softplus(){};
Softplus::~Softplus(){};

void Softplus::softplus_mean_var(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int start_chunk,
                                 int end_chunk, std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a)
/*
 */
{
    float tmp;
    for (int col = start_chunk; col < end_chunk; col++) {
        mu_a[col] = logf(1 + expf(mu_z[col]));
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = tmp * var_z[col] * tmp;
    }
}

void Softplus::softplus_mean_var_mp(std::vector<float> &mu_z,
                                    std::vector<float> &var_z, int n,
                                    unsigned int num_threads,
                                    std::vector<float> &mu_a,
                                    std::vector<float> &jcb,
                                    std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(&Softplus::softplus_mean_var, this, std::ref(mu_z),
                        std::ref(var_z), start_chunk, end_chunk, std::ref(mu_a),
                        std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void Softplus::forward(HiddenStates &input_states, HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->softplus_mean_var(input_states.mu_z, input_states.var_z, start_chunk,
                            end_chunk, output_states.mu_a, output_states.jcb,
                            output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
LeakyRelu::LeakyRelu(){};
LeakyRelu::~LeakyRelu(){};

void LeakyRelu::leaky_relu_mean_var(std::vector<float> &mu_z,
                                    std::vector<float> &var_z, float alpha,
                                    int start_chunk, int end_chunk,
                                    std::vector<float> &mu_a,
                                    std::vector<float> &jcb,
                                    std::vector<float> &var_a)
/*
 */
{
    float zeroPad = 0;
    float onePad = 1;
    float tmp;
    int col;
    for (col = start_chunk; col < end_chunk; col++) {
        tmp = std::max(mu_z[col], zeroPad);
        if (tmp == 0) {
            mu_a[col] = alpha * mu_z[col];
            jcb[col] = alpha;
            var_a[col] = alpha * var_z[col] * alpha;
        } else {
            mu_a[col] = tmp;
            jcb[col] = onePad;
            var_a[col] = var_z[col];
        }
    }
}

void LeakyRelu::leaky_relu_mean_var_mp(std::vector<float> &mu_z,
                                       std::vector<float> &var_z, float alpha,
                                       int n, unsigned int num_threads,
                                       std::vector<float> &mu_a,
                                       std::vector<float> &jcb,
                                       std::vector<float> &var_a)
/*
 */
{
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_chunk, end_chunk;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_chunk = n_batch * i;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_chunk = n_batch * i + rem_batch;
            end_chunk = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(&LeakyRelu::leaky_relu_mean_var, this, std::ref(mu_z),
                        std::ref(var_z), alpha, start_chunk, end_chunk,
                        std::ref(mu_a), std::ref(jcb), std::ref(var_a));
    }

    for (int i = 0; i < num_threads; i++) {
        if (threads[i].joinable()) {
            threads[i].join();
        }
    }
}

void LeakyRelu::forward(HiddenStates &input_states, HiddenStates &output_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << std::endl;
        throw std::invalid_argument("Error: Input state size is zero.");
    }

    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.size;
    this->leaky_relu_mean_var(input_states.mu_z, input_states.var_z,
                              start_chunk, end_chunk, this->alpha,
                              output_states.mu_a, output_states.jcb,
                              output_states.var_a);

    // Copy activation mean and jacobian to the class member for backward pass
    for (int i = 0; i < output_states.size; i++) {
        this->mu_a[i] = output_states.mu_a[i];
        this->jcb[i] = output_states.jcb[i];
    }
}
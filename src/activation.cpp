///////////////////////////////////////////////////////////////////////////////
// File:         activation_layer_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      March 22, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/activation.h"
#ifdef USE_CUDA
#include "activation_cuda.cuh"
#endif

void relu_mean_var(std::vector<float> const &mu_z,
                   std::vector<float> const &var_z, int start_chunk,
                   int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a)
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

void relu_mean_var_mp(std::vector<float> const &mu_z,
                      std::vector<float> const &var_z, int n,
                      unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            relu_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                          var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void sigmoid_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int start_chunk, int end_chunk, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
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

void sigmoid_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int n, unsigned int num_threads,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            sigmoid_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                             var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                   int start_chunk, int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a)
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

void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int n, unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            tanh_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                          var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           float omega_tol, int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
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
                omega * var_z_til + omega * (1 - omega) * powf(mu_z_til, 2);
            // jcb[i] = powf(omega * kappa, 0.5); // Approximate formulation
            jcb[i] =
                (((pow(mu_z[i], 2) + var_z[i]) *  // Exact form. (Huber, 2020)
                      normcdf_cpu(mu_z[i] / pow(var_z[i], 0.5)) +
                  mu_z[i] * var_z[i] *
                      normpdf_cpu(0.0f, mu_z[i], pow(var_z[i], 0.5))) -
                 (mu_a[i] * mu_z[i])) /
                var_z[i];
        } else {
            mu_a[i] = omega_tol;
            var_a[i] =
                omega * var_z_til + omega * (1 - omega) * powf(omega_tol, 2);
            jcb[i] = 0.0f;
        }
    }
}

void mixture_relu_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, float omega_tol, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back(
            [=, &mu_z, &var_z, &omega_tol, &mu_a, &jcb, &var_a] {
                mixture_relu_mean_var(mu_z, var_z, omega_tol, start_chunk,
                                      end_chunk, mu_a, jcb, var_a);
            });
    }
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, float omega_tol,
                              int start_chunk, int end_chunk,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = start_chunk; i < end_chunk; i++) {
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
        jcb[i] = omega * 0.5;
    }
}
void mixture_sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, float omega_tol,
                                 int n, unsigned int num_threads,
                                 std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back(
            [=, &mu_z, &var_z, &omega_tol, &mu_a, &jcb, &var_a] {
                mixture_sigmoid_mean_var(mu_z, var_z, omega_tol, start_chunk,
                                         end_chunk, mu_a, jcb, var_a);
            });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           float omega_tol, int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
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
        jcb[i] = omega;
    }
}

void mixture_tanh_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, float omega_tol, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back(
            [=, &mu_z, &var_z, &omega_tol, &mu_a, &jcb, &var_a] {
                mixture_tanh_mean_var(mu_z, var_z, omega_tol, start_chunk,
                                      end_chunk, mu_a, jcb, var_a);
            });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void softplus_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a)
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

void softplus_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            softplus_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a, jcb,
                              var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void leaky_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         float alpha, int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
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

void leaky_relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            float alpha, int n, unsigned int num_threads,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a)
/*
 */
{
    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &alpha, &mu_a, &jcb, &var_a] {
            leaky_relu_mean_var(mu_z, var_z, alpha, start_chunk, end_chunk,
                                mu_a, jcb, var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void softmax_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int no, int batch_size, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    float sum, max_m, max_v;
    int idx;
    for (int i = 0; i < batch_size; i++) {
        sum = 0.0f;
        idx = i * no;
        auto max_idx =
            std::max_element(mu_z.begin() + idx, mu_z.begin() + idx + no) -
            mu_z.begin();
        max_m = mu_z[max_idx];
        max_v = var_z[max_idx];
        for (int j = 0; j < no; j++) {
            mu_a[idx + j] = expf(mu_z[idx + j] - max_m);
            sum += mu_a[idx + j];
        }
        for (int j = 0; j < no; j++) {
            mu_a[idx + j] = mu_a[idx + j] / sum;
            jcb[idx + j] = mu_a[idx + j] * (1 - mu_a[idx + j]);
            // TODO: double check on covariance formulation
            var_a[idx + j] =
                jcb[idx + j] * (var_z[idx + j] + max_v) * jcb[idx + j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////

ReLU::ReLU(){};
ReLU::~ReLU(){};

std::string ReLU::get_layer_info() const
/*
 */
{
    return "ReLU()";
}

std::string ReLU::get_layer_name() const
/*
 */
{
    return "ReLU";
}

LayerType ReLU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ReLU::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states, BaseTempStates &temp_states)
/*
 */
{
    // Validate input. TODO: to be removed
    if (input_states.size == 0) {
        std::cerr << "Error in file: " << __FILE__ << " at line: " << __LINE__
                  << ". Reason: Invalid input state size (size is 0).\n";
        throw std::invalid_argument("Error: Invalid input state size");
    }

    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    if (this->num_threads > 1) {
        relu_mean_var_mp(input_states.mu_a, input_states.var_a, end_chunk,
                         this->num_threads, output_states.mu_a,
                         output_states.jcb, output_states.var_a);
    } else {
        relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                      end_chunk, output_states.mu_a, output_states.jcb,
                      output_states.var_a);
    }

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> ReLU::to_cuda() {
    this->device = "cuda";
    return std::make_unique<ReLUCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
Sigmoid::Sigmoid(){};
Sigmoid::~Sigmoid(){};

std::string Sigmoid::get_layer_info() const
/*
 */
{
    return "Sigmoid()";
}

std::string Sigmoid::get_layer_name() const
/*
 */
{
    return "Sigmoid";
}

LayerType Sigmoid::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Sigmoid::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    sigmoid_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                     end_chunk, output_states.mu_a, output_states.jcb,
                     output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Sigmoid::to_cuda() {
    this->device = "cuda";
    return std::make_unique<SigmoidCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
Tanh::Tanh() {}
Tanh::~Tanh() {}

std::string Tanh::get_layer_info() const
/*
 */
{
    return "Tanh()";
}

std::string Tanh::get_layer_name() const
/*
 */

{
    return "Tanh";
}

LayerType Tanh::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Tanh::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states, BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    tanh_mean_var(input_states.mu_a, input_states.var_a, start_chunk, end_chunk,
                  output_states.mu_a, output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Tanh::to_cuda() {
    this->device = "cuda";
    return std::make_unique<TanhCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture ReLU
////////////////////////////////////////////////////////////////////////////////
MixtureRelu::MixtureRelu() {}
MixtureRelu::~MixtureRelu() {}

std::string MixtureRelu::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureRelu::get_layer_name() const
/*
 */

{
    return "MixtureReLU";
}

LayerType MixtureRelu::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureRelu::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_relu_mean_var(
        input_states.mu_a, input_states.var_a, this->omega_tol, start_chunk,
        end_chunk, output_states.mu_a, output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MixtureRelu::to_cuda() {
    this->device = "cuda";
    return std::make_unique<MixtureReluCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoid::MixtureSigmoid(){};
MixtureSigmoid::~MixtureSigmoid(){};

std::string MixtureSigmoid::get_layer_info() const
/*
 */
{
    return "MixtureSigmoid()";
}

std::string MixtureSigmoid::get_layer_name() const
/*
 */

{
    return "MixtureSigmoid";
}

LayerType MixtureSigmoid::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureSigmoid::forward(BaseHiddenStates &input_states,
                             BaseHiddenStates &output_states,
                             BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_sigmoid_mean_var(
        input_states.mu_a, input_states.var_a, this->omega_tol, start_chunk,
        end_chunk, output_states.mu_a, output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MixtureSigmoid::to_cuda() {
    this->device = "cuda";
    return std::make_unique<MixtureSigmoidCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanh::MixtureTanh(){};
MixtureTanh::~MixtureTanh(){};

std::string MixtureTanh::get_layer_info() const
/*
 */
{
    return "MixtureTanh()";
}

std::string MixtureTanh::get_layer_name() const
/*
 */

{
    return "MixtureTanh";
}

LayerType MixtureTanh::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureTanh::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_tanh_mean_var(
        input_states.mu_a, input_states.var_a, this->omega_tol, start_chunk,
        end_chunk, output_states.mu_a, output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> MixtureTanh::to_cuda() {
    this->device = "cuda";
    return std::make_unique<MixtureTanhCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
Softplus::Softplus(){};
Softplus::~Softplus(){};
std::string Softplus::get_layer_info() const
/*
 */
{
    return "Softplus()";
}

std::string Softplus::get_layer_name() const
/*
 */

{
    return "Softplus";
}

LayerType Softplus::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Softplus::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    softplus_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                      end_chunk, output_states.mu_a, output_states.jcb,
                      output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Softplus::to_cuda() {
    this->device = "cuda";
    return std::make_unique<SoftplusCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
LeakyRelu::LeakyRelu(){};
LeakyRelu::~LeakyRelu(){};

std::string LeakyRelu::get_layer_info() const
/*
 */
{
    return "leakyReLU()";
}

std::string LeakyRelu::get_layer_name() const
/*
 */

{
    return "leakReLU";
}

LayerType LeakyRelu::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyRelu::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
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
    int end_chunk = input_states.actual_size * input_states.block_size;
    leaky_relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                        end_chunk, this->alpha, output_states.mu_a,
                        output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LeakyRelu::to_cuda() {
    this->device = "cuda";
    return std::make_unique<LeakyReluCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Stable Softmax
////////////////////////////////////////////////////////////////////////////////
Softmax::Softmax() {}
Softmax::~Softmax() {}
std::string Softmax::get_layer_info() const
/*
 */
{
    return "Softmax()";
}

std::string Softmax::get_layer_name() const
/*
 */

{
    return "Softmax";
}

LayerType Softmax::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Softmax::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
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
    int batch_size = input_states.size / input_states.block_size;
    softmax_mean_var(input_states.mu_a, input_states.var_a,
                     input_states.block_size, batch_size, output_states.mu_a,
                     output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Softmax::to_cuda() {
    this->device = "cuda";
    return std::make_unique<SoftmaxCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
RemaxA::RemaxA() {}
RemaxA::~RemaxA() {}

std::string RemaxA::get_layer_info() const
/*
 */
{
    return "RemaxA()";
}

std::string RemaxA::get_layer_name() const
/*
 */

{
    return "RemaxA";
}

void RemaxA::to_log(std::vector<float> &mu_m, std::vector<float> &var_m, int no,
                    int B, std::vector<float> &mu_log,
                    std::vector<float> &var_log)
/*
 */
{
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_var =
                logf(1.0f + (var_m[i * no + j] / powf(mu_m[i * no + j], 2)));
            tmp_mu = logf(mu_m[i * no + j]) - 0.5 * tmp_var;
            mu_log[i * no + j] = tmp_mu;
            var_log[i * no + j] = tmp_var;
        }
    }
}

void RemaxA::sum_class_hidden_states(std::vector<float> &mu_m,
                                     std::vector<float> &var_m, int no, int B,
                                     std::vector<float> &mu_sum,
                                     std::vector<float> &var_sum)
/*
 */
{
    float sum_mu, sum_var;
    for (int i = 0; i < B; i++) {
        sum_mu = 0.0f;
        sum_var = 0.0f;
        for (int j = 0; j < no; j++) {
            sum_mu += mu_m[i * no + j];
            sum_var += var_m[i * no + j];
        }
        mu_sum[i] = sum_mu;
        var_sum[i] = sum_var;
    }
}

void RemaxA::compute_cov_log_logsum(std::vector<float> &mu_m,
                                    std::vector<float> &var_m,
                                    std::vector<float> &mu_sum, int no, int B,
                                    std::vector<float> &cov_log_logsum)
/*
 */
{
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            cov_log_logsum[i * no + j] =
                logf(1.0f + var_m[i * no + j] * (1.0f / mu_sum[i]) *
                                (1.0f / mu_m[i * no + j]));
        }
    }
}

void RemaxA::compute_remax_prob(std::vector<float> &mu_log,
                                std::vector<float> &var_log,
                                std::vector<float> &mu_logsum,
                                std::vector<float> &var_logsum,
                                std::vector<float> &cov_log_logsum, int no,
                                int B, std::vector<float> &mu_a,
                                std::vector<float> &var_a)
/*
 */
{
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_mu = mu_log[i * no + j] - mu_logsum[i];
            tmp_var = var_log[i * no + j] + var_logsum[i] -
                      2 * cov_log_logsum[i * no + j];
            mu_a[i * no + j] = expf(tmp_mu + 0.5 * tmp_var);
            var_a[i * no + j] =
                expf(tmp_mu + 0.5 * tmp_var) * (expf(tmp_var) - 1.0f);
        }
    }
}
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
    float zero_pad = 0.0f;
    float one_pad = 1.0f;
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
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a)
/*
 */
{
    float std_z, alpha, pdf_alpha, cdf_alpha;
    for (int i = start_chunk; i < end_chunk; i++) {
        // Reused components for moments calculations
        std_z = powf(var_z[i], 0.5);
        alpha = mu_z[i] / std_z;
        pdf_alpha = normpdf_cpu(alpha, 0.0f, 1.0f);
        cdf_alpha = normcdf_cpu(alpha);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = mu_z[i] * cdf_alpha + std_z * pdf_alpha;
        var_a[i] = -powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] -
                   mu_z[i] * std_z * pdf_alpha +
                   (var_z[i] - powf(mu_z[i], 2)) * cdf_alpha;
        jcb[i] = cdf_alpha;
    }
}

void mixture_relu_mean_var_v2(const std::vector<float> &mu_z,
                              const std::vector<float> &var_z, int start_chunk,
                              int end_chunk, float threshold,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*TODO: to be reviewed
 */
{
    float std_z, alpha, pdf_alpha, cdf_alpha;
    for (int i = start_chunk; i < end_chunk; i++) {
        // Reused components for moments calculations
        std_z = powf(var_z[i], 0.5);
        alpha = mu_z[i] / std_z;
        pdf_alpha = normpdf_cpu(alpha, 0.0f, 1.0f);
        cdf_alpha = normcdf_cpu(alpha);

        // Ensure numerical stability
        pdf_alpha = std::max(pdf_alpha, threshold);
        cdf_alpha = std::max(cdf_alpha, threshold);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = std::max(1e-6f, mu_z[i] * cdf_alpha + std_z * pdf_alpha);
        var_a[i] =
            std::max(0.000001f, -powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] -
                                    mu_z[i] * std_z * pdf_alpha +
                                    (var_z[i] - powf(mu_z[i], 2)) * cdf_alpha);
        jcb[i] = cdf_alpha;
    }
}

void mixture_relu_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
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

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            mixture_relu_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a,
                                  jcb, var_a);
        });
    }
    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a)
/*
 */
{
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    for (int i = start_chunk; i < end_chunk; i++) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[i], 0.5);
        alpha_l = (1.0f + mu_z[i]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[i]) / std_z;  // Upper truncation
        cdf_l = normcdf_cpu(alpha_l);
        cdf_u = normcdf_cpu(alpha_u);
        pdf_l = normpdf_cpu(alpha_l, 0.0f, 1.0f);
        pdf_u = normpdf_cpu(alpha_u, 0.0f, 1.0f);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = (mu_z[i] + 1) * cdf_l + (mu_z[i] - 1) * cdf_u +
                  std_z * (pdf_l - pdf_u) - mu_z[i];
        var_a[i] =
            std::max(0.000001f,
                     (cdf_l * (var_z[i] - powf(mu_z[i], 2) - 2 * mu_z[i] - 1) +
                      cdf_u * (var_z[i] - powf(mu_z[i], 2) + 2 * mu_z[i] - 1) +
                      std_z * (pdf_u * (mu_z[i] - 1) - pdf_l * (mu_z[i] + 1)) -
                      powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] +
                      powf(mu_z[i], 2) - var_z[i] + 2) /
                         4.0f);
        mu_a[i] = mu_a[i] / 2.0f + 0.5f;
        jcb[i] = (cdf_u + cdf_l - 1) / 2.0f;
    }
}
void mixture_sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int n,
                                 unsigned int num_threads,
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

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            mixture_sigmoid_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a,
                                     jcb, var_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void mixture_tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a)
/*
 */
{
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    for (int i = start_chunk; i < end_chunk; i++) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[i], 0.5);
        alpha_l = (1.0f + mu_z[i]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[i]) / std_z;  // Upper truncation
        cdf_l = normcdf_cpu(alpha_l);
        cdf_u = normcdf_cpu(alpha_u);
        pdf_l = normpdf_cpu(alpha_l, 0.0f, 1.0f);
        pdf_u = normpdf_cpu(alpha_u, 0.0f, 1.0f);

        // Moments calculations (L. Alric, 2024)
        mu_a[i] = (mu_z[i] + 1) * cdf_l + (mu_z[i] - 1) * cdf_u +
                  std_z * (pdf_l - pdf_u) - mu_z[i];
        var_a[i] = std::max(
            0.000001f,
            cdf_l * (var_z[i] - powf(mu_z[i], 2) - 2 * mu_z[i] - 1) +
                cdf_u * (var_z[i] - powf(mu_z[i], 2) + 2 * mu_z[i] - 1) +
                std_z * (pdf_u * (mu_z[i] - 1) - pdf_l * (mu_z[i] + 1)) -
                powf(mu_a[i], 2) + 2 * mu_a[i] * mu_z[i] + powf(mu_z[i], 2) -
                var_z[i] + 2);
        jcb[i] = cdf_u + cdf_l - 1;
    }
}

void mixture_tanh_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
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

        threads.emplace_back([=, &mu_z, &var_z, &mu_a, &jcb, &var_a] {
            mixture_tanh_mean_var(mu_z, var_z, start_chunk, end_chunk, mu_a,
                                  jcb, var_a);
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

void even_exp_mean_var(std::vector<float> const &mu_z,
                       std::vector<float> const &var_z,
                       std::vector<float> &jcb_z, int start_chunk,
                       int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &var_a, std::vector<float> &jcb_a)

{
    for (int i = start_chunk; i < end_chunk; i++) {
        if (i % 2 == 0) {
            mu_a[i] = mu_z[i];
            var_a[i] = var_z[i];
            jcb_a[i] = jcb_z[i];
        } else {
            mu_a[i] = expf(mu_z[i] + 0.5 * var_z[i]);
            var_a[i] = expf(2 * mu_z[i] + var_z[i]) * (expf(var_z[i]) - 1);
            jcb_a[i] = var_z[i] * mu_a[i];
        }
    }
}

void even_exp_mean_var_mp(std::vector<float> const &mu_z,
                          std::vector<float> const &var_z,
                          std::vector<float> &jcb_z, int n,
                          unsigned int num_threads, std::vector<float> &mu_a,
                          std::vector<float> &var_a,
                          std::vector<float> &jcb_a) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk =
            i * n_per_thread + std::min(static_cast<int>(i), extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &jcb_z, &mu_a, &var_a, &jcb_a] {
            even_exp_mean_var(mu_z, var_z, jcb_z, start_chunk, end_chunk, mu_a,
                              var_a, jcb_a);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////

ReLU::ReLU() {};
ReLU::~ReLU() {};

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
std::unique_ptr<BaseLayer> ReLU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<ReLUCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
Sigmoid::Sigmoid() {};
Sigmoid::~Sigmoid() {};

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
std::unique_ptr<BaseLayer> Sigmoid::to_cuda(int device_idx) {
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
std::unique_ptr<BaseLayer> Tanh::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<TanhCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture ReLU
////////////////////////////////////////////////////////////////////////////////
MixtureReLU::MixtureReLU() {}
MixtureReLU::~MixtureReLU() {}

std::string MixtureReLU::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureReLU::get_layer_name() const
/*
 */

{
    return "MixtureReLU";
}

LayerType MixtureReLU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureReLU::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
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
std::unique_ptr<BaseLayer> MixtureReLU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<MixtureReLUCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoid::MixtureSigmoid() {};
MixtureSigmoid::~MixtureSigmoid() {};

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
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_sigmoid_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
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
std::unique_ptr<BaseLayer> MixtureSigmoid::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<MixtureSigmoidCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanh::MixtureTanh() {};
MixtureTanh::~MixtureTanh() {};

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
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    mixture_tanh_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
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
std::unique_ptr<BaseLayer> MixtureTanh::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<MixtureTanhCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
Softplus::Softplus() {};
Softplus::~Softplus() {};
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
std::unique_ptr<BaseLayer> Softplus::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<SoftplusCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
LeakyReLU::LeakyReLU() {};
LeakyReLU::~LeakyReLU() {};

std::string LeakyReLU::get_layer_info() const
/*
 */
{
    return "leakyReLU()";
}

std::string LeakyReLU::get_layer_name() const
/*
 */

{
    return "leakReLU";
}

LayerType LeakyReLU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyReLU::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states)
/*
 */
{
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
std::unique_ptr<BaseLayer> LeakyReLU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<LeakyReLUCuda>();
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
std::unique_ptr<BaseLayer> Softmax::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<SoftmaxCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
void to_log(std::vector<float> &mu_m, std::vector<float> &var_m,
            int hidden_size, int batch_size, std::vector<float> &mu_log,
            std::vector<float> &var_log)
/*
 */
{
    float tmp_mu, tmp_var;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            tmp_var = logf(1.0f + (var_m[i * hidden_size + j] /
                                   powf(mu_m[i * hidden_size + j], 2)));
            tmp_mu = logf(mu_m[i * hidden_size + j]) - 0.5 * tmp_var;

            mu_log[i * hidden_size + j] = tmp_mu;
            var_log[i * hidden_size + j] = tmp_var;
        }
    }
}

void compute_mean_var_sum(std::vector<float> &mu_m, std::vector<float> &var_m,
                          int hidden_size, int batch_size,
                          std::vector<float> &mu_sum,
                          std::vector<float> &var_sum)
/*
 */
{
    float sum_mu, sum_var;
    for (int i = 0; i < batch_size; i++) {
        sum_mu = 0.0f;
        sum_var = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum_mu += mu_m[i * hidden_size + j];
            sum_var += var_m[i * hidden_size + j];
        }
        mu_sum[i] = sum_mu;
        var_sum[i] = sum_var;
    }
}

void compute_cov_log_m_mt(const std::vector<float> &mu_m,
                          const std::vector<float> &var_m,
                          const std::vector<float> &mu_mt, int hidden_size,
                          int batch_size, std::vector<float> &cov_log_m_mt)
/*Compute covariance \cov(\lnM, \lnMt).
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            cov_log_m_mt[i * hidden_size + j] =
                logf(1.0f + var_m[i * hidden_size + j] * (1.0f / mu_mt[i]) *
                                (1.0f / mu_m[i * hidden_size + j]));
        }
    }
}

void compute_remax_mean_var(const std::vector<float> &mu_log_m,
                            const std::vector<float> &var_log_m,
                            const std::vector<float> &mu_log_mt,
                            const std::vector<float> &var_log_mt,
                            const std::vector<float> &cov_log_m_mt,
                            int hidden_size, int batch_size,
                            std::vector<float> &mu_a, std::vector<float> &var_a)
/*Compute mean and variance for remax.
 */
{
    float tmp_mu = 0.0f, tmp_var = 0.0f, sum_mu = 0.0f, sum_var = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        sum_mu = 0.0f;
        sum_var = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            tmp_mu = mu_log_m[i * hidden_size + j] - mu_log_mt[i];
            tmp_var = var_log_m[i * hidden_size + j] + var_log_mt[i] -
                      2 * cov_log_m_mt[i * hidden_size + j];
            sum_mu += tmp_mu;
            mu_a[i * hidden_size + j] = expf(tmp_mu + 0.5 * tmp_var);
            var_a[i * hidden_size + j] = expf(tmp_var) - 1.0f;
        }

        for (int j = 0; j < hidden_size; j++) {
            float tmp_mu_norm = mu_a[i * hidden_size + j] / sum_mu;
            var_a[i * hidden_size + j] *= tmp_mu_norm * tmp_mu_norm;
        }
    }
}

void compute_cov_a_z(
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &var_z, const std::vector<float> &mu_m,
    const std::vector<float> &var_m, const std::vector<float> &var_log_m,
    const std::vector<float> &cov_log_m_mt, const std::vector<float> &cdfn,
    int hidden_size, int batch_size, std::vector<float> &cov_a_z)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float cov_log_a_log_m = var_log_m[i * hidden_size + j] -
                                    cov_log_m_mt[i * hidden_size + j];
            float cov_a_m = (expf(cov_log_a_log_m) - 1.0f) *
                            mu_a[i * hidden_size + j] *
                            mu_m[i * hidden_size + j];

            cov_a_z[i * hidden_size + j] =
                std::min(powf(var_a[i * hidden_size + j], 0.5f) *
                             powf(var_z[i * hidden_size + j], 0.5f),
                         cov_a_m / cdfn[i * hidden_size + j]);
            cov_a_z[i * hidden_size + j] /= var_z[i * hidden_size + j];
        }
    }
}

void compute_cov_a_z_v2(
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &var_z, const std::vector<float> &mu_m,
    const std::vector<float> &var_m, const std::vector<float> &var_log_m,
    const std::vector<float> &cov_log_m_mt, const std::vector<float> &cdfn,
    int hidden_size, int batch_size, std::vector<float> &cov_a_z)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float cov_log_a_log_m = var_log_m[i * hidden_size + j] -
                                    cov_log_m_mt[i * hidden_size + j];
            float cov_a_m = (expf(cov_log_a_log_m) - 1.0f) *
                            mu_a[i * hidden_size + j] *
                            mu_m[i * hidden_size + j];

            cov_a_z[i * hidden_size + j] = std::min(
                powf(var_a[i * hidden_size + j], 0.5f) *
                    powf(var_z[i * hidden_size + j], 0.5f),
                cov_a_m * cdfn[i * hidden_size + j] *
                    var_z[i * hidden_size + j] * cdfn[i * hidden_size + j] /
                    var_m[i * hidden_size + j]);
            cov_a_z[i * hidden_size + j] /= var_z[i * hidden_size + j];
        }
    }
}

Remax::Remax() {}
Remax::~Remax() {}

std::string Remax::get_layer_info() const
/*
 */
{
    return "Remax()";
}

std::string Remax::get_layer_name() const
/*
 */

{
    return "Remax";
}

LayerType Remax::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Remax::forward(BaseHiddenStates &input_states,
                    BaseHiddenStates &output_states,
                    BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_states.block_size;
    int hidden_size = input_states.actual_size;

    if (this->batch_size_ != batch_size) {
        this->batch_size_ = batch_size;
        this->mu_m.resize(batch_size * hidden_size, 0.0f);
        this->var_m.resize(batch_size * hidden_size, 0.0f);
        this->jcb_m.resize(batch_size * hidden_size, 0.0f);
        this->mu_log_m.resize(batch_size * hidden_size, 0.0f);
        this->var_log_m.resize(batch_size * hidden_size, 0.0f);
        this->mu_mt.resize(batch_size, 0.0f);
        this->var_mt.resize(batch_size, 0.0f);
        this->mu_log_mt.resize(batch_size, 0.0f);
        this->var_log_mt.resize(batch_size, 0.0f);
        this->cov_log_m_mt.resize(batch_size * hidden_size, 0.0f);
    }
    // Compute mean and variance of M. NOTE: jcb_m = cdfn
    int start_chunk = 0;
    int end_chunk = batch_size * hidden_size;
    std::vector<float> var_a_tmp(batch_size * hidden_size, 0.0f);
    for (int i = 0; i < batch_size * hidden_size; i++) {
        if (input_states.jcb[i] != 0.0f) {
            var_a_tmp[i] = input_states.var_a[i] + 0.0f;
        } else {
            var_a_tmp[i] = input_states.var_a[i];
        }
    }
    mixture_relu_mean_var_v2(input_states.mu_a, var_a_tmp, start_chunk,
                             end_chunk, this->threshold, this->mu_m,
                             this->jcb_m, this->var_m);

    // Compute mean and variance of Mt
    compute_mean_var_sum(this->mu_m, this->var_m, hidden_size, batch_size,
                         this->mu_mt, this->var_mt);

    // Compute mean and variance of log(M)
    to_log(this->mu_m, this->var_m, hidden_size, batch_size, this->mu_log_m,
           this->var_log_m);

    // Compute mean and variance of log(Mt)
    to_log(this->mu_mt, this->var_mt, 1, batch_size, this->mu_log_mt,
           this->var_log_mt);

    // Compute covariance of log(M) and log(Mt)
    compute_cov_log_m_mt(this->mu_m, this->var_m, this->mu_mt, hidden_size,
                         batch_size, this->cov_log_m_mt);

    // Compute mean and variance of A
    compute_remax_mean_var(this->mu_log_m, this->var_log_m, this->mu_log_mt,
                           this->var_log_mt, this->cov_log_m_mt, hidden_size,
                           batch_size, output_states.mu_a, output_states.var_a);

    // Compute covariance of A and Z i.e., Jacobian.
    compute_cov_a_z(output_states.mu_a, output_states.var_a, input_states.var_a,
                    this->mu_m, this->var_m, this->var_log_m,
                    this->cov_log_m_mt, this->jcb_m, hidden_size, batch_size,
                    output_states.jcb);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

////////////////////////////////////////////////////////////////////////////////
/// EvenExp
////////////////////////////////////////////////////////////////////////////////
EvenExp::EvenExp() {}
EvenExp::~EvenExp() {}

std::string EvenExp::get_layer_info() const
/*
 */
{
    return "EvenExp()";
}

std::string EvenExp::get_layer_name() const
/*
 */

{
    return "EvenExp";
}

LayerType EvenExp::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void EvenExp::forward(BaseHiddenStates &input_states,
                      BaseHiddenStates &output_states,
                      BaseTempStates &temp_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    if (this->num_threads > 1) {
        even_exp_mean_var_mp(input_states.mu_a, input_states.var_a,
                             input_states.jcb, end_chunk, this->num_threads,
                             output_states.mu_a, output_states.var_a,
                             output_states.jcb);
    } else {
        even_exp_mean_var(input_states.mu_a, input_states.var_a,
                          input_states.jcb, start_chunk, end_chunk,
                          output_states.mu_a, output_states.var_a,
                          output_states.jcb);
    }

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> EvenExp::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<EvenExpCuda>();
}
#endif

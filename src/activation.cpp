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
    constexpr float SQRT_2PI = 2.5066282746310002f;
    for (int i = start_chunk; i < end_chunk; i++) {
        float tmp_mu_z = mu_z[i];
        std_z = powf(var_z[i], 0.5);
        alpha = tmp_mu_z / std_z;
        pdf_alpha = (1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha);
        cdf_alpha = normcdf_cpu(alpha);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = tmp_mu_z * cdf_alpha + std_z * pdf_alpha;
        mu_a[i] = fmaxf(0.000001f, tmp_mu_a);
        var_a[i] =
            fmaxf(0.000001f, -tmp_mu_a * tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z -
                                 tmp_mu_z * std_z * pdf_alpha +
                                 (var_z[i] - tmp_mu_z * tmp_mu_z) * cdf_alpha);

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

void celu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                   int start_chunk, int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a)
/*
 */
{
    constexpr float SQRT_2PI = 2.5066282746310002f;
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    constexpr float ALPHA = 0.2f;  // slope of negative part

    for (int i = start_chunk; i < end_chunk; i++) {
        float mz = mu_z[i];
        float varz = var_z[i];

        // 1) std-dev and standardize
        float sz = sqrtf(varz);
        float z = mz / sz;

        // 2) shift amount
        float a = sz / ALPHA;
        float z_a = z + a;
        float z_2a = z + 2.0f * a;

        // 3) φ(z) and tail probs P[Z < −x] = 0.5*erfc(x/√2), clamped
        float phi_z = expf(-0.5f * z * z) / SQRT_2PI;

        float tail_z = 0.5f * erfcf(z * INV_SQRT2);
        float tail_za = 0.5f * erfcf(z_a * INV_SQRT2);
        float tail_z2a = 0.5f * erfcf(z_2a * INV_SQRT2);

        // 4) analytic ratios instead of φ(z)/φ(z+k·a)
        float exp_a = expf(a * z + 0.5f * (a * a));
        float exp_2a = expf(2.0f * a * z + 0.5f * (2.0f * a) * (2.0f * a));

        // 5) Mean E[CELU(z)]
        float mean_d =
            mz + sz * phi_z - (ALPHA + mz) * tail_z + ALPHA * tail_za * exp_a;

        // 6) Second moment E[CELU(z)²]
        float E2 = mz * mz + mz * sz * phi_z + varz -
                   2.0f * ALPHA * ALPHA * tail_za * exp_a +
                   ALPHA * ALPHA * tail_z2a * exp_2a +
                   (ALPHA * ALPHA - mz * mz - varz) * tail_z;

        // 7) Variance = E2 – mean², with floor
        float var_d = E2 - mean_d * mean_d;

        // 8) Covariance Cov[z, CELU(z)] = varz * (P[Z> -z] + tail_za·exp_a)
        float cov_d = varz * ((1.0f - tail_z) + tail_za * exp_a);

        // 9) Jacobian in [0,1]
        float jcb_d = cov_d / varz;
        jcb[i] = jcb_d;
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

void exp_mean_var(std::vector<float> const &mu_z,
                  std::vector<float> const &var_z, std::vector<float> &jcb_z,
                  int start_chunk, int end_chunk, std::vector<float> &mu_a,
                  std::vector<float> &var_a, std::vector<float> &jcb_a,
                  float scale, float shift)

{
    for (int i = start_chunk; i < end_chunk; i++) {
        float new_mu = mu_z[i] * scale + shift;
        float new_var = var_z[i] * scale * scale;

        mu_a[i] = expf(new_mu + 0.5 * new_var);
        var_a[i] = expf(2 * new_mu + new_var) * (expf(new_var) - 1);
        jcb_a[i] = mu_a[i] * scale;
    }
}

void exp_mean_var_mp(std::vector<float> const &mu_z,
                     std::vector<float> const &var_z, std::vector<float> &jcb_z,
                     int n, unsigned int num_threads, std::vector<float> &mu_a,
                     std::vector<float> &var_a, std::vector<float> &jcb_a,
                     float scale, float shift) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk =
            i * n_per_thread + std::min(static_cast<int>(i), extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_z, &var_z, &jcb_z, &mu_a, &var_a, &jcb_a] {
            exp_mean_var(mu_z, var_z, jcb_z, start_chunk, end_chunk, mu_a,
                         var_a, jcb_a, scale, shift);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void agvi_backward_chunk(int start_chunk, int end_chunk,
                         BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         const BaseHiddenStates &stored_output_states,
                         const BaseHiddenStates &stored_inner_output_states,
                         const BaseHiddenStates &stored_input_states,
                         bool overfit_mu) {
    /*
     * Processes a chunk of the backward pass for the AGVI layer.
     * This function is designed to be called by a single thread.
     *
     * @param start_chunk The starting index for the chunk.
     * @param end_chunk The ending index for the chunk.
     * @param input_delta_states Deltas from the subsequent layer.
     * @param output_delta_states Deltas to be passed to the preceding layer.
     * @param stored_output_states The output states from the forward pass.
     * @param stored_inner_output_states The inner activation's output from the
     * forward pass.
     * @param stored_input_states The input states from the forward pass.
     * @param overfit_mu Flag to control the Jacobian for the mean delta.
     */
    for (int i = start_chunk; i < end_chunk; i++) {
        // 1. Retrieve stored values from the forward pass.
        float mu_a = stored_output_states.mu_a[i];
        float var_a = stored_output_states.var_a[i];

        // V2_bar_tilde
        float mu_v2_bar_tilde = stored_inner_output_states.mu_a[i];
        float var_v2_bar_tilde = stored_inner_output_states.var_a[i];
        float jcb_v2_bar_tilde = stored_inner_output_states.jcb[i];

        // Deltas from the subsequent layer
        float incoming_delta_mu = input_delta_states.delta_mu[i];
        float incoming_delta_var = input_delta_states.delta_var[i];

        // Prior variance for Z (from the even input stream)
        float var_z = stored_input_states.var_a[i * 2];

        // 2. Perform the backward message passing calculations.

        // Compute the prior predictive PDF for v2
        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 =
            3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde;

        // V ~ N(0, mu_v2)
        float mu_v = 0.0f;
        float var_v = mu_v2;

        // Compute the posterior mean and variance for A (output)
        float mu_a_pos = mu_a + incoming_delta_mu * var_a;
        float var_a_pos = var_a + incoming_delta_var * var_a * var_a;

        // Compute the posterior mean and variance for V
        float Jv = var_v / var_a;
        float mu_v_pos = mu_v + Jv * (mu_a_pos - mu_a);
        float var_v_pos = var_v + Jv * Jv * (var_a_pos - var_a);

        // Compute the posterior mean and variance for V2
        float mu_v2_pos = mu_v_pos * mu_v_pos + var_v_pos;
        float var_v2_pos = 2.0f * var_v_pos * var_v_pos +
                           4.0f * var_v_pos * mu_v_pos * mu_v_pos;

        // Compute the posterior mean and variance for V2_bar_tilde
        float Jv2_bar_tilde = var_v2_bar_tilde / var_v2;
        float mu_v2_bar_tilde_pos =
            mu_v2_bar_tilde + Jv2_bar_tilde * (mu_v2_pos - mu_v2);
        float var_v2_bar_tilde_pos =
            var_v2_bar_tilde +
            Jv2_bar_tilde * Jv2_bar_tilde * (var_v2_pos - var_v2);

        // 3. Define the output indices for the even (Z) and odd (V2_bar) stream
        int even_idx = 2 * i;
        int odd_idx = 2 * i + 1;

        // 4. Compute and write deltas for V2_bar (the odd input stream).
        float Jv2_bar = jcb_v2_bar_tilde / var_v2_bar_tilde;
        output_delta_states.delta_mu[odd_idx] =
            Jv2_bar * (mu_v2_bar_tilde_pos - mu_v2_bar_tilde);
        output_delta_states.delta_var[odd_idx] =
            Jv2_bar * Jv2_bar * (var_v2_bar_tilde_pos - var_v2_bar_tilde);

        // --- MODIFIED ---
        // 5. Compute and write deltas for Z (the even input stream).
        float Jz = 1.0f / var_a;
        float Jz_mu;
        if (overfit_mu) {
            // Use different J for the mean to allow overfitting on it
            Jz_mu = 1.0f / var_z;
        } else {
            // Use the same J for the mean and variance
            Jz_mu = Jz;
        }
        output_delta_states.delta_mu[even_idx] = Jz_mu * (mu_a_pos - mu_a);
        output_delta_states.delta_var[even_idx] = Jz * Jz * (var_a_pos - var_a);
    }
}

void agvi_backward_mp(int n, unsigned int num_threads,
                      BaseDeltaStates &input_delta_states,
                      BaseDeltaStates &output_delta_states,
                      const BaseHiddenStates &stored_output_states,
                      const BaseHiddenStates &stored_inner_output_states,
                      const BaseHiddenStates &stored_input_states,
                      bool overfit_mu) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk = i * n_per_thread + std::min(i, (unsigned int)extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &input_delta_states, &output_delta_states,
                              &stored_output_states,
                              &stored_inner_output_states,
                              &stored_input_states] {
            agvi_backward_chunk(start_chunk, end_chunk, input_delta_states,
                                output_delta_states, stored_output_states,
                                stored_inner_output_states, stored_input_states,
                                overfit_mu);
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
/// CELU
////////////////////////////////////////////////////////////////////////////////
CELU::CELU() {};
CELU::~CELU() {};

std::string CELU::get_layer_info() const
/*
 */
{
    return "CELU()";
}

std::string CELU::get_layer_name() const
/*
 */

{
    return "CELU";
}

LayerType CELU::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void CELU::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states, BaseTempStates &temp_states)
/*
 */
{
    // TODO: replace this function by the multiprocessing one
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    celu_mean_var(input_states.mu_a, input_states.var_a, start_chunk, end_chunk,
                  output_states.mu_a, output_states.jcb, output_states.var_a);

    // Save activation mean and jacobian to the class member for backward pass
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> CELU::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<CELUCuda>();
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

            mu_a[i * hidden_size + j] =
                std::max(0.000001f, expf(tmp_mu + 0.5 * tmp_var));
            sum_mu += mu_a[i * hidden_size + j];
            var_a[i * hidden_size + j] = expf(tmp_var) - 1.0f;
        }

        for (int j = 0; j < hidden_size; j++) {
            float tmp_mu_norm = mu_a[i * hidden_size + j] / sum_mu;
            mu_a[i * hidden_size + j] = tmp_mu_norm;
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

            // Original formula
            cov_a_z[i * hidden_size + j] =
                fminf(powf(var_a[i * hidden_size + j], 0.5f) *
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
                cov_a_m * var_z[i * hidden_size + j] *
                    cdfn[i * hidden_size + j] / var_m[i * hidden_size + j]);
            cov_a_z[i * hidden_size + j] /= var_z[i * hidden_size + j];
        }
    }
}

void compute_cov_a_z_v3(
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
                         cov_a_m * var_z[i * hidden_size + j] /
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
    mixture_relu_mean_var(input_states.mu_a, input_states.var_a, start_chunk,
                          end_chunk, this->mu_m, this->jcb_m, this->var_m);

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

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Remax::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<RemaxCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// ClosedFormSoftmax
////////////////////////////////////////////////////////////////////////////////
void compute_mean_var_exp_sum(const std::vector<float> &mu_z,
                              const std::vector<float> &var_z, int hidden_size,
                              int batch_size, std::vector<float> &mu_e_sum,
                              std::vector<float> &var_e_sum)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum_mu += expf(mu_z[i * hidden_size + j] +
                           0.5 * var_z[i * hidden_size + j]);
            sum_var += expf(2 * mu_z[i * hidden_size + j] +
                            var_z[i * hidden_size + j]) *
                       (expf(var_z[i * hidden_size + j]) - 1.0f);
        }
        mu_e_sum[i] = sum_mu;
        var_e_sum[i] = sum_var;
    }
}

void compute_mean_var_log_a(
    const std::vector<float> &mu_z, const std::vector<float> &var_z,
    const std::vector<float> &mu_log_e_sum,
    const std::vector<float> &var_log_e_sum, const std::vector<float> &mu_e_sum,
    const std::vector<float> &var_e_sum, int hidden_size, int batch_size,
    std::vector<float> &mu_log_a, std::vector<float> &var_log_a,
    std::vector<float> &cov_log_a_z)
/*
 */
{
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float cov_e_e_sum = expf(2 * mu_z[i * hidden_size + j] +
                                     var_z[i * hidden_size + j]) *
                                (expf(var_z[i * hidden_size + j]) - 1.0f);
            float mu_e = expf(mu_z[i * hidden_size + j] +
                              0.5 * var_z[i * hidden_size + j]);

            float tmp_inverse_mu = 1.0f / (mu_e_sum[i] * mu_e);
            float cov_z_log_e_sum = logf(1.0f + cov_e_e_sum * tmp_inverse_mu);
            mu_log_a[i * hidden_size + j] =
                mu_z[i * hidden_size + j] - mu_log_e_sum[i];
            var_log_a[i * hidden_size + j] = var_z[i * hidden_size + j] +
                                             var_log_e_sum[i] -
                                             2.0f * cov_z_log_e_sum;
            cov_log_a_z[i * hidden_size + j] =
                var_z[i * hidden_size + j] - cov_z_log_e_sum;
        }
    }
}

void compute_cfsoftmax_mean_var(
    const std::vector<float> &mu_log_a, const std::vector<float> &var_log_a,
    const std::vector<float> &cov_log_a_z, const std::vector<float> &var_z,
    int hidden_size, int batch_size, std::vector<float> &mu_a,
    std::vector<float> &var_a, std::vector<float> &jcb_a) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float tmp_mu = expf(mu_log_a[i * hidden_size + j] +
                                0.5 * var_log_a[i * hidden_size + j]);
            mu_a[i * hidden_size + j] = tmp_mu;
            var_a[i * hidden_size + j] =
                (expf(var_log_a[i * hidden_size + j]) - 1.0f) * tmp_mu * tmp_mu;

            // TODO: Need to used normalized mean?
            jcb_a[i * hidden_size + j] = tmp_mu *
                                         cov_log_a_z[i * hidden_size + j] /
                                         var_z[i * hidden_size + j];
        }
    }
}

ClosedFormSoftmax::ClosedFormSoftmax() {}
ClosedFormSoftmax::~ClosedFormSoftmax() {}

std::string ClosedFormSoftmax::get_layer_info() const
/*
 */
{
    return "ClosedFormSoftmax()";
}

std::string ClosedFormSoftmax::get_layer_name() const
/*
 */
{
    return "ClosedFormSoftmax";
}

LayerType ClosedFormSoftmax::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ClosedFormSoftmax::forward(BaseHiddenStates &input_states,
                                BaseHiddenStates &output_states,
                                BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_states.block_size;
    int hidden_size = input_states.actual_size;

    if (this->batch_size_ != batch_size) {
        this->batch_size_ = batch_size;
        this->mu_e_sum.resize(batch_size, 0.0f);
        this->var_e_sum.resize(batch_size, 0.0f);
        this->cov_z_log_e_sum.resize(batch_size * hidden_size, 0.0f);
        this->mu_log_e_sum.resize(batch_size, 0.0f);
        this->var_log_e_sum.resize(batch_size, 0.0f);
        this->cov_log_a_z.resize(batch_size * hidden_size, 0.0f);
        this->mu_log_a.resize(batch_size * hidden_size, 0.0f);
        this->var_log_a.resize(batch_size * hidden_size, 0.0f);
    }

    compute_mean_var_exp_sum(input_states.mu_a, input_states.var_a, hidden_size,
                             batch_size, this->mu_e_sum, this->var_e_sum);
    to_log(this->mu_e_sum, this->var_e_sum, 1, batch_size, this->mu_log_e_sum,
           this->var_log_e_sum);

    compute_mean_var_log_a(
        input_states.mu_a, input_states.var_a, this->mu_log_e_sum,
        this->var_log_e_sum, this->mu_e_sum, this->var_e_sum, hidden_size,
        batch_size, this->mu_log_a, this->var_log_a, this->cov_log_a_z);

    compute_cfsoftmax_mean_var(this->mu_log_a, this->var_log_a,
                               this->cov_log_a_z, input_states.var_a,
                               hidden_size, batch_size, output_states.mu_a,
                               output_states.var_a, output_states.jcb);

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> ClosedFormSoftmax::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<ClosedFormSoftmaxCuda>();
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// SplitActivation (formerly EvenExp)
////////////////////////////////////////////////////////////////////////////////

SplitActivation::SplitActivation(std::shared_ptr<BaseLayer> odd_layer,
                                 std::shared_ptr<BaseLayer> even_layer)
    : odd_layer(odd_layer), even_layer(even_layer) {
    if (!odd_layer) {
        // It's crucial that the odd_layer exists.
        // We could throw an exception here for robustness.
        std::cerr << "Error: SplitActivation must be initialized with a valid "
                     "layer for odd-indexed positions."
                  << std::endl;
    }
}

SplitActivation::~SplitActivation() {}

std::string SplitActivation::get_layer_info() const {
    std::string even_layer_name = "Identity";
    if (even_layer) {
        even_layer_name = even_layer->get_layer_name();
    }
    return "SplitActivation(odd=" + odd_layer->get_layer_name() +
           ", even=" + even_layer_name + ")";
}

std::string SplitActivation::get_layer_name() const {
    return "SplitActivation";
}

LayerType SplitActivation::get_layer_type() const {
    return LayerType::Activation;
}

void SplitActivation::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states) {
    // 1. Calculate the total number of elements to process.
    int total_elements = input_states.actual_size * input_states.block_size;
    int odd_count = total_elements / 2;
    int even_count = total_elements - odd_count;

    // 2. Prepare temporary hidden states for the odd and even data streams.
    BaseHiddenStates odd_input_states;
    odd_input_states.mu_a.reserve(odd_count);
    odd_input_states.var_a.reserve(odd_count);
    odd_input_states.jcb.reserve(odd_count);

    BaseHiddenStates even_input_states;
    even_input_states.mu_a.reserve(even_count);
    even_input_states.var_a.reserve(even_count);
    even_input_states.jcb.reserve(even_count);

    // 3. Split the input data into odd and even streams.
    for (int i = 0; i < total_elements; ++i) {
        if (i % 2 == 0) {  // Even indices
            even_input_states.mu_a.push_back(input_states.mu_a[i]);
            even_input_states.var_a.push_back(input_states.var_a[i]);
            even_input_states.jcb.push_back(input_states.jcb[i]);
        } else {  // Odd indices
            odd_input_states.mu_a.push_back(input_states.mu_a[i]);
            odd_input_states.var_a.push_back(input_states.var_a[i]);
            odd_input_states.jcb.push_back(input_states.jcb[i]);
        }
    }

    // Set metadata for the temporary states.
    odd_input_states.block_size = input_states.block_size;
    odd_input_states.actual_size = odd_count / input_states.block_size;
    even_input_states.block_size = input_states.block_size;
    even_input_states.actual_size = even_count / input_states.block_size;

    // 4. Apply the activation layers to their respective streams.

    // Process the odd stream using the mandatory odd_layer.
    BaseHiddenStates odd_output_states;
    odd_output_states.mu_a.resize(odd_count);
    odd_output_states.var_a.resize(odd_count);
    odd_output_states.jcb.resize(odd_count);
    odd_layer->forward(odd_input_states, odd_output_states, temp_states);

    // Process the even stream.
    BaseHiddenStates even_output_states;
    if (even_layer) {
        // If an even_layer is provided, use it.
        even_output_states.mu_a.resize(even_count);
        even_output_states.var_a.resize(even_count);
        even_output_states.jcb.resize(even_count);
        even_layer->forward(even_input_states, even_output_states, temp_states);
    } else {
        // If no even_layer is provided, apply an identity transformation
        // by moving the input states to the output states.
        even_output_states = std::move(even_input_states);
    }

    // 5. Merge the processed streams back into the final output_states.
    int odd_idx = 0;
    int even_idx = 0;
    for (int i = 0; i < total_elements; ++i) {
        if (i % 2 == 0) {  // Even indices
            output_states.mu_a[i] = even_output_states.mu_a[even_idx];
            output_states.var_a[i] = even_output_states.var_a[even_idx];
            output_states.jcb[i] = even_output_states.jcb[even_idx];
            even_idx++;
        } else {  // Odd indices
            output_states.mu_a[i] = odd_output_states.mu_a[odd_idx];
            output_states.var_a[i] = odd_output_states.var_a[odd_idx];
            output_states.jcb[i] = odd_output_states.jcb[odd_idx];
            odd_idx++;
        }
    }

    // 6. Update layer and output_states metadata to match the input.
    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

void SplitActivation::save(std::ofstream &file) {
    // Save the state of the inner layers.
    if (odd_layer) {
        odd_layer->save(file);
    }
    if (even_layer) {
        even_layer->save(file);
    }
}

void SplitActivation::load(std::ifstream &file) {
    // Load the state of the inner layers.
    if (odd_layer) {
        odd_layer->load(file);
    }
    if (even_layer) {
        even_layer->load(file);
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> SplitActivation::to_cuda(int device_idx) {
    this->device = "cuda";
    // Convert inner layers to their CUDA equivalents.
    auto cuda_odd_layer = odd_layer->to_cuda(device_idx);
    std::unique_ptr<BaseLayer> cuda_even_layer = nullptr;
    if (even_layer) {
        cuda_even_layer = even_layer->to_cuda(device_idx);
    }
    // Note: This assumes a SplitActivationCuda class exists.
    return std::make_unique<SplitActivationCuda>(std::move(cuda_odd_layer),
                                                 std::move(cuda_even_layer));
    // return nullptr; // Placeholder until SplitActivationCuda is implemented
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// Exp
////////////////////////////////////////////////////////////////////////////////
Exp::Exp(float scale, float shift) : scale(scale), shift(shift) {}
Exp::~Exp() {}

std::string Exp::get_layer_info() const
/*
 */
{
    return "Exp()";
}

std::string Exp::get_layer_name() const
/*
 */

{
    return "Exp";
}

LayerType Exp::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void Exp::forward(BaseHiddenStates &input_states,
                  BaseHiddenStates &output_states, BaseTempStates &temp_states)
/*
 */
{
    int start_chunk = 0;
    int end_chunk = input_states.actual_size * input_states.block_size;
    if (this->num_threads > 1) {
        exp_mean_var_mp(input_states.mu_a, input_states.var_a, input_states.jcb,
                        end_chunk, this->num_threads, output_states.mu_a,
                        output_states.var_a, output_states.jcb, scale, shift);
    } else {
        exp_mean_var(input_states.mu_a, input_states.var_a, input_states.jcb,
                     start_chunk, end_chunk, output_states.mu_a,
                     output_states.var_a, output_states.jcb, scale, shift);
    }

    this->input_size = input_states.actual_size;
    this->output_size = input_states.actual_size;

    // Update number of actual states.
    output_states.size = input_states.size;
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> Exp::to_cuda(int device_idx) {
    this->device = "cuda";
    return std::make_unique<ExpCuda>(scale, shift);
}
#endif

////////////////////////////////////////////////////////////////////////////////
/// AGVI (Approximate Gaussian Variance Inference)
////////////////////////////////////////////////////////////////////////////////
AGVI::AGVI(std::shared_ptr<BaseLayer> activation_layer, bool overfit_mu,
           bool agvi)
    : m_activation_layer(activation_layer),
      m_overfit_mu(overfit_mu),
      m_agvi(agvi) {
    if (!m_activation_layer ||
        m_activation_layer->get_layer_type() != LayerType::Activation) {
        std::cerr << "Error: AGVI layer must be initialized with a valid "
                     "activation layer."
                  << std::endl;
        m_activation_layer = std::make_shared<ReLU>();
    }
}

AGVI::~AGVI() {}

std::string AGVI::get_layer_info() const {
    return "AGVI(" + m_activation_layer->get_layer_name() + ")";
}

std::string AGVI::get_layer_name() const { return "AGVI"; }

LayerType AGVI::get_layer_type() const { return LayerType::AGVI; }

void AGVI::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states,
                   BaseTempStates &temp_states) {
    // The AGVI layer's logic is a wrapper for another activation function.
    // 1. Take odd-indexed positions from the input.
    // 2. Pass them through the inner activation function.
    // 3. Take the output mean of the inner activation and add it to the
    //    variance of the corresponding even-indexed position of the input.
    // 4. The output of the AGVI layer is only the even-indexed positions.

    int total_elements = input_states.actual_size * input_states.block_size;
    int odd_count = total_elements / 2;
    int even_count = total_elements / 2;

    // The output size is halved as we only propagate the even stream.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size / 2;
    output_states.size = input_states.size;

    // Prepare temporary vectors for odd positions
    std::vector<float> odd_mu, odd_var, odd_jcb;
    odd_mu.reserve(odd_count);
    odd_var.reserve(odd_count);
    odd_jcb.reserve(odd_count);

    // Copy odd-indexed elements to a temporary state
    for (int i = 0; i < total_elements; ++i) {
        if (i % 2 != 0) {  // Check for odd positions
            odd_mu.push_back(input_states.mu_a[i]);
            odd_var.push_back(input_states.var_a[i]);
            odd_jcb.push_back(input_states.jcb[i]);
        }
    }

    // Create temporary BaseHiddenStates for the inner activation layer
    BaseHiddenStates odd_input_states;
    odd_input_states.mu_a = std::move(odd_mu);
    odd_input_states.var_a = std::move(odd_var);
    odd_input_states.jcb = std::move(odd_jcb);
    odd_input_states.actual_size = odd_count / input_states.block_size;
    odd_input_states.block_size = input_states.block_size;

    // Create temporary BaseHiddenStates for the output of the inner layer
    BaseHiddenStates odd_output_states;
    odd_output_states.mu_a.resize(odd_count);
    odd_output_states.var_a.resize(odd_count);
    odd_output_states.jcb.resize(odd_count);

    // Call the forward pass of the inner activation layer on the odd positions
    m_activation_layer->forward(odd_input_states, odd_output_states,
                                temp_states);

    // Store the output of the inner layer for use in the backward pass.
    m_stored_inner_output_states = odd_output_states;

    int even_pos_idx = 0;
    for (int i = 0; i < total_elements; ++i) {
        if (i % 2 == 0) {  // Check for even positions
            output_states.mu_a[even_pos_idx] = input_states.mu_a[i];
            output_states.jcb[even_pos_idx] = input_states.jcb[i];
            // The output variance is the input variance
            // + the mean of the corresponding odd position
            if (m_agvi) {
                output_states.var_a[even_pos_idx] =
                    input_states.var_a[i] +
                    m_stored_inner_output_states.mu_a[even_pos_idx];
            } else {
                output_states.var_a[even_pos_idx] = input_states.var_a[i];
            }
            even_pos_idx++;
        }
    }

    // Store states for backward pass
    m_stored_output_states = output_states;
    m_stored_input_states = input_states;

    // Update layer input and output sizes
    this->input_size = input_states.actual_size;
    this->output_size = output_states.actual_size;
}

void AGVI::backward(BaseDeltaStates &input_delta_states,
                    BaseDeltaStates &output_delta_states,
                    BaseTempStates &temp_states, bool state_update) {
    int total_output_size = this->output_size * input_delta_states.block_size;

    // Decide whether to use multiprocessing
    if (this->num_threads > 1) {
        agvi_backward_mp(total_output_size, this->num_threads,
                         input_delta_states, output_delta_states,
                         m_stored_output_states, m_stored_inner_output_states,
                         m_stored_input_states, m_overfit_mu);
    } else {
        agvi_backward_chunk(0, total_output_size, input_delta_states,
                            output_delta_states, m_stored_output_states,
                            m_stored_inner_output_states, m_stored_input_states,
                            m_overfit_mu);
    }

    // Remove the stored states after the backward pass is complete to free
    // memory.
    m_stored_inner_output_states.mu_a.clear();
    m_stored_inner_output_states.var_a.clear();
    m_stored_inner_output_states.jcb.clear();
    m_stored_output_states.mu_a.clear();
    m_stored_output_states.var_a.clear();
    m_stored_output_states.jcb.clear();
    m_stored_input_states.mu_a.clear();
    m_stored_input_states.var_a.clear();
    m_stored_input_states.jcb.clear();

    // The output deltas now have the same full size as the original input to
    // this layer.
    output_delta_states.actual_size = this->input_size;
    output_delta_states.block_size = input_delta_states.block_size;
    output_delta_states.size = input_delta_states.size;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> AGVI::to_cuda(int device_idx) {
    this->device = "cuda";
    auto cuda_inner_layer = m_activation_layer->to_cuda(device_idx);
    return std::make_unique<AGVICuda>(std::move(cuda_inner_layer), m_overfit_mu,
                                      m_agvi);
}
#endif

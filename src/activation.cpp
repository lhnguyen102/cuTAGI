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
        if (mu_z[col] <= 0.0f) {
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
        mu_a[i] = fmax((mu_z[i] + 1) * cdf_l + (mu_z[i] - 1) * cdf_u +
                           std_z * (pdf_l - pdf_u) - mu_z[i],
                       0.000001f);
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

// // Double sided mRelu
// void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
//                               std::vector<float> &var_z, int start_chunk,
//                               int end_chunk, std::vector<float> &mu_a,
//                               std::vector<float> &jcb,
//                               std::vector<float> &var_a) {

//     // Mathematical constants for single-precision float
//     constexpr float SQRT_2PI = 2.5066282746310002f;
//     constexpr float INV_SQRT_2 = 0.7071067811865475f;
//     constexpr float TINY_FLOAT = 0.00000001f; // Small constant to prevent
//     division by zero
//     // Sigmoid bounds
//     constexpr float lower_bound = 0.0f;  // Corresponds to sigmoid(0)
//     constexpr float upper_bound = 1.0f;  // Corresponds to sigmoid(∞)

//     // Loop over the specified chunk of the vectors
//     for (int i = start_chunk; i < end_chunk; ++i) {
//         // Load input mean and variance for the current element
//         float mu_y_i = mu_z[i];  // Ensure non-negative mean
//         float var_y_i = std::fmax(var_z[i], 0.00000001f); // Ensure
//         non-negative variance float sigma_y_i = std::pow(var_y_i, 0.5f);

//         // Standardize the bounds
//         float alpha = (lower_bound - mu_y_i) / sigma_y_i; // Corresponds to
//         z_lower float beta = (upper_bound - mu_y_i) / sigma_y_i;   //
//         Corresponds to z_upper

//         // Compute PDF and CDF values
//         // float cdf_alpha = normcdf_cpu(alpha);
//         float cdf_alpha = 0.5f * (1.0f + std::erf(alpha * INV_SQRT_2));
//         // float cdf_beta = normcdf_cpu(beta);
//         float cdf_beta = 0.5f * (1.0f + std::erf(beta * INV_SQRT_2));
//         // float pdf_alpha = normpdf_cpu(alpha);
//         float pdf_alpha = (1.0f / SQRT_2PI) * std::exp(-0.5f * alpha *
//         alpha);
//         // float pdf_beta = normpdf_cpu(beta);
//         float pdf_beta = (1.0f / SQRT_2PI) * std::exp(-0.5f * beta * beta);

//         float cdf_diff = cdf_beta - cdf_alpha;

//         // ---- 1. Calculate Mean E[z] ----
//         float tmp_mu_z = std::fmax(lower_bound * cdf_alpha +
//                          mu_y_i * cdf_diff +
//                          sigma_y_i * (pdf_alpha - pdf_beta) +
//                          upper_bound * (1.0f - cdf_beta), 0.000001f);

//         // ---- 2. Calculate Second Moment E[z^2] ----
//         float term1_Ez2 = lower_bound * lower_bound * cdf_alpha;
//         float term2_Ez2 = (mu_y_i * mu_y_i + var_y_i) * cdf_diff;
//         float term3_Ez2 = -sigma_y_i * ((mu_y_i + upper_bound) * pdf_beta -
//         (mu_y_i + lower_bound) * pdf_alpha); float term4_Ez2 = upper_bound *
//         upper_bound * (1.0f - cdf_beta); float Ez2 = term1_Ez2 + term2_Ez2 +
//         term3_Ez2 + term4_Ez2;

//         // ---- 3. Calculate Variance Var[z] ----
//         float tmp_var_z = Ez2 - tmp_mu_z * tmp_mu_z;

//         // ---- 4. Calculate Cross Moment E[y*z] to find Cov(y, z) ----
//         float term1_Eyz = lower_bound * (mu_y_i * cdf_alpha - sigma_y_i *
//         pdf_alpha);
//         // term2_Eyz is identical to term2_Ez2
//         // term3_Eyz is identical to term3_Ez2
//         float term4_Eyz = upper_bound * (mu_y_i * (1.0f - cdf_beta) +
//         sigma_y_i * pdf_beta); float Eyz = term1_Eyz + term2_Ez2 + term3_Ez2
//         + term4_Eyz;

//         float cov_yz = Eyz - tmp_mu_z * mu_y_i;

//         // Store the final results in the output vectors
//         mu_a[i] = tmp_mu_z;
//         var_a[i] = std::fmax(tmp_var_z, 0.000001f);
//         // jcb[i] = cov_yz / var_y_i;
//         jcb[i] = cdf_diff;
//     }
// }

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
    // --- Constants ---
    constexpr float SQRT_2PI = 2.5066282746310002f;
    constexpr float INV_SQRT2 = 0.7071067811865475f;
    constexpr float ALPHA = 0.5f;  // Slope parameter of the activation

    // --- Numerical stability constants ---
    constexpr float EPSILON = 1e-8f;  // For preventing division by zero
    constexpr float MIN_VAL = 1e-6f;  // Floor for output mean and variance
    constexpr float STABILITY_THRESHOLD =
        30.0f;  // Threshold for using stable approximation

    for (int i = start_chunk; i < end_chunk; i++) {
        const float mz = mu_z[i];
        const float varz = var_z[i];

        // 1. Standardize input variable, with protection against zero variance.
        const float sz = sqrtf(varz + EPSILON);
        const float z_norm = mz / sz;  // Standardized mean, z ~ N(z_norm, 1)

        // 2. Precompute common terms.
        // `phi_z` is the PDF of a standard normal distribution evaluated at
        // `z_norm`.
        const float phi_z = expf(-0.5f * z_norm * z_norm) / SQRT_2PI;

        // `tail_z` is the upper tail probability P(Z > z_norm).
        const float tail_z = 0.5f * erfcf(z_norm * INV_SQRT2);

        // `a` is a scaled standard deviation.
        const float a = sz / ALPHA;
        const float z_a = z_norm + a;
        const float z_2a = z_norm + 2.0f * a;

        // 3. Calculate unstable products in a numerically stable way.
        // The products are of the form `exp(B) * P(Z > A)`, which can cause
        // `inf * 0`. We use a stable asymptotic approximation for large A.

        // Product 1: P(Z > z_a) * exp(a*z_norm + 0.5*a²)
        float prod1;
        if (z_a > STABILITY_THRESHOLD) {
            prod1 = phi_z / z_a;
        } else {
            const float tail_za = 0.5f * erfcf(z_a * INV_SQRT2);
            const float exp_arg1 = mz / ALPHA + 0.5f * varz / (ALPHA * ALPHA);
            prod1 = tail_za * expf(exp_arg1);
        }

        // Product 2: P(Z > z_2a) * exp(2*a*z_norm + 2*a²)
        float prod2;
        if (z_2a > STABILITY_THRESHOLD) {
            prod2 = phi_z / z_2a;
        } else {
            const float tail_z2a = 0.5f * erfcf(z_2a * INV_SQRT2);
            const float exp_arg2 = 2.0f * (mz / ALPHA + varz / (ALPHA * ALPHA));
            prod2 = tail_z2a * expf(exp_arg2);
        }

        // 4. Calculate mean of the activation E[f(Z)]
        const float mean_d =
            mz + sz * phi_z - (ALPHA + mz) * tail_z + ALPHA * prod1;

        // 5. Calculate second moment E[f(Z)²]
        const float E2 = mz * mz + mz * sz * phi_z + varz -
                         2.0f * ALPHA * ALPHA * prod1 + ALPHA * ALPHA * prod2 +
                         (ALPHA * ALPHA - mz * mz - varz) * tail_z;

        // 6. Calculate variance Var(f(Z)) = E[f(Z)²] - (E[f(Z)])²
        // This subtraction can suffer from catastrophic cancellation. Flooring
        // is a safeguard.
        const float var_d = E2 - mean_d * mean_d;

        // 7. Calculate covariance term: Cov[Z, f(Z)] / Var(Z)
        const float jcb_d = (1.0f - tail_z) + prod1;

        // 8. Store results
        // Note: Original code clamped the mean. This is preserved, though the
        // mean of a CELU-like function can be negative.
        mu_a[i] = fmaxf(MIN_VAL, mean_d + ALPHA);
        var_a[i] = fmaxf(MIN_VAL, var_d);
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
        mu_a[col] = fmax(logf(1 + expf(mu_z[col])), 0.000001f);
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = fmax(tmp * var_z[col] * tmp, 0.000001f);
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

        mu_a[i] = fmax(expf(new_mu + 0.5 * new_var), 0.000001f);
        var_a[i] =
            fmax(expf(2 * new_mu + new_var) * (expf(new_var) - 1), 0.000001f);
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
                         const BaseHiddenStates &stored_even_output_states,
                         bool overfit_mu, bool has_even_layer) {
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
        // 0. Define the output indices for the even (Z) and odd (V2_bar) stream
        int even_idx = 2 * i;
        int odd_idx = 2 * i + 1;

        // 1. Retrieve stored values from the forward pass.
        float mu_zv = stored_output_states.mu_a[i];
        float var_zv = stored_output_states.var_a[i];

        // V2_bar_tilde
        float mu_v2_bar_tilde = stored_inner_output_states.mu_a[i];
        float var_v2_bar_tilde = stored_inner_output_states.var_a[i];
        float jcb_v2_bar_tilde = stored_inner_output_states.jcb[i];

        // Deltas from the subsequent layer
        float incoming_delta_mu = input_delta_states.delta_mu[i];
        float incoming_delta_var = input_delta_states.delta_var[i];

        if (std::isnan(incoming_delta_mu) || std::isnan(incoming_delta_var) ||
            std::isinf(incoming_delta_mu) || std::isinf(incoming_delta_var)) {
            output_delta_states.delta_mu[even_idx] = 0.0f;
            output_delta_states.delta_var[even_idx] = 0.0f;
            output_delta_states.delta_mu[odd_idx] = 0.0f;
            output_delta_states.delta_var[odd_idx] = 0.0f;
            continue;
        }

        // Use variance from even output stream (after optional even layer)
        float var_z = fmax(stored_even_output_states.var_a[i], 0.000001f);

        // 2. Perform the backward message passing calculations.

        // Compute the prior predictive PDF for v2
        float mu_v2 = mu_v2_bar_tilde;
        float var_v2 =
            3.0f * var_v2_bar_tilde + 2.0f * mu_v2_bar_tilde * mu_v2_bar_tilde;

        // V ~ N(0, mu_v2)
        float mu_v = 0.0f;
        float var_v = mu_v2;

        float mu_v_pos = var_v * incoming_delta_mu;
        float var_v_pos = fmax(var_v + var_v * incoming_delta_var * var_v,
                               0.000001f);  // Prevent negative variance

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

        // 4. Compute and write deltas for V2_bar (the odd input stream).
        float Jv2_bar = jcb_v2_bar_tilde / var_v2_bar_tilde;
        if (std::isnan(Jv2_bar) || std::isinf(Jv2_bar)) {
            Jv2_bar = 0.0f;
        }
        output_delta_states.delta_mu[odd_idx] =
            Jv2_bar * (mu_v2_bar_tilde_pos - mu_v2_bar_tilde);
        output_delta_states.delta_var[odd_idx] =
            Jv2_bar * Jv2_bar * (var_v2_bar_tilde_pos - var_v2_bar_tilde);

        float mu_zv_pos = var_zv * incoming_delta_mu;
        float var_zv_pos = var_zv * incoming_delta_var * var_zv;

        // 5. Compute and write deltas for Z (the even input stream).
        float Jz = has_even_layer ? (stored_even_output_states.jcb[i]) : 1.0f;
        float delta_mu_overfit = Jz / var_zv * (mu_zv_pos);
        float delta_var = Jz / var_zv * (var_zv_pos)*Jz / var_zv;

        if (overfit_mu) {
            // Use different variance for the mean to allow overfitting on it
            delta_mu_overfit = Jz / var_z * (mu_zv_pos);
        }

        if (std::isnan(delta_mu_overfit) || std::isinf(delta_mu_overfit)) {
            delta_mu_overfit = 0.0f;
        }
        if (std::isnan(delta_var) || std::isinf(delta_var)) {
            delta_var = 0.0f;
        }

        output_delta_states.delta_mu[even_idx] = delta_mu_overfit;
        output_delta_states.delta_var[even_idx] = delta_var;
    }
}

void agvi_backward_mp(int n, unsigned int num_threads,
                      BaseDeltaStates &input_delta_states,
                      BaseDeltaStates &output_delta_states,
                      const BaseHiddenStates &stored_output_states,
                      const BaseHiddenStates &stored_inner_output_states,
                      const BaseHiddenStates &stored_input_states,
                      const BaseHiddenStates &stored_even_output_states,
                      bool overfit_mu, bool has_even_layer) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = n / num_threads;
    int extra = n % num_threads;

    for (unsigned int i = 0; i < num_threads; ++i) {
        int start_chunk = i * n_per_thread + std::min(i, (unsigned int)extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &input_delta_states, &output_delta_states,
                              &stored_output_states,
                              &stored_inner_output_states, &stored_input_states,
                              &stored_even_output_states] {
            agvi_backward_chunk(start_chunk, end_chunk, input_delta_states,
                                output_delta_states, stored_output_states,
                                stored_inner_output_states, stored_input_states,
                                stored_even_output_states, overfit_mu,
                                has_even_layer);
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

    // Get statistics of Z_output before activation
    float mean_mu_z =
        std::accumulate(input_states.mu_a.begin(),
                        input_states.mu_a.begin() + input_states.actual_size,
                        0.0f) /
        input_states.actual_size;
    float mean_var_z =
        std::accumulate(input_states.var_a.begin(),
                        input_states.var_a.begin() + input_states.actual_size,
                        0.0f) /
        input_states.actual_size;
    float std_mu_z = 0.0f, std_var_z = 0.0f;

    float min_mu_z =
        *std::min_element(input_states.mu_a.begin(),
                          input_states.mu_a.begin() + input_states.actual_size);
    float max_mu_z =
        *std::max_element(input_states.mu_a.begin(),
                          input_states.mu_a.begin() + input_states.actual_size);

    for (int i = 0; i < input_states.actual_size; i++) {
        std_mu_z += (input_states.mu_a[i] - mean_mu_z) *
                    (input_states.mu_a[i] - mean_mu_z);
        std_var_z += (input_states.var_a[i] - mean_var_z) *
                     (input_states.var_a[i] - mean_var_z);
    }
    std_mu_z = sqrtf(std_mu_z / input_states.actual_size);
    std_var_z = sqrtf(std_var_z / input_states.actual_size);

    std::cout << "Remax layer: mean_mu_z = " << mean_mu_z
              << ", std_mu_z = " << std_mu_z << ", mean_var_z = " << mean_var_z
              << ", std_var_z = " << std_var_z << std::endl;

    std::cout << "Remax layer: min_mu_z = " << min_mu_z
              << ", max_mu_z = " << max_mu_z << std::endl;

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
    // jcb is not needed for this intermediate state

    BaseHiddenStates even_input_states;
    even_input_states.mu_a.reserve(even_count);
    even_input_states.var_a.reserve(even_count);
    // jcb is not needed for this intermediate state

    // 3. Split the input data into odd and even streams.
    for (int i = 0; i < total_elements; ++i) {
        if (i % 2 == 0) {  // Even indices
            even_input_states.mu_a.push_back(input_states.mu_a[i]);
            even_input_states.var_a.push_back(input_states.var_a[i]);
        } else {  // Odd indices
            odd_input_states.mu_a.push_back(input_states.mu_a[i]);
            odd_input_states.var_a.push_back(input_states.var_a[i]);
        }
    }

    // Set metadata for the temporary states.
    odd_input_states.block_size = input_states.block_size;
    odd_input_states.actual_size = odd_count / input_states.block_size;
    even_input_states.block_size = input_states.block_size;
    even_input_states.actual_size = even_count / input_states.block_size;

    // 4. Process the odd stream FIRST to get its output mean. ✨
    BaseHiddenStates odd_output_states;
    odd_output_states.mu_a.resize(odd_count);
    odd_output_states.var_a.resize(odd_count);
    odd_output_states.jcb.resize(odd_count);
    odd_layer->forward(odd_input_states, odd_output_states, temp_states);

    // 5. MODIFICATION: Add the mean of the activated odd units to the variance
    // of the even units. 🧠
    for (int i = 0; i < even_count; ++i) {
        even_input_states.var_a[i] += odd_output_states.mu_a[i];
    }

    // 6. Process the even stream with its modified variance.
    BaseHiddenStates even_output_states;
    if (even_layer) {
        // If an even_layer is provided, use it.
        even_output_states.mu_a.resize(even_count);
        even_output_states.var_a.resize(even_count);
        even_output_states.jcb.resize(even_count);
        even_layer->forward(even_input_states, even_output_states, temp_states);
    } else {
        // If no even_layer is provided, apply an identity transformation
        // by moving the input states (with modified variance) to the output
        // states.
        even_output_states = std::move(even_input_states);
        even_output_states.jcb.assign(even_count,
                                      1.0f);  // Jacobian of identity is 1
    }

    // 7. Merge the processed streams back into the final output_states.
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

    // 8. Update layer and output_states metadata to match the input.
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
AGVI::AGVI(std::shared_ptr<BaseLayer> odd_layer,
           std::shared_ptr<BaseLayer> even_layer, bool overfit_mu, bool agvi)
    : m_odd_layer(odd_layer),
      m_even_layer(even_layer),
      m_overfit_mu(overfit_mu),
      m_agvi(agvi) {
    if (!m_odd_layer ||
        m_odd_layer->get_layer_type() != LayerType::Activation) {
        std::cerr << "Error: AGVI layer must be initialized with a valid "
                     "odd activation layer."
                  << std::endl;
        m_odd_layer = std::make_shared<ReLU>();
    }
}

AGVI::~AGVI() {}

std::string AGVI::get_layer_info() const {
    std::string even_name =
        m_even_layer ? m_even_layer->get_layer_name() : std::string("Identity");
    return "AGVI(odd=" + m_odd_layer->get_layer_name() + ", even=" + even_name +
           ")";
}

std::string AGVI::get_layer_name() const { return "AGVI"; }

LayerType AGVI::get_layer_type() const { return LayerType::AGVI; }

void AGVI::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states,
                   BaseTempStates &temp_states) {
    // The AGVI layer now supports separate odd and optional even activations.
    // Output consists of the even stream only, with var possibly augmented by
    // the odd stream activation mean (AGVI noise model).

    int total_elements = input_states.actual_size * input_states.block_size;
    int half_count = total_elements / 2;

    // The output size is halved as we only propagate the even stream.
    output_states.block_size = input_states.block_size;
    output_states.actual_size = input_states.actual_size / 2;
    output_states.size = input_states.size;

    // Prepare split streams
    BaseHiddenStates odd_input_states, even_input_states;
    odd_input_states.mu_a.reserve(half_count);
    odd_input_states.var_a.reserve(half_count);
    odd_input_states.jcb.reserve(half_count);
    even_input_states.mu_a.reserve(half_count);
    even_input_states.var_a.reserve(half_count);
    even_input_states.jcb.reserve(half_count);

    for (int i = 0; i < total_elements; ++i) {
        if ((i % 2) == 0) {
            even_input_states.mu_a.push_back(input_states.mu_a[i]);
            even_input_states.var_a.push_back(input_states.var_a[i]);
            even_input_states.jcb.push_back(input_states.jcb[i]);
        } else {
            odd_input_states.mu_a.push_back(input_states.mu_a[i]);
            odd_input_states.var_a.push_back(input_states.var_a[i]);
            odd_input_states.jcb.push_back(input_states.jcb[i]);
        }
    }
    odd_input_states.block_size = input_states.block_size;
    even_input_states.block_size = input_states.block_size;
    odd_input_states.actual_size = half_count / input_states.block_size;
    even_input_states.actual_size = half_count / input_states.block_size;

    // Process odd activation
    BaseHiddenStates odd_output_states;
    odd_output_states.mu_a.resize(half_count);
    odd_output_states.var_a.resize(half_count);
    odd_output_states.jcb.resize(half_count);
    m_odd_layer->forward(odd_input_states, odd_output_states, temp_states);
    m_stored_inner_output_states = odd_output_states;

    // Process even activation (optional). Default is identity.
    BaseHiddenStates even_output_states;
    if (m_even_layer) {
        even_output_states.mu_a.resize(half_count);
        even_output_states.var_a.resize(half_count);
        even_output_states.jcb.resize(half_count);
        m_even_layer->forward(even_input_states, even_output_states,
                              temp_states);
    } else {
        even_output_states = std::move(even_input_states);
    }
    m_stored_even_output_states = even_output_states;

    for (int i = 0; i < half_count; ++i) {
        // Even stream value (possibly activated by even layer)
        output_states.mu_a[i] = m_stored_even_output_states.mu_a[i];
        output_states.jcb[i] = 1.0f;  // Reset Jacobian to 1.0f for AGVI
        if (m_agvi) {
            output_states.var_a[i] = m_stored_even_output_states.var_a[i] +
                                     m_stored_inner_output_states.mu_a[i];
        } else {
            output_states.var_a[i] = m_stored_even_output_states.var_a[i];
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
                         m_stored_input_states, m_stored_even_output_states,
                         m_overfit_mu, m_even_layer != nullptr);
    } else {
        agvi_backward_chunk(0, total_output_size, input_delta_states,
                            output_delta_states, m_stored_output_states,
                            m_stored_inner_output_states, m_stored_input_states,
                            m_stored_even_output_states, m_overfit_mu,
                            m_even_layer != nullptr);
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
    auto cuda_odd_layer = m_odd_layer->to_cuda(device_idx);
    std::unique_ptr<BaseLayer> cuda_even_layer = nullptr;
    if (m_even_layer) {
        cuda_even_layer = m_even_layer->to_cuda(device_idx);
    }
    auto cuda_layer = std::make_unique<AGVICuda>(std::move(cuda_odd_layer),
                                                 std::move(cuda_even_layer),
                                                 m_overfit_mu, m_agvi);
    cuda_layer->set_overfit_mu(this->m_overfit_mu);
    cuda_layer->set_agvi(this->m_agvi);
    return cuda_layer;
}
#endif

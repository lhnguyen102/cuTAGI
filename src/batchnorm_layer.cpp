#include "../include/batchnorm_layer.h"

#include "../include/custom_logger.h"
#include "../include/param_init.h"

#ifdef USE_CUDA
#include "../include/batchnorm_layer_cuda.cuh"
#endif
#include <cmath>
#include <thread>

////////////////////////////////////////////////////////////////////////////////
/// CPU kernels for Batch Norm
////////////////////////////////////////////////////////////////////////////////
void running_mean_var(const std::vector<float> &mu_s,
                      const std::vector<float> &var_s, float momentum,
                      int start_chunk, int end_chunk, std::vector<float> &mu_ra,
                      std::vector<float> &var_ra)
/*Copute the running average for the normalization layers.
 */
{
    for (int col = start_chunk; col < end_chunk; col++) {
        mu_ra[col] = mu_ra[col] * momentum + mu_s[col] * (1 - momentum);
        var_ra[col] = var_ra[col] * momentum + var_s[col] * (1 - momentum);
    }
}

void running_mean_var_mp(const std::vector<float> &mu_s,
                         const std::vector<float> &var_s, float momentum,
                         int num_states, const int num_threads,
                         std::vector<float> &mu_ra, std::vector<float> &var_ra)
/*Copute the running average for the normalization layers.
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = num_states / num_threads;
    int extra = num_states % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_s, &var_s, &mu_ra, &var_ra] {
            running_mean_var(mu_s, var_s, momentum, start_chunk, end_chunk,
                             mu_ra, var_ra);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm_stat_mean_var(const std::vector<float> &mu_a,
                             const std::vector<float> &var_a, int ni,
                             int batch_size, int start_chunk, int end_chunk,
                             std::vector<float> &mu_s,
                             std::vector<float> &var_s)
/*Compute sample mean and variance of activation units of full-connected layer
for each batch.
*/
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[col + i * ni];
            sum_var += var_a[col + i * ni];
        }
        mu_s[col] = sum_mu / batch_size;
        var_s[col] = sum_var;
    }
}

void batchnorm_sample_var(const std::vector<float> &mu_a,
                          const std::vector<float> &mu_s,
                          const std::vector<float> &var_s, int ni,
                          int batch_size, int start_chunk, int end_chunk,
                          std::vector<float> &var)
/*Compute statistical mean and variance of activation units for full-connected
layer for each batch.
*/
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum = 0;
        for (int i = 0; i < batch_size; i++) {
            sum += (mu_a[col + i * ni] - mu_s[col]) *
                   (mu_a[col + i * ni] - mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (batch_size - 1);
    }
}

void batchnorm_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int ni, int start_chunk, int end_chunk,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*Compute mean of product WA of batch-normalization layer.
 */
{
    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < ni; col++) {
            int idx = col + row * ni;
            float inv_sqrt_var_ra = 1.0f / std::sqrt(var_ra[col] + epsilon);
            float mu_a_tilde = mu_a[idx] - mu_ra[col];

            mu_z[idx] = inv_sqrt_var_ra * mu_a_tilde * mu_w[col];
            var_z[idx] = inv_sqrt_var_ra * inv_sqrt_var_ra *
                         (var_a[idx] * (mu_w[col] * mu_w[col] + var_w[col]) +
                          var_w[col] * mu_a_tilde * mu_a_tilde);
            if (bias) {
                mu_z[idx] += mu_b[col];
                var_z[idx] += var_b[col];
            }
        }
    }
}

void batchnorm2d_stat_mean_var(const std::vector<float> &mu_a,
                               const std::vector<float> &var_a, int wihi,
                               int fi, int batch_size, int start_chunk,
                               int end_chunk, std::vector<float> &mu_s,
                               std::vector<float> &var_s)
/*Compute sample mean and variance of activation units for batch-normalization
layer.
*/
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0;
        float sum_var = 0;
        for (int i = 0; i < wihi * batch_size; i++)  // n = wihi*B
        {
            sum_mu += mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sum_var += var_a[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        mu_s[col] = sum_mu / (wihi * batch_size);
        var_s[col] = sum_var;
    }
}

void batchnorm2d_sample_var(const std::vector<float> &mu_a,
                            const std::vector<float> &mu_s,
                            const std::vector<float> &var_s, int wihi, int fi,
                            int batch_size, int start_chunk, int end_chunk,
                            std::vector<float> &var)
/*Compute statistical mean and variance of activation units for
batch-normalization layer.
*/
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum = 0;
        for (int i = 0; i < wihi * batch_size; i++) {
            sum += (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]) *
                   (mu_a[(i / wihi) * wihi * fi + i % wihi + col * wihi] -
                    mu_s[col]);
        }
        var[col] = (sum + var_s[col]) / (wihi * batch_size - 1);
    }
}

void batchnorm2d_fwd_mean_var(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int wihi, int fi, int batch_size, int start_chunk,
    int end_chunk, std::vector<float> &mu_z, std::vector<float> &var_z)
/*Compute mean of product WA of batch-normalization. Note that the previous
layer is a convolutional layer.
*/
{
    int k = wihi;
    // m = fi * batch_size;
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / std::sqrt(var_ra[row % fi] + epsilon);
        float mu_ra_term = mu_ra[row % fi];
        float mu_w_term = mu_w[row % fi];

        for (int col = 0; col < k; col++)  // k = wihi, m = fi*B
        {
            int idx = col + row * k;
            float mu_a_tilde = mu_a[idx] - mu_ra_term;

            mu_z[idx] = inv_sqrt_var_ra * mu_a_tilde * mu_w_term;

            var_z[idx] =
                inv_sqrt_var_ra * inv_sqrt_var_ra *
                (var_a[idx] * (mu_w_term * mu_w_term + var_w[row % fi]) +
                 var_w[row % fi] * mu_a_tilde * mu_a_tilde);

            if (bias) {
                mu_z[idx] += mu_b[row % fi];
                var_z[idx] += var_b[row % fi];
            }
        }
    }
}

void batchnorm_bwd_delta_z(const std::vector<float> &mu_w,
                           const std::vector<float> &jcb,
                           const std::vector<float> &var_ra,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu,
                           std::vector<float> &delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is full-connected layer.
*/
{
    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < ni; col++) {
            float tmp = (1 / std::sqrt(var_ra[col] + epsilon)) * mu_w[col] *
                        jcb[col + row * ni];

            delta_mu[col + row * ni] = tmp * delta_mu_out[col + row * ni];

            delta_var[col + row * ni] =
                tmp * delta_var_out[col + row * ni] * tmp;
        }
    }
}

void batchnorm2d_bwd_delta_z(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_ra, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int batch_size, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu, std::vector<float> &delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is convolutional layer.
 */
{
    // m = fi * batch_size;
    for (int row = start_chunk; row < end_chunk; row++)  // k = wihi, m = fi*B
    {
        float inv_sqrt_var_ra = 1.0f / std::sqrt(var_ra[row % fi] + epsilon);
        for (int col = 0; col < wihi; col++) {
            int idx = col + row * wihi;
            float tmp = inv_sqrt_var_ra * mu_w[row % fi] * jcb[idx];

            delta_mu[idx] = tmp * delta_mu_out[idx];
            delta_var[idx] = tmp * delta_var_out[idx] * tmp;
        }
    }
}

void batchnorm_bwd_delta_w(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to full-connected layer.
*/
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0;
        float sum_var = 0;
        float inv_sqrt_var_ra = 1.0f / std::sqrt(var_ra[col] + epsilon);
        for (int i = 0; i < batch_size; i++) {
            float tmp = inv_sqrt_var_ra * (mu_a[col + i * ni] - mu_ra[col]) *
                        var_w[col];
            sum_mu += tmp * delta_mu_out[col + i * ni];
            sum_var += tmp * delta_var_out[col + i * ni] * tmp;
        }
        delta_mu_w[col] = sum_mu;
        delta_var_w[col] = sum_var;
    }
}

void batchnorm_bwd_delta_b(const std::vector<float> &var_b,
                           const std::vector<float> &delta_mu_out,
                           const std::vector<float> &delta_var_out,
                           float epsilon, int ni, int batch_size,
                           int start_chunk, int end_chunk,
                           std::vector<float> &delta_mu_b,
                           std::vector<float> &delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to full-connected layer.
*/
{
    for (int col = start_chunk; col < end_chunk; col++) {
        float sum_mu = 0.0f;
        float sum_var = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            float tmp = var_b[col];
            sum_mu += tmp * delta_mu_out[col + i * ni];
            sum_var += tmp * delta_var_out[col + i * ni] * tmp;
        }
        delta_mu_b[col] = sum_mu;
        delta_var_b[col] = sum_var;
    }
}

void batchnorm2d_bwd_delta_w(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int batch_size, int start_chunk, int end_chunk,
    std::vector<float> &delta_mu_w, std::vector<float> &delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to convolutional layer.
*/
{
    // m = batch_size * fi;
    for (int row = start_chunk; row < end_chunk; row++) {
        float inv_sqrt_var_ra = 1.0f / std::sqrt(var_ra[row % fi] + epsilon);
        float mu_ra_term = mu_ra[row % fi];
        for (int col = 0; col < wihi; col++)  // k = wihi, m = fi*B
        {
            int idx = col + row * wihi;
            float tmp =
                inv_sqrt_var_ra * (mu_a[idx] - mu_ra_term) * var_w[row % fi];

            delta_mu_w[idx] = tmp * delta_mu_out[idx];
            delta_var_w[idx] = tmp * delta_var_out[idx] * tmp;
        }
    }
}

void batchnorm2d_bwd_delta_b(const std::vector<float> &var_b,
                             const std::vector<float> &delta_mu_out,
                             const std::vector<float> &delta_var_out,
                             float epsilon, int wihi, int fi, int start_chunk,
                             int end_chunk, std::vector<float> &delta_mu_b,
                             std::vector<float> &delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to convolutional layer.
*/
{
    for (int row = start_chunk; row < end_chunk; row++) {
        for (int col = 0; col < wihi; col++)  // k = wihi, m = fi*B
        {
            float tmp = var_b[row % fi];

            delta_mu_b[col + row * wihi] = tmp * delta_mu_out[col + row * wihi];
            delta_var_b[col + row * wihi] =
                tmp * delta_var_out[col + row * wihi] * tmp;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Multiprocessing kernels for batch norm
////////////////////////////////////////////////////////////////////////////////
void batchnorm_stat_mean_var_mp(const std::vector<float> &mu_a,
                                const std::vector<float> &var_a, int ni,
                                int batch_size, const int num_threads,
                                std::vector<float> &mu_s,
                                std::vector<float> &var_s)
/*Compute sample mean and variance of activation units of full-connected layer
for each batch.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &var_a, &mu_s, &var_s] {
            batchnorm_stat_mean_var(mu_a, var_a, ni, batch_size, start_chunk,
                                    end_chunk, mu_s, var_s);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm_sample_var_mp(const std::vector<float> &mu_a,
                             const std::vector<float> &mu_s,
                             const std::vector<float> &var_s, int ni,
                             int batch_size, const int num_threads,
                             std::vector<float> &var)
/*
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &mu_s, &var_s, &var] {
            batchnorm_sample_var(mu_a, mu_s, var_s, ni, batch_size, start_chunk,
                                 end_chunk, var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int ni, int batch_size, const int num_threads,
    std::vector<float> &mu_z, std::vector<float> &var_z)
/*Compute mean of product WA of batch-normalization layer.
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &mu_ra, &var_ra, &mu_z, &var_z] {
            batchnorm_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a, mu_ra,
                                   var_ra, bias, epsilon, ni, start_chunk,
                                   end_chunk, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm2d_stat_mean_var_mp(const std::vector<float> &mu_a,
                                  const std::vector<float> &var_a, int wihi,
                                  int fi, int batch_size, const int num_threads,
                                  std::vector<float> &mu_s,
                                  std::vector<float> &var_s)
/*Compute sample mean and variance of activation units for batch-normalization
layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = fi / num_threads;
    int extra = fi % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &var_a, &mu_s, &var_s] {
            batchnorm2d_stat_mean_var(mu_a, var_a, wihi, fi, batch_size,
                                      start_chunk, end_chunk, mu_s, var_s);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm2d_sample_var_mp(const std::vector<float> &mu_a,
                               const std::vector<float> &mu_s,
                               const std::vector<float> &var_s, int wihi,
                               int fi, int batch_size, const int num_threads,
                               std::vector<float> &var)
/*Compute statistical mean and variance of activation units for
batch-normalization layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = fi / num_threads;
    int extra = fi % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_a, &mu_s, &var_s, &var] {
            batchnorm2d_sample_var(mu_a, mu_s, var_s, wihi, fi, batch_size,
                                   start_chunk, end_chunk, var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm2d_fwd_mean_var_mp(
    const std::vector<float> &mu_w, const std::vector<float> &var_w,
    const std::vector<float> &mu_b, const std::vector<float> &var_b,
    const std::vector<float> &mu_a, const std::vector<float> &var_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    bool bias, float epsilon, int wihi, int fi, int batch_size,
    const int num_threads, std::vector<float> &mu_z, std::vector<float> &var_z)
/*Compute mean of product WA of batch-normalization. Note that the previous
layer is a convolutional layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = (fi * batch_size) / num_threads;
    int extra = (fi * batch_size) % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &mu_ra, &var_ra, &mu_z, &var_z] {
            batchnorm2d_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a,
                                     mu_ra, var_ra, bias, epsilon, wihi, fi,
                                     batch_size, start_chunk, end_chunk, mu_z,
                                     var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_ra, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, const int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is full-connected layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = batch_size / num_threads;
    int extra = batch_size % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &jcb, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu, &delta_var] {
            batchnorm_bwd_delta_z(mu_w, jcb, var_ra, delta_mu_out,
                                  delta_var_out, epsilon, ni, start_chunk,
                                  end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm2d_bwd_delta_z_mp(
    const std::vector<float> &mu_w, const std::vector<float> &jcb,
    const std::vector<float> &var_ra, const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int batch_size, const int num_threads, std::vector<float> &delta_mu,
    std::vector<float> &delta_var)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is convolutional layer.
 */
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = (fi * batch_size) / num_threads;
    int extra = (fi * batch_size) % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &jcb, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu, &delta_var] {
            batchnorm2d_bwd_delta_z(
                mu_w, jcb, var_ra, delta_mu_out, delta_var_out, epsilon, wihi,
                fi, batch_size, start_chunk, end_chunk, delta_mu, delta_var);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm_bwd_delta_w_mp(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int ni,
    int batch_size, const int num_threads, std::vector<float> &delta_mu_w,
    std::vector<float> &delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to full-connected layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &mu_a, &mu_ra, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu_w, &delta_var_w] {
            batchnorm_bwd_delta_w(var_w, mu_a, mu_ra, var_ra, delta_mu_out,
                                  delta_var_out, epsilon, ni, batch_size,
                                  start_chunk, end_chunk, delta_mu_w,
                                  delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm_bwd_delta_b_mp(const std::vector<float> &var_b,
                              const std::vector<float> &delta_mu_out,
                              const std::vector<float> &delta_var_out,
                              float epsilon, int ni, int batch_size,
                              const int num_threads,
                              std::vector<float> &delta_mu_b,
                              std::vector<float> &delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to full-connected layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = ni / num_threads;
    int extra = ni % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_b, &delta_mu_out, &delta_var_out,
                              &delta_mu_b, &delta_var_b] {
            batchnorm_bwd_delta_b(var_b, delta_mu_out, delta_var_out, epsilon,
                                  ni, batch_size, start_chunk, end_chunk,
                                  delta_mu_b, delta_var_b);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm2d_bwd_delta_w_mp(
    const std::vector<float> &var_w, const std::vector<float> &mu_a,
    const std::vector<float> &mu_ra, const std::vector<float> &var_ra,
    const std::vector<float> &delta_mu_out,
    const std::vector<float> &delta_var_out, float epsilon, int wihi, int fi,
    int batch_size, const int num_threads, std::vector<float> &delta_mu_w,
    std::vector<float> &delta_var_w)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to convolutional layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = (fi * batch_size) / num_threads;
    int extra = (fi * batch_size) % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_w, &mu_a, &mu_ra, &var_ra, &delta_mu_out,
                              &delta_var_out, &delta_mu_w, &delta_var_w] {
            batchnorm2d_bwd_delta_w(var_w, mu_a, mu_ra, var_ra, delta_mu_out,
                                    delta_var_out, epsilon, wihi, fi,
                                    batch_size, start_chunk, end_chunk,
                                    delta_mu_w, delta_var_w);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void batchnorm2d_bwd_delta_b_mp(const std::vector<float> &var_b,
                                const std::vector<float> &delta_mu_out,
                                const std::vector<float> &delta_var_out,
                                float epsilon, int wihi, int fi, int batch_size,
                                const int num_threads,
                                std::vector<float> &delta_mu_b,
                                std::vector<float> &delta_var_b)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to convolutional layer.
*/
{
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = (fi * batch_size) / num_threads;
    int extra = (fi * batch_size) % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &var_b, &delta_mu_out, &delta_var_out,
                              &delta_mu_b, &delta_var_b] {
            batchnorm2d_bwd_delta_b(var_b, delta_mu_out, delta_var_out, epsilon,
                                    wihi, fi, start_chunk, end_chunk,
                                    delta_mu_b, delta_var_b);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//// Batch Norm
////////////////////////////////////////////////////////////////////////////////
BatchNorm2d::BatchNorm2d(int num_features, float eps, float momentum, bool bias,
                         float gain_weight, float gain_bias)
    : num_features(num_features),
      epsilon(eps),
      momentum(momentum),
      gain_w(gain_weight),
      gain_b(gain_bias)

/*
 */
{
    this->bias = bias;
    this->init_weight_bias();
    this->allocate_running_mean_var();
    if (this->training) {
        this->allocate_param_delta();
    }
}

BatchNorm2d::~BatchNorm2d()
/*
 */
{}

std::string BatchNorm2d::get_layer_info() const
/*
 */
{
    return "BatchNorm2d()";
}

std::string BatchNorm2d::get_layer_name() const
/*
 */
{
    return "BatchNorm2d";
}

LayerType BatchNorm2d::get_layer_type() const
/*
 */
{
    return LayerType::Norm;
}

void BatchNorm2d::init_weight_bias()
/*
 */
{
    this->num_weights = this->num_features;
    this->num_biases = this->bias ? this->num_features : 0;
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_norm("", this->gain_w, this->gain_b,
                              this->num_features, this->num_features,
                              this->num_weights, this->num_biases);
}

void BatchNorm2d::allocate_running_mean_var()
/*
 */
{
    // For inference, we use the running average during the training
    if (this->mu_ra.size() == 0) {
        this->mu_ra.resize(this->num_features, 0.0f);
        this->var_ra.resize(this->num_features, 1.0f);
    }

    this->mu_norm_batch.resize(this->num_features, 0.0f);
    this->var_norm_batch.resize(this->num_features, 0.0f);
}

void BatchNorm2d::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->input_size == 0 || this->output_size == 0) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }
    float _momentum = this->momentum;
    if (this->first_batch) {
        if (this->training) {
            _momentum = 0.0f;
        }
        this->first_batch = false;
    }
    // Assign output dimensions
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;

    const std::vector<float> &mu_target =
        this->training ? this->mu_norm_batch : this->mu_ra;
    const std::vector<float> &var_target =
        this->training ? this->var_norm_batch : this->var_ra;

    if (this->num_threads == 1) {
        // This condition might not be robust!
        if (this->num_features != this->in_channels) {
            if (this->training) {
                batchnorm_stat_mean_var(input_states.mu_a, input_states.var_a,
                                        this->input_size, batch_size, 0,
                                        this->input_size, this->mu_norm_batch,
                                        temp_states.tmp_2);

                batchnorm_sample_var(input_states.mu_a, this->mu_norm_batch,
                                     temp_states.tmp_2, this->input_size,
                                     batch_size, 0, this->input_size,
                                     this->var_norm_batch);

                running_mean_var(this->mu_norm_batch, this->var_norm_batch,
                                 _momentum, 0, this->input_size, this->mu_ra,
                                 this->var_ra);
            }
            batchnorm_fwd_mean_var(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, mu_target, var_target,
                this->bias, this->epsilon, this->input_size, 0, batch_size,
                output_states.mu_a, output_states.var_a);

        } else {
            int wihi = this->in_height * this->in_width;
            if (this->training) {
                batchnorm2d_stat_mean_var(
                    input_states.mu_a, input_states.var_a, wihi,
                    this->in_channels, batch_size, 0, this->in_channels,
                    this->mu_norm_batch, temp_states.tmp_2);

                batchnorm2d_sample_var(input_states.mu_a, this->mu_norm_batch,
                                       temp_states.tmp_2, wihi,
                                       this->in_channels, batch_size, 0,
                                       this->in_channels, this->var_norm_batch);

                running_mean_var(this->mu_norm_batch, this->var_norm_batch,
                                 _momentum, 0, this->in_channels, this->mu_ra,
                                 this->var_ra);
            }

            int end_chunk = this->in_channels * batch_size;
            batchnorm2d_fwd_mean_var(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, mu_target, var_target,
                this->bias, this->epsilon, wihi, this->in_channels, batch_size,
                0, end_chunk, output_states.mu_a, output_states.var_a);
        }
    } else {
        if (this->num_features != this->in_channels) {
            if (this->training) {
                batchnorm_stat_mean_var_mp(
                    input_states.mu_a, input_states.var_a, this->input_size,
                    batch_size, this->num_threads, this->mu_norm_batch,
                    temp_states.tmp_2);

                batchnorm_sample_var_mp(input_states.mu_a, this->mu_norm_batch,
                                        temp_states.tmp_2, this->input_size,
                                        batch_size, this->num_threads,
                                        this->var_norm_batch);

                running_mean_var_mp(this->mu_norm_batch, this->var_norm_batch,
                                    momentum, this->input_size,
                                    this->num_threads, this->mu_ra,
                                    this->var_ra);
            }

            batchnorm_fwd_mean_var_mp(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, mu_target, var_target,
                this->bias, this->epsilon, this->input_size, batch_size,
                this->num_threads, output_states.mu_a, output_states.var_a);

        } else {
            int wihi = this->in_height * this->in_width;
            if (this->training) {
                batchnorm2d_stat_mean_var_mp(
                    input_states.mu_a, input_states.var_a, wihi,
                    this->in_channels, batch_size, this->num_threads,
                    this->mu_norm_batch, temp_states.tmp_2);

                batchnorm2d_sample_var_mp(
                    input_states.mu_a, this->mu_norm_batch, temp_states.tmp_2,
                    wihi, this->in_channels, batch_size, this->num_threads,
                    this->var_norm_batch);

                running_mean_var_mp(this->mu_norm_batch, this->var_norm_batch,
                                    momentum, this->in_channels,
                                    this->num_threads, this->mu_ra,
                                    this->var_ra);
            }

            int end_chunk = this->in_channels * batch_size;
            batchnorm2d_fwd_mean_var_mp(
                this->mu_w, this->var_w, this->mu_b, this->var_b,
                input_states.mu_a, input_states.var_a, mu_target, var_target,
                this->bias, this->epsilon, wihi, this->in_channels, batch_size,
                this->num_threads, output_states.mu_a, output_states.var_a);
        }
    }
    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void BatchNorm2d::backward(BaseDeltaStates &input_delta_states,
                           BaseDeltaStates &output_delta_states,
                           BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    int batch_size = input_delta_states.block_size;

    if (state_udapte) {
        if (this->num_threads == 1) {
            if (this->in_channels == 0) {
                batchnorm_bwd_delta_z(
                    this->mu_w, this->bwd_states->jcb, this->var_norm_batch,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, this->input_size, 0, batch_size,
                    output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            } else {
                int wihi = this->in_width * this->in_height;
                int end_chunk = this->in_channels * batch_size;

                batchnorm2d_bwd_delta_z(
                    this->mu_w, this->bwd_states->jcb, this->var_norm_batch,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, wihi, this->in_channels, batch_size, 0,
                    end_chunk, output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            }
        } else {
            if (this->in_channels == 0) {
                batchnorm_bwd_delta_z_mp(
                    this->mu_w, this->bwd_states->jcb, this->var_norm_batch,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, this->input_size, batch_size,
                    this->num_threads, output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            } else {
                int wihi = this->in_width * this->in_height;
                int end_chunk = this->in_channels * batch_size;

                batchnorm2d_bwd_delta_z_mp(
                    this->mu_w, this->bwd_states->jcb, this->var_norm_batch,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->epsilon, wihi, this->in_channels, batch_size,
                    this->num_threads, output_delta_states.delta_mu,
                    output_delta_states.delta_var);
            }
        }
    }

    if (this->param_update) {
        if (this->num_threads == 1) {
            if (this->in_channels == 0) {
                batchnorm_bwd_delta_w(
                    this->var_w, this->bwd_states->mu_a, this->mu_norm_batch,
                    this->var_norm_batch, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon,
                    this->input_size, batch_size, 0, this->input_size,
                    this->delta_mu_w, this->delta_var_w);

                if (this->bias) {
                    batchnorm_bwd_delta_b(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon,
                        this->input_size, batch_size, 0, this->input_size,
                        this->delta_mu_b, this->delta_var_b);
                }

            } else {
                int wihi = this->in_width * this->in_height;
                int end_chunk = this->in_channels * batch_size;

                batchnorm2d_bwd_delta_w(
                    this->var_w, this->bwd_states->mu_a, this->mu_norm_batch,
                    this->var_norm_batch, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon, wihi,
                    this->in_channels, batch_size, 0, end_chunk,
                    temp_states.tmp_1, temp_states.tmp_2);

                delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                this->in_channels, batch_size, this->delta_mu_w,
                                this->delta_var_w);

                if (this->bias) {
                    batchnorm2d_bwd_delta_b(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon, wihi,
                        this->in_channels, 0, end_chunk, temp_states.tmp_1,
                        temp_states.tmp_2);

                    delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                    this->in_channels, batch_size,
                                    this->delta_mu_b, this->delta_var_b);
                }
            }
        } else {
            if (this->in_channels == 0) {
                batchnorm_bwd_delta_w_mp(
                    this->var_w, this->bwd_states->mu_a, this->mu_norm_batch,
                    this->var_norm_batch, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon,
                    this->input_size, batch_size, this->num_threads,
                    this->delta_mu_w, this->delta_var_w);

                if (this->bias) {
                    batchnorm_bwd_delta_b_mp(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon,
                        this->input_size, batch_size, this->num_threads,
                        this->delta_mu_b, this->delta_var_b);
                }

            } else {
                int wihi = this->in_width * this->in_height;
                int end_chunk = this->in_channels * batch_size;

                batchnorm2d_bwd_delta_w_mp(
                    this->var_w, this->bwd_states->mu_a, this->mu_norm_batch,
                    this->var_norm_batch, input_delta_states.delta_mu,
                    input_delta_states.delta_var, this->epsilon, wihi,
                    this->in_channels, batch_size, this->num_threads,
                    temp_states.tmp_1, temp_states.tmp_2);

                delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                this->in_channels, batch_size, this->delta_mu_w,
                                this->delta_var_w);

                if (this->bias) {
                    batchnorm2d_bwd_delta_b_mp(
                        this->var_b, input_delta_states.delta_mu,
                        input_delta_states.delta_var, this->epsilon, wihi,
                        this->in_channels, batch_size, this->num_threads,
                        temp_states.tmp_1, temp_states.tmp_2);

                    delta_param_sum(temp_states.tmp_1, temp_states.tmp_2, wihi,
                                    this->in_channels, batch_size,
                                    this->delta_mu_b, this->delta_var_b);
                }
            }
        }
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> BatchNorm2d::to_cuda() {
    this->device = "cuda";
    return std::make_unique<BatchNorm2dCuda>(this->num_features, this->epsilon,
                                             this->momentum, this->bias,
                                             this->gain_w, this->gain_b);
}
#endif

void BatchNorm2d::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }

    // Save the name length and name
    auto layer_name = this->get_layer_info();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (const auto &m_ra : this->mu_ra) {
        file.write(reinterpret_cast<const char *>(&m_ra), sizeof(m_ra));
    }
    for (const auto &v_ra : this->var_ra) {
        file.write(reinterpret_cast<const char *>(&v_ra), sizeof(v_ra));
    }
}

void BatchNorm2d::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_info();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }

    // Running average for nomalization
    for (auto &m_ra : this->mu_ra) {
        file.read(reinterpret_cast<char *>(&m_ra), sizeof(m_ra));
    }
    for (auto &v_ra : this->var_ra) {
        file.read(reinterpret_cast<char *>(&v_ra), sizeof(v_ra));
    }

    this->num_weights = this->mu_w.size();
    this->num_biases = this->mu_b.size();
    if (this->training) {
        this->allocate_param_delta();
    }
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
BatchNorm2d::get_norm_mean_var() {
    std::vector<std::vector<float>> mu_ras = {this->mu_ra};
    std::vector<std::vector<float>> var_ras = {this->var_ra};
    std::vector<std::vector<float>> mu_norms = {this->mu_norm_batch};
    std::vector<std::vector<float>> var_norms = {this->var_norm_batch};

    return std::make_tuple(mu_ras, var_ras, mu_norms, var_norms);
}

///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun_cpu.cpp
// Description:  Activation function (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 11, 2022
// Updated:      December 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/activation_fun_cpu.h"

void no_act_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                         int zpos, int n, std::vector<float> &ma,
                         std::vector<float> &J, std::vector<float> &Sa)
/* No activation function

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    zpos: Input-hidden-state position for this layer in the weight vector
          of network
    n: Number of hidden units for this layer
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian matrix
*/
{
    float onePad = 1;
    for (int col = 0; col < n; col++) {
        ma[col + zpos] = mz[col + zpos];
        J[col + zpos] = onePad;
        Sa[col + zpos] = Sz[col + zpos];
    }
}

void tanh_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa) {
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = tanhf(mz[col + zpos]);
        ma[col + zpos] = tmp;
        J[col + zpos] = (1 - powf(tmp, 2));
        Sa[col + zpos] =
            (1 - powf(tmp, 2)) * Sz[col + zpos] * (1 - powf(tmp, 2));
    }
}

void sigmoid_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int n, std::vector<float> &ma,
                          std::vector<float> &J, std::vector<float> &Sa) {
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        ma[col + zpos] = tmp;
        J[col + zpos] = tmp * (1 - tmp);
        Sa[col + zpos] = tmp * (1 - tmp) * Sz[col + zpos] * tmp * (1 - tmp);
    }
}

void relu_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
        ma[col + zpos] = tmp;
        if (tmp == 0) {
            J[col + zpos] = zeroPad;
            Sa[col + zpos] = zeroPad;
        } else {
            J[col + zpos] = onePad;
            Sa[col + zpos] = Sz[col + zpos];
        }
    }
}

void softplus_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                           int zpos, int n, std::vector<float> &ma,
                           std::vector<float> &J, std::vector<float> &Sa) {
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        ma[col + zpos] = logf(1 + expf(mz[col + zpos]));
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        J[col + zpos] = tmp;
        Sa[col + zpos] = tmp * Sz[col + zpos] * tmp;
    }
}

void leakyrelu_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                            float alpha, int zpos, int n,
                            std::vector<float> &ma, std::vector<float> &J,
                            std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
        if (tmp == 0) {
            ma[col + zpos] = alpha * mz[col + zpos];
            J[col + zpos] = alpha;
            Sa[col + zpos] = alpha * Sz[col + zpos] * alpha;
        } else {
            ma[col + zpos] = tmp;
            J[col + zpos] = onePad;
            Sa[col + zpos] = Sz[col + zpos];
        }
    }
}

void mixture_relu_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                      float omega_tol, int zpos, int n, std::vector<float> &ma,
                      std::vector<float> &J, std::vector<float> &Sa) {
    float alpha, beta, omega, kappa, mz_til, Sz_til;
    for (int i = 0; i < n; i++) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mz[zpos + i] / powf(Sz[zpos + i], 0.5);
        omega = std::max(1 - normcdf_cpu(alpha), omega_tol);
        beta = normpdf_cpu(alpha, 0.0f, 1.0f) / omega;
        kappa = 1 + alpha * beta - powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[zpos + i] + beta * powf(Sz[zpos + i], 0.5);
        Sz_til = kappa * Sz[zpos + i];

        // Activation distribution
        ma[zpos + i] = omega * mz_til;
        Sa[zpos + i] = omega * Sz_til + omega * (1 - omega) * powf(mz_til, 2);
        J[zpos + i] = powf(omega * kappa, 0.5);
    }
}

void mixture_bounded_relu_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                              float omega_tol, int zpos, int n,
                              std::vector<float> &ma, std::vector<float> &J,
                              std::vector<float> &Sa) {
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = 0; i < n; i++) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
        alpha_upper = (1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
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
        mz_til = mz[zpos + i] - beta * pow(Sz[zpos + i], 0.5);
        Sz_til = kappa * Sz[zpos + i];

        // Activation distribution
        ma[zpos + i] = omega * mz_til - cdf_lower + (1 - cdf_upper);
        Sa[zpos + i] = omega * Sz_til + omega * powf(mz_til - ma[zpos + i], 2) +
                       cdf_lower * powf(1 + ma[zpos + i], 2) +
                       (1 - cdf_upper) * powf(1 - ma[zpos + i], 2);
        J[zpos + i] = powf(omega * kappa, 0.5);
    }
}

void mixture_sigmoid_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                         float omega_tol, int zpos, int n,
                         std::vector<float> &ma, std::vector<float> &J,
                         std::vector<float> &Sa) {
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = 0; i < n; i++) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
        alpha_upper = (1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
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
        mz_til = mz[zpos + i] - beta * pow(Sz[zpos + i], 0.5);
        Sz_til = kappa * Sz[zpos + i];

        // Activation distribution
        ma[zpos + i] =
            (omega * mz_til - cdf_lower + (1 - cdf_upper)) / 2.0f + 0.5f;
        Sa[zpos + i] =
            (omega * Sz_til + omega * powf(mz_til - ma[zpos + i], 2) +
             cdf_lower * powf(1 + ma[zpos + i], 2) +
             (1 - cdf_upper) * powf(1 - ma[zpos + i], 2)) /
            4.0f;
        J[zpos + i] = powf(omega * kappa, 0.5);
    }
}

void exp_fun_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                 std::vector<float> &ma, std::vector<float> &Sa,
                 std::vector<float> &Cza)
/* Exponential function y = exp(x)

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    Cza: Covariance between hidden states and activation units
*/
{
    float tmp_m, tmp_S;
    for (int i = 0; i < mz.size(); i++) {
        tmp_m = mz[i];
        tmp_S = Sz[i];
        ma[i] = exp(mz[i] + 0.5 * Sz[i]);
        Sa[i] = exp(2 * tmp_m + tmp_S) * (exp(tmp_S) - 1);
        Cza[i] = tmp_S * exp(tmp_m + 0.5 * tmp_S);
    }
}

void act_full_cov(std::vector<float> &Sz_f, std::vector<float> &J, int no,
                  int B, int z_pos_out, std::vector<float> &Sa_f)
/*Activate the full covariance.

Args:
    Sz_f: Full-covariance matrix for hidden states
    J: Jacobian matrix
    no: Output node
    B: Number of batches
    z_pos_out: Output-hidden-state position for this layer in the weight vector
        of network
    Sa_f: Full-covariance matrix for activation units

*/
{
    int col, row, idx;
    for (row = 0; row < no * B; row++) {
        for (col = 0; col < no; col++) {
            if (col <= (row % no)) {
                idx = no * col - ((col * (col + 1)) / 2) + row % no +
                      (row / no) * (((no + 1) * no) / 2);
                Sa_f[idx] = Sz_f[idx] *
                            J[row % no + (row / no) * no + z_pos_out] *
                            J[col + (row / no) * no + z_pos_out];
            }
        }
    }
}

void no_act_full_cov(std::vector<float> &Sz_f, int no, int B,
                     std::vector<float> &Sa_f)
/* No activation layer*/
{
    int col;
    for (col = 0; col < (no * (no + 1)) / 2 * B; col++) {
        Sa_f[col] = Sz_f[col];
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void no_act_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                            int zpos, int start_idx, int end_idx,
                            std::vector<float> &ma, std::vector<float> &J,
                            std::vector<float> &Sa)

{
    int col;
    float onePad = 1;
    for (col = start_idx; col < end_idx; col++) {
        ma[col + zpos] = mz[col + zpos];
        J[col + zpos] = onePad;
        Sa[col + zpos] = Sz[col + zpos];
    }
}

void no_act_mean_var_multithreading(std::vector<float> &mz,
                                    std::vector<float> &Sz, int z_pos, int n,
                                    unsigned int NUM_THREADS,
                                    std::vector<float> &ma,
                                    std::vector<float> &J,
                                    std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(no_act_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void tanh_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int start_idx, int end_idx,
                          std::vector<float> &ma, std::vector<float> &J,
                          std::vector<float> &Sa) {
    int col;
    float tmp = 0;
    for (col = start_idx; col < end_idx; col++) {
        tmp = tanhf(mz[col + zpos]);
        ma[col + zpos] = tmp;
        J[col + zpos] = (1 - tmp * tmp);
        Sa[col + zpos] = (1 - tmp * tmp) * Sz[col + zpos] * (1 - tmp * tmp);
    }
}

void tanh_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  unsigned int NUM_THREADS,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(tanh_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void sigmoid_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                             int zpos, int start_idx, int end_idx,
                             std::vector<float> &ma, std::vector<float> &J,
                             std::vector<float> &Sa) {
    int col;
    float tmp;
    for (col = start_idx; col < end_idx; col++) {
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        ma[col + zpos] = tmp;
        J[col + zpos] = tmp * (1 - tmp);
        Sa[col + zpos] = tmp * (1 - tmp) * Sz[col + zpos] * tmp * (1 - tmp);
    }
}

void sigmoid_mean_var_multithreading(std::vector<float> &mz,
                                     std::vector<float> &Sz, int z_pos, int n,
                                     unsigned int NUM_THREADS,
                                     std::vector<float> &ma,
                                     std::vector<float> &J,
                                     std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(sigmoid_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void no_act_full_cov_worker(std::vector<float> &Sz_f, int start_idx,
                            int end_idx, std::vector<float> &Sa_f) {
    int col;
    for (col = start_idx; col < end_idx; col++) {
        Sa_f[col] = Sz_f[col];
    }
}

void no_act_full_cov_multithreading(std::vector<float> &Sz_f, int no, int B,
                                    unsigned int NUM_THREADS,
                                    std::vector<float> &Sa_f) {
    const int tot_ops = (no * (no + 1)) / 2 * B;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(no_act_full_cov_worker, std::ref(Sz_f),
                                 start_idx, end_idx, std::ref(Sa_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void relu_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int start_idx, int end_idx,
                          std::vector<float> &ma, std::vector<float> &J,
                          std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp;
    int col;
    for (col = start_idx; col < end_idx; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
        ma[col + zpos] = tmp;
        if (tmp == 0) {
            J[col + zpos] = zeroPad;
            Sa[col + zpos] = zeroPad;
        } else {
            J[col + zpos] = onePad;
            Sa[col + zpos] = Sz[col + zpos];
        }
    }
}

void relu_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  unsigned int NUM_THREADS,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(relu_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void softplus_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                              int zpos, int start_idx, int end_idx,
                              std::vector<float> &ma, std::vector<float> &J,
                              std::vector<float> &Sa) {
    float tmp;
    int col;
    for (col = start_idx; col < end_idx; col++) {
        ma[col + zpos] = logf(1 + expf(mz[col + zpos]));
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        J[col + zpos] = tmp;
        Sa[col + zpos] = tmp * Sz[col + zpos] * tmp;
    }
}

void softplus_mean_var_multithreading(std::vector<float> &mz,
                                      std::vector<float> &Sz, int z_pos, int n,
                                      unsigned int NUM_THREADS,
                                      std::vector<float> &ma,
                                      std::vector<float> &J,
                                      std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(softplus_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void leakyrelu_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                               float alpha, int zpos, int start_idx,
                               int end_idx, std::vector<float> &ma,
                               std::vector<float> &J, std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp;
    int col;
    for (col = start_idx; col < end_idx; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
        if (tmp == 0) {
            ma[col + zpos] = alpha * mz[col + zpos];
            J[col + zpos] = alpha;
            Sa[col + zpos] = alpha * Sz[col + zpos] * alpha;
        } else {
            ma[col + zpos] = tmp;
            J[col + zpos] = onePad;
            Sa[col + zpos] = Sz[col + zpos];
        }
    }
}

void leakyrelu_mean_var_multithreading(
    std::vector<float> &mz, std::vector<float> &Sz, float alpha, int z_pos,
    int n, unsigned int NUM_THREADS, std::vector<float> &ma,
    std::vector<float> &J, std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(leakyrelu_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), alpha, z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void act_full_cov_worker(std::vector<float> &Sz_f, std::vector<float> &J,
                         int no, int B, int z_pos_out, int start_idx,
                         int end_idx, std::vector<float> &Sa_f) {
    int col, row, idx;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / no;
        col = j % no;
        if (col <= (row % no)) {
            idx = no * col - ((col * (col + 1)) / 2) + row % no +
                  (row / no) * (((no + 1) * no) / 2);

            Sa_f[idx] = Sz_f[idx] * J[row % no + (row / no) * no + z_pos_out] *
                        J[col + (row / no) * no + z_pos_out];
        }
    }
}

void act_full_cov_multithreading(std::vector<float> &Sz_f,
                                 std::vector<float> &J, int no, int B,
                                 int z_pos_out, unsigned int NUM_THREADS,
                                 std::vector<float> &Sa_f) {
    const int tot_ops = no * B * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(act_full_cov_worker, std::ref(Sz_f), std::ref(J), no, B,
                        z_pos_out, start_idx, end_idx, std::ref(Sa_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void activate_hidden_states(Network &net, NetState &state, int j) {
    int B = net.batch_size;
    int no = net.nodes[j];
    int ni = net.nodes[j - 1];
    int z_pos_out = net.z_pos[j];
    int z_pos_in = net.z_pos[j - 1];
    int w_pos_in = net.w_pos[j - 1];
    int b_pos_in = net.b_pos[j - 1];
    int no_B = no * B;

    // Handle multiple input sequences from LSTM layer
    if (net.layers[j - 1] == net.layer_names.lstm) {
        ni = net.nodes[j - 1] * net.input_seq_len;
    }
    if (net.layers[j] == net.layer_names.lstm) {
        no_B = no * B * net.input_seq_len;
    }
    if (net.activations[j] == net.act_names.tanh)  // tanh
    {
        if (no * B > net.min_operations && net.multithreading) {
            tanh_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                         net.num_cpu_threads, state.ma, state.J,
                                         state.Sa);
        } else {
            tanh_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                              state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.sigmoid)  // sigmoid
    {
        if (no * B > net.min_operations && net.multithreading) {
            sigmoid_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                            net.num_cpu_threads, state.ma,
                                            state.J, state.Sa);

        } else {
            sigmoid_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                 state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.relu)  // ReLU
    {
        if (no * B > net.min_operations && net.multithreading) {
            relu_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                         net.num_cpu_threads, state.ma, state.J,
                                         state.Sa);
        } else {
            relu_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                              state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.mrelu)  // mReLU
    {
        // TODO: Build multithreading for mReLU
        mixture_relu_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out, no_B,
                         state.ma, state.J, state.Sa);

    } else if (net.activations[j] == net.act_names.mbrelu)  // mbReLU
    {
        // TODO: Build multithreading for mbReLU
        mixture_bounded_relu_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out,
                                 no_B, state.ma, state.J, state.Sa);

    } else if (net.activations[j] == net.act_names.msigmoid)  // mbReLU
    {
        // TODO: Build multithreading for msigmoid
        mixture_sigmoid_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out, no_B,
                            state.ma, state.J, state.Sa);

    } else if (net.activations[j] == net.act_names.softplus)  // softplus
    {
        if (no * B > net.min_operations && net.multithreading) {
            softplus_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                             no_B, net.num_cpu_threads,
                                             state.ma, state.J, state.Sa);

        } else {
            softplus_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                  state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.leakyrelu)  // leaky ReLU
    {
        if (no * B > net.min_operations && net.multithreading) {
            leakyrelu_mean_var_multithreading(
                state.mz, state.Sz, net.alpha, z_pos_out, no_B,
                net.num_cpu_threads, state.ma, state.J, state.Sa);
        } else {
            leakyrelu_mean_var_cpu(state.mz, state.Sz, net.alpha, z_pos_out,
                                   no_B, state.ma, state.J, state.Sa);
        }
    } else  // no activation
    {
        if (no * B > net.min_operations && net.multithreading) {
            no_act_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                           net.num_cpu_threads, state.ma,
                                           state.J, state.Sa);
        } else {
            no_act_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                state.J, state.Sa);
        }
    }

    // Full-covariance mode
    if (net.is_full_cov) {
        if (net.activations[j] == 0) {
            if (no * B * no > net.min_operations && net.multithreading) {
                no_act_full_cov_multithreading(state.Sz_f, no, B,
                                               net.num_cpu_threads, state.Sa_f);
            } else {
                no_act_full_cov(state.Sz_f, no, B, state.Sa_f);
            }
        } else {
            if (((no * (no + 1) / 2) * B) > net.min_operations &&
                net.multithreading) {
                act_full_cov_multithreading(state.Sz_f, state.J, no, B,
                                            z_pos_out, net.num_cpu_threads,
                                            state.Sa_f);
            } else {
                act_full_cov(state.Sz_f, state.J, no, B, z_pos_out, state.Sa_f);
            }
        }
    }
}

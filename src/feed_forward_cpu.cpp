///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cpp
// Description:  CPU version for forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      June 13, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/feed_forward_cpu.h"

void fc_mean_cpu(std::vector<float> &mw, std::vector<float> &mb,
                 std::vector<float> &ma, int w_pos, int b_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k, std::vector<float> &mz)
/*Compute mean of product WA for full connected layer

Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    w_pos: Weight position for this layer in the weight vector of network
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_in: Input-hidden-state position for this layer in the hidden-state
        vector of network
    z_pos_out: Output-hidden-state position for this layer in the hidden-state
        vector of network
    n: Input node
    m: Output node
    k: Number of batches
*/
{
    float sum = 0;
    float ma_tmp = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                ma_tmp = ma[n * col + i + z_pos_in];
                sum += mw[row * n + i + w_pos] * ma_tmp;
            }
            mz[col * m + row + z_pos_out] = sum + mb[row + b_pos];
        }
    }
}

void fc_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                std::vector<float> &Sb, std::vector<float> &ma,
                std::vector<float> &Sa, int w_pos, int b_pos, int z_pos_in,
                int z_pos_out, int m, int n, int k, std::vector<float> &Sz)
/*Compute variance of product WA for full connected layer

Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Variance of the biases
    ma: Mean of activation units
    Sa: Variance of activation units
    Sz: Variance of hidden states
    w_pos: Weight position for this layer in the weight vector of network
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_in: Input-hidden-state position for this layer in the hidden-state
        vector of network
    z_pos_out: Output-hidden-state position for this layer in the hidden-state
        vector of network
    n: Input node
    m: Output node
    k: Number of batches
*/
{
    float sum = 0;
    float ma_tmp = 0;
    float Sa_tmp = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                ma_tmp = ma[n * col + i + z_pos_in];
                Sa_tmp = Sa[n * col + i + z_pos_in];
                sum += (mw[row * n + i + w_pos] * mw[row * n + i + w_pos] +
                        Sw[row * n + i + w_pos]) *
                           Sa_tmp +
                       Sw[row * n + i + w_pos] * ma_tmp * ma_tmp;
            }
            Sz[col * m + row + z_pos_out] = sum + Sb[row + b_pos];
        }
    }
}

void fc_full_cov_cpu(std::vector<float> &mw, std::vector<float> &Sa_f,
                     int w_pos, int no, int ni, int B,
                     std::vector<float> &Sz_fp)
/* Compute full covariance matrix for fully-connected layer.

Args:
    mw: Mean of weights
    Sa_f: Full-covariance matrix of activation units for the previous layer
    w_pos: Weight position for this layer in the weight vector of network
    no: Output node
    ni: Input node
    B: Number of batches
    Sz_fp: Partial full-covariance matrix of hidden states of current
        layer
 */
{
    int tu, col, row, k;
    float sum, Sa_in;
    for (int row = 0; row < no * B; row++) {
        for (int col = 0; col < no; col++) {
            if (col <= (row % no)) {
                sum = 0;
                for (int i = 0; i < ni * ni; i++) {
                    int row_in = i / ni;
                    int col_in = i % ni;
                    if (row_in > col_in)  // lower triangle
                    {
                        tu = (ni * col_in - ((col_in * (col_in + 1)) / 2) +
                              row_in);
                    } else {
                        tu = (ni * row_in - ((row_in * (row_in + 1)) / 2) +
                              col_in);
                    }
                    Sa_in = Sa_f[tu + (row / no) * (ni * (ni + 1)) / 2];

                    sum += mw[i % ni + (row % no) * ni + w_pos] * Sa_in *
                           mw[i / ni + (col % no) * ni + w_pos];
                }
                k = no * col - ((col * (col + 1)) / 2) + row % no +
                    (row / no) * (((no + 1) * no) / 2);
                Sz_fp[k] = sum;
            }
        }
    }
}

void fc_full_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                     std::vector<float> &Sb, std::vector<float> &ma,
                     std::vector<float> &Sa, std::vector<float> &Sz_fp,
                     int w_pos, int b_pos, int z_pos_in, int z_pos_out, int no,
                     int ni, int B, std::vector<float> &Sz,
                     std::vector<float> &Sz_f)
/* Add diagonal terms to the full covariance matrix.

Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Variance of biases
    Sz_fp: Partial full-covariance matrix of hidden states of current
                layer
    w_pos: Weight position for this layer in the weight vector of network
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
        of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
        of network
    no: Output node
    ni: Input node
    B: Number of batches
    Sz: Diagonal covariance matrix for hidden states
    Sz_f: Full-covariance matrix for hidden states
 */
{
    int col, row, i, k;
    float sum, final_sum;
    for (int j = 0; j < (no * (no + 1) / 2) * B; j++) {
        Sz_f[j] = Sz_fp[j];
    }
    for (row = 0; row < no; row++) {
        for (col = 0; col < B; col++) {
            sum = 0;
            for (i = 0; i < ni; i++) {
                sum += Sw[row * ni + i + w_pos] * Sa[ni * col + i + z_pos_in] +
                       Sw[row * ni + i + w_pos] * ma[ni * col + i + z_pos_in] *
                           ma[ni * col + i + z_pos_in];
            }
            k = no * row - (row * (row - 1)) / 2 + col * (no * (no + 1)) / 2;
            final_sum = sum + Sb[row + b_pos] + Sz_fp[k];
            Sz[col * no + row + z_pos_out] = final_sum;
            Sz_f[k] = final_sum;
        }
    }
}

void partition_fc_mean_var(std::vector<float> &mw, std::vector<float> &Sw,
                           std::vector<float> &mb, std::vector<float> &Sb,
                           std::vector<float> &ma, std::vector<float> &Sa,
                           int w_pos, int b_pos, int z_pos_in, int z_pos_out,
                           int m, int n, int k, int start_idx, int end_idx,
                           std::vector<float> &mz, std::vector<float> &Sz) {
    float ma_tmp;
    float Sa_tmp;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / k;
        int col = i % k;
        float sum_mz = 0.0f;
        float sum_Sz = 0.0f;
        for (int j = 0; j < n; j++) {
            ma_tmp = ma[n * col + j + z_pos_in];
            Sa_tmp = Sa[n * col + j + z_pos_in];
            sum_mz += mw[row * n + j + w_pos] * ma_tmp;
            sum_Sz += (mw[row * n + j + w_pos] * mw[row * n + j + w_pos] +
                       Sw[row * n + j + w_pos]) *
                          Sa_tmp +
                      Sw[row * n + j + w_pos] * ma_tmp * ma_tmp;
        }
        mz[col * m + row + z_pos_out] = sum_mz + mb[row + b_pos];
        Sz[col * m + row + z_pos_out] = sum_Sz + Sb[row + b_pos];
    }
}

void fc_mean_var_multithreading(std::vector<float> &mw, std::vector<float> &Sw,
                                std::vector<float> &mb, std::vector<float> &Sb,
                                std::vector<float> &ma, std::vector<float> &Sa,
                                int w_pos, int b_pos, int z_pos_in,
                                int z_pos_out, int m, int n, int k,
                                std::vector<float> &mz, std::vector<float> &Sz)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    const int tot_ops = m * k;
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
        threads[i] = std::thread(
            partition_fc_mean_var, std::ref(mw), std::ref(Sw), std::ref(mb),
            std::ref(Sb), std::ref(ma), std::ref(Sa), w_pos, b_pos, z_pos_in,
            z_pos_out, m, n, k, start_idx, end_idx, std::ref(mz), std::ref(Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void partition_full_cov(std::vector<float> &mw, std::vector<float> &Sa_f,
                        int w_pos, int no, int ni, int B, int start_idx,
                        int end_idx, std::vector<float> &Sz_fp) {
    int tu, col, row, k;
    float Sa_in;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / ((no * B - 1) % no + 1);
        col = j % ((no * B - 1) % no + 1);
        float sum = 0.0f;
        for (int i = 0; i < ni * ni; i++) {
            if ((i / ni) > (i % ni))  // Upper triangle
            {
                tu = (ni * (i % ni) - (((i % ni) * (i % ni + 1)) / 2) + i / ni);
            } else {
                tu = (ni * (i / ni) - (((i / ni) * (i / ni + 1)) / 2) + i % ni);
            }
            Sa_in = Sa_f[tu + (row / no) * (ni * (ni + 1)) / 2];
            sum += mw[i % ni + (row % no) * ni + w_pos] * Sa_in *
                   mw[i / ni + (col % no) * ni + w_pos];
        }
        k = no * col - ((col * (col + 1)) / 2) + row % no +
            (row / no) * (((no + 1) * no) / 2);
        Sz_fp[k] = sum;
    }
}

void fc_full_cov_multithreading(std::vector<float> &mw,
                                std::vector<float> &Sa_f, int w_pos, int no,
                                int ni, int B, std::vector<float> &Sz_fp) {
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
            std::thread(partition_full_cov, std::ref(mw), std::ref(Sa_f), w_pos,
                        no, ni, B, start_idx, end_idx, std::ref(Sz_fp));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void partition_fc_full_var(std::vector<float> &mw, std::vector<float> &Sw,
                           std::vector<float> &Sb, std::vector<float> &ma,
                           std::vector<float> &Sa, std::vector<float> &Sz_fp,
                           int w_pos, int b_pos, int z_pos_in, int z_pos_out,
                           int no, int ni, int B, int start_idx, int end_idx,
                           std::vector<float> &Sz, std::vector<float> &Sz_f) {
    int col, row, i, k;
    float final_sum;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / B;
        col = j % B;
        float sum = 0.0f;
        for (i = 0; i < ni; i++) {
            sum += Sw[row * ni + i + w_pos] * Sa[ni * col + i + z_pos_in] +
                   Sw[row * ni + i + w_pos] * ma[ni * col + i + z_pos_in] *
                       ma[ni * col + i + z_pos_in];
        }
        k = no * row - (row * (row - 1)) / 2 + col * (no * (no + 1)) / 2;
        final_sum = sum + Sb[row + b_pos] + Sz_fp[k];
        Sz[col * no + row + z_pos_out] = final_sum;
        Sz_f[k] = final_sum;
    }
}

void fc_full_var_multithreading(std::vector<float> &mw, std::vector<float> &Sw,
                                std::vector<float> &Sb, std::vector<float> &ma,
                                std::vector<float> &Sa,
                                std::vector<float> &Sz_fp, int w_pos, int b_pos,
                                int z_pos_in, int z_pos_out, int no, int ni,
                                int B, std::vector<float> &Sz,
                                std::vector<float> &Sz_f) {
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    const int tot_ops = no * B;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;

    for (int j = 0; j < (no * (no + 1) / 2) * B; j++) {
        Sz_f[j] = Sz_fp[j];
    }
    std::thread threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(partition_fc_full_var, std::ref(mw),
                                 std::ref(Sw), std::ref(Sb), std::ref(ma),
                                 std::ref(Sa), std::ref(Sz_fp), w_pos, b_pos,
                                 z_pos_in, z_pos_out, no, ni, B, start_idx,
                                 end_idx, std::ref(Sz), std::ref(Sz_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ACTIVATION
////////////////////////////////////////////////////////////////////////////////
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

void partition_no_act_mean_var(std::vector<float> &mz, std::vector<float> &Sz,
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
                                    std::vector<float> &ma,
                                    std::vector<float> &J,
                                    std::vector<float> &Sa)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_no_act_mean_var, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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

void partition_tanh_mean_var(std::vector<float> &mz, std::vector<float> &Sz,
                             int zpos, int start_idx, int end_idx,
                             std::vector<float> &ma, std::vector<float> &J,
                             std::vector<float> &Sa) {
    int col;
    float tmp = 0;
    for (col = start_idx; col < end_idx; col++) {
        tmp = tanhf(mz[col + zpos]);
        ma[col + zpos] = tmp;
        J[col + zpos] = (1 - powf(tmp, 2));
        Sa[col + zpos] =
            (1 - powf(tmp, 2)) * Sz[col + zpos] * (1 - powf(tmp, 2));
    }
}

void tanh_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_tanh_mean_var, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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

void partition_sigmoid_mean_var(std::vector<float> &mz, std::vector<float> &Sz,
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
                                     std::vector<float> &ma,
                                     std::vector<float> &J,
                                     std::vector<float> &Sa)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_sigmoid_mean_var, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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

void partition_relu_mean_var(std::vector<float> &mz, std::vector<float> &Sz,
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
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_relu_mean_var, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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

void partition_softplus_mean_var(std::vector<float> &mz, std::vector<float> &Sz,
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
                                      std::vector<float> &ma,
                                      std::vector<float> &J,
                                      std::vector<float> &Sa)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_softplus_mean_var, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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

void partition_leakyrelu_mean_var(std::vector<float> &mz,
                                  std::vector<float> &Sz, float alpha, int zpos,
                                  int start_idx, int end_idx,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa) {
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

void leakyrelu_mean_var_multithreading(std::vector<float> &mz,
                                       std::vector<float> &Sz, float alpha,
                                       int z_pos, int n, std::vector<float> &ma,
                                       std::vector<float> &J,
                                       std::vector<float> &Sa)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_leakyrelu_mean_var, std::ref(mz),
                                 std::ref(Sz), alpha, z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
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

void partition_act_full_cov(std::vector<float> &Sz_f, std::vector<float> &J,
                            int no, int B, int z_pos_out, int start_idx,
                            int end_idx, std::vector<float> &Sa_f) {
    int col, row, idx;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / ((no * B - 1) % no + 1);
        col = j % ((no * B - 1) % no + 1);

        idx = no * col - ((col * (col + 1)) / 2) + row % no +
              (row / no) * (((no + 1) * no) / 2);

        Sa_f[idx] = Sz_f[idx] * J[row % no + (row / no) * no + z_pos_out] *
                    J[col + (row / no) * no + z_pos_out];
    }
}

void act_full_cov_multithreading(std::vector<float> &Sz_f,
                                 std::vector<float> &J, int no, int B,
                                 int z_pos_out, std::vector<float> &Sa_f) {
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
            std::thread(partition_act_full_cov, std::ref(Sz_f), std::ref(J), no,
                        B, z_pos_out, start_idx, end_idx, std::ref(Sa_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void partition_no_act_full_cov(std::vector<float> &Sz_f, int start_idx,
                               int end_idx, std::vector<float> &Sa_f) {
    int col;
    for (col = start_idx; col < end_idx; col++) {
        Sa_f[col] = Sz_f[col];
    }
}

void no_act_full_cov_multithreading(std::vector<float> &Sz_f, int no, int B,
                                    std::vector<float> &Sa_f) {
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
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
        threads[i] = std::thread(partition_no_act_full_cov, std::ref(Sz_f),
                                 start_idx, end_idx, std::ref(Sa_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

//////////////////////////////////////////////////////////////////////
/// INITIALIZE STATES
//////////////////////////////////////////////////////////////////////
void initialize_states_cpu(std::vector<float> &x, std::vector<float> &Sx,
                           std::vector<float> &Sx_f, int ni, int B,
                           NetState &state)
/* Insert input data to network's states

Args:
    x: Input data:
    Sx: Variance of input data i.e. in the common case, Sx=0
    Sx_f: Full covariance of input data i.e. in the common case, Sx_f=0
    state: Network's state
    niB: Number of hidden units x number of batches for input layer
 */
{
    for (int col = 0; col < ni * B; col++) {
        state.mz[col] = x[col];
        state.Sz[col] = Sx[col];
        state.ma[col] = x[col];
        state.Sa[col] = Sx[col];
        state.J[col] = 1;
    }
    if (Sx_f.size() > 0) {
        for (int col = 0; col < (ni * (ni + 1) / 2) * B; col++) {
            state.Sz_f[col] = Sx_f[col];
            state.Sa_f[col] = Sx_f[col];
        }
    }
}

void partition_states_init(std::vector<float> &x, std::vector<float> &Sx,
                           int start_idx, int end_idx, NetState &state)
// TODO*: Decompose state in different vector
{
    for (int col = start_idx; col < end_idx; col++) {
        state.mz[col] = x[col];
        state.Sz[col] = Sx[col];
        state.ma[col] = x[col];
        state.Sa[col] = Sx[col];
        state.J[col] = 1;
    }
}
void initialize_states_multithreading(std::vector<float> &x,
                                      std::vector<float> &Sx, int niB,
                                      NetState &state)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    const int n_batch = niB / NUM_THREADS;
    const int rem_batch = niB % NUM_THREADS;
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
            std::thread(partition_states_init, std::ref(x), std::ref(Sx),
                        start_idx, end_idx, std::ref(state));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

//////////////////////////////////////////////////////////////////////
/// TAGI-FEEDFORWARD PASS
//////////////////////////////////////////////////////////////////////
void feed_forward_cpu(Network &net, Param &theta, IndexOut &idx,
                      NetState &state)
/*
  Update Network's hidden states using TAGI.

  Args:
    net: Network architecture
    theta: Network's weights and biases
    idx: Indices for network e.g. see indices.cpp

  Returns:
    state: Hidden state of network
 */
{
    int ni, no, no_B, z_pos_in, z_pos_out, w_pos_in, b_pos_in;
    int B = net.batch_size;
    for (int j = 1; j < net.layers.size(); j++) {
        no = net.nodes[j];
        ni = net.nodes[j - 1];
        z_pos_out = net.z_pos[j];
        z_pos_in = net.z_pos[j - 1];
        w_pos_in = net.w_pos[j - 1];
        b_pos_in = net.b_pos[j - 1];
        no_B = no * B;

        //**
        // 1: Full connected
        //
        if (net.layers[j] == net.layer_names.fc) {
            if (!net.is_full_cov) {
                if (no * B > 1000 && net.multithreading) {
                    fc_mean_var_multithreading(
                        theta.mw, theta.Sw, theta.mb, theta.Sb, state.ma,
                        state.Sa, w_pos_in, b_pos_in, z_pos_in, z_pos_out, no,
                        ni, B, state.mz, state.Sz);

                } else {
                    fc_mean_cpu(theta.mw, theta.mb, state.ma, w_pos_in,
                                b_pos_in, z_pos_in, z_pos_out, no, ni, B,
                                state.mz);
                    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.ma, state.Sa,
                               w_pos_in, b_pos_in, z_pos_in, z_pos_out, no, ni,
                               B, state.Sz);
                }
            } else {
                if (no * B * no > 1000 && net.multithreading) {
                    fc_mean_var_multithreading(
                        theta.mw, theta.Sw, theta.mb, theta.Sb, state.ma,
                        state.Sa, w_pos_in, b_pos_in, z_pos_in, z_pos_out, no,
                        ni, B, state.mz, state.Sz);

                    fc_full_cov_multithreading(theta.mw, state.Sa_f, w_pos_in,
                                               no, ni, B, state.Sz_fp);

                    fc_full_var_multithreading(
                        theta.mw, theta.Sw, theta.Sb, state.ma, state.Sa,
                        state.Sz_fp, w_pos_in, b_pos_in, z_pos_in, z_pos_out,
                        no, ni, B, state.Sz, state.Sz_f);

                } else {
                    fc_mean_cpu(theta.mw, theta.mb, state.ma, w_pos_in,
                                b_pos_in, z_pos_in, z_pos_out, no, ni, B,
                                state.mz);
                    fc_full_cov_cpu(theta.mw, state.Sa_f, w_pos_in, no, ni, B,
                                    state.Sz_fp);

                    fc_full_var_cpu(theta.mw, theta.Sw, theta.Sb, state.ma,
                                    state.Sa, state.Sz_fp, w_pos_in, b_pos_in,
                                    z_pos_in, z_pos_out, no, ni, B, state.Sz,
                                    state.Sz_f);
                }
            }
        }

        //**
        // Activation
        //
        if (net.activations[j] == 1)  // tanh
        {
            if (no * B > 1000 && net.multithreading) {
                no_act_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                               no_B, state.ma, state.J,
                                               state.Sa);
            } else {
                no_act_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B,
                                    state.ma, state.J, state.Sa);
            }
        } else if (net.activations[j] == 2)  // sigmoid
        {
            if (no * B > 1000 && net.multithreading) {
                tanh_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                             no_B, state.ma, state.J, state.Sa);

            } else {
                tanh_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                  state.J, state.Sa);
            }
        } else if (net.activations[j] == 4)  // ReLU
        {
            if (no * B > 1000 && net.multithreading) {
                relu_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                             no_B, state.ma, state.J, state.Sa);
            } else {
                relu_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                  state.J, state.Sa);
            }
        } else if (net.activations[j] == 5)  // softplus
        {
            if (no * B > 1000 && net.multithreading) {
                softplus_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                                 no_B, state.ma, state.J,
                                                 state.Sa);

            } else {
                softplus_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B,
                                      state.ma, state.J, state.Sa);
            }
        } else if (net.activations[j] == 6)  // leaky ReLU
        {
            if (no * B > 1000 && net.multithreading) {
                leakyrelu_mean_var_multithreading(state.mz, state.Sz, net.alpha,
                                                  z_pos_out, no_B, state.ma,
                                                  state.J, state.Sa);
            } else {
                leakyrelu_mean_var_cpu(state.mz, state.Sz, net.alpha, z_pos_out,
                                       no_B, state.ma, state.J, state.Sa);
            }
        } else  // no activation
        {
            if (no * B > 1000 && net.multithreading) {
                no_act_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                               no_B, state.ma, state.J,
                                               state.Sa);
            } else {
                no_act_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B,
                                    state.ma, state.J, state.Sa);
            }
        }

        // Full-covariance mode
        if (net.is_full_cov) {
            if (net.activations[j] == 0) {
                if (no * B * no > 1000 && net.multithreading) {
                    no_act_full_cov_multithreading(state.Sz_f, no, B,
                                                   state.Sa_f);
                } else {
                    no_act_full_cov(state.Sz_f, no, B, state.Sa_f);
                }
            } else {
                if (((no * (no + 1) / 2) * B) > 1000 && net.multithreading) {
                    act_full_cov_multithreading(state.Sz_f, state.J, no, B,
                                                z_pos_out, state.Sa_f);
                } else {
                    act_full_cov(state.Sz_f, state.J, no, B, z_pos_out,
                                 state.Sa_f);
                }
            }
        }
    }
}
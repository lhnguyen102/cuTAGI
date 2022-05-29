///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cpp
// Description:  CPU version for forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      May 28, 2022
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
    z_pos_in: Input-hidden-state position for this layer in the weight vector
              of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
               of network
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
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
             of network
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

//////////////////////////////////////////////////////////////////////
/// INITIALIZE STATES
//////////////////////////////////////////////////////////////////////
void initialize_states_cpu(std::vector<float> &x, std::vector<float> &Sx,
                           int niB, NetState &state)
/* Insert input data to network's states

Args:
    x: Input data:
    Sx: Variance of input data i.e. in the common case, Sx=0
    state: Network's state
    niB: Number of hidden units x number of batches for input layer
 */
{
    for (int col = 0; col < niB; col++) {
        state.mz[col] = x[col];
        state.Sz[col] = Sx[col];
        state.ma[col] = x[col];
        state.Sa[col] = Sx[col];
        state.J[col] = 1;
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
            if (no * B < 1000) {
                fc_mean_cpu(theta.mw, theta.mb, state.ma, w_pos_in, b_pos_in,
                            z_pos_in, z_pos_out, no, ni, B, state.mz);
                fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.ma, state.Sa,
                           w_pos_in, b_pos_in, z_pos_in, z_pos_out, no, ni, B,
                           state.Sz);
            } else {
                fc_mean_var_multithreading(
                    theta.mw, theta.Sw, theta.mb, theta.Sb, state.ma, state.Sa,
                    w_pos_in, b_pos_in, z_pos_in, z_pos_out, no, ni, B,
                    state.mz, state.Sz);
            }
        }

        //**
        // Activation
        //
        if (net.activations[j] == 1)  // tanh
        {
            if (no * B < 1000) {
                no_act_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B,
                                    state.ma, state.J, state.Sa);
            } else {
                no_act_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                               no_B, state.ma, state.J,
                                               state.Sa);
            }
        } else if (net.activations[j] == 2)  // sigmoid
        {
            if (no * B < 1000) {
                tanh_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                  state.J, state.Sa);
            } else {
                tanh_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                             no_B, state.ma, state.J, state.Sa);
            }
        } else if (net.activations[j] == 4)  // ReLU
        {
            if (no * B < 1000) {
                relu_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                  state.J, state.Sa);
            } else {
                relu_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                             no_B, state.ma, state.J, state.Sa);
            }
        } else if (net.activations[j] == 5)  // softplus
        {
            if (no * B < 1000) {
                softplus_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B,
                                      state.ma, state.J, state.Sa);
            } else {
                softplus_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                                 no_B, state.ma, state.J,
                                                 state.Sa);
            }
        } else if (net.activations[j] == 6)  // leaky ReLU
        {
            if (no * B < 1000) {
                leakyrelu_mean_var_cpu(state.mz, state.Sz, net.alpha, z_pos_out,
                                       no_B, state.ma, state.J, state.Sa);
            } else {
                leakyrelu_mean_var_multithreading(state.mz, state.Sz, net.alpha,
                                                  z_pos_out, no_B, state.ma,
                                                  state.J, state.Sa);
            }
        } else  // no activation
        {
            if (no * B < 1000) {
                no_act_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B,
                                    state.ma, state.J, state.Sa);
            } else {
                no_act_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                               no_B, state.ma, state.J,
                                               state.Sa);
            }
        }
    }
}
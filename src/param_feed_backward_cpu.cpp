///////////////////////////////////////////////////////////////////////////
// File:         param_feed_backward_cpu.cpp
// Description:  CPU version for backward pass for parametes
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      August 17, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////

#include "../include/param_feed_backward_cpu.h"

////////////////////////////////////////////////////////////////////////////////
/// FULL-CONNECTED
////////////////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMwz
void fc_delta_mw(std::vector<float> &Sw, std::vector<float> &ma,
                 std::vector<float> &delta_m, int w_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_mw)
/* Compute update quantities for the mean of weights for full-connected layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
    w_pos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    delta_mw: Updated quantities for the mean of weights
 */
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += ma[m * i + row + z_pos_in] *
                       delta_m[col + k * i + z_pos_out];
            }
            delta_mw[col * m + row + w_pos] = sum * Sw[col * m + row + w_pos];
        }
    }
}

// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
void fc_delta_Sw(std::vector<float> &Sw, std::vector<float> &ma,
                 std::vector<float> &delta_S, int w_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_Sw)
/* Compute update quantities for the variance of weights for full-connected
layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    delta_S: Inovation vector for variance i.e (S_observation - S_prediction)
    w_pos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaSw: Updated quantities for the variance of weights
*/
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += ma[m * i + row + z_pos_in] * ma[m * i + row + z_pos_in] *
                       delta_S[col + k * i + z_pos_out];
            }
            delta_Sw[col * m + row + w_pos] =
                sum * Sw[col * m + row + w_pos] * Sw[col * m + row + w_pos];
        }
    }
}

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
void fc_delta_mb(std::vector<float> &C_bz, std::vector<float> &delta_m,
                 int b_pos, int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_mb)
/* Compute update quantities for the mean of biases for full-connected layer.

Args:
    C_bz: Covariance b|Z+
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaMb: Updated quantities for the mean of biases
*/
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += delta_m[m * i + row + z_pos_out];
            }
            delta_mb[col * m + row + b_pos] = sum * C_bz[col * m + row + b_pos];
        }
    }
}

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
void fc_delta_Sb(std::vector<float> &C_bz, std::vector<float> &delta_S,
                 int b_pos, int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_Sb)
/* Compute update quantities for the variance of biases for full-connected
layer.

Args:
    C_bz: Covariance b|Z+
    delta_S: Inovation vector for variance i.e. (S_observation - S_prediction)
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
    of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaSb: Updated quantities for the variance of biases
*/
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += delta_S[m * i + row + z_pos_out];
            }
            delta_Sb[col * m + row + b_pos] =
                sum * C_bz[col * m + row + b_pos] * C_bz[col * m + row + b_pos];
        }
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void fc_delta_w_worker(std::vector<float> &Sw, std::vector<float> &ma,
                       std::vector<float> &delta_m, std::vector<float> &delta_S,
                       int w_pos, int z_pos_in, int z_pos_out, int m, int n,
                       int k, int start_idx, int end_idx,
                       std::vector<float> &delta_mw,
                       std::vector<float> &delta_Sw) {
    for (int j = start_idx; j < end_idx; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mw = 0.0f;
        float sum_Sw = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_mw +=
                ma[m * i + row + z_pos_in] * delta_m[col + k * i + z_pos_out];
            sum_Sw += ma[m * i + row + z_pos_in] * ma[m * i + row + z_pos_in] *
                      delta_S[col + k * i + z_pos_out];
        }
        delta_mw[col * m + row + w_pos] = sum_mw * Sw[col * m + row + w_pos];
        delta_Sw[col * m + row + w_pos] =
            sum_Sw * Sw[col * m + row + w_pos] * Sw[col * m + row + w_pos];
    }
}

void fc_delta_w_multithreading(std::vector<float> &Sw, std::vector<float> &ma,
                               std::vector<float> &delta_m,
                               std::vector<float> &delta_S, int w_pos,
                               int z_pos_in, int z_pos_out, int m, int n, int k,
                               unsigned int NUM_THREADS,
                               std::vector<float> &delta_mw,
                               std::vector<float> &delta_Sw)

{
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
            fc_delta_w_worker, std::ref(Sw), std::ref(ma), std::ref(delta_m),
            std::ref(delta_S), w_pos, z_pos_in, z_pos_out, m, n, k, start_idx,
            end_idx, std::ref(delta_mw), std::ref(delta_Sw));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void fc_delta_b_worker(std::vector<float> &C_bz, std::vector<float> &delta_m,
                       std::vector<float> &delta_S, int b_pos, int z_pos_out,
                       int m, int n, int k, int start_idx, int end_idx,
                       std::vector<float> &delta_mb,
                       std::vector<float> &delta_Sb)

{
    for (int j = start_idx; j < end_idx; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mb = 0.0f;
        float sum_Sb = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_mb += delta_m[m * i + row + z_pos_out];
            sum_Sb += delta_S[m * i + row + z_pos_out];
        }
        delta_mb[col * m + row + b_pos] = sum_mb * C_bz[col * m + row + b_pos];
        delta_Sb[col * m + row + b_pos] =
            sum_Sb * C_bz[col * m + row + b_pos] * C_bz[col * m + row + b_pos];
    }
}

void fc_delta_b_multithreading(std::vector<float> &C_bz,
                               std::vector<float> &delta_m,
                               std::vector<float> &delta_S, int b_pos,
                               int z_pos_out, int m, int n, int k,
                               unsigned int NUM_THREADS,
                               std::vector<float> &delta_mb,
                               std::vector<float> &delta_Sb)

{
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
        threads[i] =
            std::thread(fc_delta_b_worker, std::ref(C_bz), std::ref(delta_m),
                        std::ref(delta_S), b_pos, z_pos_out, m, n, k, start_idx,
                        end_idx, std::ref(delta_mb), std::ref(delta_Sb));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

///////////////////////////////////////////////////////////////////
/// PARAMETER BACKWARD PASS
///////////////////////////////////////////////////////////////////
void param_backward_cpu(Network &net, Param &theta, NetState &state,
                        DeltaState &d_state, IndexOut &idx, DeltaParam &d_theta)
/*Compute updated quantities for weights and biases using TAGI.

Args:
    net: Network architecture
    theta: Network's weights and biases
    state: Hidden state of Network
    d_state: Difference between prediction and observation
    idx: Indices for network e.g. see indices.cpp

Returns:
    d_theta: Updated quantities for weights and biases.
*/
{
    int no, ni, z_pos_in, z_pos_out, w_pos_in, b_pos_in;
    int B = net.batch_size;
    for (int k = net.layers.size() - 2; k >= 0; k--) {
        no = net.nodes[k + 1];
        ni = net.nodes[k];
        z_pos_out = net.z_pos[k + 1];
        z_pos_in = net.z_pos[k];
        w_pos_in = net.w_pos[k];
        b_pos_in = net.b_pos[k];

        //**
        // 1: Fully connected
        //
        if (net.layers[k + 1] == net.layer_names.fc) {
            if (ni * no > net.min_operations && net.multithreading) {
                // Compute updated quantites for weights
                fc_delta_w_multithreading(
                    theta.Sw, state.ma, d_state.delta_m, d_state.delta_S,
                    w_pos_in, z_pos_in, z_pos_out, ni, B, no,
                    net.num_cpu_threads, d_theta.delta_mw, d_theta.delta_Sw);

                // Compute updated quantities for biases
                fc_delta_b_multithreading(theta.Sb, d_state.delta_m,
                                          d_state.delta_S, b_pos_in, z_pos_out,
                                          no, B, 1, net.num_cpu_threads,
                                          d_theta.delta_mb, d_theta.delta_Sb);
            } else {
                // Compute updated quantites for weights
                fc_delta_mw(theta.Sw, state.ma, d_state.delta_m, w_pos_in,
                            z_pos_in, z_pos_out, ni, B, no, d_theta.delta_mw);
                fc_delta_Sw(theta.Sw, state.ma, d_state.delta_S, w_pos_in,
                            z_pos_in, z_pos_out, ni, B, no, d_theta.delta_Sw);

                // Compute updated quantities for biases
                fc_delta_mb(theta.Sb, d_state.delta_m, b_pos_in, z_pos_out, no,
                            B, 1, d_theta.delta_mb);
                fc_delta_Sb(theta.Sb, d_state.delta_S, b_pos_in, z_pos_out, no,
                            B, 1, d_theta.delta_Sb);
            }
        }
        //**
        // 7: LSTM
        //
        else if (net.layers[k] == net.layer_names.lstm) {
            lstm_parameter_update_cpu(net, state, theta, d_state, d_theta, k);
        }
    }
}
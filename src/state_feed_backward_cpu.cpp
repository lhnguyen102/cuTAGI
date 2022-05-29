///////////////////////////////////////////////////////////////////////////
// File:         state_feed_backward_cpu.cpp
// Description:  CPU version for backward pass for hidden state
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      May 28, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////

#include "../include/state_feed_backward_cpu.h"

void delta_mzSz_with_indices(std::vector<float> &ma, std::vector<float> &Sa,
                             std::vector<float> &Sz, std::vector<float> &J,
                             std::vector<float> &y, std::vector<float> &Sv,
                             std::vector<int> &udIdx, int zpos, int ny, int nye,
                             int n, std::vector<float> &delta_mz,
                             std::vector<float> &delta_Sz)
/* Update output layer based on selected indices.

Args:
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian vector
    y: Observation
    Sv: Observation noise
    udIdx: Selected indiced to update
    delta_mz: Updated quantities for the mean of output's hidden states
    delta_Sz: Updated quantities for the varaince of output's hidden states
    z_pos: Hidden state's position for output layer
    ny: Size of the output layer
    nye: Number of observation to be updated
    n: Number of batches x size of output layer
 */
{
    float zeroPad = 0;
    float tmp = 0;
    int idx = 0;
    for (int col = 0; col < n; col++) {
        idx = udIdx[col] + (col / nye) * ny -
              1;  // minus 1 due to matlab's indexing
        tmp = (J[idx + zpos] * Sz[idx + zpos]) / (Sa[idx + zpos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[idx] = zeroPad;
            delta_Sz[idx] = zeroPad;
        } else {
            delta_mz[idx] = tmp * (y[col] - ma[idx + zpos]);
            delta_Sz[idx] = -tmp * (J[idx + zpos] * Sz[idx + zpos]);
        }
    }
}

void partition_delta_mzSz_with_indices(
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &Sz,
    std::vector<float> &J, std::vector<float> &y, std::vector<float> &Sv,
    std::vector<int> &udIdx, int z_pos, int ny, int nye, int start_idx,
    int end_idx, std::vector<float> &delta_mz, std::vector<float> &delta_Sz)

{
    float zeroPad = 0;
    float tmp = 0;
    int idx = 0;
    for (int col = start_idx; col < end_idx; col++) {
        // minus 1 due to matlab's indexing
        idx = udIdx[col] + (col / nye) * ny - 1;
        tmp = (J[idx + z_pos] * Sz[idx + z_pos]) / (Sa[idx + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[idx] = zeroPad;
            delta_Sz[idx] = zeroPad;
        } else {
            delta_mz[idx] = tmp * (y[col] - ma[idx + z_pos]);
            delta_Sz[idx] = -tmp * (J[idx + z_pos] * Sz[idx + z_pos]);
        }
    }
}

void delta_mzSz_with_indices_multithreading(
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &Sz,
    std::vector<float> &J, std::vector<float> &y, std::vector<float> &Sv,
    std::vector<int> &udIdx, int z_pos, int ny, int nye, int n,
    std::vector<float> &delta_mz, std::vector<float> &delta_Sz)

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
        threads[i] = std::thread(
            partition_delta_mzSz_with_indices, std::ref(ma), std::ref(Sa),
            std::ref(Sz), std::ref(J), std::ref(y), std::ref(Sv),
            std::ref(udIdx), z_pos, ny, nye, start_idx, end_idx,
            std::ref(delta_mz), std::ref(delta_Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void delta_mzSz(std::vector<float> &ma, std::vector<float> &Sa,
                std::vector<float> &Sz, std::vector<float> &J,
                std::vector<float> &y, std::vector<float> &Sv, int z_pos, int n,
                std::vector<float> &delta_mz, std::vector<float> &delta_Sz) {
    float zeroPad = 0;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = (J[col + z_pos] * Sz[col + z_pos]) / (Sa[col + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[col] = zeroPad;
            delta_Sz[col] = zeroPad;
        } else {
            delta_mz[col] = tmp * (y[col] - ma[col + z_pos]);
            delta_Sz[col] = -tmp * (J[col + z_pos] * Sz[col + z_pos]);
        }
    }
}

void partition_delta_mzSz(std::vector<float> &ma, std::vector<float> &Sa,
                          std::vector<float> &Sz, std::vector<float> &J,
                          std::vector<float> &y, std::vector<float> &Sv,
                          int z_pos, int start_idx, int end_idx,
                          std::vector<float> &delta_mz,
                          std::vector<float> &delta_Sz) {
    float zeroPad = 0;
    float tmp = 0;
    for (int col = start_idx; col < end_idx; col++) {
        tmp = (J[col + z_pos] * Sz[col + z_pos]) / (Sa[col + z_pos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mz[col] = zeroPad;
            delta_Sz[col] = zeroPad;
        } else {
            delta_mz[col] = tmp * (y[col] - ma[col + z_pos]);
            delta_Sz[col] = -tmp * (J[col + z_pos] * Sz[col + z_pos]);
        }
    }
}

void delta_mzSz_multithreading(std::vector<float> &ma, std::vector<float> &Sa,
                               std::vector<float> &Sz, std::vector<float> &J,
                               std::vector<float> &y, std::vector<float> &Sv,
                               int z_pos, int n, std::vector<float> &delta_mz,
                               std::vector<float> &delta_Sz) {
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
        threads[i] = std::thread(
            partition_delta_mzSz, std::ref(ma), std::ref(Sa), std::ref(Sz),
            std::ref(J), std::ref(y), std::ref(Sv), z_pos, start_idx, end_idx,
            std::ref(delta_mz), std::ref(delta_Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// INOVATION VECTOR
////////////////////////////////////////////////////////////////////////////////
void inovation_mean(std::vector<float> &Sz, std::vector<float> &delta_mz,
                    int z_pos, int z_delta_pos, int n,
                    std::vector<float> &delta_m)
/* Compute the mean of the inovation vector.

Args:
    Sz: Variance of hidden states
    delta_mz: Updated quantities for the mean of output's hidden states
    z_pos: Hidden state's position for output
    z_delta_pos: Position of the inovation vector for this layer
    n: Number of hidden states for input x number of batches
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
*/
{
    float zeroPad = 0;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = delta_mz[col] / Sz[col + z_pos];
        if (isinf(tmp) || isnan(tmp)) {
            delta_m[col + z_delta_pos] = zeroPad;
        } else {
            delta_m[col + z_delta_pos] = tmp;
        }
    }
}

void inovation_var(std::vector<float> &Sz, std::vector<float> &delta_Sz,
                   int z_pos, int z_delta_pos, int n,
                   std::vector<float> &delta_S)
/* Compute the variance of the inovation vector.

Args:
    Sz: Variance of hidden states
    delta_Sz: Updated quantities for the variance of output's hidden states
    z_pos: Hidden state's position for output
    z_delta_pos: Position of the inovation vector for this layer
    n: Number of hidden states for input x number of batches
    delta_S: Inovation vector for variance i.e. (M_observation - M_prediction)
*/
{
    float zeroPad = 0;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = delta_Sz[col] / Sz[col + z_pos];
        if (isinf(tmp) || isnan(tmp)) {
            delta_S[col + z_delta_pos] = zeroPad;
        } else {
            delta_S[col + z_delta_pos] = tmp / Sz[col + z_pos];
        }
    }
}

void partition_inovation(std::vector<float> &Sz, std::vector<float> &delta_mz,
                         std::vector<float> &delta_Sz, int z_pos,
                         int z_delta_pos, int start_idx, int end_idx,
                         std::vector<float> &delta_m,
                         std::vector<float> &delta_S)

{
    float zeroPad = 0;
    float tmp_mz = 0;
    float tmp_Sz = 0;
    for (int col = start_idx; col < end_idx; col++) {
        tmp_mz = delta_mz[col] / Sz[col + z_pos];
        tmp_Sz = delta_Sz[col] / Sz[col + z_pos];
        if (isinf(tmp_mz) || isnan(tmp_mz) || isinf(tmp_Sz) || isnan(tmp_Sz)) {
            delta_m[col + z_delta_pos] = zeroPad;
            delta_S[col + z_delta_pos] = zeroPad;
        } else {
            delta_m[col + z_delta_pos] = tmp_mz;
            delta_S[col + z_delta_pos] = tmp_Sz / Sz[col + z_pos];
        }
    }
}

void inovation_multithreading(std::vector<float> &Sz,
                              std::vector<float> &delta_mz,
                              std::vector<float> &delta_Sz, int z_pos,
                              int z_delta_pos, int n,
                              std::vector<float> &delta_m,
                              std::vector<float> &delta_S)

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
        threads[i] =
            std::thread(partition_inovation, std::ref(Sz), std::ref(delta_mz),
                        std::ref(delta_Sz), z_pos, z_delta_pos, start_idx,
                        end_idx, std::ref(delta_m), std::ref(delta_S));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// FULL-CONNECTED
////////////////////////////////////////////////////////////////////////////////
void fc_delta_mz(std::vector<float> &mw, std::vector<float> &Sz,
                 std::vector<float> &J, std::vector<float> &delta_m, int w_pos,
                 int z_pos_in, int z_pos_out, int ni, int no, int B,
                 std::vector<float> &delta_mz)
/* Compute the updated quatitites of the mean of the hidden states.

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    J: Jacobian vector
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
    w_pos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    ni: Number of hidden units for input
    B: Number of batches
    no: Number of hidden units for output
    delta_mz: Updated quantities for the mean of output's hidden states
*/
{
    float sum = 0;
    for (int row = 0; row < ni; row++) {
        for (int col = 0; col < B; col++) {
            sum = 0;
            for (int i = 0; i < no; i++) {
                sum += mw[ni * i + row + w_pos] *
                       delta_m[col * no + i + z_pos_out];
            }
            delta_mz[col * ni + row] = sum * Sz[col * ni + row + z_pos_in] *
                                       J[col * ni + row + z_pos_in];
        }
    }
}

void fc_delta_Sz(std::vector<float> &mw, std::vector<float> &Sz,
                 std::vector<float> &J, std::vector<float> &delta_S, int w_pos,
                 int z_pos_in, int z_pos_out, int ni, int no, int B,
                 std::vector<float> &delta_Sz)
/* Compute the updated quatitites for the variance of the hidden states.

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    J: Jacobian vector
    delta_S: Inovation vector for variance i.e. (S_observation - S_prediction)
    wpos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    ni: Number of hidden units for input
    B: Number of batches
    no: Number of hidden units for output
    delta_Sz: Updated quantities for the varaince of output's hidden states
*/
{
    float sum = 0;
    for (int row = 0; row < ni; row++) {
        for (int col = 0; col < B; col++) {
            sum = 0;
            for (int i = 0; i < no; i++) {
                sum += mw[ni * i + row + w_pos] *
                       delta_S[col * no + i + z_pos_out] *
                       mw[ni * i + row + w_pos];
            }
            delta_Sz[col * ni + row] = sum * Sz[col * ni + row + z_pos_in] *
                                       Sz[col * ni + row + z_pos_in] *
                                       J[col * ni + row + z_pos_in] *
                                       J[col * ni + row + z_pos_in];
        }
    }
}

void partition_fc_delta_mzSz(std::vector<float> &mw, std::vector<float> &Sz,
                             std::vector<float> &J, std::vector<float> &delta_m,
                             std::vector<float> &delta_S, int w_pos,
                             int z_pos_in, int z_pos_out, int ni, int no, int B,
                             int start_idx, int end_idx,
                             std::vector<float> &delta_mz,
                             std::vector<float> &delta_Sz)

{
    for (int j = start_idx; j < end_idx; j++) {
        int row = j / B;
        int col = j % B;
        float sum_mz = 0.0f;
        float sum_Sz = 0.0f;
        for (int i = 0; i < no; i++) {
            sum_mz +=
                mw[ni * i + row + w_pos] * delta_m[col * no + i + z_pos_out];

            sum_Sz += mw[ni * i + row + w_pos] *
                      delta_S[col * no + i + z_pos_out] *
                      mw[ni * i + row + w_pos];
        }
        delta_mz[col * ni + row] = sum_mz * Sz[col * ni + row + z_pos_in] *
                                   J[col * ni + row + z_pos_in];
        delta_Sz[col * ni + row] = sum_Sz * Sz[col * ni + row + z_pos_in] *
                                   Sz[col * ni + row + z_pos_in] *
                                   J[col * ni + row + z_pos_in] *
                                   J[col * ni + row + z_pos_in];
    }
}

void fc_delta_mzSz_multithreading(std::vector<float> &mw,
                                  std::vector<float> &Sz, std::vector<float> &J,
                                  std::vector<float> &delta_m,
                                  std::vector<float> &delta_S, int w_pos,
                                  int z_pos_in, int z_pos_out, int ni, int no,
                                  int B, std::vector<float> &delta_mz,
                                  std::vector<float> &delta_Sz)

{
    unsigned int NUM_THREADS = std::thread::hardware_concurrency();
    const int tot_ops = ni * B;
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
            std::thread(partition_fc_delta_mzSz, std::ref(mw), std::ref(Sz),
                        std::ref(J), std::ref(delta_m), std::ref(delta_S),
                        w_pos, z_pos_in, z_pos_out, ni, no, B, start_idx,
                        end_idx, std::ref(delta_mz), std::ref(delta_Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}
///////////////////////////////////////////////////////////////////////////
/// STATE BACKWARD PASS
///////////////////////////////////////////////////////////////////////////
void state_backward_cpu(Network &net, Param &theta, NetState &state,
                        IndexOut &idx, Obs &obs, DeltaState &d_state)
/*Compute the updated quantities for network's hidden states using TAGI.

  Args:
    net: Network architecture
    theta: Network's weights and biases
    state: Hidden state of network
    idx: Indices for network e.g. see indices.cpp
    obs: Observations

  Returns:
    d_state: Updated quantities for network's hidden states.
 */
{
    // Compute updated quantities for the output layer's hidden state
    int n_state_last_layer = net.batch_size * net.nodes.back();
    if (net.is_output_ud) {
        if (!net.is_idx_ud) {
            if (n_state_last_layer < 1000) {
                delta_mzSz(state.ma, state.Sa, state.Sz, state.J, obs.y_batch,
                           obs.V_batch, net.z_pos.back(), n_state_last_layer,
                           d_state.delta_mz, d_state.delta_Sz);
            } else {
                delta_mzSz_multithreading(state.ma, state.Sa, state.Sz, state.J,
                                          obs.y_batch, obs.V_batch,
                                          net.z_pos.back(), n_state_last_layer,
                                          d_state.delta_mz, d_state.delta_Sz);
            }
        } else {
            int n_state_last_layer_e = net.nye * net.batch_size;
            if (n_state_last_layer < 1000) {
                delta_mzSz_with_indices(
                    state.ma, state.Sa, state.Sz, state.J, obs.y_batch,
                    obs.V_batch, obs.idx_ud_batch, net.z_pos.back(),
                    net.nodes.back(), net.nye, n_state_last_layer_e,
                    d_state.delta_mz, d_state.delta_Sz);
            } else {
                delta_mzSz_with_indices_multithreading(
                    state.ma, state.Sa, state.Sz, state.J, obs.y_batch,
                    obs.V_batch, obs.idx_ud_batch, net.z_pos.back(),
                    net.nodes.back(), net.nye, n_state_last_layer_e,
                    d_state.delta_mz, d_state.delta_Sz);
            }
        }
    } else {
        d_state.delta_mz = obs.y_batch;
        d_state.delta_Sz = obs.V_batch;
    }

    // Compute inovation vector
    if (n_state_last_layer < 1000) {
        inovation_mean(state.Sz, d_state.delta_mz, net.z_pos.back(),
                       net.z_pos.back(), n_state_last_layer, d_state.delta_m);
        inovation_var(state.Sz, d_state.delta_Sz, net.z_pos.back(),
                      net.z_pos.back(), n_state_last_layer, d_state.delta_S);
    } else {
        inovation_multithreading(state.Sz, d_state.delta_mz, d_state.delta_Sz,
                                 net.z_pos.back(), net.z_pos.back(),
                                 n_state_last_layer, d_state.delta_m,
                                 d_state.delta_S);
    }

    int no, ni, niB, z_pos_in, z_pos_out, w_pos_in;
    int B = net.batch_size;
    for (int k = net.layers.size() - 1; k >= net.last_backward_layer; k--) {
        no = net.nodes[k + 1];
        ni = net.nodes[k];
        z_pos_out = net.z_pos[k + 1];
        z_pos_in = net.z_pos[k];
        w_pos_in = net.w_pos[k];
        niB = ni * B;
        //**
        // 1: Full connected
        //
        if (net.layers[k + 1] == net.layer_names.fc) {
            if (niB < 1000) {
                fc_delta_mz(theta.mw, state.Sz, state.J, d_state.delta_m,
                            w_pos_in, z_pos_in, z_pos_out, ni, no, B,
                            d_state.delta_mz);
                fc_delta_Sz(theta.mw, state.Sz, state.J, d_state.delta_S,
                            w_pos_in, z_pos_in, z_pos_out, ni, no, B,
                            d_state.delta_Sz);
            } else {
                fc_delta_mzSz_multithreading(
                    theta.mw, state.Sz, state.J, d_state.delta_m,
                    d_state.delta_S, w_pos_in, z_pos_in, z_pos_out, ni, no, B,
                    d_state.delta_mz, d_state.delta_Sz);
            }
        }
        if (niB < 1000) {
            inovation_mean(state.Sz, d_state.delta_mz, z_pos_in, z_pos_in, niB,
                           d_state.delta_m);
            inovation_var(state.Sz, d_state.delta_Sz, z_pos_in, z_pos_in, niB,
                          d_state.delta_S);
        } else {
            inovation_multithreading(state.Sz, d_state.delta_mz,
                                     d_state.delta_Sz, z_pos_in, z_pos_in, niB,
                                     d_state.delta_m, d_state.delta_S);
        }
    }
}
///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_forward_cpu.cpp
// Description:  Long-Short Term Memory (LSTM) forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 03, 2022
// Updated:      August 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/lstm_feed_forward_cpu.h"

void cov_input_cell_states_cpu(std::vector<float> &Sha, std::vector<float> &mw,
                               std::vector<float> &Ji_ga,
                               std::vector<float> &Jc_ga, int w_pos_i, int ni,
                               int no, int n_seq, int B,
                               std::vector<float> &Ci_c)
/*Compute covariance between input gates and cell states*/
{
    float sum;
    int k;
    for (int i = 0; i < no * B; i++) {
        for (int l = 0; l < n_seq; l++) {
            sum = 0;
            for (int j = 0; j < ni * B; j++) {
                k = j % ni + (i % no) * ni;
                sum += Sha[j + l * n_seq] * mw[w_pos_i + k + ni * no] *
                       mw[w_pos_i + k + 2 * ni * no];
            }
            Ci_c[i * n_seq + l] =
                Ji_ga[i * n_seq + l] * sum * Jc_ga[i * n_seq + l];
        }
    }
}

void cell_state_mean_var_cpu(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int no, int n_seq, int B, std::vector<float> &mc,
    std::vector<float> &Sc)
/*Compute cell states for the current state*/
{
    int m;
    for (int i = 0; i < no * B; i++) {
        for (int l = 0; l < n_seq; l++) {
            m = i * n_seq + l;
            mc_ga[m] = mf_ga[m] * mc_prev[m] + mi_ga[m] * mc_ga[m] + Ci_c[m];
            Sc_ga[m] = Sc_prev[m] * mf_ga[m] * mf_ga[m] +
                       Sc_prev[m] * Sf_ga[m] +
                       Sf_ga[m] * mc_prev[m] * mc_prev[m] +
                       Sc_ga[m] * mi_ga[m] * mi_ga[m] + Si_ga[m] * Sc_ga[m] +
                       Si_ga[m] * mc_ga[m] * mc_ga[m] + powf(Ci_c[m], 2) +
                       2 * Ci_c[m] * mi_ga[m] * mc_ga[m];
        }
    }
}

void cov_output_tanh_cell_states_cpu(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &mc_a,
    std::vector<float> &Sc_a, std::vector<float> &Jc_a,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int z_pos_o,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no,
    int n_seq, int B, std::vector<float> &Co_tanh_c)
/*Compute convariance between output gate and tanh(cell states)*/
{
    float sum_fo, sum_io, sum_oc;
    int k, m;
    for (int i = 0; i < no * B; i++) {
        for (int l = 0; l < n_seq; l++) {
            sum_fo = 0;
            sum_io = 0;
            sum_oc = 0;
            for (int j = 0; j < ni * B; j++) {
                k = j % ni + (i % no) * ni;
                sum_fo +=
                    Sha[j + l * n_seq] * mw[w_pos_f + k] * mw[w_pos_o + k];
                sum_io +=
                    Sha[j + l * n_seq] * mw[w_pos_i + k] * mw[w_pos_o + k];
                sum_oc +=
                    Sha[j + l * n_seq] * mw[w_pos_c + k] * mw[w_pos_o + k];
            }
            m = i * n_seq + l;
            Co_tanh_c[m] =
                Jc_a[m] * Jo_ga[m] * sum_fo * Jf_ga[m] * mc_prev[m + z_pos_o] +
                Jc_a[m] * sum_io * Ji_ga[m] * mc_ga[m] +
                Jc_a[m] * sum_oc * Jc_ga[m] * mi_ga[m];
        }
    }
}

void hidden_state_mean_var_lstm_cpu(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int z_pos_o, int no, int n_seq, int B,
    std::vector<float> &mz, std::vector<float> &Sz)
/*Compute mean and variance for hidden states of the LSTM layer*/
{
    int m;
    for (int i = 0; i < no * B; i++) {
        for (int l = 0; l < n_seq; l++) {
            m = i * n_seq + l;
            mz[m + z_pos_o] = mo_ga[m] * mc_a[m] + Co_tanh_c[m];
            Sz[m + z_pos_o] =
                Sc_a[m] * mo_ga[m] * mo_ga[m] + Sc_a[m] * So_ga[m] +
                So_ga[m] * mc_a[m] * mc_a[m] + powf(Co_tanh_c[m], 2) +
                2 * Co_tanh_c[m] * mo_ga[m] * mc_a[m];
        }
    }
}

void cat_states_and_activations(std::vector<float> &a, std::vector<float> &b,
                                int n, int m, int z_pos_a, int z_pos_b,
                                std::vector<float> &c)
/*Concatenate two vectors*/
{
    for (int i = 0; i < n; i++) {
        c[i] = a[i + z_pos_a];
    }

    for (int j = 0; j < m; j++) {
        c[j + n] = b[j + z_pos_b];
    }
}

void lstm_mean_var_cpu(Network &net, NetState &state, Param &theta, int l)
/*Steps for computing hiiden states mean and covariance for the lstm layer*/
{
    // Initialization
    int ni = net.nodes[l];
    int no = net.nodes[l + 1];
    int z_pos_i = net.z_pos[l];
    int z_pos_o = net.z_pos[l + 1];
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    int z_pos_ga = 0;
    int no_b_seq = no * net.batch_size * net.num_seq;

    // Concatenate the hidden states from the previous time step and activations
    // from the previous layer
    cat_states_and_activations(state.ma, state.lstm_state.mh_prev, ni, no,
                               z_pos_i, z_pos_o, state.lstm_state.mha);
    cat_states_and_activations(state.Sa, state.lstm_state.Sh_prev, ni, no,
                               z_pos_i, z_pos_o, state.lstm_state.Sha);

    // Forget gate
    w_pos_f = net.w_pos[l];
    b_pos_f = net.b_pos[l];
    fc_mean_cpu(theta.mw, theta.mb, state.lstm_state.mha, w_pos_f, b_pos_f,
                z_pos_ga, z_pos_ga, no, ni, net.batch_size,
                state.lstm_state.mf_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm_state.mha,
               state.lstm_state.Sha, w_pos_f, b_pos_f, z_pos_ga, z_pos_ga, no,
               ni, net.batch_size, state.lstm_state.Sf_ga);

    sigmoid_mean_var_cpu(state.lstm_state.mf_ga, state.lstm_state.Sf_ga,
                         z_pos_ga, no_b_seq, state.lstm_state.mf_ga,
                         state.lstm_state.Jf_ga, state.lstm_state.Sf_ga);

    // Input gate
    w_pos_i = net.w_pos[l] + ni * no;
    b_pos_i = net.b_pos[l] + ni * no;
    fc_mean_cpu(theta.mw, theta.mb, state.lstm_state.mha, w_pos_i, b_pos_i,
                z_pos_ga, z_pos_ga, no, ni, net.batch_size,
                state.lstm_state.mi_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm_state.mha,
               state.lstm_state.Sha, w_pos_i, b_pos_i, z_pos_ga, z_pos_ga, no,
               ni, net.batch_size, state.lstm_state.Si_ga);

    sigmoid_mean_var_cpu(state.lstm_state.mi_ga, state.lstm_state.Si_ga,
                         z_pos_ga, no_b_seq, state.lstm_state.mi_ga,
                         state.lstm_state.Ji_ga, state.lstm_state.Si_ga);

    // Cell state gate
    w_pos_c = net.w_pos[l] + 2 * ni * no;
    b_pos_c = net.b_pos[l] + 2 * ni * no;
    fc_mean_cpu(theta.mw, theta.mb, state.lstm_state.mha, w_pos_c, b_pos_c,
                z_pos_ga, z_pos_ga, no, ni, net.batch_size,
                state.lstm_state.mc_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm_state.mha,
               state.lstm_state.Sha, w_pos_c, b_pos_c, z_pos_ga, z_pos_ga, no,
               ni, net.batch_size, state.lstm_state.Sc_ga);

    tanh_mean_var_cpu(state.lstm_state.mc_ga, state.lstm_state.Sc_ga, z_pos_ga,
                      no_b_seq, state.lstm_state.mc_ga, state.lstm_state.Jc_ga,
                      state.lstm_state.Sc_ga);

    // Output gate
    w_pos_o = net.w_pos[l] + 3 * ni * no;
    b_pos_o = net.b_pos[l] + 3 * ni * no;
    fc_mean_cpu(theta.mw, theta.mb, state.lstm_state.mha, w_pos_o, b_pos_o,
                z_pos_ga, z_pos_ga, no, ni, net.batch_size,
                state.lstm_state.mo_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm_state.mha,
               state.lstm_state.Sha, w_pos_o, b_pos_o, z_pos_ga, z_pos_ga, no,
               ni, net.batch_size, state.lstm_state.So_ga);

    sigmoid_mean_var_cpu(state.lstm_state.mo_ga, state.lstm_state.So_ga,
                         z_pos_ga, no_b_seq, state.lstm_state.mo_ga,
                         state.lstm_state.Jo_ga, state.lstm_state.So_ga);

    // Cov(input gate, cell state gate)
    cov_input_cell_states_cpu(state.lstm_state.Sha, theta.mw,
                              state.lstm_state.Ji_ga, state.lstm_state.Jc_ga,
                              w_pos_i, ni, no, net.num_seq, net.batch_size,
                              state.lstm_state.Ci_c);

    // Mean and variance for the current cell states
    cell_state_mean_var_cpu(
        state.lstm_state.mf_ga, state.lstm_state.Sf_ga, state.lstm_state.mi_ga,
        state.lstm_state.Si_ga, state.lstm_state.mc_ga, state.lstm_state.Sc_ga,
        state.lstm_state.mc_prev, state.lstm_state.Sc_prev,
        state.lstm_state.Ci_c, no, net.num_seq, net.batch_size,
        state.lstm_state.mc, state.lstm_state.Sc);

    tanh_mean_var_cpu(state.lstm_state.mc, state.lstm_state.Sc, z_pos_ga,
                      no_b_seq, state.lstm_state.mc, state.lstm_state.Jc,
                      state.lstm_state.Sc);

    // Cov(output gate, tanh(cell states))
    cov_output_tanh_cell_states_cpu(
        theta.mw, state.lstm_state.Sha, state.lstm_state.mc_prev,
        state.lstm_state.mc, state.lstm_state.Sc, state.lstm_state.Jc,
        state.lstm_state.Jf_ga, state.lstm_state.mi_ga, state.lstm_state.Ji_ga,
        state.lstm_state.mc_ga, state.lstm_state.Jc_ga, state.lstm_state.Jo_ga,
        z_pos_o, w_pos_f, w_pos_i, w_pos_c, w_pos_o, ni, no, net.num_seq,
        net.batch_size, state.lstm_state.Co_tanh_c);

    // Mean and variance for hidden states
    hidden_state_mean_var_lstm_cpu(
        state.lstm_state.mo_ga, state.lstm_state.So_ga, state.lstm_state.mc,
        state.lstm_state.Sc, state.lstm_state.Co_tanh_c, z_pos_o, no,
        net.num_seq, net.batch_size, state.mz, state.Sz);
}

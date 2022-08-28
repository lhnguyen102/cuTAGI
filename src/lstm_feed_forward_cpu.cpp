///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_forward_cpu.cpp
// Description:  Long-Short Term Memory (LSTM) forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 03, 2022
// Updated:      August 27, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/lstm_feed_forward_cpu.h"

void cov_input_cell_states_cpu(std::vector<float> &Sha, std::vector<float> &mw,
                               std::vector<float> &Ji_ga,
                               std::vector<float> &Jc_ga, int z_pos_o,
                               int w_pos_i, int w_pos_c, int ni, int no,
                               int n_seq, int B, std::vector<float> &Ci_c)
/*Compute covariance between input gates and cell states. Note that we store the
   hidden state vector as follows: z = [seq1, seq2, ..., seq n] where seq's
   shape = [1, no * B]
*/
{
    float sum;
    int k, i, m;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < n_seq; y++) {
            for (int z = 0; z < no; z++) {
                sum = 0;
                for (int j = 0; j < ni + no; j++) {
                    k = j + z * (ni + no);
                    m = j + y * (ni + no) + x * (n_seq * (ni + no));
                    sum += mw[w_pos_i + k] * Sha[m] * mw[w_pos_c + k];
                }
                i = z + y * no + x * n_seq * no;
                Ci_c[i] = Ji_ga[i + z_pos_o] * sum * Jc_ga[i + z_pos_o];
            }
        }
    }
}

void cell_state_mean_var_cpu(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int z_pos_o, int no, int n_seq, int B,
    std::vector<float> &mc, std::vector<float> &Sc)
/*Compute cell states for the current state*/
{
    int m, k;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < n_seq; y++) {
            for (int z = 0; z < no; z++) {
                k = z + y * no + x * no * n_seq;
                m = k + z_pos_o;
                mc[m] = mf_ga[m] * mc_prev[m] + mi_ga[m] * mc_ga[m] + Ci_c[k];
                Sc[m] = Sc_prev[m] * mf_ga[m] * mf_ga[m] +
                        Sc_prev[m] * Sf_ga[m] +
                        Sf_ga[m] * mc_prev[m] * mc_prev[m] +
                        Sc_ga[m] * mi_ga[m] * mi_ga[m] + Si_ga[m] * Sc_ga[m] +
                        Si_ga[m] * mc_ga[m] * mc_ga[m] + powf(Ci_c[k], 2) +
                        2 * Ci_c[k] * mi_ga[m] * mc_ga[m];
            }
        }
    }
}

void cov_output_tanh_cell_states_cpu(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jc_a,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int z_pos_o_lstm,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no,
    int n_seq, int B, std::vector<float> &Co_tanh_c)
/*Compute convariance between output gates & tanh(cell states)
 */
// TODO. DOUBLE CHECK if prev_mc is hidden state or activation unit
{
    float sum_fo, sum_io, sum_oc;
    int k, m, i;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < n_seq; y++) {
            for (int z = 0; z < no; z++) {
                sum_fo = 0;
                sum_io = 0;
                sum_oc = 0;
                for (int j = 0; j < ni; j++) {
                    k = j + z * (ni + no);
                    m = j + y * (ni + no) + x * (n_seq * (ni + no));
                    sum_fo += mw[w_pos_f + k] * Sha[m] * mw[w_pos_o + k];
                    sum_io += mw[w_pos_i + k] * Sha[m] * mw[w_pos_o + k];
                    sum_oc += mw[w_pos_c + k] * Sha[m] * mw[w_pos_o + k];
                }
                i = z + y * no + x * n_seq * no + z_pos_o_lstm;
                Co_tanh_c[i - z_pos_o_lstm] =
                    Jc_a[i] * (Jo_ga[i] * sum_fo * Jf_ga[i] * mc_prev[i] +
                               Jo_ga[i] * sum_io * Ji_ga[i] * mc_ga[i] +
                               Jo_ga[i] * sum_oc * Jc_ga[i] * mi_ga[i]);
            }
        }
    }
}

void hidden_state_mean_var_lstm_cpu(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int z_pos_o, int z_pos_o_lstm, int no,
    int n_seq, int B, std::vector<float> &mz, std::vector<float> &Sz)
/*Compute mean and variance for hidden states of the LSTM layer*/
{
    int m, k, j;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < n_seq; y++) {
            for (int z = 0; z < no; z++) {
                j = z + y * no + x * no * n_seq;
                m = j + z_pos_o_lstm;
                k = z + y * no + x * no * n_seq + z_pos_o;
                mz[k] = mo_ga[m] * mc_a[m] + Co_tanh_c[j];
                Sz[k] = Sc_a[m] * mo_ga[m] * mo_ga[m] + Sc_a[m] * So_ga[m] +
                        So_ga[m] * mc_a[m] * mc_a[m] + powf(Co_tanh_c[j], 2) +
                        2 * Co_tanh_c[j] * mo_ga[m] * mc_a[m];
            }
        }
    }
}

void to_prev_states(std::vector<float> &curr, std::vector<float> &prev)
/*Transfer data from current cell & hidden to previous cell & hidden states
   which are used for the next step*/
{
    for (int i = 0; i < curr.size(); i++) {
        prev[i] = curr[i];
    }
}

void lstm_state_forward_cpu(Network &net, NetState &state, Param &theta, int l)
/*Steps for computing hiiden states mean and covariance for the lstm layer

NOTE: Weight & bias vector for lstm is defined following
            w = [w_f, w_i, w_c, w_o] & b = [b_f, b_i, b_c, b_o]
*/
{
    // Initialization
    int ni = net.nodes[l - 1];
    int no = net.nodes[l];
    int z_pos_i = net.z_pos[l - 1];
    int z_pos_o = net.z_pos[l];
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    int z_pos_o_lstm = net.z_pos_lstm[l];
    int z_pos_i_lstm = 0;
    int no_b_seq = no * net.batch_size * net.input_seq_len;
    int ni_c = ni + no;

    // Concatenate the hidden states from the previous time step and activations
    // from the previous layer
    cat_activations_and_prev_states(state.ma, state.lstm.mh_prev, ni, no,
                                    z_pos_i, z_pos_o_lstm, state.lstm.mha);
    cat_activations_and_prev_states(state.Sa, state.lstm.Sh_prev, ni, no,
                                    z_pos_i, z_pos_o_lstm, state.lstm.Sha);

    // Forget gate
    w_pos_f = net.w_pos[l - 1];
    b_pos_f = net.b_pos[l - 1];
    fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_f, b_pos_f,
                z_pos_i_lstm, z_pos_o_lstm, no, ni_c, net.batch_size,
                state.lstm.mf_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
               w_pos_f, b_pos_f, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
               net.batch_size, state.lstm.Sf_ga);

    sigmoid_mean_var_cpu(state.lstm.mf_ga, state.lstm.Sf_ga, z_pos_o_lstm,
                         no_b_seq, state.lstm.mf_ga, state.lstm.Jf_ga,
                         state.lstm.Sf_ga);

    // Input gate
    w_pos_i = net.w_pos[l - 1] + ni_c * no;
    b_pos_i = net.b_pos[l - 1] + no;
    fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_i, b_pos_i,
                z_pos_i_lstm, z_pos_o_lstm, no, ni_c, net.batch_size,
                state.lstm.mi_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
               w_pos_i, b_pos_i, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
               net.batch_size, state.lstm.Si_ga);

    sigmoid_mean_var_cpu(state.lstm.mi_ga, state.lstm.Si_ga, z_pos_o_lstm,
                         no_b_seq, state.lstm.mi_ga, state.lstm.Ji_ga,
                         state.lstm.Si_ga);

    // Cell state gate
    w_pos_c = net.w_pos[l - 1] + 2 * ni_c * no;
    b_pos_c = net.b_pos[l - 1] + 2 * no;
    fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_c, b_pos_c,
                z_pos_i_lstm, z_pos_o_lstm, no, ni_c, net.batch_size,
                state.lstm.mc_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
               w_pos_c, b_pos_c, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
               net.batch_size, state.lstm.Sc_ga);

    tanh_mean_var_cpu(state.lstm.mc_ga, state.lstm.Sc_ga, z_pos_o_lstm,
                      no_b_seq, state.lstm.mc_ga, state.lstm.Jc_ga,
                      state.lstm.Sc_ga);

    // Output gate
    w_pos_o = net.w_pos[l - 1] + 3 * ni_c * no;
    b_pos_o = net.b_pos[l - 1] + 3 * no;
    fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_o, b_pos_o,
                z_pos_i_lstm, z_pos_o_lstm, no, ni_c, net.batch_size,
                state.lstm.mo_ga);

    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
               w_pos_o, b_pos_o, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
               net.batch_size, state.lstm.So_ga);

    sigmoid_mean_var_cpu(state.lstm.mo_ga, state.lstm.So_ga, z_pos_o_lstm,
                         no_b_seq, state.lstm.mo_ga, state.lstm.Jo_ga,
                         state.lstm.So_ga);

    // Cov(input gate, cell state gate)
    cov_input_cell_states_cpu(state.lstm.Sha, theta.mw, state.lstm.Ji_ga,
                              state.lstm.Jc_ga, z_pos_o_lstm, w_pos_i, w_pos_c,
                              ni, no, net.input_seq_len, net.batch_size,
                              state.lstm.Ci_c);

    // Mean and variance for the current cell states
    cell_state_mean_var_cpu(
        state.lstm.mf_ga, state.lstm.Sf_ga, state.lstm.mi_ga, state.lstm.Si_ga,
        state.lstm.mc_ga, state.lstm.Sc_ga, state.lstm.mc_prev,
        state.lstm.Sc_prev, state.lstm.Ci_c, z_pos_o_lstm, no,
        net.input_seq_len, net.batch_size, state.lstm.mc, state.lstm.Sc);

    tanh_mean_var_cpu(state.lstm.mc, state.lstm.Sc, z_pos_o_lstm, no_b_seq,
                      state.lstm.mca, state.lstm.Jca, state.lstm.Sca);

    // Cov(output gate, tanh(cell states))
    cov_output_tanh_cell_states_cpu(
        theta.mw, state.lstm.Sha, state.lstm.mc_prev, state.lstm.Jca,
        state.lstm.Jf_ga, state.lstm.mi_ga, state.lstm.Ji_ga, state.lstm.mc_ga,
        state.lstm.Jc_ga, state.lstm.Jo_ga, z_pos_o_lstm, w_pos_f, w_pos_i,
        w_pos_c, w_pos_o, ni, no, net.input_seq_len, net.batch_size,
        state.lstm.Co_tanh_c);

    // Mean and variance for hidden states
    hidden_state_mean_var_lstm_cpu(
        state.lstm.mo_ga, state.lstm.So_ga, state.lstm.mca, state.lstm.Sca,
        state.lstm.Co_tanh_c, z_pos_o, z_pos_o_lstm, no, net.input_seq_len,
        net.batch_size, state.mz, state.Sz);
}

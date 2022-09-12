///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_forward_cpu.cpp
// Description:  Long-Short Term Memory (LSTM) forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 03, 2022
// Updated:      September 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/lstm_feed_forward_cpu.h"

void cov_input_cell_states_cpu(std::vector<float> &Sha, std::vector<float> &mw,
                               std::vector<float> &Ji_ga,
                               std::vector<float> &Jc_ga, int z_pos_o,
                               int w_pos_i, int w_pos_c, int ni, int no,
                               int seq_len, int B, std::vector<float> &Ci_c)
/*Compute covariance between input gates and cell states. Note that we store the
   hidden state vector as follows: z = [seq1, seq2, ..., seq n] where seq's
   shape = [1, no * B]
*/
{
    float sum;
    int k, i, m;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                sum = 0;
                for (int j = 0; j < ni + no; j++) {
                    k = j + z * (ni + no);
                    m = j + y * (ni + no) + x * (seq_len * (ni + no));
                    sum += mw[w_pos_i + k] * Sha[m] * mw[w_pos_c + k];
                }
                i = z + y * no + x * seq_len * no;
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
    std::vector<float> &Ci_c, int z_pos_o, int no, int seq_len, int B,
    std::vector<float> &mc, std::vector<float> &Sc)
/*Compute cell states for the current state*/
{
    int m, k;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                k = z + y * no + x * no * seq_len;
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
    int seq_len, int B, std::vector<float> &Co_tanh_c)
/*Compute convariance between output gates & tanh(cell states)
 */
// TODO: DOUBLE CHECK if prev_mc is hidden state or activation unit
{
    float sum_fo, sum_io, sum_oc;
    int k, m, i;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                sum_fo = 0;
                sum_io = 0;
                sum_oc = 0;
                for (int j = 0; j < ni; j++) {
                    k = j + z * (ni + no);
                    m = j + y * (ni + no) + x * (seq_len * (ni + no));
                    sum_fo += mw[w_pos_f + k] * Sha[m] * mw[w_pos_o + k];
                    sum_io += mw[w_pos_i + k] * Sha[m] * mw[w_pos_o + k];
                    sum_oc += mw[w_pos_c + k] * Sha[m] * mw[w_pos_o + k];
                }
                i = z + y * no + x * seq_len * no + z_pos_o_lstm;
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
    int seq_len, int B, std::vector<float> &mz, std::vector<float> &Sz)
/*Compute mean and variance for hidden states of the LSTM layer*/
{
    int m, k, j;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                j = z + y * no + x * no * seq_len;
                m = j + z_pos_o_lstm;
                k = z + y * no + x * no * seq_len + z_pos_o;
                mz[k] = mo_ga[m] * mc_a[m] + Co_tanh_c[j];
                Sz[k] = Sc_a[m] * mo_ga[m] * mo_ga[m] + Sc_a[m] * So_ga[m] +
                        So_ga[m] * mc_a[m] * mc_a[m] + powf(Co_tanh_c[j], 2) +
                        2 * Co_tanh_c[j] * mo_ga[m] * mc_a[m];
            }
        }
    }
}

void to_prev_states_cpu(std::vector<float> &curr, int n, int z_pos,
                        int z_pos_lstm, std::vector<float> &prev)
/*Transfer data from current cell & hidden to previous cell & hidden states
   which are used for the next step*/
{
    for (int i = 0; i < n; i++) {
        prev[i + z_pos_lstm] = curr[i + z_pos];
    }
}

void cat_activations_and_prev_states_cpu(std::vector<float> &a,
                                         std::vector<float> &b, int n, int m,
                                         int seq_len, int B, int z_pos_a,
                                         int z_pos_b, std::vector<float> &c)
/*Concatenate two vectors*/
{
    for (int k = 0; k < B; k++) {
        for (int s = 0; s < seq_len; s++) {
            for (int i = 0; i < n; i++) {
                c[i + s * (n + m) + k * (n + m) * seq_len] =
                    a[i + z_pos_a + s * n + k * seq_len * n];
            }

            for (int j = 0; j < m; j++) {
                c[j + n + s * (n + m) + k * (n + m) * seq_len] =
                    b[j + z_pos_b + s * m + k * m * seq_len];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void cov_input_cell_states_worker(std::vector<float> &Sha,
                                  std::vector<float> &mw,
                                  std::vector<float> &Ji_ga,
                                  std::vector<float> &Jc_ga, int z_pos_o,
                                  int w_pos_i, int w_pos_c, int ni, int no,
                                  int seq_len, int B, int start_idx,
                                  int end_idx, std::vector<float> &Ci_c) {
    float sum;
    int k, i, m, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;
        sum = 0;
        for (int j = 0; j < ni + no; j++) {
            k = j + z * (ni + no);
            m = j + y * (ni + no) + x * (seq_len * (ni + no));
            sum += mw[w_pos_i + k] * Sha[m] * mw[w_pos_c + k];
        }
        i = z + y * no + x * seq_len * no;
        Ci_c[i] = Ji_ga[i + z_pos_o] * sum * Jc_ga[i + z_pos_o];
    }
}

void cov_input_cell_states_mp(std::vector<float> &Sha, std::vector<float> &mw,
                              std::vector<float> &Ji_ga,
                              std::vector<float> &Jc_ga, int z_pos_o,
                              int w_pos_i, int w_pos_c, int ni, int no,
                              int seq_len, int B, int NUM_THREADS,
                              std::vector<float> &Ci_c) {
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(cov_input_cell_states_worker, std::ref(Sha),
                                 std::ref(mw), std::ref(Ji_ga), std::ref(Jc_ga),
                                 z_pos_o, w_pos_i, w_pos_c, ni, no, seq_len, B,
                                 start_idx, end_idx, std::ref(Ci_c));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void cell_state_mean_var_worker(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int z_pos_o, int no, int seq_len, int start_idx,
    int end_idx, std::vector<float> &mc, std::vector<float> &Sc) {
    int m, k, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;

        k = z + y * no + x * no * seq_len;
        m = k + z_pos_o;
        mc[m] = mf_ga[m] * mc_prev[m] + mi_ga[m] * mc_ga[m] + Ci_c[k];
        Sc[m] = Sc_prev[m] * mf_ga[m] * mf_ga[m] + Sc_prev[m] * Sf_ga[m] +
                Sf_ga[m] * mc_prev[m] * mc_prev[m] +
                Sc_ga[m] * mi_ga[m] * mi_ga[m] + Si_ga[m] * Sc_ga[m] +
                Si_ga[m] * mc_ga[m] * mc_ga[m] + powf(Ci_c[k], 2) +
                2 * Ci_c[k] * mi_ga[m] * mc_ga[m];
    }
}

void cell_state_mean_var_mp(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int z_pos_o, int no, int seq_len, int B,
    int NUM_THREADS, std::vector<float> &mc, std::vector<float> &Sc) {
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            cell_state_mean_var_worker, std::ref(mf_ga), std::ref(Sf_ga),
            std::ref(mi_ga), std::ref(Si_ga), std::ref(mc_ga), std::ref(Sc_ga),
            std::ref(mc_prev), std::ref(Sc_prev), std::ref(Ci_c), z_pos_o, no,
            seq_len, start_idx, end_idx, std::ref(mc), std::ref(Sc));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void cov_output_tanh_cell_states_worker(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jca,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int z_pos_o_lstm,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no,
    int seq_len, int start_idx, int end_idx, std::vector<float> &Co_tanh_c) {
    float sum_fo, sum_io, sum_oc;
    int k, m, i, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;
        sum_fo = 0;
        sum_io = 0;
        sum_oc = 0;
        for (int j = 0; j < ni; j++) {
            k = j + z * (ni + no);
            m = j + y * (ni + no) + x * (seq_len * (ni + no));
            sum_fo += mw[w_pos_f + k] * Sha[m] * mw[w_pos_o + k];
            sum_io += mw[w_pos_i + k] * Sha[m] * mw[w_pos_o + k];
            sum_oc += mw[w_pos_c + k] * Sha[m] * mw[w_pos_o + k];
        }
        i = z + y * no + x * seq_len * no + z_pos_o_lstm;
        Co_tanh_c[i - z_pos_o_lstm] =
            Jca[i] * (Jo_ga[i] * sum_fo * Jf_ga[i] * mc_prev[i] +
                      Jo_ga[i] * sum_io * Ji_ga[i] * mc_ga[i] +
                      Jo_ga[i] * sum_oc * Jc_ga[i] * mi_ga[i]);
    }
}

void cov_output_tanh_cell_states_mp(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jca,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int z_pos_o_lstm,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no,
    int seq_len, int B, int NUM_THREADS, std::vector<float> &Co_tanh_c) {
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            cov_output_tanh_cell_states_worker, std::ref(mw), std::ref(Sha),
            std::ref(mc_prev), std::ref(Jca), std::ref(Jf_ga), std::ref(mi_ga),
            std::ref(Ji_ga), std::ref(mc_ga), std::ref(Jc_ga), std::ref(Jo_ga),
            z_pos_o_lstm, w_pos_f, w_pos_i, w_pos_c, w_pos_o, ni, no, seq_len,
            start_idx, end_idx, std::ref(Co_tanh_c));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void hidden_state_mean_var_lstm_worker(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int z_pos_o, int z_pos_o_lstm, int no,
    int seq_len, int start_idx, int end_idx, std::vector<float> &mz,
    std::vector<float> &Sz) {
    int m, k, j, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;

        j = z + y * no + x * no * seq_len;
        m = j + z_pos_o_lstm;
        k = z + y * no + x * no * seq_len + z_pos_o;
        mz[k] = mo_ga[m] * mc_a[m] + Co_tanh_c[j];
        Sz[k] = Sc_a[m] * mo_ga[m] * mo_ga[m] + Sc_a[m] * So_ga[m] +
                So_ga[m] * mc_a[m] * mc_a[m] + powf(Co_tanh_c[j], 2) +
                2 * Co_tanh_c[j] * mo_ga[m] * mc_a[m];
    }
}

void hidden_state_mean_var_lstm_mp(std::vector<float> &mo_ga,
                                   std::vector<float> &So_ga,
                                   std::vector<float> &mc_a,
                                   std::vector<float> &Sc_a,
                                   std::vector<float> &Co_tanh_c, int z_pos_o,
                                   int z_pos_o_lstm, int no, int seq_len, int B,
                                   int NUM_THREADS, std::vector<float> &mz,
                                   std::vector<float> &Sz) {
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] =
            std::thread(hidden_state_mean_var_lstm_worker, std::ref(mo_ga),
                        std::ref(So_ga), std::ref(mc_a), std::ref(Sc_a),
                        std::ref(Co_tanh_c), z_pos_o, z_pos_o_lstm, no, seq_len,
                        start_idx, end_idx, std::ref(mz), std::ref(Sz));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void cat_activations_and_prev_states_worker(std::vector<float> &a,
                                            std::vector<float> &b, int n, int m,
                                            int seq_len, int B, int z_pos_a,
                                            int z_pos_b, int start_idx,
                                            int end_idx, std::vector<float> &c)
/*Concatenate two vectors*/
{
    int k, s;
    for (int t = start_idx; t < end_idx; t++) {
        k = t / seq_len;
        s = t % seq_len;

        for (int i = 0; i < n; i++) {
            c[i + s * (n + m) + k * (n + m) * seq_len] =
                a[i + z_pos_a + s * n + k * seq_len * n];
        }

        for (int j = 0; j < m; j++) {
            c[j + n + s * (n + m) + k * (n + m) * seq_len] =
                b[j + z_pos_b + s * m + k * m * seq_len];
        }
    }
}

void cat_activations_and_prev_states_mp(std::vector<float> &a,
                                        std::vector<float> &b, int n, int m,
                                        int seq_len, int B, int z_pos_a,
                                        int z_pos_b, int NUM_THREADS,
                                        std::vector<float> &c) {
    const int tot_ops = B * seq_len;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            cat_activations_and_prev_states_worker, std::ref(a), std::ref(b), n,
            m, seq_len, B, z_pos_a, z_pos_b, start_idx, end_idx, std::ref(c));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void save_prev_states_cpu(Network &net, NetState &state)
/*Save the hidden and cell states of the previous time step*/
{
    for (int j = 1; j < net.layers.size(); j++) {
        if (net.layers[j] == net.layer_names.lstm) {
            int no_b_seq = net.nodes[j] * net.batch_size * net.input_seq_len;
            int z_pos_o_lstm = net.z_pos_lstm[j];
            int z_pos_o = net.z_pos[j];
            to_prev_states_cpu(state.lstm.mc, no_b_seq, z_pos_o_lstm,
                               z_pos_o_lstm, state.lstm.mc_prev);
            to_prev_states_cpu(state.lstm.Sc, no_b_seq, z_pos_o_lstm,
                               z_pos_o_lstm, state.lstm.Sc_prev);

            to_prev_states_cpu(state.mz, no_b_seq, z_pos_o, z_pos_o_lstm,
                               state.lstm.mh_prev);
            to_prev_states_cpu(state.Sz, no_b_seq, z_pos_o, z_pos_o_lstm,
                               state.lstm.Sh_prev);
        }
    }
}

void lstm_state_forward_cpu(Network &net, NetState &state, Param &theta, int l)
/*Steps for computing hidden states mean and covariance for the lstm layer

NOTE: Weight & bias vector for lstm is defined following
            w = [w_f, w_i, w_c, w_o] & b = [b_f, b_i, b_c, b_o]
*/
// TODO: Fix cycling import between feed_forward_cpu and lstm_feed_forward_cpu
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
    int ni_seq = ni * net.input_seq_len;
    int no_seq = no * net.input_seq_len;
    int b_seq = net.batch_size * net.input_seq_len;

    // Forget gate
    w_pos_f = net.w_pos[l - 1];
    b_pos_f = net.b_pos[l - 1];

    w_pos_i = net.w_pos[l - 1] + ni_c * no;
    b_pos_i = net.b_pos[l - 1] + no;

    w_pos_c = net.w_pos[l - 1] + 2 * ni_c * no;
    b_pos_c = net.b_pos[l - 1] + 2 * no;

    w_pos_o = net.w_pos[l - 1] + 3 * ni_c * no;
    b_pos_o = net.b_pos[l - 1] + 3 * no;

    if (net.multithreading && no_b_seq > net.min_operations) {
        // Concatenate the hidden states from the previous time step and
        // activations from the previous layer.
        cat_activations_and_prev_states_mp(state.ma, state.lstm.mh_prev, ni, no,
                                           net.input_seq_len, net.batch_size,
                                           z_pos_i, z_pos_o_lstm,
                                           net.num_cpu_threads, state.lstm.mha);
        cat_activations_and_prev_states_mp(state.Sa, state.lstm.Sh_prev, ni, no,
                                           net.input_seq_len, net.batch_size,
                                           z_pos_i, z_pos_o_lstm,
                                           net.num_cpu_threads, state.lstm.Sha);

        // Forget gate
        fc_mean_var_multithreading(theta.mw, theta.Sw, theta.mb, theta.Sb,
                                   state.lstm.mha, state.lstm.Sha, w_pos_f,
                                   b_pos_f, z_pos_i_lstm, z_pos_o_lstm, no,
                                   ni_c, b_seq, net.num_cpu_threads,
                                   state.lstm.mf_ga, state.lstm.Sf_ga);
        sigmoid_mean_var_multithreading(state.lstm.mf_ga, state.lstm.Sf_ga,
                                        z_pos_o_lstm, no_b_seq,
                                        net.num_cpu_threads, state.lstm.mf_ga,
                                        state.lstm.Jf_ga, state.lstm.Sf_ga);

        // Input gate
        fc_mean_var_multithreading(theta.mw, theta.Sw, theta.mb, theta.Sb,
                                   state.lstm.mha, state.lstm.Sha, w_pos_i,
                                   b_pos_i, z_pos_i_lstm, z_pos_o_lstm, no,
                                   ni_c, b_seq, net.num_cpu_threads,
                                   state.lstm.mi_ga, state.lstm.Si_ga);
        sigmoid_mean_var_multithreading(state.lstm.mi_ga, state.lstm.Si_ga,
                                        z_pos_o_lstm, no_b_seq,
                                        net.num_cpu_threads, state.lstm.mi_ga,
                                        state.lstm.Ji_ga, state.lstm.Si_ga);

        // Cell state gate
        fc_mean_var_multithreading(theta.mw, theta.Sw, theta.mb, theta.Sb,
                                   state.lstm.mha, state.lstm.Sha, w_pos_c,
                                   b_pos_c, z_pos_i_lstm, z_pos_o_lstm, no,
                                   ni_c, b_seq, net.num_cpu_threads,
                                   state.lstm.mc_ga, state.lstm.Sc_ga);
        tanh_mean_var_multithreading(state.lstm.mc_ga, state.lstm.Sc_ga,
                                     z_pos_o_lstm, no_b_seq,
                                     net.num_cpu_threads, state.lstm.mc_ga,
                                     state.lstm.Jc_ga, state.lstm.Sc_ga);

        // Output gate
        fc_mean_var_multithreading(theta.mw, theta.Sw, theta.mb, theta.Sb,
                                   state.lstm.mha, state.lstm.Sha, w_pos_o,
                                   b_pos_o, z_pos_i_lstm, z_pos_o_lstm, no,
                                   ni_c, b_seq, net.num_cpu_threads,
                                   state.lstm.mo_ga, state.lstm.So_ga);
        sigmoid_mean_var_multithreading(state.lstm.mo_ga, state.lstm.So_ga,
                                        z_pos_o_lstm, no_b_seq,
                                        net.num_cpu_threads, state.lstm.mo_ga,
                                        state.lstm.Jo_ga, state.lstm.So_ga);

        // Cov(input gate, cell state gate)
        cov_input_cell_states_mp(
            state.lstm.Sha, theta.mw, state.lstm.Ji_ga, state.lstm.Jc_ga,
            z_pos_o_lstm, w_pos_i, w_pos_c, ni, no, net.input_seq_len,
            net.batch_size, net.num_cpu_threads, state.lstm.Ci_c);

        // Mean and variance for the current cell states
        cell_state_mean_var_mp(
            state.lstm.mf_ga, state.lstm.Sf_ga, state.lstm.mi_ga,
            state.lstm.Si_ga, state.lstm.mc_ga, state.lstm.Sc_ga,
            state.lstm.mc_prev, state.lstm.Sc_prev, state.lstm.Ci_c,
            z_pos_o_lstm, no, net.input_seq_len, net.batch_size,
            net.num_cpu_threads, state.lstm.mc, state.lstm.Sc);
        tanh_mean_var_multithreading(state.lstm.mc, state.lstm.Sc, z_pos_o_lstm,
                                     no_b_seq, net.num_cpu_threads,
                                     state.lstm.mca, state.lstm.Jca,
                                     state.lstm.Sca);

        // Cov(output gate, tanh(cell states))
        cov_output_tanh_cell_states_mp(
            theta.mw, state.lstm.Sha, state.lstm.mc_prev, state.lstm.Jca,
            state.lstm.Jf_ga, state.lstm.mi_ga, state.lstm.Ji_ga,
            state.lstm.mc_ga, state.lstm.Jc_ga, state.lstm.Jo_ga, z_pos_o_lstm,
            w_pos_f, w_pos_i, w_pos_c, w_pos_o, ni, no, net.input_seq_len,
            net.batch_size, net.num_cpu_threads, state.lstm.Co_tanh_c);

        // Mean and variance for hidden states
        hidden_state_mean_var_lstm_mp(
            state.lstm.mo_ga, state.lstm.So_ga, state.lstm.mca, state.lstm.Sca,
            state.lstm.Co_tanh_c, z_pos_o, z_pos_o_lstm, no, net.input_seq_len,
            net.batch_size, net.num_cpu_threads, state.mz, state.Sz);
        int check = 1;

    } else {
        cat_activations_and_prev_states_cpu(
            state.ma, state.lstm.mh_prev, ni, no, net.input_seq_len,
            net.batch_size, z_pos_i, z_pos_o_lstm, state.lstm.mha);
        cat_activations_and_prev_states_cpu(
            state.Sa, state.lstm.Sh_prev, ni, no, net.input_seq_len,
            net.batch_size, z_pos_i, z_pos_o_lstm, state.lstm.Sha);

        fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_f, b_pos_f,
                    z_pos_i_lstm, z_pos_o_lstm, no, ni_c, b_seq,
                    state.lstm.mf_ga);
        fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
                   w_pos_f, b_pos_f, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
                   b_seq, state.lstm.Sf_ga);
        sigmoid_mean_var_cpu(state.lstm.mf_ga, state.lstm.Sf_ga, z_pos_o_lstm,
                             no_b_seq, state.lstm.mf_ga, state.lstm.Jf_ga,
                             state.lstm.Sf_ga);

        fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_i, b_pos_i,
                    z_pos_i_lstm, z_pos_o_lstm, no, ni_c, b_seq,
                    state.lstm.mi_ga);
        fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
                   w_pos_i, b_pos_i, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
                   b_seq, state.lstm.Si_ga);
        sigmoid_mean_var_cpu(state.lstm.mi_ga, state.lstm.Si_ga, z_pos_o_lstm,
                             no_b_seq, state.lstm.mi_ga, state.lstm.Ji_ga,
                             state.lstm.Si_ga);

        fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_c, b_pos_c,
                    z_pos_i_lstm, z_pos_o_lstm, no, ni_c, b_seq,
                    state.lstm.mc_ga);
        fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
                   w_pos_c, b_pos_c, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
                   b_seq, state.lstm.Sc_ga);
        tanh_mean_var_cpu(state.lstm.mc_ga, state.lstm.Sc_ga, z_pos_o_lstm,
                          no_b_seq, state.lstm.mc_ga, state.lstm.Jc_ga,
                          state.lstm.Sc_ga);

        fc_mean_cpu(theta.mw, theta.mb, state.lstm.mha, w_pos_o, b_pos_o,
                    z_pos_i_lstm, z_pos_o_lstm, no, ni_c, b_seq,
                    state.lstm.mo_ga);
        fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.lstm.mha, state.lstm.Sha,
                   w_pos_o, b_pos_o, z_pos_i_lstm, z_pos_o_lstm, no, ni_c,
                   b_seq, state.lstm.So_ga);
        sigmoid_mean_var_cpu(state.lstm.mo_ga, state.lstm.So_ga, z_pos_o_lstm,
                             no_b_seq, state.lstm.mo_ga, state.lstm.Jo_ga,
                             state.lstm.So_ga);

        // Cov(input gate, cell state gate)
        cov_input_cell_states_cpu(state.lstm.Sha, theta.mw, state.lstm.Ji_ga,
                                  state.lstm.Jc_ga, z_pos_o_lstm, w_pos_i,
                                  w_pos_c, ni, no, net.input_seq_len,
                                  net.batch_size, state.lstm.Ci_c);

        // Mean and variance for the current cell states
        cell_state_mean_var_cpu(
            state.lstm.mf_ga, state.lstm.Sf_ga, state.lstm.mi_ga,
            state.lstm.Si_ga, state.lstm.mc_ga, state.lstm.Sc_ga,
            state.lstm.mc_prev, state.lstm.Sc_prev, state.lstm.Ci_c,
            z_pos_o_lstm, no, net.input_seq_len, net.batch_size, state.lstm.mc,
            state.lstm.Sc);

        tanh_mean_var_cpu(state.lstm.mc, state.lstm.Sc, z_pos_o_lstm, no_b_seq,
                          state.lstm.mca, state.lstm.Jca, state.lstm.Sca);

        // Cov(output gate, tanh(cell states))
        cov_output_tanh_cell_states_cpu(
            theta.mw, state.lstm.Sha, state.lstm.mc_prev, state.lstm.Jca,
            state.lstm.Jf_ga, state.lstm.mi_ga, state.lstm.Ji_ga,
            state.lstm.mc_ga, state.lstm.Jc_ga, state.lstm.Jo_ga, z_pos_o_lstm,
            w_pos_f, w_pos_i, w_pos_c, w_pos_o, ni, no, net.input_seq_len,
            net.batch_size, state.lstm.Co_tanh_c);

        // Mean and variance for hidden states
        hidden_state_mean_var_lstm_cpu(
            state.lstm.mo_ga, state.lstm.So_ga, state.lstm.mca, state.lstm.Sca,
            state.lstm.Co_tanh_c, z_pos_o, z_pos_o_lstm, no, net.input_seq_len,
            net.batch_size, state.mz, state.Sz);
        int check = 1;
    }
}

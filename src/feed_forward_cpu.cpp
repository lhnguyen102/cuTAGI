///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cpp
// Description:  CPU version for forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      December 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void fc_mean_var_worker(std::vector<float> &mw, std::vector<float> &Sw,
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
                                unsigned int NUM_THREADS,
                                std::vector<float> &mz, std::vector<float> &Sz)

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
            fc_mean_var_worker, std::ref(mw), std::ref(Sw), std::ref(mb),
            std::ref(Sb), std::ref(ma), std::ref(Sa), w_pos, b_pos, z_pos_in,
            z_pos_out, m, n, k, start_idx, end_idx, std::ref(mz), std::ref(Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void full_cov_worker(std::vector<float> &mw, std::vector<float> &Sa_f,
                     int w_pos, int no, int ni, int B, int start_idx,
                     int end_idx, std::vector<float> &Sz_fp) {
    int tu, col, row, k;
    float Sa_in;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / no;
        col = j % no;
        if (col <= (row % no)) {
            float sum = 0.0f;
            for (int i = 0; i < ni * ni; i++) {
                if ((i / ni) > (i % ni))  // Upper triangle
                {
                    tu = (ni * (i % ni) - (((i % ni) * (i % ni + 1)) / 2) +
                          i / ni);
                } else {
                    tu = (ni * (i / ni) - (((i / ni) * (i / ni + 1)) / 2) +
                          i % ni);
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

void fc_full_cov_multithreading(std::vector<float> &mw,
                                std::vector<float> &Sa_f, int w_pos, int no,
                                int ni, int B, unsigned int NUM_THREADS,
                                std::vector<float> &Sz_fp) {
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
            std::thread(full_cov_worker, std::ref(mw), std::ref(Sa_f), w_pos,
                        no, ni, B, start_idx, end_idx, std::ref(Sz_fp));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void fc_full_var_worker(std::vector<float> &mw, std::vector<float> &Sw,
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
                                int B, unsigned int NUM_THREADS,
                                std::vector<float> &Sz,
                                std::vector<float> &Sz_f) {
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
        threads[i] = std::thread(fc_full_var_worker, std::ref(mw), std::ref(Sw),
                                 std::ref(Sb), std::ref(ma), std::ref(Sa),
                                 std::ref(Sz_fp), w_pos, b_pos, z_pos_in,
                                 z_pos_out, no, ni, B, start_idx, end_idx,
                                 std::ref(Sz), std::ref(Sz_f));
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

void initialize_full_states_cpu(
    std::vector<float> &mz_init, std::vector<float> &Sz_init,
    std::vector<float> &ma_init, std::vector<float> &Sa_init,
    std::vector<float> &J_init, std::vector<float> &mz, std::vector<float> &Sz,
    std::vector<float> &ma, std::vector<float> &Sa, std::vector<float> &J) {
    for (int i = 0; i < mz_init.size(); i++) {
        mz[i] = mz_init[i];
        Sz[i] = Sz_init[i];
        ma[i] = ma_init[i];
        Sa[i] = Sa_init[i];
        J[i] = J_init[i];
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
                                      unsigned int NUM_THREADS, NetState &state)

{
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
/// FEED FORWARD
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
        if (net.layers[j - 1] == net.layer_names.lstm) {
            ni = net.nodes[j - 1] * net.input_seq_len;
        }
        z_pos_out = net.z_pos[j];
        z_pos_in = net.z_pos[j - 1];
        w_pos_in = net.w_pos[j - 1];
        b_pos_in = net.b_pos[j - 1];
        no_B = no * B;

        //**
        // 1: Fully connected
        //
        if (net.layers[j] == net.layer_names.fc) {
            if (!net.is_full_cov) {
                if (no * B > net.min_operations && net.multithreading) {
                    fc_mean_var_multithreading(
                        theta.mw, theta.Sw, theta.mb, theta.Sb, state.ma,
                        state.Sa, w_pos_in, b_pos_in, z_pos_in, z_pos_out, no,
                        ni, B, net.num_cpu_threads, state.mz, state.Sz);

                } else {
                    fc_mean_cpu(theta.mw, theta.mb, state.ma, w_pos_in,
                                b_pos_in, z_pos_in, z_pos_out, no, ni, B,
                                state.mz);
                    fc_var_cpu(theta.mw, theta.Sw, theta.Sb, state.ma, state.Sa,
                               w_pos_in, b_pos_in, z_pos_in, z_pos_out, no, ni,
                               B, state.Sz);
                }
            } else {
                if (no * B * no > net.min_operations && net.multithreading) {
                    fc_mean_var_multithreading(
                        theta.mw, theta.Sw, theta.mb, theta.Sb, state.ma,
                        state.Sa, w_pos_in, b_pos_in, z_pos_in, z_pos_out, no,
                        ni, B, net.num_cpu_threads, state.mz, state.Sz);

                    fc_full_cov_multithreading(theta.mw, state.Sa_f, w_pos_in,
                                               no, ni, B, net.num_cpu_threads,
                                               state.Sz_fp);

                    fc_full_var_multithreading(
                        theta.mw, theta.Sw, theta.Sb, state.ma, state.Sa,
                        state.Sz_fp, w_pos_in, b_pos_in, z_pos_in, z_pos_out,
                        no, ni, B, net.num_cpu_threads, state.Sz, state.Sz_f);

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
        // 7: LSTM
        //
        else if (net.layers[j] == net.layer_names.lstm) {
            no_B = no * B * net.input_seq_len;
            lstm_state_forward_cpu(net, state, theta, j);
        }

        //**
        // Activation
        //
        activate_hidden_states_cpu(net, state, j);

        // Activation derivatives
        if (net.collect_derivative) {
            compute_activation_derivatives_cpu(net, state, j);
        }
    }
    // Split the output layer into output & noise hidden states
    if (net.noise_type.compare("heteros") == 0) {
        // Split hidden state of output layer
        get_output_hidden_states_ni_cpu(state.ma, net.nodes.back(),
                                        net.z_pos.back(),
                                        state.noise_state.ma_mu);
        get_output_hidden_states_ni_cpu(state.Sa, net.nodes.back(),
                                        net.z_pos.back(),
                                        state.noise_state.Sa_mu);
        get_output_hidden_states_ni_cpu(state.Sz, net.nodes.back(),
                                        net.z_pos.back(),
                                        state.noise_state.Sz_mu);
        get_output_hidden_states_ni_cpu(state.J, net.nodes.back(),
                                        net.z_pos.back(),
                                        state.noise_state.J_mu);

        get_noise_hidden_states_cpu(state.ma, net.nodes.back(),
                                    net.z_pos.back(),
                                    state.noise_state.ma_v2b_prior);
        get_noise_hidden_states_cpu(state.Sa, net.nodes.back(),
                                    net.z_pos.back(),
                                    state.noise_state.Sa_v2b_prior);
        get_noise_hidden_states_cpu(state.J, net.nodes.back(), net.z_pos.back(),
                                    state.noise_state.J_v2);

        // Activate observation noise squared using exponential fun for ensuring
        // the positive values
        exp_fun_cpu(state.noise_state.ma_v2b_prior,
                    state.noise_state.Sa_v2b_prior,
                    state.noise_state.ma_v2b_prior,
                    state.noise_state.Sa_v2b_prior, state.noise_state.Cza_v2);

    } else if (net.noise_type.compare("homosce") == 0) {
        // Assign value to the nosie states
        get_output_hidden_states_cpu(state.ma, net.z_pos.back(),
                                     state.noise_state.ma_mu);
        get_output_hidden_states_cpu(state.Sa, net.z_pos.back(),
                                     state.noise_state.Sa_mu);
        get_output_hidden_states_cpu(state.J, net.z_pos.back(),
                                     state.noise_state.J_mu);
        get_output_hidden_states_cpu(state.Sz, net.z_pos.back(),
                                     state.noise_state.Sz_mu);
    } else {
    }
}
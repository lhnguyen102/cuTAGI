///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cpp
// Description:  CPU version for forward pass
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 17, 2022
// Updated:      June 05, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/feed_forward_cpu.h"

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
        // 8: MHA
        //
        else if (net.layers[j] == net.layer_names.mha) {
            self_attention_forward_cpu(net, state, theta, j);
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
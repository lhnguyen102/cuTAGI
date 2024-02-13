///////////////////////////////////////////////////////////////////////////
// File:         param_feed_backward_cpu.cpp
// Description:  CPU version for backward pass for parametes
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      September 10, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////

#include "../include/param_feed_backward_cpu.h"
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////
/// PARAMETER BACKWARD
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
    // Reset delta value vector to zero
    d_theta.reset_zero();
    std::vector<bool> J_bool(state.J.begin(), state.J.end());

    for (int k = net.layers.size() - 2; k >= 0; k--) {
        no = net.nodes[k + 1];
        ni = net.nodes[k];
        // Handle multiple input sequences from LSTM layer
        if (net.layers[k] == net.layer_names.lstm) {
            ni = net.nodes[k] * net.input_seq_len;
        }
        z_pos_out = net.z_pos[k + 1];
        z_pos_in = net.z_pos[k];
        w_pos_in = net.w_pos[k];
        b_pos_in = net.b_pos[k];

        // k-th layer non-zero input idx
        std::vector<int> J_in_idx;
        for(int i = z_pos_in; i < z_pos_in + B * ni; i++) {
            if(state.J[i] != 0) J_in_idx.push_back(i);
        }
        // k-th layer non-zero output idx
        std::vector<int> J_out_idx;
        for(int i = z_pos_out; i < z_pos_out + B * no; i++) {
            if(state.J[i] != 0) J_out_idx.push_back(i);
        }

        /*std::cout << "J_in_idx[" << k << "]: ";
        for (int i = 0; i < J_in_idx.size(); i++) {
            std::cout << J_in_idx[i] << " ";
        }
        std::cout << '\n';
        std::cout << "J_out_idx[" << k << "]: ";
        for (int i = 0; i < J_out_idx.size(); i++) {
            std::cout << J_out_idx[i] << " ";
        }
        std::cout << '\n' << '\n';*/


        /*std::cout << "B * ni = " << B * ni << '\n';
        std::cout << "B * no = " << B * no << '\n';
        std::cout << "J_in_idx.size() = " << J_in_idx.size() << '\n';
        std::cout << "J_out_idx.size() = " << J_out_idx.size() << '\n';
        std::cout << '\n' << '\n';*/

        // Zero-initialize the delta_mw vector
        std::fill(d_theta.delta_mw.begin(),
                    d_theta.delta_mw.begin() + w_pos_in, 0);

        /*
        std::vector<bool> J_in(state.J.begin() + z_pos_in,
                                state.J.begin() + z_pos_in + B * ni);
        std::vector<bool> J_out(state.J.begin() + z_pos_out,
                                 state.J.begin() + z_pos_out + B * no);

        std::cout << "J_out[" << k << "]: ";
        for (int i = 0; i < J_out.size(); i++) {
            std::cout << J_out[i] << " ";
        }
        std::cout << '\n';

        std::cout << "J_in[" << k << "]: ";
        for (int i = 0; i < J_in.size(); i++) {
            std::cout << J_in[i] << " ";
        }
        std::cout << '\n';

        std::cout << "J_in_idx[" << k << "]: ";
        for (int i = 0; i < J_in_idx.size(); i++) {
            std::cout << J_in_idx[i] << " ";
        }
        std::cout << '\n';
        std::cout << "J_out_idx[" << k << "]: ";
        for (int i = 0; i < J_out_idx.size(); i++) {
            std::cout << J_out_idx[i] << " ";
        }
        std::cout << '\n' << '\n';

        std::cout << "J_bool: ";
        for (int i = 0; i < J_bool.size(); i++) {
            std::cout << J_bool[i] << " ";
        }
        std::cout << '\n' << '\n';*/

        //**
        // 1: Fully connected
        //
        if (net.layers[k + 1] == net.layer_names.fc) {
            if (ni * no > net.min_operations && net.multithreading) {
                // Compute updated quantites for weights
                fc_delta_w_multithreading(
                    theta.Sw, state.ma, d_state.delta_m, d_state.delta_S,
                    w_pos_in, z_pos_in, z_pos_out, ni, B, no,
                    net.num_cpu_threads, d_theta.delta_mw, d_theta.delta_Sw,
                    state.J);

                // Compute updated quantities for biases
                fc_delta_b_multithreading(theta.Sb, d_state.delta_m,
                                          d_state.delta_S, b_pos_in, z_pos_in,
                                          z_pos_out, no, B, 1, net.num_cpu_threads,
                                          d_theta.delta_mb, d_theta.delta_Sb);
            } else {
                // Compute updated quantites for weigh ts
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
        else if (net.layers[k + 1] == net.layer_names.lstm) {
            lstm_parameter_update_cpu(net, state, theta, d_state, d_theta, k);
        }
        //**
        // 8: MHA
        //
        else if (net.layers[k + 1] == net.layer_names.mha) {
            update_self_attention_param(net, theta, state, d_state, d_theta, k);
        }
    }
}
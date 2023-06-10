///////////////////////////////////////////////////////////////////////////
// File:         param_feed_backward_cpu.cpp
// Description:  CPU version for backward pass for parametes
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 18, 2022
// Updated:      April 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////

#include "../include/param_feed_backward_cpu.h"

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
///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_backward.cu
// Description:  Long-Short Term Memory (LSTM) state backward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      September 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/lstm_feed_backward.cuh"

__global__ void lstm_delta_mean_var_z(
    float const *Sz, float const *mw, float const *Jf_ga, float const *mi_ga,
    float const *Ji_ga, float const *mc_ga, float const *Jc_ga,
    float const *mo_ga, float const *Jo_ga, float const *mc_prev,
    float const *mca, float const *Jca, float const *delta_m,
    float const *delta_S, int z_pos_i, int z_pos_o, int z_pos_o_lstm,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int no, int ni,
    int seq_len, int B, float *delta_mz, float *delta_Sz)
/*Compute the updated quatitites of the mean of the hidden states for lstm
   layer*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_mf, sum_mi, sum_mc, sum_mo, sum_Sz;
    float Czz_f, Czz_i, Czz_c, Czz_o;
    int k, m, i, x, y;
    if (row < B * seq_len && col < ni) {
        x = row / seq_len;
        y = row % seq_len;

        sum_mf = 0;
        sum_mi = 0;
        sum_mc = 0;
        sum_mo = 0;
        sum_Sz = 0;
        for (int j = 0; j < no; j++) {
            k = j + x * no * seq_len + y * no + z_pos_o_lstm;
            i = j + x * no * seq_len + y * no + z_pos_o;

            // Forget gate
            Czz_f = Jca[k] * mo_ga[k] * Jf_ga[k] *
                    mw[(ni + no) * j + col + w_pos_f] * mc_prev[k];
            sum_mf += Czz_f * delta_m[i];

            // Input gate
            Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                    mw[(ni + no) * j + col + w_pos_i] * mc_ga[k];
            sum_mi += Czz_i * delta_m[i];

            // Cell state gate
            Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                    mw[(ni + no) * j + col + w_pos_c] * mi_ga[k];
            sum_mc += Czz_c * delta_m[i];

            // Output gate
            Czz_o = Jo_ga[k] * mw[(ni + no) * j + col + w_pos_o] * mca[k];
            sum_mo += Czz_o * delta_m[i];
            sum_Sz += powf(Czz_f + Czz_i + Czz_c + Czz_o, 2) * delta_S[i];
        }

        // Updating quantities
        m = x * ni * seq_len + y * ni + col;
        delta_mz[m] = (sum_mf + sum_mi + sum_mc + sum_mo) * Sz[m + z_pos_i];
        delta_Sz[m] = Sz[m + z_pos_i] * sum_Sz * Sz[m + z_pos_i];
    }
}

__global__ void lstm_delta_mean_var_w(
    float const *Sw, float const *mha, float const *Jf_ga, float const *mi_ga,
    float const *Ji_ga, float const *mc_ga, float const *Jc_ga,
    float const *mo_ga, float const *Jo_ga, float const *mc_prev,
    float const *mca, float const *Jc, float const *delta_m,
    float const *delta_S, int z_pos_o, int z_pos_o_lstm, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int no, int ni, int seq_len, int B,
    float *delta_mw, float *delta_Sw)
/*Compute updating quantities of the weight parameters for lstm layer */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, m, l, i, x, y;
    if (row < (ni + no) && col < no) {
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int t = 0; t < B * seq_len; t++) {
            x = t / seq_len;
            y = t % seq_len;

            k = col + y * seq_len + no * seq_len * x + z_pos_o_lstm;
            i = col + y * seq_len + no * seq_len * x + z_pos_o;
            l = row + y * (ni + no) + (ni + no) * seq_len * x;

            // Forget gate
            Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k] * mha[l];
            sum_mf += Cwa_f * delta_m[i];
            sum_Sf += Cwa_f * delta_S[i] * Cwa_f;

            // Input gate
            Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k] * mha[l];
            sum_mi += Cwa_i * delta_m[i];
            sum_Si += Cwa_i * delta_S[i] * Cwa_i;

            // Cell state gate
            Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k] * mha[l];
            sum_mc += Cwa_c * delta_m[i];
            sum_Sc += Cwa_c * delta_S[i] * Cwa_c;

            // Output gate
            Cwa_o = Jo_ga[k] * mca[k] * mha[l];
            sum_mo += Cwa_o * delta_m[i];
            sum_So += Cwa_o * delta_S[i] * Cwa_o;
        }
        // Updating quantities for weights
        m = col * (ni + no) + row;
        delta_mw[m + w_pos_f] = sum_mf * Sw[m + w_pos_f];
        delta_Sw[m + w_pos_f] = Sw[m + w_pos_f] * sum_Sf * Sw[m + w_pos_f];

        delta_mw[m + w_pos_i] = sum_mi * Sw[m + w_pos_i];
        delta_Sw[m + w_pos_i] = Sw[m + w_pos_i] * sum_Si * Sw[m + w_pos_i];

        delta_mw[m + w_pos_c] = sum_mc * Sw[m + w_pos_c];
        delta_Sw[m + w_pos_c] = Sw[m + w_pos_c] * sum_Sc * Sw[m + w_pos_c];

        delta_mw[m + w_pos_o] = sum_mo * Sw[m + w_pos_o];
        delta_Sw[m + w_pos_o] = Sw[m + w_pos_o] * sum_So * Sw[m + w_pos_o];
    }
}

__global__ void lstm_delta_mean_var_b(
    float const *Sb, float const *Jf_ga, float const *mi_ga, float const *Ji_ga,
    float const *mc_ga, float const *Jc_ga, float const *mo_ga,
    float const *Jo_ga, float const *mc_prev, float const *mca, float const *Jc,
    float const *delta_m, float const *delta_S, int z_pos_o, int z_pos_o_lstm,
    int b_pos_f, int b_pos_i, int b_pos_c, int b_pos_o, int no, int seq_len,
    int B, float *delta_mb, float *delta_Sb)
/*Compute updating quantities of the bias for the lstm layer */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, i, x, y;
    if (col < no) {
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int t = 0; t < B * seq_len; t++) {
            x = t / seq_len;
            y = t % seq_len;

            k = col + y * seq_len + no * seq_len * x + z_pos_o_lstm;
            i = col + y * seq_len + no * seq_len * x + z_pos_o;

            // Forget gate
            Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k];
            sum_mf += Cwa_f * delta_m[i];
            sum_Sf += Cwa_f * delta_S[i] * Cwa_f;

            // Input gate
            Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k];
            sum_mi += Cwa_i * delta_m[i];
            sum_Si += Cwa_i * delta_S[i] * Cwa_i;

            // Cell state gate
            Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k];
            sum_mc += Cwa_c * delta_m[i];
            sum_Sc += Cwa_c * delta_S[i] * Cwa_c;

            // Output gate
            Cwa_o = Jo_ga[k] * mca[k];
            sum_mo += Cwa_o * delta_m[i];
            sum_So += Cwa_o * delta_S[i] * Cwa_o;
        }
        // Updating quantities for biases
        delta_mb[col + b_pos_f] = sum_mf * Sb[col + b_pos_f];
        delta_Sb[col + b_pos_f] =
            Sb[col + b_pos_f] * sum_Sf * Sb[col + b_pos_f];

        delta_mb[col + b_pos_i] = sum_mi * Sb[col + b_pos_i];
        delta_Sb[col + b_pos_i] =
            Sb[col + b_pos_i] * sum_Si * Sb[col + b_pos_i];

        delta_mb[col + b_pos_c] = sum_mc * Sb[col + b_pos_c];
        delta_Sb[col + b_pos_c] =
            Sb[col + b_pos_c] * sum_Sc * Sb[col + b_pos_c];

        delta_mb[col + b_pos_o] = sum_mo * Sb[col + b_pos_o];
        delta_Sb[col + b_pos_o] =
            Sb[col + b_pos_o] * sum_So * Sb[col + b_pos_o];
    }
}

void lstm_state_update(Network &net, StateGPU &state, ParamGPU &theta,
                       DeltaStateGPU &d_state, int l)
/*Update lstm's hidden states*/
{
    // Initialization
    int ni = net.nodes[l];
    int no = net.nodes[l + 1];
    int z_pos_i = net.z_pos[l];
    int z_pos_o = net.z_pos[l + 1];
    int z_pos_o_lstm = net.z_pos_lstm[l + 1];
    int w_pos_f, w_pos_i, w_pos_c, w_pos_o;
    int ni_c = ni + no;
    int b_seq = net.batch_size * net.input_seq_len;

    w_pos_f = net.w_pos[l];
    w_pos_i = net.w_pos[l] + ni_c * no;
    w_pos_c = net.w_pos[l] + 2 * ni_c * no;
    w_pos_o = net.w_pos[l] + 3 * ni_c * no;

    // Launch kernel
    int THREADS = net.num_gpu_threads;
    unsigned int gridRow_cov = (b_seq + THREADS - 1) / THREADS;
    unsigned int gridCol_cov = (ni + THREADS - 1) / THREADS;
    dim3 dimGrid_cov(gridCol_cov, gridRow_cov);
    dim3 dimBlock(THREADS, THREADS);

    lstm_delta_mean_var_z<<<dimGrid_cov, dimBlock>>>(
        state.d_Sz, theta.d_mw, state.lstm.d_Jf_ga, state.lstm.d_mi_ga,
        state.lstm.d_Ji_ga, state.lstm.d_mc_ga, state.lstm.d_Jc_ga,
        state.lstm.d_mo_ga, state.lstm.d_Jo_ga, state.lstm.d_mc_prev,
        state.lstm.d_mca, state.lstm.d_Jca, d_state.d_delta_m,
        d_state.d_delta_S, z_pos_i, z_pos_o, z_pos_o_lstm, w_pos_f, w_pos_i,
        w_pos_c, w_pos_o, no, ni, net.input_seq_len, net.batch_size,
        d_state.d_delta_mz, d_state.d_delta_Sz);
}

void lstm_parameter_update(Network &net, StateGPU &state, ParamGPU &theta,
                           DeltaStateGPU &d_state, DeltaParamGPU &d_theta,
                           int l)
/*Update lstm's parameters*/
{
    // Initialization
    int ni = net.nodes[l];
    int no = net.nodes[l + 1];
    int z_pos_i = net.z_pos[l];
    int z_pos_o = net.z_pos[l + 1];
    int z_pos_o_lstm = net.z_pos_lstm[l + 1];
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    int ni_c = ni + no;

    w_pos_f = net.w_pos[l];
    b_pos_f = net.b_pos[l];
    w_pos_i = net.w_pos[l] + ni_c * no;
    b_pos_i = net.b_pos[l] + no;
    w_pos_c = net.w_pos[l] + 2 * ni_c * no;
    b_pos_c = net.b_pos[l] + 2 * no;
    w_pos_o = net.w_pos[l] + 3 * ni_c * no;
    b_pos_o = net.b_pos[l] + 3 * no;

    // Launch kernel
    int THREADS = net.num_gpu_threads;
    unsigned int BLOCKS = (no + THREADS - 1) / THREADS;
    unsigned int gridRow = (ni + no + +THREADS - 1) / THREADS;
    unsigned int gridCol = (no + THREADS - 1) / THREADS;
    dim3 dimGrid(gridCol, gridRow);
    dim3 dimBlock(THREADS, THREADS);

    // Concatenate the hidden states from the previous time step and
    // activations from the previous layer
    unsigned int gridRow_cat = (net.batch_size + THREADS - 1) / THREADS;
    unsigned int gridCol_cat = (net.input_seq_len + THREADS - 1) / THREADS;
    dim3 dimGrid_cat(gridCol_cat, gridRow_cat);
    cat_activations_and_prev_states<<<dimGrid_cat, dimBlock>>>(
        state.d_ma, state.lstm.d_mh_prev, ni, no, net.input_seq_len,
        net.batch_size, z_pos_i, z_pos_o_lstm, state.lstm.d_mha);

    lstm_delta_mean_var_w<<<dimGrid, dimBlock>>>(
        theta.d_Sw, state.lstm.d_mha, state.lstm.d_Jf_ga, state.lstm.d_mi_ga,
        state.lstm.d_Ji_ga, state.lstm.d_mc_ga, state.lstm.d_Jc_ga,
        state.lstm.d_mo_ga, state.lstm.d_Jo_ga, state.lstm.d_mc_prev,
        state.lstm.d_mca, state.lstm.d_Jca, d_state.d_delta_m,
        d_state.d_delta_S, z_pos_o, z_pos_o_lstm, w_pos_f, w_pos_i, w_pos_c,
        w_pos_o, no, ni, net.input_seq_len, net.batch_size, d_theta.d_delta_mw,
        d_theta.d_delta_Sw);

    lstm_delta_mean_var_b<<<BLOCKS, THREADS>>>(
        theta.d_Sb, state.lstm.d_Jf_ga, state.lstm.d_mi_ga, state.lstm.d_Ji_ga,
        state.lstm.d_mc_ga, state.lstm.d_Jc_ga, state.lstm.d_mo_ga,
        state.lstm.d_Jo_ga, state.lstm.d_mc_prev, state.lstm.d_mca,
        state.lstm.d_Jca, d_state.d_delta_m, d_state.d_delta_S, z_pos_o,
        z_pos_o_lstm, b_pos_f, b_pos_i, b_pos_c, b_pos_o, no, net.input_seq_len,
        net.batch_size, d_theta.d_delta_mb, d_theta.d_delta_Sb);
}

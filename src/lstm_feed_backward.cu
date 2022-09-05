///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_backward.cu
// Description:  Long-Short Term Memory (LSTM) state backward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      September 05, 2022
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
    int k, m, l, i, t, x, y;
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
    int k, l, i, t, x, y;
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

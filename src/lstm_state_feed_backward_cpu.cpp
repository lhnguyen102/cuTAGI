///////////////////////////////////////////////////////////////////////////////
// File:         lstm_state_feed_backward_cpu.cpp
// Description:  Long-Short Term Memory (LSTM) state backward pass in TAGI
//               (cpu version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2022
// Updated:      August 11, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/lstm_state_feed_backward_cpu.h"

void lstm_delta_mean_var_z(std::vector<float> &Sz, std::vector<float> &mw,
                           std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
                           std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
                           std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
                           std::vector<float> &Jo_ga,
                           std::vector<float> &mc_prev, std::vector<float> &mc,
                           std::vector<float> &Jc, std::vector<float> &delta_m,
                           std::vector<float> &delta_S, int z_pos_i,
                           int z_pos_o, int w_pos_f, int w_pos_i, int w_pos_c,
                           int w_pos_o, int no, int ni, int n_seq, int B,
                           std::vector<float> &delta_mz,
                           std::vector<float> &delta_Sz)
/*Compute the updated quatitites of the mean of the hidden states for lstm
   layer*/
{
    float sum_mf, sum_Sf, sum_mi, sum_Si, sum_mc, sum_Sc, sum_mo, sum_So;
    float Czz_f, Czz_i, Czz_c, Czz_o;
    int k, m;
    // TODO Forget about number of sequences
    for (int row = 0; row < ni; row++) {
        for (int col = 0; col < B; col++) {
            sum_mf = 0;
            sum_Sf = 0;
            sum_mi = 0;
            sum_Si = 0;
            sum_mc = 0;
            sum_Sc = 0;
            sum_mo = 0;
            sum_So = 0;
            for (int i = 0; i < no; i++) {
                k = col * no + i + z_pos_o;

                // Forget gate
                Czz_f = Jc[k] * mo_ga[k] * Jf_ga[k] *
                        mw[ni * i + row + w_pos_f] * mc_prev[k];
                sum_mf += Czz_f * delta_m[k];
                sum_Sf += Czz_f * delta_S[k] * Czz_f;

                // Input gate
                Czz_i = Jc[k] * mo_ga[k] * Ji_ga[k] *
                        mw[ni * i + row + w_pos_i] * mc_ga[k];
                sum_mi += Czz_i * delta_m[k];
                sum_Si += Czz_i * delta_S[k] * Czz_i;

                // Cell gate
                Czz_c = Jc[k] * mo_ga[k] * Jc_ga[k] *
                        mw[ni * i + row + w_pos_c] * mi_ga[k];
                sum_mc += Czz_c * delta_m[k];
                sum_Sc += Czz_c * delta_S[k] * Czz_c;

                // Output gate
                Czz_o = Jo_ga[k] * mw[ni * i + row + w_pos_o] * mc[k];
                sum_mo += Czz_o * delta_m[k];
                sum_So += Czz_o * delta_S[k] * Czz_o;
            }

            // Updating quantities
            m = col * ni + row + z_pos_i;
            delta_mz[col * ni + row] =
                (sum_mf + sum_mi + sum_mc + sum_mo) * Sz[m];
            delta_Sz[col * ni + row] =
                Sz[m] * (sum_Sf + sum_Si + sum_Sc + sum_So) * Sz[m];
        }
    }
}

void lstm_delta_mean_var_wb(
    std::vector<float> &Sw, std::vector<float> &Sb, std::vector<float> &ma,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
    std::vector<float> &Jo_ga, std::vector<float> &mc_prev,
    std::vector<float> &mc, std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int z_pos_i, int z_pos_o, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int no, int ni, int n_seq, int B,
    std::vector<float> &delta_mw, std::vector<float> &delta_Sw) {}
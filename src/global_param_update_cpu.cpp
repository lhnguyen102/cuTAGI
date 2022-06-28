///////////////////////////////////////////////////////////////////////////////
// File:         global_param_update_cpu.cpp
// Description:  CPU version for global parameter update
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 21, 2021
// Updated:      May 21, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/global_param_update_cpu.h"

void update_weight_cpu(std::vector<float> &delta_mw,
                       std::vector<float> &delta_Sw, int n,
                       std::vector<float> &mw, std::vector<float> &Sw)
/* Update network's weights
 */
{
    for (int col = 0; col < n; col++) {
        mw[col] += delta_mw[col];
        Sw[col] += delta_Sw[col];
    }
}

void update_bias_cpu(std::vector<float> &delta_mb, std::vector<float> &delta_Sb,
                     int n, std::vector<float> &mb, std::vector<float> &Sb)
/* Update network's biases
 */
{
    for (int col = 0; col < n; col++) {
        mb[col] += delta_mb[col];
        Sb[col] += delta_Sb[col];
    }
}

void global_param_update_cpu(DeltaParam &d_theta, int wN, int bN, int wN_sc,
                             int bN_sc, Param &theta)
/* Update network's parameters.
 */
{
    // Common network
    update_weight_cpu(d_theta.delta_mw, d_theta.delta_Sw, wN, theta.mw,
                      theta.Sw);
    update_bias_cpu(d_theta.delta_mb, d_theta.delta_Sb, bN, theta.mb, theta.Sb);

    if (wN_sc > 0) {
        update_weight_cpu(d_theta.delta_mw_sc, d_theta.delta_Sw_sc, wN_sc,
                          theta.mw_sc, theta.Sw_sc);
        update_bias_cpu(d_theta.delta_mb_sc, d_theta.delta_Sb_sc, bN_sc,
                        theta.mb_sc, theta.Sb_sc);
    }
}

///////////////////////////////////////////////////////////////////////////////
// File:         global_param_update_cpu.cpp
// Description:  CPU version for global parameter update
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      May 21, 2021
// Updated:      March 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/global_param_update_cpu.h"
void update_weight_cpu(std::vector<float> &delta_mw,
                       std::vector<float> &delta_Sw, float cap_factor, int n,
                       std::vector<float> &mw, std::vector<float> &Sw)
/* Update network's weights
 */
{
    float delta_m_sign, delta_S_sign, delta_bar;
    for (int col = 0; col < n; col++) {
        delta_m_sign = (delta_mw[col] > 0) - (delta_mw[col] < 0);
        delta_S_sign = (delta_Sw[col] > 0) - (delta_Sw[col] < 0);
        delta_bar = powf(Sw[col], 0.5) / cap_factor;
        mw[col] += delta_m_sign * std::min(std::abs(delta_mw[col]), delta_bar);
        Sw[col] += delta_S_sign * std::min(std::abs(delta_Sw[col]), delta_bar);
    }
}

void update_bias_cpu(std::vector<float> &delta_mb, std::vector<float> &delta_Sb,
                     float cap_factor, int n, std::vector<float> &mb,
                     std::vector<float> &Sb)
/* Update network's biases*/
{
    float delta_m_sign, delta_S_sign, delta_bar;
    for (int col = 0; col < n; col++) {
        delta_m_sign = (delta_mb[col] > 0) - (delta_mb[col] < 0);
        delta_S_sign = (delta_Sb[col] > 0) - (delta_Sb[col] < 0);
        delta_bar = powf(Sb[col], 0.5) / cap_factor;
        mb[col] += delta_m_sign * std::min(std::abs(delta_mb[col]), delta_bar);
        Sb[col] += delta_S_sign * std::min(std::abs(delta_Sb[col]), delta_bar);
    }
}

void global_param_update_cpu(DeltaParam &d_theta, float cap_factor, int wN,
                             int bN, int wN_sc, int bN_sc, Param &theta)
/* Update network's parameters.
 */
{
    // Common network
    update_weight_cpu(d_theta.delta_mw, d_theta.delta_Sw, cap_factor, wN,
                      theta.mw, theta.Sw);
    update_bias_cpu(d_theta.delta_mb, d_theta.delta_Sb, cap_factor, bN,
                    theta.mb, theta.Sb);

    // Residual network
    if (wN_sc > 0) {
        update_weight_cpu(d_theta.delta_mw_sc, d_theta.delta_Sw_sc, cap_factor,
                          wN_sc, theta.mw_sc, theta.Sw_sc);
        update_bias_cpu(d_theta.delta_mb_sc, d_theta.delta_Sb_sc, cap_factor,
                        bN_sc, theta.mb_sc, theta.Sb_sc);
    }
}

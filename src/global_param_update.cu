///////////////////////////////////////////////////////////////////////////////
// File:         globalParamUpdate.cu
// Description:  global parameter update in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 28, 2021
// Updated:      March 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/global_param_update.cuh"

__global__ void update_weight(float const *mw_0, float const *Sw_0,
                              float const *delta_mw, float const *delta_Sw,
                              float cap_factor, int n, float *mw, float *Sw) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float delta_m_sign, delta_S_sign, delta_bar;
    if (col < n) {
        delta_m_sign = (delta_mw[col] > 0) - (delta_mw[col] < 0);
        delta_S_sign = (delta_Sw[col] > 0) - (delta_Sw[col] < 0);
        delta_bar = powf(Sw[col], 0.5) / cap_factor;
        mw[col] =
            mw_0[col] + delta_m_sign * min(fabsf(delta_mw[col]), delta_bar);
        Sw[col] =
            Sw_0[col] + delta_S_sign * min(fabsf(delta_Sw[col]), delta_bar);
    }
}

__global__ void update_bias(float const *mb_0, float const *Sb_0,
                            float const *delta_mb, float const *delta_Sb,
                            float cap_factor, int n, float *mb, float *Sb) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float delta_m_sign, delta_S_sign, delta_bar;
    if (col < n) {
        delta_m_sign = (delta_mb[col] > 0) - (delta_mb[col] < 0);
        delta_S_sign = (delta_Sb[col] > 0) - (delta_Sb[col] < 0);
        delta_bar = powf(Sb[col], 0.5) / cap_factor;
        mb[col] =
            mb_0[col] + delta_m_sign * min(fabsf(delta_mb[col]), delta_bar);
        Sb[col] =
            Sb_0[col] + delta_S_sign * min(fabsf(delta_Sb[col]), delta_bar);
    }
}

void global_param_update(DeltaParamGPU &d_theta, float cap_factor, int wN,
                         int bN, int wN_sc, int bN_sc, int THREADS,
                         ParamGPU &theta) {
    // Launch kernel
    int BLOCKS_W = (wN + THREADS - 1) / THREADS;
    int BLOCKS_B = (bN + THREADS - 1) / THREADS;

    // update weights
    update_weight<<<BLOCKS_W, THREADS>>>(
        theta.d_mw, theta.d_Sw, d_theta.d_delta_mw, d_theta.d_delta_Sw,
        cap_factor, wN, theta.d_mw, theta.d_Sw);
    // update bias
    update_bias<<<BLOCKS_B, THREADS>>>(theta.d_mb, theta.d_Sb,
                                       d_theta.d_delta_mb, d_theta.d_delta_Sb,
                                       cap_factor, bN, theta.d_mb, theta.d_Sb);

    if (wN_sc > 0) {
        int BLOCKS_W_SC = (wN_sc + THREADS - 1) / THREADS;
        int BLOCKS_B_SC = (bN_sc + THREADS - 1) / THREADS;

        // update weights
        update_weight<<<BLOCKS_W_SC, THREADS>>>(
            theta.d_mw_sc, theta.d_Sw_sc, d_theta.d_delta_mw_sc,
            d_theta.d_delta_Sw_sc, cap_factor, wN_sc, theta.d_mw_sc,
            theta.d_Sw_sc);

        // update bias
        update_bias<<<BLOCKS_B_SC, THREADS>>>(
            theta.d_mb_sc, theta.d_Sb_sc, d_theta.d_delta_mb_sc,
            d_theta.d_delta_Sb_sc, cap_factor, bN_sc, theta.d_mb_sc,
            theta.d_Sb_sc);
    }
}

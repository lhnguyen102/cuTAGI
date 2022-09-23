///////////////////////////////////////////////////////////////////////////////
// File:         globalParamUpdate.cu
// Description:  global parameter update in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 28, 2021
// Updated:      June 12, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////
#include "../include/global_param_update.cuh"

__global__ void update_weight(float const *mw_0, float const *Sw_0,
                              float const *deltaMw, float const *deltaSw, int n,
                              float *mw, float *Sw) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        mw[col] = mw_0[col] + deltaMw[col];
        Sw[col] = Sw_0[col] + deltaSw[col];
    }
}

__global__ void update_bias(float const *mb_0, float const *Sb_0,
                            float const *deltaMb, float const *deltaSb, int n,
                            float *mb, float *Sb) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        mb[col] = mb_0[col] + deltaMb[col];
        Sb[col] = Sb_0[col] + deltaSb[col];
    }
}

void globalParamUpdate(DeltaParamGPU &d_theta, int wN, int bN, int wN_sc,
                       int bN_sc, int THREADS, ParamGPU &theta) {
    // Launch kernel
    int BLOCKS_W = (wN + THREADS - 1) / THREADS;
    int BLOCKS_B = (bN + THREADS - 1) / THREADS;

    // update weights
    update_weight<<<BLOCKS_W, THREADS>>>(theta.d_mw, theta.d_Sw,
                                         d_theta.d_delta_mw, d_theta.d_delta_Sw,
                                         wN, theta.d_mw, theta.d_Sw);
    // update bias
    update_bias<<<BLOCKS_B, THREADS>>>(theta.d_mb, theta.d_Sb,
                                       d_theta.d_delta_mb, d_theta.d_delta_Sb,
                                       bN, theta.d_mb, theta.d_Sb);

    if (wN_sc > 0) {
        int BLOCKS_W_SC = (wN_sc + THREADS - 1) / THREADS;
        int BLOCKS_B_SC = (bN_sc + THREADS - 1) / THREADS;

        // update weights
        update_weight<<<BLOCKS_W_SC, THREADS>>>(
            theta.d_mw_sc, theta.d_Sw_sc, d_theta.d_delta_mw_sc,
            d_theta.d_delta_Sw_sc, wN_sc, theta.d_mw_sc, theta.d_Sw_sc);

        // update bias
        update_bias<<<BLOCKS_B_SC, THREADS>>>(
            theta.d_mb_sc, theta.d_Sb_sc, d_theta.d_delta_mb_sc,
            d_theta.d_delta_Sb_sc, bN_sc, theta.d_mb_sc, theta.d_Sb_sc);
    }
}

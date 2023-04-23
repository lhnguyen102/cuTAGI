///////////////////////////////////////////////////////////////////////////////
// File:         param_feed_backward.cuh
// Description:  Header file for paramerer feed backward in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 22, 2022
// Updated:      September 09, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <cuda.h>

#include "data_transfer.cuh"
#include "lstm_feed_backward.cuh"
#include "net_prop.h"
#include "struct_var.h"

////////////////////////////////////////////////////////////////////////////////
/// FULL-CONNECTED
////////////////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMwz
__global__ void fcDeltaMw(float const *Sw, float const *ma, float const *deltaM,
                          int wpos, int zposIn, int zposOut, int m, int n,
                          int k, float *deltaMw);

// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
__global__ void fcDeltaSw(float const *Sw, float const *ma, float const *deltaS,
                          int wpos, int zposIn, int zposOut, int m, int n,
                          int k, float *deltaSw);

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
__global__ void fcDeltaMb(float const *Cbz, float const *deltaM, int bpos,
                          int zposOut, int m, int n, int k, float *deltaMb);

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
__global__ void fcDeltaSb(float const *Cbz, float const *deltaS, int bpos,
                          int zposOut, int m, int n, int k, float *deltaSb);

////////////////////////////////////////////////////////////////////////////////
/// CONVOLUTIONAL
////////////////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
__global__ void convDeltaMw(float const *Sw, float const *ma,
                            float const *deltaM, int const *aidx, int wpos,
                            int apos, int aidxpos, int m, int n, int k,
                            int woho, int wihi, int fi, int ki2, int padIdx,
                            float *deltaMw);

// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
__global__ void convDeltaSw(float const *Sw, float const *ma,
                            float const *deltaS, int const *aidx, int wpos,
                            int apos, int aidxpos, int m, int n, int k,
                            int woho, int wihi, int fi, int ki2, int padIdx,
                            float *deltaSw);

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
__global__ void convDeltaMb(float const *Cbz, float const *deltaM, int bpos,
                            int m, int n, int k, float *deltaMb);

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
__global__ void convDeltaSb(float const *Cbz, float const *deltaS, int bpos,
                            int m, int n, int k, float *deltaSb);

__global__ void permuteMeanVar(float const *deltaMinit, float const *deltaSinit,
                               float *deltaM, float *deltaS, int zpos, int woho,
                               int kp, int B);

///////////////////////////////////////////////////////////////////////////
/// NORMALIZATION
///////////////////////////////////////////////////////////////////////////
// Batch Normalization
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void convbnDeltaMwSw(float const *Sw, float const *ma,
                                float const *mhat, float const *Shat,
                                float const *deltaM, float const *deltaS,
                                float epsilon, int wpos, int zposIn,
                                int zposOut, int rapos, int wihi, int fi, int m,
                                int k, float *deltaMw, float *deltaSw);

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void convbnDeltaMbSb(float const *Sb, float const *deltaM,
                                float const *deltaS, float epsilon, int bpos,
                                int zposOut, int wihi, int fi, int m, int k,
                                float *deltaMb, float *deltaSb);

// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void fcbnDeltaMwSw(float const *Sw, float const *ma,
                              float const *mhat, float const *Shat,
                              float const *deltaM, float const *deltaS,
                              float epsilon, int wpos, int zposIn, int zposOut,
                              int rapos, int ni, int B, float *deltaMw,
                              float *deltaSw);

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void fcbnDeltaMbSb(float const *Sb, float const *deltaM,
                              float const *deltaS, float epsilon, int bpos,
                              int zposOut, int ni, int B, float *deltaMb,
                              float *deltaSb);

// Layer Normalization
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void convlnDeltaMwSw(float const *Sw, float const *ma,
                                float const *mhat, float const *Shat,
                                float const *deltaM, float const *deltaS,
                                float epsilon, int wpos, int zposIn,
                                int zposOut, int rapos, int wihi, int m, int k,
                                float *deltaMw, float *deltaSw);

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void convlnDeltaMbSb(float const *Sb, float const *deltaM,
                                float const *deltaS, float epsilon, int bpos,
                                int zposOut, int wihi, int m, int k,
                                float *deltaMb, float *deltaSb);

// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void fclnDeltaMwSw(float const *Sw, float const *ma,
                              float const *mhat, float const *Shat,
                              float const *deltaM, float const *deltaS,
                              float epsilon, int wpos, int zposIn, int zposOut,
                              int rapos, int ni, int B, float *deltaMw,
                              float *deltaSw);

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void fclnDeltaMbSb(float const *Sb, float const *deltaM,
                              float const *deltaS, float epsilon, int bpos,
                              int zposOut, int ni, int B, float *deltaMb,
                              float *deltaSb);

__global__ void deltaParamSum(float const *deltaMe, float const *deltaSe,
                              int startpos, int wihi, int fi, int n,
                              float *deltaM, float *deltaS);

///////////////////////////////////////////////////////////////////
/// PARAMETER BACKWARD PASS
///////////////////////////////////////////////////////////////////
void paramBackward(Network &net, ParamGPU &theta, StateGPU &state,
                   DeltaStateGPU &d_state, IndexGPU &idx,
                   DeltaParamGPU &d_theta);

///////////////////////////////////////////////////////////////////////////////
// File:         global_param_update.cu
// Description:  global parameter update in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 28, 2021
// Updated:      January 22, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include <global_param_update.cuh>

__global__ void update_weight(float const *mw_0, float const *Sw_0,
    float const *deltaMw, float const *deltaSw, int n, float *mw, float *Sw)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < n)
    {
      mw[col] = mw_0[col] + deltaMw[col];
      Sw[col] = Sw_0[col] + deltaSw[col];
    }
}

__global__ void update_bias(float const *mb_0, float const *Sb_0,
    float const *deltaMb, float const *deltaSb, int n, float *mb, float *Sb)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < n)
    {
      mb[col] = mb_0[col] + deltaMb[col];
      Sb[col] = Sb_0[col] + deltaSb[col];
    }
}

void globalParamUpdate(float const *mw_0, float const *Sw_0, float const *mb_0,
    float const *Sb_0, float const *mwx_0, float const *Swx_0,
    float const *mbx_0, float const *Sbx_0, float const *deltaMw,
    float const *deltaSw, float const *deltaMb, float const *deltaSb,
    float const *deltaMwx, float const *deltaSwx, float const *deltaMbx,
    float const *deltaSbx, int wxupdate, int wN, int bN, int wxN, int bxN,
    float *mw, float *Sw, float *mb, float *Sb, float *mwx, float *Swx,
    float *mbx, float *Sbx)
{
  // Launch kernel
  int THREADS = 16;
  unsigned gridColW = (wN + THREADS - 1) / THREADS;
  unsigned gridColB = (bN + THREADS - 1) / THREADS;
  unsigned gridRow  = (1 + THREADS - 1) / THREADS;
  dim3 dimGridW(gridColW, gridRow);
  dim3 dimGridB(gridColB, gridRow);
  dim3 dimBlock(THREADS, THREADS);

  // update weights
  update_weight<<<dimGridW, dimBlock>>>(mw_0, Sw_0, deltaMw, deltaSw, wN, mw,
    Sw);

  // update bias
  update_bias<<<dimGridB, dimBlock>>>(mb_0, Sb_0, deltaMb, deltaSb, bN, mb, Sb);

  if (wxupdate == 1)// 1 = true. Need to figure out how to work with boolean data type
  {
    unsigned gridColWx = (wxN + THREADS - 1) / THREADS;
    unsigned gridColBx = (bxN + THREADS - 1) / THREADS;
    dim3 dimGridWx(gridColWx, gridRow);
    dim3 dimGridBx(gridColBx, gridRow);

    // update weights
    update_weight<<<dimGridWx, dimBlock>>>(mwx_0, Swx_0, deltaMwx, deltaSwx,
      wxN, mwx, Swx);

    // update bias
    update_bias<<<dimGridBx, dimBlock>>>(mbx_0, Sbx_0, deltaMbx, deltaSbx, bxN,
      mbx, Sbx);
  }
}


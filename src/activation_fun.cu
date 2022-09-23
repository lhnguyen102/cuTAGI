///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cu
// Description:  Activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      September 07, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/activation_fun.cuh"

__global__ void noActMeanVar(float const *mz, float const *Sz, float *ma,
                             float *J, float *Sa, int zpos, int n)
/* No activation function

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian matrix
    zpos: Input-hidden-state position for this layer in the weight vector
          of network
    n: Number of hidden units for this layer
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float onePad = 1;
    if (col < n && row < 1) {
        ma[col + zpos] = mz[col + zpos];
        J[col + zpos] = onePad;
        Sa[col + zpos] = Sz[col + zpos];
    }
}

__global__ void tanhMeanVar(float const *mz, float const *Sz, float *ma,
                            float *J, float *Sa, int zpos, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < n) {
        tmp = tanhf(mz[col + zpos]);
        ma[col + zpos] = tmp;
        J[col + zpos] = (1 - powf(tmp, 2));
        Sa[col + zpos] =
            (1 - powf(tmp, 2)) * Sz[col + zpos] * (1 - powf(tmp, 2));
    }
}

__global__ void sigmoidMeanVar(float const *mz, float const *Sz, float *ma,
                               float *J, float *Sa, int zpos, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < n) {
        tmp = 1.0 / (1.0 + expf(-mz[col + zpos]));
        ma[col + zpos] = tmp;
        J[col + zpos] = tmp * (1 - tmp);
        Sa[col + zpos] = tmp * (1 - tmp) * Sz[col + zpos] * tmp * (1 - tmp);
    }
}

__global__ void reluMeanVar(float const *mz, float const *Sz, float *ma,
                            float *J, float *Sa, int zpos, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float onePad = 1;
    float tmp = 0;
    if (col < n) {
        tmp = max(mz[col + zpos], zeroPad);
        ma[col + zpos] = tmp;
        if (tmp == 0) {
            J[col + zpos] = zeroPad;
            Sa[col + zpos] = zeroPad;
        } else {
            J[col + zpos] = onePad;
            Sa[col + zpos] = Sz[col + zpos];
        }
    }
}

__global__ void softplusMeanVar(float const *mz, float const *Sz, float *ma,
                                float *J, float *Sa, int zpos, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < n) {
        ma[col + zpos] = logf(1 + expf(mz[col + zpos]));
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        J[col + zpos] = tmp;
        Sa[col + zpos] = tmp * Sz[col + zpos] * tmp;
    }
}

__global__ void leakyreluMeanVar(float const *mz, float const *Sz, float alpha,
                                 float *ma, float *J, float *Sa, int zpos,
                                 int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float onePad = 1;
    float tmp = 0;
    if (col < n) {
        tmp = max(mz[col + zpos], zeroPad);
        if (tmp == 0) {
            ma[col + zpos] = alpha * mz[col + zpos];
            J[col + zpos] = alpha;
            Sa[col + zpos] = alpha * Sz[col + zpos] * alpha;
        } else {
            ma[col + zpos] = tmp;
            J[col + zpos] = onePad;
            Sa[col + zpos] = Sz[col + zpos];
        }
    }
}

__global__ void exp_fun(float const *mz, float const *Sz, int n, float *ma,
                        float *Sa, float *Cza) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_m = 0.0f;
    float tmp_S = 0.0f;
    if (col < n) {
        tmp_m = mz[col];
        tmp_S = Sz[col];
        ma[col] = expf(tmp_m + 0.5 * tmp_S);
        Sa[col] = expf(2 * tmp_m + tmp_S) * (expf(tmp_S) - 1.0f);
        Cza[col] = tmp_S * expf(tmp_m + 0.5 * tmp_S);
    }
}

__global__ void actFullCov(float const *Szf, float const *J, int no, int B,
                           int zposOut, float *Saf)
/*Activate the full covariance.

Args:
    Szf: Full-covariance matrix for hidden states
    J: Jacobian matrix
    no: Output node
    B: Number of batches
    zposOut: Output-hidden-state position for this layer in the weight vector
        of network
    Saf: Full-covariance matrix for activation units

*/

{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = 0;
    if (col <= (row % no) && row < no * B) {
        idx = no * col - ((col * (col + 1)) / 2) + row % no +
              (row / no) * (((no + 1) * no) / 2);
        Saf[idx] = Szf[idx] * J[row % no + (row / no) * no + zposOut] *
                   J[col + (row / no) * no + zposOut];
    }
}
__global__ void noActFullCov(float const *Szf, float *Saf, int Nf) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Nf) {
        Saf[col] = Szf[col];
    }
}
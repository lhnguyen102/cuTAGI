///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cu
// Description:  Activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      December 07, 2022
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

__global__ void mixture_relu(float const *mz, float const *Sz, float omega_tol,
                             int zpos, int n, float *ma, float *J, float *Sa) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha, beta, omega, kappa, mz_til, Sz_til;
    float pi = 3.141592;  // pi number
    if (col < n) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mz[zpos + col] / powf(Sz[zpos + col], 0.5);
        omega = max(1.0f - normcdff(alpha), omega_tol);
        beta = (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha, 2) / 2.0f) /
               omega;
        kappa = 1.0f + alpha * beta - powf(beta, 2);

        // Gaussian mixture's parameters
        mz_til = mz[zpos + col] + beta * powf(Sz[zpos + col], 0.5);
        Sz_til = kappa * Sz[zpos + col];

        // Activation distribution
        ma[zpos + col] = omega * mz_til;
        Sa[zpos + col] =
            omega * Sz_til + omega * (1.0f - omega) * powf(mz_til, 2);
        J[zpos + col] = powf(omega * kappa, 0.5);
    }
}

__global__ void mixture_bounded_relu(float const *mz, float const *Sz,
                                     float omega_tol, int zpos, int n,
                                     float *ma, float *J, float *Sa) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number
    if (col < n) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + col]) / pow(Sz[zpos + col], 0.5);
        alpha_upper = (1.0f - mz[zpos + col]) / pow(Sz[zpos + col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = pdf_upper - pdf_lower;
        kappa = 1 -
                (pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[zpos + col] - beta * powf(Sz[zpos + col], 0.5);
        Sz_til = kappa * Sz[zpos + col];

        // Activation distribution
        ma[zpos + col] = omega * mz_til - cdf_lower + (1 - cdf_upper);
        Sa[zpos + col] = omega * Sz_til +
                         omega * powf(mz_til - ma[zpos + col], 2) +
                         cdf_lower * powf(1 + ma[zpos + col], 2) +
                         (1 - cdf_upper) * powf(1 - ma[zpos + col], 2);
        J[zpos + col] = powf(omega * kappa, 0.5);
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
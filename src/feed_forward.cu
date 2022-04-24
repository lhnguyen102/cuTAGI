///////////////////////////////////////////////////////////////////////////////
// File:         feed_forward.cu
// Description:  forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      June 13, 2021
// Updated:      April 23, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/feed_forward.cuh"

////////////////////////////////////////////////////////////////////////////////
// FULL-CONNECTED
////////////////////////////////////////////////////////////////////////////////
__global__ void fcMean(float const *mw, float const *mb, float const *ma,
                       float *mz, int wpos, int bpos, int zposIn, int zposOut,
                       int m, int n, int k)
/*Compute mean of product WA for full connected layer
Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
             of network
    n: Input node
    m: Output node
    k: Number of batches

 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    float ma_tmp = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            ma_tmp = ma[n * col + i + zposIn];
            if (ma_tmp != 0) {
                sum += mw[row * n + i + wpos] * ma_tmp;
            }
        }
        mz[col * m + row + zposOut] = sum + mb[row + bpos];
    }
}

__global__ void fcVar(float const *mw, float const *Sw, float const *Sb,
                      float const *ma, float const *Sa, float *Sz, int wpos,
                      int bpos, int zposIn, int zposOut, int m, int n, int k)
/*Compute variance of product WA for full connected layer
Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Varaince of the biases
    ma: Mean of activation units
    Sa: Variance of activation units
    Sz: Variance of hidden states
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
             of network
    n: Input node
    m: Output node
    k: Number of batches

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    float ma_tmp = 0;
    float Sa_tmp = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            ma_tmp = ma[n * col + i + zposIn];
            Sa_tmp = Sa[n * col + i + zposIn];
            sum += (mw[row * n + i + wpos] * mw[row * n + i + wpos] +
                    Sw[row * n + i + wpos]) *
                       Sa_tmp +
                   Sw[row * n + i + wpos] * ma_tmp * ma_tmp;
        }
        Sz[col * m + row + zposOut] = sum + Sb[row + bpos];
    }
}
////////////////////////////////////////////////////////////////////////////////
// CONVOLUTIONAL
////////////////////////////////////////////////////////////////////////////////
__global__ void convMean(float const *mw, float const *mb, float const *ma,
                         int const *aidx, float *mz, int wpos, int bpos,
                         int zposIn, int zposOut, int aidxpos, int woho, int fo,
                         int wihi, int fi, int ki2, int B, int n, int k,
                         int padIdx)
/*Compute mean of product WA for convolutional layer
Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    aidx: Activation indices for mean product WA
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    aidxpos: Position of weight indices for mean product WA
    woho: Width x heights for the output layer
    fo: Number of filters for the output layer
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    ki2: Kernel size x kernel size
    B: Number of batches
    n: ki2 x fi
    k: woho x B
    padIdx: Size of the hidden state vector for this layer + 1

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    if (col < k && row < fo) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[(col % woho) * ki2 + i % ki2 + aidxpos] +
                       (i / ki2) * wihi + (col / woho) * wihi * fi;
            if (aidx_tmp < padIdx) {
                sum += mw[row * n + i + wpos] * ma[aidx_tmp - 1 + zposIn];
            }
        }
        mz[woho * (col / woho) * fo + col % woho + row * woho + zposOut] =
            sum + mb[row + bpos];
    }
}

__global__ void convVar(float const *mw, float const *Sw, float const *Sb,
                        float const *ma, float const *Sa, int const *aidx,
                        float *Sz, int wpos, int bpos, int zposIn, int zposOut,
                        int aidxpos, int woho, int fo, int wihi, int fi,
                        int ki2, int B, int n, int k, int padIdx)
/*Compute variance of product WA for convolutional layer
Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Varaince of the biases
    ma: Mean of activation units
    Sa: Variance of activation units
    Sz: Variance of hidden states
    aidx: Acrivation indices for mean product WA
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    aidxpos: Position of activation indices for mean product WA
    woho: Width x heights for the output layer
    fo: Number of filters for the output layer
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    ki2: Kernel size x kernel size
    B: Number of batches
    n: ki2 x fi
    k: woho x B
    padIdx: Size of the hidden state vector for this layer + 1

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    float ma_tmp = 0;
    float Sa_tmp = 0;
    float mw_tmp = 0;
    float Sw_tmp = 0;
    if (col < k && row < fo) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[(col % woho) * ki2 + i % ki2 + aidxpos] +
                       (i / ki2) * wihi + (col / woho) * wihi * fi;
            if (aidx_tmp < padIdx) {
                ma_tmp = ma[aidx_tmp - 1 + zposIn];
                Sa_tmp = Sa[aidx_tmp - 1 + zposIn];
                mw_tmp = mw[row * n + i + wpos];
                Sw_tmp = Sw[row * n + i + wpos];
                sum += (mw_tmp * mw_tmp + Sw_tmp) * Sa_tmp +
                       Sw_tmp * ma_tmp * ma_tmp;
            }
        }
        Sz[woho * (col / woho) * fo + col % woho + row * woho + zposOut] =
            sum + Sb[row + bpos];
    }
}

__global__ void convMeanNoBiases(float const *mw, float const *ma,
                                 int const *aidx, float *mz, int wpos,
                                 int zposIn, int zposOut, int aidxpos, int woho,
                                 int fo, int wihi, int fi, int ki2, int B,
                                 int n, int k, int padIdx)
/*Compute the meanof product WAfor the convolutional layer WITHOUT biases.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    if (col < k && row < fo) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[(col % woho) * ki2 + i % ki2 + aidxpos] +
                       (i / ki2) * wihi + (col / woho) * wihi * fi;
            if (aidx_tmp < padIdx) {
                sum += mw[row * n + i + wpos] * ma[aidx_tmp - 1 + zposIn];
            }
        }
        mz[woho * (col / woho) * fo + col % woho + row * woho + zposOut] = sum;
    }
}

__global__ void convVarNoBiases(float const *mw, float const *Sw,
                                float const *ma, float const *Sa,
                                int const *aidx, float *Sz, int wpos,
                                int zposIn, int zposOut, int aidxpos, int woho,
                                int fo, int wihi, int fi, int ki2, int B, int n,
                                int k, int padIdx)
/*Compute the product variance WA for the convolutional layer WITHOUT biases.
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    if (col < k && row < fo) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[(col % woho) * ki2 + i % ki2 + aidxpos] +
                       (i / ki2) * wihi + (col / woho) * wihi * fi;
            if (aidx_tmp < padIdx) {
                sum += (mw[row * n + i + wpos] * mw[row * n + i + wpos] +
                        Sw[row * n + i + wpos]) *
                           Sa[aidx_tmp - 1 + zposIn] +
                       Sw[row * n + i + wpos] * ma[aidx_tmp - 1 + zposIn] *
                           ma[aidx_tmp - 1 + zposIn];
            }
        }
        Sz[woho * (col / woho) * fo + col % woho + row * woho + zposOut] = sum;
    }
}
////////////////////////////////////////////////////////////////////////////////
// TRANSPOSE CONVOLUTIONAL
////////////////////////////////////////////////////////////////////////////////
__global__ void tconvMean(float const *mw, float const *mb, float const *ma,
                          int const *widx, int const *aidx, int wpos, int bpos,
                          int zposIn, int zposOut, int widxpos, int aidxpos,
                          int woho, int fo, int wihi, int fi, int ki, int rf,
                          int B, float *mz)
/*Compute mean of product WA for transpose convolutional layer
Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    widx: Weight indices for mean product WA
    aidx: Activation indices for mean product WA
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    widxpos: Position of weight indices for mean product WA
    aidxpos: Position of activation indices for mean product WA
    woho: Width x heights for the output layer
    fo: Number of filters for the output layer
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    ki: Kernel size
    rf: Number of columns of weight indices for mean product WA
    B: Number of batches

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    int widx_tmp = 0;
    if (col < woho * fo && row < B)  // k = woho * fo
    {
        for (int i = 0; i < rf * fi; i++)  // n = ? * fi
        {
            widx_tmp = widx[(col % woho) * rf + i % rf + widxpos] +
                       (col / woho) * ki * ki + (i / rf) * ki * ki * fo + wpos -
                       1;  // minus 1 due to matlab's indexing

            aidx_tmp = aidx[(col % woho) * rf + i % rf + aidxpos] +
                       row * wihi * fi + (i / rf) * wihi - 1;

            if (aidx_tmp + 1 < wihi * fi * B + 1) {
                sum += mw[widx_tmp] * ma[aidx_tmp + zposIn];
            }
        }
        mz[col + row * woho * fo + zposOut] = sum + mb[col / woho + bpos];
    }
}

__global__ void tconvVar(float const *mw, float const *Sw, float const *Sb,
                         float const *ma, float const *Sa, int const *widx,
                         int const *aidx, int wpos, int bpos, int zposIn,
                         int zposOut, int widxpos, int aidxpos, int woho,
                         int fo, int wihi, int fi, int ki, int rf, int B,
                         float *Sz)
/*Compute variance of product WA for convolutional layer
Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Varaince of the biases
    ma: Mean of activation units
    Sa: Variance of activation units
    Sz: Variance of hidden states
    widx: Weight indices for mean product WA
    aidx: Activation indices for mean product WA
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    widxpos: Position of weight indices for mean product WA
    aidxpos: Position of activation indices for mean product WA
    woho: Width x heights for the output layer
    fo: Number of filters for the output layer
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    ki: Kernel size
    rf: Number of columns of weight indices for mean product WA
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int widx_tmp = 0;
    int aidx_tmp = 0;
    if (col < woho * fo && row < B) {
        for (int i = 0; i < rf * fi; i++) {
            widx_tmp = widx[(col % woho) * rf + i % rf + widxpos] +
                       (col / woho) * ki * ki + (i / rf) * ki * ki * fo + wpos -
                       1;
            aidx_tmp = aidx[(col % woho) * rf + i % rf + aidxpos] +
                       row * wihi * fi + (i / rf) * wihi - 1;

            if (aidx_tmp + 1 < wihi * fi * B + 1) {
                sum += (mw[widx_tmp] * mw[widx_tmp] + Sw[widx_tmp]) *
                           Sa[aidx_tmp + zposIn] +
                       Sw[widx_tmp] * ma[aidx_tmp + zposIn] *
                           ma[aidx_tmp + zposIn];
            }
        }
        Sz[col + row * woho * fo + zposOut] = sum + Sb[col / woho + bpos];
    }
}

////////////////////////////////////////////////////////////////////////////////
// AVERAGE POOLING
////////////////////////////////////////////////////////////////////////////////
__global__ void apMeanVarOverlap(float const *ma, float const *Sa,
                                 int const *aidx, float *mz, float *Sz,
                                 int zposIn, int zposOut, int aidxpos, int woho,
                                 int wihi, int ki2, int k, int padIdx)
/*Compute product mean & variance WA for average pooling for the case where
there is the overlap when sliding kernel size.

Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    Sa: Variance of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    aidx: Activation indices for mean product WA
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    aidxpos: Position of weight indices for mean product WA
    woho: Width x heights for the output layer
    wihi: Width x heights for the input layer
    ki2: Kernel size x kernel size
    k: woho x fo x B, where B: Number of batches
    padIdx: Size of the hidden state vector for this layer + 1
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumMz = 0;
    float sumSz = 0;
    int aidx_tmp = 0;
    if (col < k && row < 1) {
        for (int i = 0; i < ki2; i++) {
            aidx_tmp =
                aidx[col % woho + woho * i + aidxpos] + (col / woho) * wihi;
            if (aidx_tmp < padIdx) {
                sumMz += ma[aidx_tmp - 1 +
                            zposIn];  // minus 1 due to matlab's indexing
                sumSz += Sa[aidx_tmp - 1 + zposIn];
            }
        }
        mz[col + zposOut] = sumMz / ki2;
        Sz[col + zposOut] = sumSz / (ki2 * ki2);
    }
}
__global__ void apMeanVar(float const *ma, float const *Sa, int const *aidx,
                          float *mz, float *Sz, int zposIn, int zposOut,
                          int aidxpos, int woho, int wihi, int ki2, int k)
/* Compute product mean & variance WA for average pooling for the case there
is no overlap when sliding kernel size.
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumMz = 0;
    float sumSz = 0;
    int aidx_tmp = 0;
    if (col < k && row < 1) {
        for (int i = 0; i < ki2; i++) {
            aidx_tmp = aidx[col % woho + woho * i + aidxpos] +
                       (col / woho) * wihi -
                       1;  // minus 1 due to matlab's indexing
            sumMz += ma[aidx_tmp + zposIn];
            sumSz += Sa[aidx_tmp + zposIn];
        }
        mz[col + zposOut] = sumMz / ki2;
        Sz[col + zposOut] = sumSz / (ki2 * ki2);
    }
}
////////////////////////////////////////////////////////////////////////////////
/// NORMALIZATION
////////////////////////////////////////////////////////////////////////////////
// Conv. layer normalization
// These functions compute statistical mean & variance for conv. l.n. layer
__global__ void convlnStatMeanVar(float const *ma, float const *Sa, float *ms,
                                  float *Ss, int zpos, int spos, int wihi,
                                  int fi, int B)
/*Compute sample mean and variance of activation units for layer-normalization
layer. Note that the previous layer is a conolutional layer.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples e.g. ms = mean(ma)
    Ss: Variance of samples
    zpos: Input-hidden-state postision for this layer in the weight vector
          of network
    spos: Position of sample mean and varance in the vector for entire network
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumM = 0;
    float sumS = 0;
    if (col < B && row < 1) {
        for (int i = 0; i < wihi * fi; i++)  // n = wihi*B
        {
            sumM += ma[col * wihi * fi + i + zpos];
            sumS += Sa[col * wihi * fi + i + zpos];
        }
        ms[col + spos] = sumM / (wihi * fi);
        Ss[col + spos] = sumS;
    }
}
__global__ void convlnStatSampleVar(float const *ma, float const *ms,
                                    float const *Ss, float *S, int zpos,
                                    int spos, int wihi, int fi, int B)
/*Compute statistical mean and variance of activation units for
layer-normalization layer. Note that the previous layer is a conolutional layer.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples
    Ss: Variance of samples
    S: Statistical vatiance
    zpos: Input-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of sample mean and varance in the vector for entire network
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < B && row < 1) {
        for (int i = 0; i < wihi * fi; i++) {
            sum += (ma[col * wihi * fi + i + zpos] - ms[col + spos]) *
                   (ma[col * wihi * fi + i + zpos] - ms[col + spos]);
        }
        S[col + spos] = (sum + Ss[col + spos]) / (wihi * fi - 1);
    }
}
// These function compute TAGI-feedforward pass for conv. l.n. layer
__global__ void convlnMean(float const *mw, float const *mb, float const *ma,
                           float const *mra, float const *Sra, float epsilon,
                           float *mz, float *Sz, int wpos, int bpos,
                           int zposOut, int zposIn, int spos, int wihi, int m,
                           int k)
/*Compute mean of product WA for convolutional layer. Note that we consider
hidden states within layer as samples.


Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    spos: Position of statstical mean & variance
    wihi: Width x heights for the input layer
    m: Number of hidden units for output
    k: wihi x fi where fi is the number of filters for input

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi * fi, m = B
    {
        mz[col + row * k + zposOut] =
            (1 / sqrtf(Sra[row + spos] + epsilon)) *
                (ma[col + row * k + zposIn] - mra[row + spos]) *
                mw[col / wihi + wpos] +
            mb[col / wihi + bpos];
    }
}
__global__ void convlnVar(float const *mw, float const *Sw, float const *mb,
                          float const *Sb, float const *ma, float const *Sa,
                          float const *mra, float const *Sra, float epsilon,
                          float *mz, float *Sz, int wpos, int bpos, int zposOut,
                          int zposIn, int spos, int wihi, int m, int k)
/*Compute variance of product WA  for convolutional layer. Note that we consider
hidden states within layer as samples.

Args:
    mw: Mean of weights
    Sw: Variance of weights
    mb: Mean of the biases
    Sb: Variance of biases
    ma: Mean of activation units
    Sa: Variance of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    spos: Position of statstical mean & variance
    wihi: Width x heights for the input layer
    m: Number of hidden units for output
    k: wihi x fi where fi is the number of filters for input

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m) {
        Sz[col + row * k + zposOut] =
            (1 / (Sra[row + spos] + epsilon)) *
                (Sa[col + row * k + zposIn] * mw[col / wihi + wpos] *
                     mw[col / wihi + wpos] +
                 Sw[col / wihi + wpos] *
                     (ma[col + row * k + zposIn] * ma[col + row * k + zposIn] -
                      mra[row + spos] * mra[row + spos] +
                      Sa[col + row * k + zposIn])) +
            Sb[col / wihi + bpos];
    }
}
// FC Layer Normalization
// These functions compute statistical mean & variance for fc l.n. layer
__global__ void fclnStatMeanVar(float const *ma, float const *Sa, float *ms,
                                float *Ss, int zpos, int spos, int ni, int B)
/*Compute sample mean and variance of activation units for full-connected layer.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples e.g. ms = mean(ma)
    Ss: Variance of samples
    zpos: Input-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of sample mean and varance in the vector for entire network
    ni: Number of hidden units for inputs
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumM = 0;
    float sumS = 0;
    if (col < B && row < 1) {
        for (int i = 0; i < ni; i++)  // n = wihi*B
        {
            sumM += ma[col * ni + i + zpos];
            sumS += Sa[col * ni + i + zpos];
        }
        ms[col + spos] = sumM / ni;
        Ss[col + spos] = sumS;
    }
}
__global__ void fclnStatSampleVar(float const *ma, float const *ms,
                                  float const *Ss, float *S, int zpos, int spos,
                                  int ni, int B)
/*Compute statistical mean and variance of activation units for full-connected
layer.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples
    Ss: Variance of samples
    S: Statistical vatiance
    zpos: Input-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of sample mean and varance in the vector for entire network
    ni: Number of hidden units for inputs
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < B && row < 1) {
        for (int i = 0; i < ni; i++) {
            sum += (ma[col * ni + i + zpos] - ms[col + spos]) *
                   (ma[col * ni + i + zpos] - ms[col + spos]);
        }
        S[col + spos] = (sum + Ss[col + spos]) / (ni - 1);
    }
}
// These functions compute TAGI-feedforward pass for fc l.n. layer
__global__ void fclnMean(float const *mw, float const *mb, float const *ma,
                         float const *mra, float const *Sra, float epsilon,
                         float *mz, float *Sz, int wpos, int bpos, int zposOut,
                         int zposIn, int spos, int ni, int B)
/*Compute mean o fproduct WA of layer-normalization. Note that the previous
layer is the full-connected layer.


Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalization layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    spos: Position of statstical mean & variance
    ni: Number of hidden units
    B: Number of batches

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < B) {
        mz[col + row * ni + zposOut] =
            (1 / sqrtf(Sra[row + spos] + epsilon)) *
                (ma[col + row * ni + zposIn] - mra[row + spos]) *
                mw[col + wpos] +
            mb[col + bpos];
    }
}
__global__ void fclnVar(float const *mw, float const *Sw, float const *mb,
                        float const *Sb, float const *ma, float const *Sa,
                        float const *mra, float const *Sra, float epsilon,
                        float *mz, float *Sz, int wpos, int bpos, int zposOut,
                        int zposIn, int spos, int ni, int B)
/*Compute variance of product WA of layer-normalization for convolutional layer.
Note that the previous layer is a full-connected layer.

Args:
    mw: Mean of weights
    Sw: Variance of weights
    mb: Mean of the biases
    Sb: Variance of biases
    ma: Mean of activation units
    Sa: Variance of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    spos: Position of statstical mean & variance
    ni: Number of hidden units
    B: Number of batches

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < B) {
        Sz[col + row * ni + zposOut] =
            (1 / (Sra[row + spos] + epsilon)) *
                (Sa[col + row * ni + zposIn] * mw[col + wpos] * mw[col + wpos] +
                 Sw[col + wpos] * (ma[col + row * ni + zposIn] *
                                       ma[col + row * ni + zposIn] -
                                   mra[row + spos] * mra[row + spos] +
                                   Sa[col + row * ni + zposIn])) +
            Sb[col + bpos];
    }
}
// Conv. batch normalization
// These functions compute the statistical mean & variance for conv. b.n layer
__global__ void convbnStatMeanVar(float const *ma, float const *Sa, float *ms,
                                  float *Ss, int zpos, int spos, int wihi,
                                  int fi, int B)
/*Compute sample mean and variance of activation units for batch-normalization
layer.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples e.g. ms = mean(ma)
    Ss: Variance of samples
    zpos: Input-hidden-state postision for this layer in the weight vector
          of network
    spos: Position of sample mean and varance in the vector for entire network
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumM = 0;
    float sumS = 0;
    if (col < fi && row < 1) {
        for (int i = 0; i < wihi * B; i++)  // n = wihi*B
        {
            sumM += ma[(i / wihi) * wihi * fi + i % wihi + col * wihi + zpos];
            sumS += Sa[(i / wihi) * wihi * fi + i % wihi + col * wihi + zpos];
        }
        ms[col + spos] = sumM / (wihi * B);
        Ss[col + spos] = sumS;
    }
}
__global__ void convbnStatSampleVar(float const *ma, float const *ms,
                                    float const *Ss, float *S, int zpos,
                                    int spos, int wihi, int fi, int B)
/*Compute statistical mean and variance of activation units for
batch-normalization layer.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples
    Ss: Variance of samples
    S: Statistical vatiance
    zpos: Input-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of sample mean and varance in the vector for entire network
    wihi: Width x heights for the input layer
    fi: Number of filters for the input layer
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < fi && row < 1) {
        for (int i = 0; i < wihi * B; i++) {
            sum += (ma[(i / wihi) * wihi * fi + i % wihi + col * wihi + zpos] -
                    ms[col + spos]) *
                   (ma[(i / wihi) * wihi * fi + i % wihi + col * wihi + zpos] -
                    ms[col + spos]);
        }
        S[col + spos] = (sum + Ss[col + spos]) / (wihi * B - 1);
    }
}
// These two functions are TAGI-feedforward pass for conv. b.n. layer
__global__ void convbnMean(float const *mw, float const *mb, float const *ma,
                           float const *mra, float const *Sra, float epsilon,
                           float *mz, float *Sz, int wpos, int bpos,
                           int zposOut, int zposIn, int spos, int wihi, int fi,
                           int m, int k)
/*Compute mean of product WA of batch-normalization. Note that the previous
layer is a convolutional layer.

Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
    of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of statstical mean & variance
    wihi: Width x heights for the input layer
    m: fi x B
    k: wihi
*/
// TODO: Remove input variable k
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        mz[col + row * k + zposOut] =
            (1 / sqrtf(Sra[row % fi + spos] + epsilon)) *
                (ma[col + row * k + zposIn] - mra[row % fi + spos]) *
                mw[row % fi + wpos] +
            mb[row % fi + bpos];
    }
}
__global__ void convbnVar(float const *mw, float const *Sw, float const *mb,
                          float const *Sb, float const *ma, float const *Sa,
                          float const *mra, float const *Sra, float epsilon,
                          float *mz, float *Sz, int wpos, int bpos, int zposOut,
                          int zposIn, int spos, int wihi, int fi, int m, int k)
/*Compute variance of product WA of batch-normalization layer.Note that the
previous layer is a convolutional layer.

Args:
    mw: Mean of weights
    Sw: Variance of weights
    mb: Mean of the biases
    Sb: Variance of biases
    ma: Mean of activation units
    Sa: Variance of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
    of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of statstical mean & variance
    wihi: Width x heights for the input layer
    m: fi x B
    k: wihi

*/
// TODO: Remove input variable k
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m) {
        Sz[col + row * k + zposOut] =
            (1 / (Sra[row % fi + spos] + epsilon)) *
                (Sa[col + row * k + zposIn] * mw[row % fi + wpos] *
                     mw[row % fi + wpos] +
                 Sw[row % fi + wpos] *
                     (ma[col + row * k + zposIn] * ma[col + row * k + zposIn] -
                      mra[row % fi + spos] * mra[row % fi + spos] +
                      Sa[col + row * k + zposIn])) +
            Sb[row % fi + bpos];
    }
}
// Full-connected batch normalization
// These function compute the statistical mean & variance  for fc b.n. layer
__global__ void fcbnStatMeanVar(float const *ma, float const *Sa, float *ms,
                                float *Ss, int zpos, int spos, int ni, int B)
/*Compute sample mean and variance of activation units of full-connected layer
for each batch.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples e.g. ms = mean(ma)
    Ss: Variance of samples
    zpos: Input-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of sample mean and varance in the vector for entire network
    ni: Number of hidden units for inputs
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumM = 0;
    float sumS = 0;
    if (col < ni && row < 1) {
        for (int i = 0; i < B; i++)  // n = wihi*B
        {
            sumM += ma[col + i * ni + zpos];
            sumS += Sa[col + i * ni + zpos];
        }
        ms[col + spos] = sumM / B;
        Ss[col + spos] = sumS;
    }
}
__global__ void fcbnStatSampleVar(float const *ma, float const *ms,
                                  float const *Ss, float *S, int zpos, int spos,
                                  int ni, int B)
/*Compute statistical mean and variance of activation units for full-connected
layer for each batch.

Args:
    ma: Mean of activation units
    Sa: Variance of activation units
    ms: Mean of samples
    Ss: Variance of samples
    S: Statistical vatiance
    zpos: Input-hidden-state postision for this layer in the weight vector
    of network
    spos: Position of sample mean and varance in the vector for entire network
    ni: Number of hidden units for inputs
    B: Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < ni && row < 1) {
        for (int i = 0; i < B; i++) {
            sum += (ma[col + i * ni + zpos] - ms[col + spos]) *
                   (ma[col + i * ni + zpos] - ms[col + spos]);
        }
        S[col] = (sum + Ss[col + spos]) / (B - 1);
    }
}
// These functions compute TAGI-feedforward pass for fc b.n. layer
__global__ void fcbnMean(float const *mw, float const *mb, float const *ma,
                         float const *mra, float const *Sra, float epsilon,
                         float *mz, float *Sz, int wpos, int bpos, int zposOut,
                         int zposIn, int spos, int ni, int B)
/*Compute pmean of product WA of batch-normalization layer. Note that the
previous layer is a full-connected layer.

Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    spos: Position of statstical mean & variance
    ni: Number of hidden units
    B: Number of batches

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < B) {
        mz[col + row * ni + zposOut] =
            (1 / sqrtf(Sra[col + spos] + epsilon)) *
                (ma[col + row * ni + zposIn] - mra[col + spos]) *
                mw[col + wpos] +
            mb[col + bpos];
    }
}
__global__ void fcbnVar(float const *mw, float const *Sw, float const *mb,
                        float const *Sb, float const *ma, float const *Sa,
                        float const *mra, float const *Sra, float epsilon,
                        float *mz, float *Sz, int wpos, int bpos, int zposOut,
                        int zposIn, int spos, int ni, int B)
/*Compute variance of product WA of batch-normalization. Note that the previous
layer is a full-connected layer.

Args:
    mw: Mean of weights
    Sw: Variance of weights
    mb: Mean of the biases
    Sb: Variance of biases
    ma: Mean of activation units
    Sa: Variance of activation units
    mz: Mean of hidden states
    Sz: Variance of hidden states
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalzation layers
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight postision for this layer in the weight vector of network
    bpos: Bias postision for this layer in the bias vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
             of network
    spos: Position of statstical mean & variance
    ni: Number of hidden units
    B: Number of batches

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ni && row < B) {
        Sz[col + row * ni + zposOut] =
            (1 / (Sra[col + spos] + epsilon)) *
                (Sa[col + row * ni + zposIn] * mw[col + wpos] * mw[col + wpos] +
                 Sw[col + wpos] * (ma[col + row * ni + zposIn] *
                                       ma[col + row * ni + zposIn] -
                                   mra[col + spos] * mra[col + spos] +
                                   Sa[col + row * ni + zposIn])) +
            Sb[col + bpos];
    }
}
// Running average for statistical mean and variance
__global__ void raMeanVar(float const *ms, float const *Ss,
                          float const *mraprev, float const *Sraprev,
                          float momentum, float *mra, float *Sra, int spos,
                          int N)
/*Copute the running average for the normalization layers.

Args:
    ms: New statistical mean of samples
    Ss: New statistical variance of samples
    mraprev: Previous mean for the normalzation layers
    Sraprev: Previous statistical variance for the normalization layers
    momentum: Running average factor
    mra: Statistical mean for the normalzation layers
    Sra: Statistical variance for the normalization layers
    spos: Position of statstical mean & variance
    N: Size of mra

 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        // mra[col + spos] = mraprev[col + spos];
        // Sra[col + spos] = Sraprev[col + spos];
        mra[col + spos] =
            mraprev[col + spos] * momentum + ms[col + spos] * (1 - momentum);
        Sra[col + spos] =
            Sraprev[col + spos] * momentum + Ss[col + spos] * (1 - momentum);
    }
}
////////////////////////////////////////////////////////////////////////////////
/// RESIDUAL NETWORKS
////////////////////////////////////////////////////////////////////////////////
__global__ void twoPlusSc(float const *m1, float const *S1, float const *m2,
                          float const *S2, float *m, float *S, int pos1,
                          int pos2, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        m[col + pos1] = m1[col + pos2] + m2[col + pos2];
        S[col + pos1] = S1[col + pos2] + S2[col + pos2];
    }
}
__global__ void duplicateMeanVarWithIndex(float const *m1, float const *S1,
                                          float *m, float *S, int posIn,
                                          int posOut, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        m[col + posOut] = m1[col + posIn];
        S[col + posOut] = S1[col + posIn];
    }
}

////////////////////////////////////////////////////////////////////////////////
/// ACTIVATION
////////////////////////////////////////////////////////////////////////////////
__global__ void noActMeanVar(float const *mz, float const *Sz, float *ma,
                             float *J, float *Sa, int zpos, int n)
/* No activation function

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian matrix
    zpos: Input-hidden-state postision for this layer in the weight vector
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
        tmp = 1 / (1 + expf(-mz[col + zpos]));
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
//////////////////////////////////////////////////////////////////////
/// INITIALIZE STATES
//////////////////////////////////////////////////////////////////////
__global__ void initializeStates(float const *x, float const *Sx, float *mz,
                                 float *Sz, float *ma, float *Sa, float *J,
                                 int niB)
/* Insert input data to network's states

Args:
    x: Input data:
    Sx: Varaince of input data i.e. in the common case, Sx=0
    mz: Mean of hidden states
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian matrix
    niB: Number of hidden units x number of batches for input layer

 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < niB) {
        mz[col] = x[col];
        Sz[col] = Sx[col];
        ma[col] = x[col];
        Sa[col] = Sx[col];
        J[col] = 1;
    }
}

__global__ void initializeFullStates(float const *mz_0, float const *Sz_0,
                                     float const *ma_0, float const *Sa_0,
                                     float const *J_0, int niB, int zposIn,
                                     float *mz, float *Sz, float *ma, float *Sa,
                                     float *J)
/* Insert full initial state to network's states. This is commoly used for
multiple networks that connect to each other.

Args:
    mz_0: Mean if hidden states for input layer
    Sz_0: Variance of hidden states for input layer
    ma_0: Mean of activation units for input layer
    Sa_0: Variance of activation units for input layer
    J_0: Jacobian matrix for input layer
    mz: Mean of hidden states
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian matrix
    niB: Number of hidden units x number of batches for input layer

*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < niB) {
        mz[col] = mz_0[col + zposIn];
        Sz[col] = Sz_0[col + zposIn];
        ma[col] = ma_0[col + zposIn];
        Sa[col] = Sa_0[col + zposIn];
        J[col] = J_0[col + zposIn];
    }
}
//////////////////////////////////////////////////////////////////////
/// GET OUTPUT HIDDEN STATE
//////////////////////////////////////////////////////////////////////
__global__ void getOutputHiddenStates(float const *mz, float const *Sz,
                                      float const *ma, float const *Sa,
                                      float const *J, int zpos, int N,
                                      float *mz_op, float *Sz_op, float *ma_op,
                                      float *Sa_op, float *J_op)
/*Get states for the output layers.
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        mz_op[col] = mz[col + zpos];
        Sz_op[col] = Sz[col + zpos];
        ma_op[col] = ma[col + zpos];
        Sa_op[col] = Sa[col + zpos];
        J_op[col] = J[col + zpos];
    }
}

//////////////////////////////////////////////////////////////////////
/// TAGI-FEEDFORWARD PASS
//////////////////////////////////////////////////////////////////////
void feedForward(Network &net, ParamGPU &theta, IndexGPU &idx, StateGPU &state)
/*
  Update Network's hidden states using TAGI.

  Args:
    net: Network architecture
    theta: Network's weights and biases
    idx: Indices for network e.g. see indices.cpp

  Returns:
    state: Hidden state of network
 */

{
    // Launch kernel
    int THREADS = 16;
    dim3 dimBlock(THREADS, THREADS);
    int B = net.batch_size;

    for (int j = 1; j < net.layers.size(); j++) {
        // Common hyperparameters
        int zposIn =
            net.z_pos[j - 1];        // location of actv. input in activ. vector
        int zposOut = net.z_pos[j];  // location of h.s. output in h.s. vector
        int wposIn = net.w_pos[j - 1];  // location of weights in param. vector
        int bposIn = net.b_pos[j - 1];  // location of biases in param. vector
        int M = net.nodes[j];           // num. of nodes for output
        int ni = net.nodes[j - 1];      // num. of nodes for input

        // Hyperparameters are requried only for CNN.
        int ki = net.kernels[j - 1];  // kernel size of input
        int fi = net.filters[j - 1];  // num. of filters for input
        int wi = net.widths[j - 1];   // width of input
        int hi = net.heights[j - 1];  // height of input
        int fo = net.filters[j];      // num. of filters for output
        int wo = net.widths[j];       // width of output
        int ho = net.heights[j];      // height of output
        int ki2 = ki * ki;
        int wihi = wi * hi;
        int woho = wo * ho;
        int padIdx = wihi * fi * B + 1;  // padding index

        // Hyperparameters for residual networks. Note that current version
        // works only with CNN layer. Future version will include other layers.
        int xsOut = net.shortcuts[j];  // index for the shorcut layer

        //**
        // 1: Full connected
        //
        if (net.layers[j] == net.layer_names.fc) {
            int N = net.nodes[j - 1];  // num. of nodes for input
            // Launch kernel
            unsigned int gridRows = (M + THREADS - 1) / THREADS;
            unsigned int gridCols = (B + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCols, gridRows);
            // Compute mean and variance
            fcMean<<<dimGrid, dimBlock>>>(theta.d_mw, theta.d_mb, state.d_ma,
                                          state.d_mz, wposIn, bposIn, zposIn,
                                          zposOut, M, N, B);

            fcVar<<<dimGrid, dimBlock>>>(
                theta.d_mw, theta.d_Sw, theta.d_Sb, state.d_ma, state.d_Sa,
                state.d_Sz, wposIn, bposIn, zposIn, zposOut, M, N, B);
        }
        //**
        // 2: Convolutional
        //
        else if (net.layers[j] == net.layer_names.conv) {
            int N = ki2 * fi;
            int K = woho * B;
            int aidxposIn = net.Fmwa_2_pos[j - 1];  // location of input in
                                                    // kernel indices vector
            // Launch kernel
            unsigned int gridRows = (fo + THREADS - 1) / THREADS;
            unsigned int gridCols = (K + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCols, gridRows);
            // Compute mean and variance
            if (net.num_biases[j] > 0) {
                convMean<<<dimGrid, dimBlock>>>(
                    theta.d_mw, theta.d_mb, state.d_ma, idx.d_Fmwa_2,
                    state.d_mz, wposIn, bposIn, zposIn, zposOut, aidxposIn,
                    woho, fo, wihi, fi, ki2, B, N, K, padIdx);

                convVar<<<dimGrid, dimBlock>>>(
                    theta.d_mw, theta.d_Sw, theta.d_Sb, state.d_ma, state.d_Sa,
                    idx.d_Fmwa_2, state.d_Sz, wposIn, bposIn, zposIn, zposOut,
                    aidxposIn, woho, fo, wihi, fi, ki2, B, N, K, padIdx);
            } else {
                convMeanNoBiases<<<dimGrid, dimBlock>>>(
                    theta.d_mw, state.d_ma, idx.d_Fmwa_2, state.d_mz, wposIn,
                    zposIn, zposOut, aidxposIn, woho, fo, wihi, fi, ki2, B, N,
                    K, padIdx);

                convVarNoBiases<<<dimGrid, dimBlock>>>(
                    theta.d_mw, theta.d_Sw, state.d_ma, state.d_Sa,
                    idx.d_Fmwa_2, state.d_Sz, wposIn, zposIn, zposOut,
                    aidxposIn, woho, fo, wihi, fi, ki2, B, N, K, padIdx);
            }
        }
        //**
        // 4: Average pooling
        //
        else if (net.layers[j] == net.layer_names.ap) {
            int K = woho * fo * B;
            int aidxposIn = net.pooling_pos[j - 1];  // location of input in
                                                     // kernel-indices vector
            // Launch kernel
            unsigned int gridRows = (1 + THREADS - 1) / THREADS;
            unsigned int gridCols = (K + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCols, gridRows);
            if (net.overlap[j - 1] == 1) {
                int padIdx = wi * hi * fi * B + 1;
                apMeanVarOverlap<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_Sa, idx.d_pooling, state.d_mz,
                    state.d_Sz, zposIn, zposOut, aidxposIn, woho, wihi, ki2, K,
                    padIdx);
            } else {
                apMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_Sa, idx.d_pooling, state.d_mz,
                    state.d_Sz, zposIn, zposOut, aidxposIn, woho, wihi, ki2, K);
            }
        }
        //**
        // 5: Layer normalization
        //
        else if (net.layers[j] == net.layer_names.ln) {
            int sposIn =
                net.ra_pos[j - 1];  // location of input in running-avg vector
            if (net.layers[j - 1] == net.layer_names.fc) {
                // Compute  statistical mean and variance
                unsigned int gridCols = (B + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCols, 1);
                fclnStatMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_Sa, state.d_ms, state.d_SsTmp, zposIn,
                    sposIn, ni, B);

                fclnStatSampleVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_ms, state.d_SsTmp, state.d_Ss, zposIn,
                    sposIn, ni, B);

                // Update running average
                raMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ms, state.d_Ss, state.d_mra_prev, state.d_Sra_prev,
                    net.ra_mt, state.d_mra, state.d_Sra, sposIn, B);

                // Compute norm-forwardpass
                unsigned int gridCols2 = (ni + THREADS - 1) / THREADS;
                dim3 dimGrid2(gridCols2, gridCols);
                fclnMean<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_mb, state.d_ma, state.d_mra,
                    state.d_Sra, net.epsilon, state.d_mz, state.d_Sz, wposIn,
                    bposIn, zposOut, zposIn, sposIn, ni, B);

                fclnVar<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_Sw, theta.d_mb, theta.d_Sb, state.d_ma,
                    state.d_Sa, state.d_mra, state.d_Sra, net.epsilon,
                    state.d_mz, state.d_Sz, wposIn, bposIn, zposOut, zposIn,
                    sposIn, ni, B);
            } else {
                // Compute new statistical mean & variance
                unsigned int gridCols = (B + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCols, 1);
                convlnStatMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_Sa, state.d_ms, state.d_SsTmp, zposIn,
                    sposIn, wihi, fi, B);

                convlnStatSampleVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_ms, state.d_SsTmp, state.d_Ss, zposIn,
                    zposIn, wihi, fi, B);

                // Update running average
                raMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ms, state.d_Ss, state.d_mra_prev, state.d_Sra_prev,
                    net.ra_mt, state.d_mra, state.d_Sra, sposIn, B);

                // Compute norm-forwardpass
                int Kln = wihi * fi;
                unsigned int gridCols2 = (Kln + THREADS - 1) / THREADS;
                // Launch kernel
                dim3 dimGrid2(gridCols2, gridCols);
                convlnMean<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_mb, state.d_ma, state.d_mra,
                    state.d_Sra, net.epsilon, state.d_mz, state.d_Sz, wposIn,
                    bposIn, zposOut, zposIn, sposIn, wihi, M, Kln);

                convlnVar<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_Sw, theta.d_mb, theta.d_Sb, state.d_ma,
                    state.d_Sa, state.d_mra, state.d_Sra, net.epsilon,
                    state.d_mz, state.d_Sz, wposIn, bposIn, zposOut, zposIn,
                    sposIn, wihi, M, Kln);
            }
        }
        //**
        // 6: Batch normalization
        //
        else if (net.layers[j] == net.layer_names.bn) {
            int sposIn =
                net.ra_pos[j - 1];  // location of running-avg for input
            if (net.layers[j - 1] == net.layer_names.fc)  // FC layer
            {
                // Compute  statistical mean and variance
                unsigned int gridCols = (ni + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCols, 1);
                fcbnStatMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_Sa, state.d_ms, state.d_SsTmp, zposIn,
                    sposIn, ni, B);

                fcbnStatSampleVar<<<dimGrid, dimBlock>>>(
                    state.d_ma, state.d_ms, state.d_SsTmp, state.d_Ss, zposIn,
                    sposIn, ni, B);

                // Compute running average (to be continued)
                raMeanVar<<<dimGrid, dimBlock>>>(
                    state.d_ms, state.d_Ss, state.d_mra_prev, state.d_Sra_prev,
                    net.ra_mt, state.d_mra, state.d_Sra, sposIn, ni);

                // Compute norm-forwardpass (need to defined mhat and shat)
                unsigned int gridRows2 = (B + THREADS - 1) / THREADS;
                dim3 dimGrid2(gridCols, gridRows2);
                fcbnMean<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_mb, state.d_ma, state.d_mra,
                    state.d_Sra, net.epsilon, state.d_mz, state.d_Sz, wposIn,
                    bposIn, zposOut, zposIn, sposIn, ni, B);

                fcbnVar<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_Sw, theta.d_mb, theta.d_Sb, state.d_ma,
                    state.d_Sa, state.d_mra, state.d_Sra, net.epsilon,
                    state.d_mz, state.d_Sz, wposIn, bposIn, zposOut, zposIn,
                    sposIn, ni, B);
            } else  // Conv. layer
            {
                // Compute new statistical mean & variance
                unsigned int gridColN = (fi + THREADS - 1) / THREADS;
                unsigned int gridRowN = (1 + THREADS - 1) / THREADS;
                dim3 dimGridN(gridColN, gridRowN);
                convbnStatMeanVar<<<dimGridN, dimBlock>>>(
                    state.d_ma, state.d_Sa, state.d_ms, state.d_SsTmp, zposIn,
                    sposIn, wihi, fi, B);

                convbnStatSampleVar<<<dimGridN, dimBlock>>>(
                    state.d_ma, state.d_ms, state.d_SsTmp, state.d_Ss, zposIn,
                    sposIn, wihi, fi, B);

                // Update running average
                raMeanVar<<<dimGridN, dimBlock>>>(
                    state.d_ms, state.d_Ss, state.d_mra, state.d_Sra, net.ra_mt,
                    state.d_mra, state.d_Sra, sposIn, fi);

                // Compute norm-forwardpass
                int Mbn = fi * B;
                int Kbn = wihi;
                unsigned int gridRows2 = (Mbn + THREADS - 1) / THREADS;
                unsigned int gridCols2 = (Kbn + THREADS - 1) / THREADS;

                // Launch kernel
                dim3 dimGrid2(gridCols2, gridRows2);
                convbnMean<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_mb, state.d_ma, state.d_mra,
                    state.d_Sra, net.epsilon, state.d_mz, state.d_Sz, wposIn,
                    bposIn, zposOut, zposIn, sposIn, wihi, fi, Mbn, Kbn);

                convbnVar<<<dimGrid2, dimBlock>>>(
                    theta.d_mw, theta.d_Sw, theta.d_mb, theta.d_Sb, state.d_ma,
                    state.d_Sa, state.d_mra, state.d_Sra, net.epsilon,
                    state.d_mz, state.d_Sz, wposIn, bposIn, zposOut, zposIn,
                    sposIn, wihi, fi, Mbn, Kbn);
            }
        }
        //**
        // 21: Transpose convolutional
        //
        else if (net.layers[j] == net.layer_names.tconv) {
            // Launch kernel
            unsigned int gridRows = (B + THREADS - 1) / THREADS;
            unsigned int gridCols = (woho * ho + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCols, gridRows);
            dim3 dimBlock(THREADS, THREADS);

            tconvMean<<<dimGrid, dimBlock>>>(
                theta.d_mw, theta.d_mb, state.d_ma, idx.d_Fmwa_1, idx.d_Fmwa_2,
                wposIn, bposIn, zposIn, zposOut, net.Fmwa_1_pos[j - 1],
                net.Fmwa_2_pos[j - 1], woho, fo, wihi, fi, ki,
                net.Fmwa_1_col[j - 1], B, state.d_mz);

            tconvVar<<<dimGrid, dimBlock>>>(
                theta.d_mw, theta.d_Sw, theta.d_Sb, state.d_ma, state.d_Sa,
                idx.d_Fmwa_1, idx.d_Fmwa_2, wposIn, bposIn, zposIn, zposOut,
                net.Fmwa_1_pos[j - 1], net.Fmwa_2_pos[j - 1], woho, fo, wihi,
                fi, ki, net.Fmwa_1_col[j - 1], B, state.d_Sz);
        } else {
            std::cout << "Layer:" << j << "\n" << std::endl;
            throw std::invalid_argument(
                "Layer is invalid valid - feed_forward");
        }

        //**
        // Residual connection for CNN (xs: x shortcut)
        //
        if (j == net.init_sc) {
            int Nxs = fo * woho * B;
            // Launch kernel
            unsigned int gridRowdxs = (1 + THREADS - 1) / THREADS;
            unsigned int gridColdxs = (Nxs + THREADS - 1) / THREADS;
            dim3 dimGriddxs(gridColdxs, gridRowdxs);
            int xsposOut = net.sc_pos[j];

            duplicateMeanVarWithIndex<<<dimGriddxs, dimBlock>>>(
                state.d_mz, state.d_Sz, state.d_mdsc, state.d_Sdsc, zposOut,
                xsposOut, Nxs);

            duplicateMeanVarWithIndex<<<dimGriddxs, dimBlock>>>(
                state.d_mz, state.d_Sz, state.d_msc, state.d_Ssc, zposOut,
                xsposOut, Nxs);
        }
        if (xsOut != -1) {
            int fixs = net.filters[xsOut];  // num. of filter for shortcut layer
            int Nxs = fo * woho * B;        // numel of output
            int xsposOut = net.sc_pos[j];   // location of output
            int zxsposIn = net.z_pos[xsOut];  // location of input
            int wixs = net.widths[xsOut];     // width of shortcut
            int hixs = net.heights[xsOut];    // height of shortcut

            // Launch kernel
            unsigned int gridRowdxs = (1 + THREADS - 1) / THREADS;
            unsigned int gridColdxs = (Nxs + THREADS - 1) / THREADS;
            dim3 dimGriddxs(gridColdxs, gridRowdxs);

            duplicateMeanVarWithIndex<<<dimGriddxs, dimBlock>>>(
                state.d_mz, state.d_Sz, state.d_mdsc, state.d_Sdsc, zposOut,
                xsposOut, Nxs);

            if (fixs != fo || wixs != wo)  // size of shortcut # output
            {
                int wihixs = wixs * hixs;
                int ki2xs = 1;  // we use a conv. 1x1 to match the output size
                int N = fixs;
                int K = woho * B;
                int padIdx = wihixs * fixs * B + 1;  // padding index
                int xsidxposIn =
                    net.Fmwa_2_sc_pos[xsOut];  // location of input in kernel
                                               // indices vector

                // TODO need to fix xsOut - 1
                int wxsposIn =
                    net.w_sc_pos[xsOut -
                                 1];  // location of weights for shortcut
                int bxsposIn =
                    net.b_sc_pos[xsOut - 1];  // location of biases for shortcut

                // Launch kernel
                unsigned int gridRowxs = (fo + THREADS - 1) / THREADS;
                unsigned int gridColxs = (K + THREADS - 1) / THREADS;
                dim3 dimGridxs(gridColxs, gridRowxs);

                // Compute mean and variance
                convMean<<<dimGridxs, dimBlock>>>(
                    theta.d_mw_sc, theta.d_mb_sc, state.d_ma, idx.d_Fmwa_2_sc,
                    state.d_msc, wxsposIn, bxsposIn, zxsposIn, xsposOut,
                    xsidxposIn, woho, fo, wihixs, fixs, ki2xs, B, N, K, padIdx);

                convVar<<<dimGridxs, dimBlock>>>(
                    theta.d_mw_sc, theta.d_Sw_sc, theta.d_Sb_sc, state.d_ma,
                    state.d_Sa, idx.d_Fmwa_2_sc, state.d_Ssc, wxsposIn,
                    bxsposIn, zxsposIn, xsposOut, xsidxposIn, woho, fo, wihixs,
                    fixs, ki2xs, B, N, K, padIdx);
            } else {
                duplicateMeanVarWithIndex<<<dimGriddxs, dimBlock>>>(
                    state.d_mz, state.d_Sz, state.d_msc, state.d_Ssc, zxsposIn,
                    xsposOut, Nxs);
            }
            // Z = X + \delta_X
            twoPlusSc<<<dimGriddxs, dimBlock>>>(
                state.d_mdsc, state.d_Sdsc, state.d_msc, state.d_Ssc,
                state.d_mz, state.d_Sz, zposOut, xsposOut, Nxs);
        }
        //**
        // Activation
        //
        // Launch kernel
        int MB = M * B;
        unsigned int BLOCKS = (MB + THREADS - 1) / THREADS;
        // Compute mean, variance, and Jacobian matrix
        if (net.activations[j] == 1)  // tanh
        {
            tanhMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                             state.d_J, state.d_Sa, zposOut,
                                             MB);
        } else if (net.activations[j] == 2)  // sigmoid
        {
            sigmoidMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz,
                                                state.d_ma, state.d_J,
                                                state.d_Sa, zposOut, MB);
        } else if (net.activations[j] == 4)  // ReLU
        {
            reluMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                             state.d_J, state.d_Sa, zposOut,
                                             MB);
        } else if (net.activations[j] == 5)  // softplus
        {
            softplusMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz,
                                                 state.d_ma, state.d_J,
                                                 state.d_Sa, zposOut, MB);
        } else if (net.activations[j] == 6)  // leaky ReLU
        {
            leakyreluMeanVar<<<BLOCKS, THREADS>>>(
                state.d_mz, state.d_Sz, net.alpha, state.d_ma, state.d_J,
                state.d_Sa, zposOut, MB);
        } else  // no activation
        {
            noActMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz,
                                              state.d_ma, state.d_J, state.d_Sa,
                                              zposOut, MB);
        }
    }
}

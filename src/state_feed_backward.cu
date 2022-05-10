///////////////////////////////////////////////////////////////////////////
// File:         state_feed_backward.cu
// Description:  forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2021
// Updated:      May 09, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2021 Luong-Ha Nguyen & James-A. Goulet. All rights reserved.
///////////////////////////////////////////////////////////////////////////

#include "../include/state_feed_backward.cuh"

////////////////////////////////////////////////////////////////////////////////
/// LAST LAYER UPDATE
////////////////////////////////////////////////////////////////////////////////
__global__ void deltaMzSzWithIndices(float const *ma, float const *Sa,
                                     float const *Sz, float const *J,
                                     float const *y, float const *Sv,
                                     int const *udIdx, float *deltaMz,
                                     float *deltaSz, int zpos, int ny, int nye,
                                     int n)
/* Update output layer based on selected indices.

Args:
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian vector
    y: Observation
    Sv: Observation noise
    udIdx: Selected indiced to update
    deltaMz: Updated quantities for the mean of output's hidden states
    deltaSz: Updated quantities for the varaince of output's hidden states
    zpos: Hidden state's position for output layer
    ny: Size of the output layer
    nye: Number of observation to be updated
    n: Number of batches x size of output layer
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float tmp = 0;
    int idx = 0;
    if (col < n) {
        idx = udIdx[col] + (col / nye) * ny -
              1;  // minus 1 due to matlab's indexing
        tmp = (J[idx + zpos] * Sz[idx + zpos]) / (Sa[idx + zpos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            deltaMz[idx] = zeroPad;
            deltaSz[idx] = zeroPad;
        } else {
            deltaMz[idx] = tmp * (y[col] - ma[idx + zpos]);
            deltaSz[idx] = -tmp * (J[idx + zpos] * Sz[idx + zpos]);
        }
    }
}
__global__ void deltaMzSz(float const *ma, float const *Sa, float const *Sz,
                          float const *J, float const *y, float const *Sv,
                          float *deltaMz, float *deltaSz, int zpos, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float tmp = 0;
    if (col < n) {
        tmp = (J[col + zpos] * Sz[col + zpos]) / (Sa[col + zpos] + Sv[col]);
        if (isinf(tmp) || isnan(tmp)) {
            deltaMz[col] = zeroPad;
            deltaSz[col] = zeroPad;
        } else {
            deltaMz[col] = tmp * (y[col] - ma[col + zpos]);
            deltaSz[col] = -tmp * (J[col + zpos] * Sz[col + zpos]);
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
/// INOVATION VECTOR
////////////////////////////////////////////////////////////////////////////////
__global__ void inovationMean(float const *Sz, float const *deltaMz,
                              float *deltaM, int zpos, int zdeltapos, int n)
/* Compute the mean of the inovation vector.

Args:
    Sz: Variance of hidden states
    deltaMz: Updated quantities for the mean of output's hidden states
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    zpos: Hidden state's position for output
    zdeltapos: Position of the inovation vector for this layer
    n: Number of hidden states for input x number of batches
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float tmp = 0;
    if (col < n) {
        tmp = deltaMz[col] / Sz[col + zpos];
        if (isinf(tmp) || isnan(tmp)) {
            deltaM[col + zdeltapos] = zeroPad;
        } else {
            deltaM[col + zdeltapos] = tmp;
        }
    }
}

__global__ void inovationVar(float const *Sz, float const *deltaSz,
                             float *deltaS, int zpos, int zdeltapos, int n)
/* Compute the variance of the inovation vector.

Args:
    Sz: Variance of hidden states
    deltaSz: Updated quantities for the variance of output's hidden states
    deltaS: Inovation vector for variance i.e. (M_observation - M_prediction)
    zpos: Hidden state's position for output
    zdeltapos: Position of the inovation vector for this layer
    n: Number of hidden states for input x number of batches
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float tmp = 0;
    if (col < n) {
        tmp = deltaSz[col] / Sz[col + zpos];
        if (isinf(tmp) || isnan(tmp)) {
            deltaS[col + zdeltapos] = zeroPad;
        }

        else {
            deltaS[col + zdeltapos] = tmp / Sz[col + zpos];
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
/// FULL-CONNECTED
////////////////////////////////////////////////////////////////////////////////
__global__ void fcDeltaMz(float const *mw, float const *Sz, float const *J,
                          float const *deltaM, float *deltaMz, int wpos,
                          int zposIn, int zposOut, int ni, int no, int B)
/* Compute the updated quatitites of the mean of the hidden states.

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaMz: Updated quantities for the mean of output's hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    ni: Number of hidden units for input
    B: Number of batches
    no: Number of hidden units for output
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < B && row < ni) {
        for (int i = 0; i < no; i++) {
            sum += mw[ni * i + row + wpos] * deltaM[col * no + i + zposOut];
        }
        deltaMz[col * ni + row] =
            sum * Sz[col * ni + row + zposIn] * J[col * ni + row + zposIn];
    }
}

__global__ void fcDeltaSz(float const *mw, float const *Sz, float const *J,
                          float const *deltaS, float *deltaSz, int wpos,
                          int zposIn, int zposOut, int ni, int no, int B)
/* Compute the updated quatitites for the variance of the hidden states.

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    deltaSz: Updated quantities for the varaince of output's hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    ni: Number of hidden units for input
    B: Number of batches
    no: Number of hidden units for output
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < B && row < ni) {
        for (int i = 0; i < no; i++) {
            sum += mw[ni * i + row + wpos] * deltaS[col * no + i + zposOut] *
                   mw[ni * i + row + wpos];
        }
        deltaSz[col * ni + row] =
            sum * Sz[col * ni + row + zposIn] * Sz[col * ni + row + zposIn] *
            J[col * ni + row + zposIn] * J[col * ni + row + zposIn];
    }
}
///////////////////////////////////////////////////////////////////////////
/// CONVOLUTIONAL
///////////////////////////////////////////////////////////////////////////
__global__ void convDeltaMz(float const *mw, float const *Sz, float const *J,
                            float const *deltaM, int const *zwidx,
                            int const *zudidx, float *deltaMz, int wpos,
                            int zposIn, int jposIn, int zposOut, int zwidxpos,
                            int zudidxpos, int woho, int fo, int wihi, int fi,
                            int ki2, int nr, int n, int k, int padIdx)
/* Compute updated quantities of the mean of hidden states for convolutional
 layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    zwidx: Weight indices for covariance Z|WA i.e. FCzwa_1
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaMz: Updated quantities for the mean of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    zwidxpos: Position of weight indices for covariance Z|WA
    zudidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    ki2: ki x ki
    nr: Number of rows of weight indices for covariance Z|WA i.e. row_zw
    n: nr x fo
    k: wihi x B
    padIdx: Size of the hidden state vector for this layer + 1
 */
// TODO: remove jposIn
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int widx_tmp = 0;
    int aidx_tmp = 0;
    if (col < k && row < fi)  // k = wihi * B
    {
        for (int i = 0; i < n; i++) {
            // indices for mw. Note that nr = n / fo
            widx_tmp = zwidx[(col % wihi) * nr + i % nr + zwidxpos] +
                       (i / nr) * ki2 * fi + row * ki2 - 1;
            // minus 1 due to matlab's indexing

            // indices for deltaM
            aidx_tmp = zudidx[col % wihi + wihi * (i % nr) + zudidxpos] +
                       (i / nr) * woho + (col / wihi) * woho * fo;
            if (aidx_tmp < padIdx) {
                sum += deltaM[aidx_tmp - 1 + zposOut] * mw[widx_tmp + wpos];
            }
        }
        deltaMz[wihi * (col / wihi) * fi + col % wihi + row * wihi] =
            sum * Sz[row * k + col] * J[row * k + col];
    }
}

// This function computes the update amount for hidden state variance
// Sz_new = Sz_old + deltaSz
__global__ void convDeltaSz(float const *mw, float const *Sz, float const *J,
                            float const *deltaS, int const *zwidx,
                            int const *zudidx, float *deltaSz, int wpos,
                            int zposIn, int jposIn, int zposOut, int zwidxpos,
                            int zudidxpos, int woho, int fo, int wihi, int fi,
                            int ki2, int nr, int n, int k, int padIdx)
/* Compute updated quantities of the variance of hidden states for
 convolutional layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for variance i.e. (S_observation - S_prediction)
    zwidx: Weight indices for covariance Z|WA i.e. FCzwa_1
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    zwidxpos: Position of weight indices for covariance Z|WA
    zudidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    ki2: ki x ki
    nr: Number of rows of weight indices for covariance Z|WA i.e. row_zw
    n: nr x fo
    k: wihi x B
    padIdx: Size of the hidden state vector for this layer + 1
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int widx_tmp = 0;
    int aidx_tmp = 0;
    if (col < k && row < fi)  // k = wihi * B
    {
        for (int i = 0; i < n; i++) {
            // indices for mw. Note that nr = n / fo
            widx_tmp = zwidx[(col % wihi) * nr + i % nr + zwidxpos] +
                       (i / nr) * ki2 * fi + row * ki2 - 1;
            // minus 1 due to matlab's indexing

            // indices for deltaS
            aidx_tmp = zudidx[col % wihi + wihi * (i % nr) + zudidxpos] +
                       (i / nr) * woho + (col / wihi) * woho * fo;
            if (aidx_tmp < padIdx) {
                sum += mw[widx_tmp + wpos] * deltaS[aidx_tmp - 1 + zposOut] *
                       mw[widx_tmp + wpos];
            }
        }
        deltaSz[wihi * (col / wihi) * fi + col % wihi + row * wihi] =
            sum * Sz[row * k + col] * J[row * k + col] * Sz[row * k + col] *
            J[row * k + col];
    }
}
__global__ void permmuteMeanVar(float const *Szinit, float const *Jinit,
                                float *Sz, float *J, int zpos, int jpos,
                                int wihi, int fi, int B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < wihi * fi && row < B) {
        // Note that (col/(w * h)) equivalent to floorf((col/(w * h)))
        // because of interger division
        Sz[wihi * (col / wihi) * B + col % wihi + row * wihi] =
            Szinit[row * wihi * fi + col + zpos];
        J[wihi * (col / wihi) * B + col % wihi + row * wihi] =
            Jinit[row * wihi * fi + col + jpos];
    }
}
///////////////////////////////////////////////////////////////////////////
/// TRANSPOSE CONVOLUTIONAL
///////////////////////////////////////////////////////////////////////////
// This function computes the update amount for hidden state mean
// mz_new = mz_old + deltaMz
__global__ void tconvDeltaMz(float const *mw, float const *Sz, float const *J,
                             float const *deltaM, int const *widx,
                             int const *zidx, int wpos, int zposIn, int zposOut,
                             int widxpos, int zidxpos, int woho, int fo,
                             int wihi, int fi, int ki, int rf, int B,
                             float *deltaMz)
/* Compute updated quantities of the mean of hidden states for transpose
 convolutional layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    widx: Weight indices for covariance Z|WA i.e. FCzwa_1
    zidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    widxpos: Position of weight indices for covariance Z|WA
    zidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    rf: Number of column of weight indices for covariance Z|WA i.e. FCzwa_1_col
    B: Number of batches
    deltaMz: Updated quantities for the mean of the hidden states
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int K = wihi * fi;
    int widx_tmp = 0;
    int zidx_tmp = 0;        // updated index (idxSzzUd)
    if (col < K && row < B)  // k = wihi * fi, m = B
    {
        for (int i = 0; i < rf * fo; i++)  // n = ki2 * fo
        {
            // indices for mw
            widx_tmp = widx[(col % wihi) * ki * ki + i % rf + widxpos] +
                       (i / rf) * ki * ki + (col / wihi) * ki * ki * fo + wpos -
                       1;  // minus 1 due to matlab's indexing

            // indices for deltaM
            zidx_tmp = zidx[(col % wihi) * ki * ki + i % rf + zidxpos] +
                       (i / rf) * woho + row * woho * fo - 1;
            if (zidx_tmp + 1 < woho * fo * B + 1) {
                sum += deltaM[zidx_tmp + zposOut] * mw[widx_tmp];
            }
        }
        // TODO: Double check the definition zposIn
        deltaMz[col + row * K] =
            sum * Sz[col + row * K + zposIn] * J[col + row * K + zposIn];
    }
}

// This function computes the update amount for hidden state variance
// Sz_new = Sz_old + deltaSz
__global__ void tconvDeltaSz(float const *mw, float const *Sz, float const *J,
                             float const *deltaS, int const *widx,
                             int const *zidx, int wpos, int zposIn, int zposOut,
                             int widxpos, int zidxpos, int woho, int fo,
                             int wihi, int fi, int ki, int rf, int B,
                             float *deltaSz)
/* Compute updated quantities of the variance of hidden states for transpose
 convolutional layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    widx: Weight indices for covariance Z|WA i.e. FCzwa_1
    zidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    widxpos: Position of weight indices for covariance Z|WA
    zidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    rf: Number of column of weight indices for covariance Z|WA i.e. FCzwa_1_col
    B: Number of batches
    deltaSz: Updated quantities for the variance of the hidden states
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int K = wihi * fi;
    int widx_tmp = 0;
    int zidx_tmp = 0;  // updated index (idxSzzUd)

    if (col < K && row < B)  // k = wihi * fi, m = B
    {
        for (int i = 0; i < rf * fo; i++)  // n = ki2 * fo
        {
            // indices for mw
            widx_tmp = widx[(col % wihi) * ki * ki + i % rf + widxpos] +
                       (i / rf) * ki * ki + (col / wihi) * ki * ki * fo + wpos -
                       1;  // minus 1 due to matlab's indexing

            // indices for deltaM
            zidx_tmp = zidx[(col % wihi) * ki * ki + i % rf + zidxpos] +
                       (i / rf) * woho + row * woho * fo - 1;

            if (zidx_tmp + 1 < woho * fo * B + 1) {
                sum += mw[widx_tmp] * deltaS[zidx_tmp + zposOut] * mw[widx_tmp];
            }
        }

        deltaSz[col + row * K] =
            sum * Sz[col + row * K + zposIn] * J[col + row * K + zposIn] *
            Sz[col + row * K + zposIn] * J[col + row * K + zposIn];
    }
}

//////////////////////////////////////////////////////////////////////////
/// AVERAGE POOLING
//////////////////////////////////////////////////////////////////////////
__global__ void apDeltaMzSzOverlap(float const *Sz, float const *J,
                                   float const *deltaM, float const *deltaS,
                                   int const *zudidx, float *deltaMz,
                                   float *deltaSz, int zposIn, int jposIn,
                                   int zposOut, int zudidxpos, int woho,
                                   int wihi, int ki2, int n, int k, int padIdx)
/* Compute updated quantities for the mean and variance of hidden states for
 average pooling layer. Note that this case the kernel size overlap each other
 when scaning images.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    zudidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    wihi: Width x height of the input image
    ki: ki x ki
    n: Number of columns of next hidden state indices for covariance
       Z|Z+ i.e. col_z_ud
    k: Number of hidden units for input x number of batches
    padIdx: Size of the hidden state vector for this layer + 1
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumDeltaMz = 0;
    float sumDeltaSz = 0;
    int zidx_tmp = 0;
    if (col < k && row < 1) {
        for (int i = 0; i < n; i++) {
            zidx_tmp =
                zudidx[col % wihi + wihi * i + zudidxpos] + (col / wihi) * woho;
            // zidx_tmp = min(zidx_tmp_raw, padIdx) - 1; // minus 1 due to
            // matlab's indexing
            if (zidx_tmp < padIdx) {
                sumDeltaMz += deltaM[zidx_tmp - 1 + zposOut];
                sumDeltaSz += deltaS[zidx_tmp - 1 + zposOut];
            }
        }
        deltaMz[col] = sumDeltaMz * J[col + jposIn] * Sz[col + zposIn] / ki2;
        deltaSz[col] = sumDeltaSz * J[col + jposIn] * Sz[col + zposIn] *
                       J[col + jposIn] * Sz[col + zposIn] / (ki2 * ki2);
    }
}

__global__ void apDeltaMzSz(float const *Sz, float const *J,
                            float const *deltaM, float const *deltaS,
                            float *deltaMz, float *deltaSz, int zposIn,
                            int jposIn, int zposOut, int wo, int ki, int ki2,
                            int m, int k)
/* Compute updated quantities for the mean and variance of hidden states for
 average pooling layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    zudidxpos: Postision of next hidden state indices for covariance Z|Z+
    wo: Width for the output image
    ki: Kernel size
    ki: ki x ki
    m: ki x wo
    k: whi x ki x B / (ki x wo)
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < k && row < m)  // k = wihi * fi * B / (k*wo); m = k*wo
    {
        deltaMz[row + col * m] = deltaM[row / ki + (col / ki) * wo + zposOut] *
                                 J[row + col * m + jposIn] *
                                 Sz[row + col * m + zposIn] / ki2;

        deltaSz[row + col * m] = deltaS[row / ki + (col / ki) * wo + zposOut] *
                                 J[row + col * m + jposIn] *
                                 Sz[row + col * m + zposIn] *
                                 J[row + col * m + jposIn] *
                                 Sz[row + col * m + zposIn] / (ki2 * ki2);
    }
}

///////////////////////////////////////////////////////////////////////////
/// NORMALIZATION
///////////////////////////////////////////////////////////////////////////
// Conv. layer normalization
__global__ void convlnDeltaMzSz(float const *mw, float const *Sz,
                                float const *J, float const *Shat,
                                float const *deltaM, float const *deltaS,
                                float epsilon, float *deltaMz, float *deltaSz,
                                int wpos, int zposIn, int jposIn, int zposOut,
                                int rapos, int wihi, int m, int k)
/* Compute updated quantities for the mean and variance of hidden states for
LAYER-NORMALIZATION layer whose the previous layer is convolutional layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    Shat: Statistical variance for the normalzation layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    rapos: Statistical mean and variance position for the
                normalization layer
    wihi: Width x height for input image
    m: Number of batches
    k: Width x height x kernel size for input image
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < k && row < m)  // k = wihi * fi, m = B
    {
        A = (1 / sqrtf(Shat[row + rapos] + epsilon)) * mw[col / wihi + wpos] *
            J[col + row * k + jposIn] * Sz[col + row * k + zposIn];

        deltaMz[col + row * k] = A * deltaM[col + row * k + zposOut];
        deltaSz[col + row * k] = A * deltaS[col + row * k + zposOut] * A;
    }
}
// FC Layer Normalization
__global__ void fclnDeltaMzSz(float const *mw, float const *Sz, float const *J,
                              float const *Shat, float const *deltaM,
                              float const *deltaS, float epsilon,
                              float *deltaMz, float *deltaSz, int wpos,
                              int zposIn, int zposOut, int rapos, int ni, int B)
/* Compute updated quantities for the mean and variance of hidden states for
LAYER-NORMALIZATION layer whose the previous layer is full-connected layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    Shat: Statistical variance for the normalzation layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    rapos: Statistical mean and variance position for the
                normalization layer
    ni: Number of hidden units for input
    k: Number of batches
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < ni && row < B) {
        A = (1 / sqrtf(Shat[row + rapos] + epsilon)) * mw[col + wpos] *
            J[col + row * ni + zposIn] * Sz[col + row * ni + zposIn];

        deltaMz[col + row * ni] = A * deltaM[col + row * ni + zposOut];

        deltaSz[col + row * ni] = A * deltaS[col + row * ni + zposOut] * A;
    }
}

// Conv. batch normalization
__global__ void convbnDeltaMzSz(float const *mw, float const *Sz,
                                float const *J, float const *Shat,
                                float const *deltaM, float const *deltaS,
                                float epsilon, float *deltaMz, float *deltaSz,
                                int wpos, int zposIn, int jposIn, int zposOut,
                                int rapos, int wihi, int fi, int m)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is convolutional layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    Shat: Statistical variance for the normalzation layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    rapos: Statistical mean and variance position for the
                normalization layer
    wihi: Width x height for input image
    fi: Number of filters for input
    m:  Number of filters for input x number of batches
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < wihi && row < m)  // k = wihi * fi, m = B
    {
        A = (1 / sqrtf(Shat[row % fi + rapos] + epsilon)) *
            mw[row % fi + wpos] * J[col + row * wihi + jposIn] *
            Sz[col + row * wihi + zposIn];

        deltaMz[col + row * wihi] = A * deltaM[col + row * wihi + zposOut];

        deltaSz[col + row * wihi] = A * deltaS[col + row * wihi + zposOut] * A;
    }
}
// Full-connected batch normalization
__global__ void fcbnDeltaMzSz(float const *mw, float const *Sz, float const *J,
                              float const *Shat, float const *deltaM,
                              float const *deltaS, float epsilon,
                              float *deltaMz, float *deltaSz, int wpos,
                              int zposIn, int jposIn, int zposOut, int rapos,
                              int ni, int B)
/* Compute updated quantities for the mean and variance of hidden states for
BATCH-NORMALIZATION layer whose the previous layer is full-connected layer.

Args:
   mw: Mean of weights
   Sz: Variance of hidden states
   J: Jacobian vector
   Shat: Statistical variance for the normalzation layers
   deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
   deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
   epsilon: Constant for normalization layer to avoid zero-division
   deltaMz: Updated quantities for the mean of the hidden states
   deltaSz: Updated quantities for the variance of the hidden states
   wpos: Weight postision for this layer in the weight vector of network
   zposIn: Input-hidden-state postision for this layer in the weight vector
           of network
   jposIn: Postision os the Jacobian vector for this layer
   zposOut: Output-hidden-state postision for this layer in the weight vector
           of network
   rapos: Statistical mean and variance position for the
               normalization layer
   ni: Number of hidden units for input
   B:  Number of batches
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < ni && row < B) {
        A = (1 / sqrtf(Shat[col + rapos] + epsilon)) * mw[col + wpos] *
            J[col + row * ni + jposIn] * Sz[col + row * ni + zposIn];

        deltaMz[col + row * ni] = A * deltaM[col + row * ni + zposOut];

        deltaSz[col + row * ni] = A * deltaS[col + row * ni + zposOut] * A;
    }
}
////////////////////////////////////////////////////////////////////////////////
/// RESIDUAL NETWORKS
////////////////////////////////////////////////////////////////////////////////
__global__ void convDeltaMzsc(float const *mw, float const *Sz, float const *J,
                              float const *deltaM, int const *widx,
                              int const *aidx, int const *zudidx,
                              float *deltaMz, int wpos, int zpos, int jpos,
                              int zposOut, int widxpos, int aidxpos,
                              int zudidxpos, int woho, int fo, int realwihi,
                              int wihi, int fi, int ki2, int nr, int n, int k,
                              int B, int padIdx)
/* Compute updated quantities for the mean of hidden states for shortcut layer
 for redisual network. Note that we appy the convolutional layer for the
 shortcut in order to match the size of the image for the current layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    widx: Weight indices for covariance Z|WA i.e. FCzwa_1
    aidx: Activaiton indices for covariance Z|WA i.e. FCzwa_2
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaMz: Updated quantities for the mean of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    widxpos: Position of weight indices for covariance Z|WA
    aidxpos: Position of activation indices for covariance Z|WA
    zudidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    realwihi: Width x height of the shortcut image
    wihi: Width x height of the input
    fi: Number of filters of the input image
    ki2: ki x ki
    nr: Number of rows of weight indices for covariance Z|WA i.e. row_zw
    n: nr x fi
    k: cz x B where cz is number of columns of next hidden state indices for
       covariance Z|Z+
    B: Number of batches
    padIdx: Size of the hidden state vector for this layer + 1
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int widx_tmp = 0;
    int aidx_tmp = 0;
    int zudidx_tmp = 0;
    if (col < k && row < fi)  // k = wihi * B
    {
        for (int i = 0; i < n; i++) {
            // indices for mw. Note that nr = n / fo
            widx_tmp = widx[(col % wihi) * nr + i % nr + widxpos] +
                       (i / nr) * ki2 * fi + row * ki2 + wpos - 1;
            // minus 1 due to matlab's indexing

            // indices for deltaM
            zudidx_tmp = zudidx[col % wihi + wihi * (i % nr) + zudidxpos] +
                         (i / nr) * woho + (col / wihi) * woho * fo;
            if (zudidx_tmp < padIdx) {
                sum += deltaM[zudidx_tmp - 1 + zposOut] * mw[widx_tmp];
                // minus 1 due to  matlab's indexing
            }
        }
        aidx_tmp = aidx[col % wihi + aidxpos] + (col / wihi) * realwihi * fi +
                   row * realwihi - 1 + zpos;

        deltaMz[aidx_tmp] = sum * Sz[row * k + col] * J[row * k + col];
    }
}

// This function computes the update amount for hidden state variance
// Sz_new = Sz_old + deltaSz
__global__ void convDeltaSzsc(float const *mw, float const *Sz, float const *J,
                              float const *deltaS, int const *widx,
                              int const *aidx, int const *zudidx,
                              float *deltaSz, int wpos, int zpos, int jpos,
                              int zposOut, int widxpos, int aidxpos,
                              int zudidxpos, int woho, int fo, int realwihi,
                              int wihi, int fi, int ki2, int nr, int n, int k,
                              int B, int padIdx)
/* Compute updated quantities for the variance of hidden states for shortcut
 layer for redisual network. Note that we appy the convolutional layer for the
 shortcut in order to match the size of the image for the current layer.

 Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    widx: Weight indices for covariance Z|WA i.e. FCzwa_1
    aidx: Activaiton indices for covariance Z|WA i.e. FCzwa_2
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight postision for this layer in the weight vector of network
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    widxpos: Position of weight indices for covariance Z|WA
    aidxpos: Position of activation indices for covariance Z|WA
    zudidxpos: Postision of next hidden state indices for covariance Z|Z+
    woho: Width x height of the output image
    fo: Number of filters of the output image
    realwihi: Width x height of the shortcut image
    wihi: Width x height of the input
    fi: Number of filters of the input image
    ki2: ki x ki
    nr: Number of rows of weight indices for covariance Z|WA i.e. row_zw
    n: nr x fi
    k: cz x B where cz is number of columns of next hidden state indices for
       covariance Z|Z+
    B: Number of batches
    padIdx: Size of the hidden state vector for this layer + 1
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int widx_tmp = 0;
    int aidx_tmp = 0;
    int zudidx_tmp = 0;
    if (col < k && row < fi)  // k = wihi * B
    {
        for (int i = 0; i < n; i++) {
            // indices for mw. Note that nr = n / fo
            widx_tmp = widx[(col % wihi) * nr + i % nr + widxpos] +
                       (i / nr) * ki2 * fi + row * ki2 + wpos - 1;
            // minus 1 due to matlab's indexing

            // indices for deltaS
            zudidx_tmp = zudidx[col % wihi + wihi * (i % nr) + zudidxpos] +
                         (i / nr) * woho + (col / wihi) * woho * fo;
            if (zudidx_tmp < padIdx) {
                sum += mw[widx_tmp] * deltaS[zudidx_tmp - 1 + zposOut] *
                       mw[widx_tmp];
                // minus 1 due to matlab's indexing
            }
        }
        aidx_tmp = aidx[col % wihi + aidxpos] + (col / wihi) * realwihi * fi +
                   row * realwihi - 1 + zpos;

        deltaSz[aidx_tmp] = sum * Sz[row * k + col] * J[row * k + col] *
                            Sz[row * k + col] * J[row * k + col];
    }
}
__global__ void permuteMeanVarsc(float const *Ssc, float const *J,
                                 int const *aidx, float *Szp, float *Jp,
                                 int zpos, int jpos, int aidxpos, int realwihi,
                                 int wihi, int fi, int B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int aidxTmp = 0;
    if (col < wihi * fi && row < B) {
        // Note that (col/(w * h)) equvalent to floorf((col/(w * h)))
        // because of interger division
        aidxTmp = aidx[col % wihi + aidxpos] + (col / wihi) * realwihi +
                  row * realwihi * fi - 1;

        Szp[wihi * (col / wihi) * B + col % wihi + row * wihi] =
            Ssc[aidxTmp + zpos];

        Jp[wihi * (col / wihi) * B + col % wihi + row * wihi] =
            J[aidxTmp + jpos];
    }
}
__global__ void scDeltaMzSz(float const *Sz, float const *J,
                            float const *deltaM, float const *deltaS,
                            float *deltaMz, float *deltaSz, int zposIn,
                            int jposIn, int zposOut, int msposIn, int N)
/*Compute updated quantities for the variance of hidden states for shortcut
 layer for redisual network. Note that in this case we simply take the hidden
 states of the shortcut layer without changing its size.

Args:
    mw: Mean of weights
    Sz: Variance of hidden states
    J: Jacobian vector
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    zposIn: Input-hidden-state postision for this layer in the weight vector
            of network
    jposIn: Postision os the Jacobian vector for this layer
    zposOut: Output-hidden-state postision for this layer in the weight vector
            of network
    msposIn: Shorcut-hidden-state position
    N: Number of hidden units for the shortcut layer x number of batches.
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float C = 0;
    if (col < N) {
        C = J[col + jposIn] * Sz[col + zposIn];
        deltaMz[col + msposIn] = C * deltaM[col + zposOut];
        deltaSz[col + msposIn] = C * deltaS[col + zposOut] * C;
    }
}
__global__ void fourPlus(float const *deltaMzx0, float const *deltaSzx0,
                         float const *deltaMz0, float const *deltaSz0,
                         float const *deltaMsc, float const *deltaSsc,
                         float const *deltaMdsc, float const *deltaSdsc,
                         float *deltaMzx, float *deltaSzx, float *deltaMz,
                         float *deltaSz, int scpos, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        deltaMzx[col] = deltaMzx0[col] + deltaMsc[col + scpos];
        deltaSzx[col] = deltaSzx0[col] + deltaSsc[col + scpos];
        deltaMz[col] = deltaMz0[col] + deltaMdsc[col + scpos];
        deltaSz[col] = deltaSz0[col] + deltaSdsc[col + scpos];
    }
}
__global__ void twoPlus(float const *deltaMz0, float const *deltaSz0,
                        float const *deltaMdsc, float const *deltaSdsc,
                        float *deltaMz, float *deltaSz, int scpos, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        deltaMz[col] = deltaMz0[col] + deltaMdsc[col + scpos];
        deltaSz[col] = deltaSz0[col] + deltaSdsc[col + scpos];
    }
}
// GET DELTA STATE FOR INPUT LAYER
__global__ void getInputDeltaState(float const *delta_m, float const *delta_S,
                                   int niB, float *delta_m_0,
                                   float *delta_S_0) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < niB) {
        delta_m_0[col] = delta_m[col];
        delta_S_0[col] = delta_S[col];
    }
}

// DEBUG FUNCITON
__global__ void duplicateMeanVar(float const *m1, float const *S1, float *m,
                                 float *S, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        m[col] = m1[col];
        S[col] = S1[col];
    }
}

///////////////////////////////////////////////////////////////////////////
/// STATE BACKWARD PASS
///////////////////////////////////////////////////////////////////////////
void stateBackward(Network &net, ParamGPU &theta, StateGPU &state,
                   IndexGPU &idx, ObsGPU &obs, DeltaStateGPU &d_state)
/*Compute the updated quantities for network's hidden states using TAGI.

  Args:
    net: Network architecture
    theta: Network's weights and biases
    state: Hidden state of network
    idx: Indices for network e.g. see indices.cpp
    obs: Observations

  Returns:
    d_state: Updated quantities for network's hidden states.
 */
{
    // Launch kernel
    int THREADS = 16;
    dim3 dimBlock(THREADS, THREADS);

    // Declare variables
    int zposL, NL, zposIn, zposOut, wposIn, no, ni, ki, fi, wi;
    int hi, fo, wo, ho;
    int ki2, wihi, woho, niB, padIdx, xsIn, nextSc2ud;
    int fisc, scposOut, scposIn, zscposIn, wisc, hisc;
    int wihisc, ki2sc, rwIn, czIn, czInB, rwInfo, wscposIn;
    int zwscidxposIn;
    int zascidxposIn, zudscidxposIn, nisc, niscB;
    int rowwIn, rowwfo, wihiB;
    int colzIn, kiwo, K, raposIn, wihifi, fiB, xposIn;
    int nL, padIdxIn, nLB;
    unsigned int gridRow, gridCol;
    int B = net.batch_size;
    int numLayers = net.layers.size();

    zposL = net.z_pos[numLayers - 1];  // minus 1 due to matlab's index
    nL = net.nodes[numLayers - 1];
    nLB = nL * B;
    // Output-layer update
    if (net.is_output_ud)  // Update hidden states for the output layer
    {
        if (!net.is_idx_ud) {
            // gridRowL = (1 + THREADS - 1) / THREADS;
            // gridColL = (nLB + THREADS - 1) / THREADS;
            // dim3 dimGridL(gridColL, gridRowL);
            int BLOCKS_UD = (nLB + THREADS - 1) / THREADS;
            deltaMzSz<<<BLOCKS_UD, THREADS>>>(
                state.d_ma, state.d_Sa, state.d_Sz, state.d_J, obs.d_y_batch,
                obs.d_V_batch, d_state.d_delta_mz, d_state.d_delta_Sz, zposL,
                nLB);
        } else  // Only update the hidden states in the indices udIdx
        {
            NL = net.nye * B;
            // Launch kernel
            // gridRowL = (1 + THREADS - 1) / THREADS;
            // gridColL = (NL + THREADS - 1) / THREADS;
            // dim3 dimGridL(gridColL, gridRowL);
            int BLOCKS_UD = (NL + THREADS - 1) / THREADS;
            deltaMzSzWithIndices<<<BLOCKS_UD, THREADS>>>(
                state.d_ma, state.d_Sa, state.d_Sz, state.d_J, obs.d_y_batch,
                obs.d_V_batch, obs.d_idx_ud_batch, d_state.d_delta_mz,
                d_state.d_delta_Sz, zposL, net.nodes[numLayers - 1], net.nye,
                NL);
        }
    }
    // Connected network such as GAN and Autoencoder
    else {
        // gridRowL = (1 + THREADS - 1) / THREADS;
        // gridColL = (nLB + THREADS - 1) / THREADS;
        // dim3 dimGridL(gridColL, gridRowL);
        int BLOCKS_UD = (nLB + THREADS - 1) / THREADS;
        duplicateMeanVar<<<BLOCKS_UD, THREADS>>>(obs.d_y_batch, obs.d_V_batch,
                                                 d_state.d_delta_mz,
                                                 d_state.d_delta_Sz, nLB);
    }
    // Inovation vector for output layer
    // unsigned int gridRowI = (1 + THREADS - 1) / THREADS;
    // unsigned int gridColI = (nLB + THREADS - 1) / THREADS;
    // dim3 dimGridI(gridColI, gridRowI);
    int BLOCKS_I = (nLB + THREADS - 1) / THREADS;
    // Inovavation vector for Z
    inovationMean<<<BLOCKS_I, THREADS>>>(state.d_Sz, d_state.d_delta_mz,
                                         d_state.d_delta_m, zposL, zposL, nLB);
    inovationVar<<<BLOCKS_I, THREADS>>>(state.d_Sz, d_state.d_delta_Sz,
                                        d_state.d_delta_S, zposL, zposL, nLB);

    // Hidden-layer update
    nextSc2ud = -1;
    for (int k = numLayers - 1; k-- > net.last_backward_layer;) {
        // General hyperparameters
        zposIn = net.z_pos[k];       // location of actv. input
        zposOut = net.z_pos[k + 1];  // location of h.s. output
        wposIn = net.w_pos[k];       // location of weights in param. vector
        no = net.nodes[k + 1];       // num. of nodes for output
        ni = net.nodes[k];           // num. of nodes for input

        // Hyperparameters are requried only for CNN.
        ki = net.kernels[k];      // kernel size of input
        fi = net.filters[k];      // num. of filters for input
        wi = net.widths[k];       // width of input
        hi = net.heights[k];      // height of input
        fo = net.filters[k + 1];  // num. of filters for output
        wo = net.widths[k + 1];   // width of output
        ho = net.heights[k + 1];  // height of output
        ki2 = ki * ki;
        wihi = wi * hi;
        woho = wo * ho;
        niB = ni * B;
        padIdx = woho * fo * B + 1;  // padding index (following TAGI matlab)

        // Hyperparameters for residual networks. Note that current version
        // works only with CNN layer. Future version will include other layers.
        xsIn = net.shortcuts[k];

        //**
        // 1: Full connected
        //
        if (net.layers[k + 1] == net.layer_names.fc) {
            // Launch kernel
            gridRow = (ni + THREADS - 1) / THREADS;
            gridCol = (B + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCol, gridRow);
            // Compute mean and variance
            fcDeltaMz<<<dimGrid, dimBlock>>>(
                theta.d_mw, state.d_Sz, state.d_J, d_state.d_delta_m,
                d_state.d_delta_mz, wposIn, zposIn, zposOut, ni, no, B);

            fcDeltaSz<<<dimGrid, dimBlock>>>(
                theta.d_mw, state.d_Sz, state.d_J, d_state.d_delta_S,
                d_state.d_delta_Sz, wposIn, zposIn, zposOut, ni, no, B);
        }

        //**
        // 2: Convolutional layer
        //
        else if (net.layers[k + 1] == net.layer_names.conv) {
            rowwIn = net.row_zw[k];
            rowwfo = rowwIn * fo;
            wihiB = wihi * B;
            // Permute deltaM and deltaS
            unsigned int gridRowP = (B + THREADS - 1) / THREADS;
            unsigned int gridColP = (wihi * fi + THREADS - 1) / THREADS;
            gridRow = (fi + THREADS - 1) / THREADS;
            gridCol = (wihiB + THREADS - 1) / THREADS;
            dim3 dimGridP(gridColP, gridRowP);
            dim3 dimGrid(gridCol, gridRow);

            // We should find another way to avoid permuting these two vectors
            // J, Sz
            if (xsIn == -1) {
                permmuteMeanVar<<<dimGridP, dimBlock>>>(
                    state.d_Sz, state.d_J, d_state.d_dummy_S, d_state.d_dummy_m,
                    zposIn, zposIn, wihi, fi, B);

                // Compute deltaMz & deltaSz
                convDeltaMz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_m, idx.d_FCzwa_1, idx.d_Szz_ud,
                    d_state.d_delta_mz, wposIn, zposIn, zposIn, zposOut,
                    net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi,
                    ki2, rowwIn, rowwfo, wihiB, padIdx);

                convDeltaSz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_S, idx.d_FCzwa_1, idx.d_Szz_ud,
                    d_state.d_delta_Sz, wposIn, zposIn, zposIn, zposOut,
                    net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi,
                    ki2, rowwIn, rowwfo, wihiB, padIdx);
            } else {
                scposIn = net.sc_pos[k];

                // Compute deltaMz & deltaSz
                permmuteMeanVar<<<dimGridP, dimBlock>>>(
                    state.d_Sdsc, state.d_J, d_state.d_dummy_S,
                    d_state.d_dummy_m, scposIn, zposIn, wihi, fi, B);

                convDeltaMz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_m, idx.d_FCzwa_1, idx.d_Szz_ud,
                    d_state.d_delta_mz, wposIn, scposIn, zposIn, zposOut,
                    net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi,
                    ki2, rowwIn, rowwfo, wihiB, padIdx);

                convDeltaSz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_S, idx.d_FCzwa_1, idx.d_Szz_ud,
                    d_state.d_delta_Sz, wposIn, scposIn, zposIn, zposOut,
                    net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi,
                    ki2, rowwIn, rowwfo, wihiB, padIdx);

                // Compute deltaMx & deltaSx
                permmuteMeanVar<<<dimGridP, dimBlock>>>(
                    state.d_Ssc, state.d_J, d_state.d_dummy_S,
                    d_state.d_dummy_m, scposIn, zposIn, wihi, fi, B);

                convDeltaMz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_m, idx.d_FCzwa_1, idx.d_Szz_ud,
                    d_state.d_delta_mzsc, wposIn, scposIn, zposIn, zposOut,
                    net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi,
                    ki2, rowwIn, rowwfo, wihiB, padIdx);

                convDeltaSz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_S, idx.d_FCzwa_1, idx.d_Szz_ud,
                    d_state.d_delta_Szsc, wposIn, scposIn, zposIn, zposOut,
                    net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi,
                    ki2, rowwIn, rowwfo, wihiB, padIdx);
            }
        }

        //**
        // 3: Average pooling
        //
        else if (net.layers[k + 1] == net.layer_names.ap) {
            if (net.overlap[k] == 1) {
                colzIn = net.col_z_ud[k];
                // Launch kernel
                gridRow = 1;
                gridCol = (niB + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);

                // Compute deltaMz, deltaSz, deltaMzsc, deltaSzsc
                if (xsIn == -1) {
                    apDeltaMzSzOverlap<<<dimGrid, dimBlock>>>(
                        state.d_Sz, state.d_J, d_state.d_delta_m,
                        d_state.d_delta_S, idx.d_Szz_ud, d_state.d_delta_mz,
                        d_state.d_delta_Sz, zposIn, zposIn, zposOut,
                        net.Szz_ud_pos[k], woho, wihi, ki2, colzIn, niB,
                        padIdx);
                } else {
                    int scposIn = net.sc_pos[k];
                    apDeltaMzSzOverlap<<<dimGrid, dimBlock>>>(
                        state.d_Sdsc, state.d_J, d_state.d_delta_m,
                        d_state.d_delta_S, idx.d_Szz_ud, d_state.d_delta_mz,
                        d_state.d_delta_Sz, scposIn, zposIn, zposOut,
                        net.Szz_ud_pos[k], woho, wihi, ki2, colzIn, niB,
                        padIdx);

                    apDeltaMzSzOverlap<<<dimGrid, dimBlock>>>(
                        state.d_Ssc, state.d_J, d_state.d_delta_m,
                        d_state.d_delta_S, idx.d_Szz_ud, d_state.d_delta_mzsc,
                        d_state.d_delta_Szsc, scposIn, zposIn, zposOut,
                        net.Szz_ud_pos[k], woho, wihi, ki2, colzIn, niB,
                        padIdx);
                }
            } else {
                kiwo = ki * wo;
                K = wihi * fi * B / kiwo;
                // Lauch kernel
                gridRow = (kiwo + THREADS - 1) / THREADS;
                gridCol = (K + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);

                // Compute deltaMz, deltaSz, deltaMzsc, deltaSzsc
                if (xsIn == -1) {
                    apDeltaMzSz<<<dimGrid, dimBlock>>>(
                        state.d_Sz, state.d_J, d_state.d_delta_m,
                        d_state.d_delta_S, d_state.d_delta_mz,
                        d_state.d_delta_Sz, zposIn, zposIn, zposOut, wo, ki,
                        ki2, kiwo, K);
                } else {
                    scposIn = net.sc_pos[k];

                    apDeltaMzSz<<<dimGrid, dimBlock>>>(
                        state.d_Sdsc, state.d_J, d_state.d_delta_m,
                        d_state.d_delta_S, d_state.d_delta_mz,
                        d_state.d_delta_Sz, scposIn, zposIn, zposOut, wo, ki,
                        ki2, kiwo, K);

                    apDeltaMzSz<<<dimGrid, dimBlock>>>(
                        state.d_Ssc, state.d_J, d_state.d_delta_m,
                        d_state.d_delta_S, d_state.d_delta_mzsc,
                        d_state.d_delta_Szsc, scposIn, zposIn, zposOut, wo, ki,
                        ki2, kiwo, K);
                }
            }
        }

        //**
        // 5: Layer normalization
        //
        else if (net.layers[k + 1] == net.layer_names.ln) {
            raposIn = net.ra_pos[k];
            if (net.layers[k] == 1) {
                gridRow = (B + THREADS - 1) / THREADS;
                gridCol = (ni + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);
                fclnDeltaMzSz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, state.d_Sz, state.d_J, state.d_Sra,
                    d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                    d_state.d_delta_mz, d_state.d_delta_Sz, wposIn, zposIn,
                    zposOut, raposIn, ni, B);
            } else if (net.layers[k] == 2) {
                wihifi = wi * hi * fi;
                // Launch kernel
                gridRow = (B + THREADS - 1) / THREADS;
                gridCol = (wihifi + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);

                // Compute deltaMz, deltaSz, deltaMzsc, deltaSzsc
                if (xsIn == -1) {
                    convlnDeltaMzSz<<<dimGrid, dimBlock>>>(
                        theta.d_mw, state.d_Sz, state.d_J, state.d_Sra,
                        d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                        d_state.d_delta_mz, d_state.d_delta_Sz, wposIn, zposIn,
                        zposIn, zposOut, raposIn, wihi, B, wihifi);
                } else {
                    scposIn = net.sc_pos[k];

                    convlnDeltaMzSz<<<dimGrid, dimBlock>>>(
                        theta.d_mw, state.d_Sdsc, state.d_J, state.d_Sra,
                        d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                        d_state.d_delta_mz, d_state.d_delta_Sz, wposIn, scposIn,
                        zposIn, zposOut, raposIn, wihi, B, wihifi);

                    convlnDeltaMzSz<<<dimGrid, dimBlock>>>(
                        theta.d_mw, state.d_Ssc, state.d_J, state.d_Sra,
                        d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                        d_state.d_delta_mzsc, d_state.d_delta_Szsc, wposIn,
                        scposIn, zposIn, zposOut, raposIn, wihi, B, wihifi);
                }
            }
        }

        //**
        // 6: Batch normalization
        //
        else if (net.layers[k + 1] == net.layer_names.bn) {
            raposIn = net.ra_pos[k];
            if (net.layers[k] == 1) {
                gridRow = (B + THREADS - 1) / THREADS;
                gridCol = (ni + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);
                fcbnDeltaMzSz<<<dimGrid, dimBlock>>>(
                    theta.d_mw, state.d_Sz, state.d_J, state.d_Sra,
                    d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                    d_state.d_delta_mz, d_state.d_delta_Sz, wposIn, zposIn,
                    zposIn, zposOut, raposIn, ni, B);
            } else if (net.layers[k] == 2) {
                fiB = fi * B;

                // Launch kernel
                gridRow = (fiB + THREADS - 1) / THREADS;
                gridCol = (wihi + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);

                // Compute deltaMz, deltaSz, deltaMzsc, deltaSzsc
                if (xsIn == -1) {
                    convbnDeltaMzSz<<<dimGrid, dimBlock>>>(
                        theta.d_mw, state.d_Sz, state.d_J, state.d_Sra,
                        d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                        d_state.d_delta_mz, d_state.d_delta_Sz, wposIn, zposIn,
                        zposIn, zposOut, raposIn, wihi, fi, fiB);
                } else {
                    scposIn = net.sc_pos[k];

                    convbnDeltaMzSz<<<dimGrid, dimBlock>>>(
                        theta.d_mw, state.d_Sdsc, state.d_J, state.d_Sra,
                        d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                        d_state.d_delta_mz, d_state.d_delta_Sz, wposIn, scposIn,
                        zposIn, zposOut, raposIn, wihi, fi, fiB);

                    convbnDeltaMzSz<<<dimGrid, dimBlock>>>(
                        theta.d_mw, state.d_Ssc, state.d_J, state.d_Sra,
                        d_state.d_delta_m, d_state.d_delta_S, net.epsilon,
                        d_state.d_delta_mzsc, d_state.d_delta_Szsc, wposIn,
                        scposIn, zposIn, zposOut, raposIn, wihi, fi, fiB);
                }
            }
        }

        //**
        // 21: Transpose convolutional
        //
        else if (net.layers[k + 1] == net.layer_names.tconv) {
            // Launch kernel
            unsigned int gridRows = (B + THREADS - 1) / THREADS;
            unsigned int gridCols = (wihi * fi + THREADS - 1) / THREADS;

            dim3 dimGrid(gridCols, gridRows);

            tconvDeltaMz<<<dimGrid, dimBlock>>>(
                theta.d_mw, state.d_Sz, state.d_J, d_state.d_delta_m,
                idx.d_FCzwa_1, idx.d_Szz_ud, wposIn, zposIn, zposOut,
                net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi, ki,
                net.FCzwa_1_col[k], B, d_state.d_delta_mz);

            tconvDeltaSz<<<dimGrid, dimBlock>>>(
                theta.d_mw, state.d_Sz, state.d_J, d_state.d_delta_S,
                idx.d_FCzwa_1, idx.d_Szz_ud, wposIn, zposIn, zposOut,
                net.FCzwa_1_pos[k], net.Szz_ud_pos[k], woho, fo, wihi, fi, ki,
                net.FCzwa_1_col[k], B, d_state.d_delta_Sz);
        }

        else {
            std::cout << "Layer:" << k + 1 << "\n" << std::endl;
            throw std::invalid_argument(
                "Layer is invalid - state_feed_backward ");
        }

        // Update the shorcut
        if (k == nextSc2ud) {
            xposIn = net.sc_pos[k];
            unsigned int gridColsc = (niB + THREADS - 1) / THREADS;
            unsigned int gridRowsc = (1 + THREADS - 1) / THREADS;
            dim3 dimGridsc(gridColsc, gridRowsc);
            if (k != net.init_sc) {
                duplicateMeanVar<<<dimGridsc, dimBlock>>>(
                    d_state.d_delta_mz, d_state.d_delta_Sz, d_state.d_dummy_m,
                    d_state.d_dummy_S, niB);

                twoPlus<<<dimGridsc, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, d_state.d_delta_mdsc,
                    d_state.d_delta_Sdsc, d_state.d_delta_mz,
                    d_state.d_delta_Sz, xposIn, niB);

                duplicateMeanVar<<<dimGridsc, dimBlock>>>(
                    d_state.d_delta_mzsc, d_state.d_delta_Szsc,
                    d_state.d_dummy_m, d_state.d_dummy_S, niB);

                twoPlus<<<dimGridsc, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, d_state.d_delta_msc,
                    d_state.d_delta_Ssc, d_state.d_delta_mzsc,
                    d_state.d_delta_Szsc, xposIn, niB);
            } else {
                duplicateMeanVar<<<dimGridsc, dimBlock>>>(
                    d_state.d_delta_mz, d_state.d_delta_Sz, d_state.d_dummy_m,
                    d_state.d_dummy_S, niB);

                twoPlus<<<dimGridsc, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, d_state.d_delta_mdsc,
                    d_state.d_delta_Sdsc, d_state.d_delta_mz,
                    d_state.d_delta_Sz, xposIn, niB);
            }
        }

        //**
        // Update shortcut's hidden states
        //
        unsigned int gridRowI = (1 + THREADS - 1) / THREADS;
        unsigned int gridColI = (niB + THREADS - 1) / THREADS;
        dim3 dimGridI(gridColI, gridRowI);
        if (xsIn != -1) {
            nextSc2ud = xsIn;
            fisc = net.filters[xsIn];  // num. of filter for shorcut layer
            scposOut = net.sc_pos[k];  // location of output
            scposIn = net.sc_pos[xsIn];
            zscposIn = net.z_pos[xsIn];  // location of input
            wisc = net.widths[xsIn];     // width of shortcut
            hisc = net.heights[xsIn];    // height of shortcut
            padIdxIn = wihi * fi * B + 1;

            // Inovavation vector For z
            inovationMean<<<dimGridI, dimBlock>>>(
                state.d_Sdsc, d_state.d_delta_mz, d_state.d_delta_m, scposOut,
                zposIn, niB);

            inovationVar<<<dimGridI, dimBlock>>>(
                state.d_Sdsc, d_state.d_delta_Sz, d_state.d_delta_S, scposOut,
                zposIn, niB);

            // Inovation vector for X
            unsigned int gridRowIsc = (1 + THREADS - 1) / THREADS;
            unsigned int gridColIsc = (niB + THREADS - 1) / THREADS;
            dim3 dimGridIsc(gridColIsc, gridRowIsc);
            inovationMean<<<dimGridIsc, dimBlock>>>(
                state.d_Ssc, d_state.d_delta_mzsc, d_state.d_delta_mx, scposOut,
                scposOut, niB);

            inovationVar<<<dimGridIsc, dimBlock>>>(
                state.d_Ssc, d_state.d_delta_Szsc, d_state.d_delta_Sx, scposOut,
                scposOut, niB);

            if (fisc != fi || wisc != wi)  // size of shortcut # output
            {
                wihisc = wisc * hisc;
                ki2sc = 1;
                rwIn = net.row_w_sc[xsIn];
                czIn = net.col_z_sc[xsIn];
                czInB = czIn * B;
                rwInfo = rwIn * fi;
                wscposIn = net.Fmwa_2_sc_pos[xsIn];
                zwscidxposIn = net.FCzwa_1_sc_pos[xsIn];
                zascidxposIn = net.FCzwa_2_sc_pos[xsIn];
                zudscidxposIn = net.Szz_ud_sc_pos[xsIn];

                // Hidden states
                unsigned int gridRowsc = (fisc + THREADS - 1) / THREADS;
                unsigned int gridColsc = (czInB + THREADS - 1) / THREADS;
                unsigned int gridRowPsc = (B + THREADS - 1) / THREADS;
                unsigned int gridColPsc =
                    (wihisc * fisc + THREADS - 1) / THREADS;
                dim3 dimGridsc(gridColsc, gridRowsc);
                dim3 dimGridPsc(gridColPsc, gridRowPsc);

                // Shorcut X
                permuteMeanVarsc<<<dimGridPsc, dimBlock>>>(
                    state.d_Ssc, state.d_J, idx.d_FCzwa_2_sc, d_state.d_dummy_S,
                    d_state.d_dummy_m, scposIn, zscposIn, zascidxposIn, wihisc,
                    czIn, fisc, B);

                convDeltaMzsc<<<dimGridsc, dimBlock>>>(
                    theta.d_mw_sc, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_mx, idx.d_FCzwa_1_sc, idx.d_FCzwa_2_sc,
                    idx.d_Szz_ud_sc, d_state.d_delta_msc, wscposIn, scposIn,
                    zscposIn, scposOut, zwscidxposIn, zascidxposIn,
                    zudscidxposIn, wihi, fi, wihisc, czIn, fisc, ki2sc, rwIn,
                    rwInfo, czInB, B, padIdxIn);

                convDeltaSzsc<<<dimGridsc, dimBlock>>>(
                    theta.d_mw_sc, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_Sx, idx.d_FCzwa_1_sc, idx.d_FCzwa_2_sc,
                    idx.d_Szz_ud_sc, d_state.d_delta_Ssc, wscposIn, scposIn,
                    zscposIn, scposOut, zwscidxposIn, zascidxposIn,
                    zudscidxposIn, wihi, fi, wihisc, czIn, fisc, ki2sc, rwIn,
                    rwInfo, czInB, B, padIdxIn);

                // Shortcut dX
                permuteMeanVarsc<<<dimGridPsc, dimBlock>>>(
                    state.d_Sdsc, state.d_J, idx.d_FCzwa_2_sc,
                    d_state.d_dummy_S, d_state.d_dummy_m, scposIn, zscposIn,
                    zascidxposIn, wihisc, czIn, fisc, B);

                convDeltaMzsc<<<dimGridsc, dimBlock>>>(
                    theta.d_mw_sc, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_mx, idx.d_FCzwa_1_sc, idx.d_FCzwa_2_sc,
                    idx.d_Szz_ud_sc, d_state.d_delta_mdsc, wscposIn, scposIn,
                    zscposIn, scposOut, zwscidxposIn, zascidxposIn,
                    zudscidxposIn, wihi, fi, wihisc, czIn, fisc, ki2sc, rwIn,
                    rwInfo, czInB, B, padIdxIn);

                convDeltaSzsc<<<dimGridsc, dimBlock>>>(
                    theta.d_mw_sc, d_state.d_dummy_S, d_state.d_dummy_m,
                    d_state.d_delta_Sx, idx.d_FCzwa_1_sc, idx.d_FCzwa_2_sc,
                    idx.d_Szz_ud_sc, d_state.d_delta_Sdsc, wscposIn, scposIn,
                    zscposIn, scposOut, zwscidxposIn, zascidxposIn,
                    zudscidxposIn, wihi, fi, wihisc, czIn, fisc, ki2sc, rwIn,
                    rwInfo, czInB, B, padIdxIn);
            } else {
                nisc = net.nodes[xsIn];
                niscB = nisc * B;
                unsigned int gridRowsc = (1 + THREADS - 1) / THREADS;
                unsigned int gridColsc = (niscB + THREADS - 1) / THREADS;
                dim3 dimGridsc(gridColsc, gridRowsc);
                if (xsIn != net.init_sc) {
                    scDeltaMzSz<<<dimGridsc, dimBlock>>>(
                        state.d_Sdsc, state.d_J, d_state.d_delta_mx,
                        d_state.d_delta_Sx, d_state.d_delta_mdsc,
                        d_state.d_delta_Sdsc, scposIn, zscposIn, scposOut,
                        scposIn, niscB);

                    scDeltaMzSz<<<dimGridsc, dimBlock>>>(
                        state.d_Ssc, state.d_J, d_state.d_delta_mx,
                        d_state.d_delta_Sx, d_state.d_delta_msc,
                        d_state.d_delta_Ssc, scposIn, zscposIn, scposOut,
                        scposIn, niscB);
                } else {
                    scDeltaMzSz<<<dimGridsc, dimBlock>>>(
                        state.d_Sz, state.d_J, d_state.d_delta_mx,
                        d_state.d_delta_Sx, d_state.d_delta_mdsc,
                        d_state.d_delta_Sdsc, zscposIn, zscposIn, scposOut,
                        scposIn, niscB);
                }
            }
        } else {
            // Inovavation vector for Z
            int BLOCKS_I = (niB + THREADS - 1) / THREADS;
            inovationMean<<<BLOCKS_I, THREADS>>>(state.d_Sz, d_state.d_delta_mz,
                                                 d_state.d_delta_m, zposIn,
                                                 zposIn, niB);

            inovationVar<<<BLOCKS_I, THREADS>>>(state.d_Sz, d_state.d_delta_Sz,
                                                d_state.d_delta_S, zposIn,
                                                zposIn, niB);
        }
    }
}

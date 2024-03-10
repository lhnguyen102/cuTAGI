///////////////////////////////////////////////////////////////////////////
// File:         state_feed_backward.cu
// Description:  forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      August 07, 2021
// Updated:      March 06, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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
    nye: Number of observation to be updated for an observation
    n: Number of batches x size of output layer
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zeroPad = 0;
    float tmp = 0;
    int idx = 0;
    if (col < n) {
        // minus 1 due to matlab's indexing
        idx = udIdx[col] + (col / nye) * ny - 1;
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
/// CLOSED-FORM SOFTMAX
////////////////////////////////////////////////////////////////////////////////
__global__ void delta_z_y_check(float const *mu_a, float const *var_a,
                                float const *cov_y_y_check, float const *y,
                                float const *var_noise, int no, int B,
                                int z_pos, float *delta_mu_zy_check,
                                float *delta_var_zy_check)
/*Compute updating quantities for \check{y}*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp, zero_pad = 0;
    int idx;
    if (row < B && col < no) {
        idx = row * no + col;
        tmp = cov_y_y_check[idx] / (var_a[idx + z_pos] + var_noise[idx]);
        if (isinf(tmp) || isnan(tmp)) {
            delta_mu_zy_check[idx] = zero_pad;
            delta_var_zy_check[idx] = zero_pad;
        } else {
            delta_mu_zy_check[idx] = tmp * (y[idx] - mu_a[idx + z_pos]);
            delta_var_zy_check[idx] = -tmp * cov_y_y_check[idx];
        }
    }
}

__global__ void delta_z_softmax(float const *cov_z_y_check,
                                float const *delta_mu, float const *delta_var,
                                int no, int B, float *delta_mu_z,
                                float *delta_var_z)
/*Compute updating quantities for hidden states for the softmax layer*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < no * B) {
        delta_mu_z[col] = cov_z_y_check[col] * delta_mu[col];
        delta_var_z[col] =
            cov_z_y_check[col] * delta_var[col] * cov_z_y_check[col];
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
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    zwidxpos: Position of weight indices for covariance Z|WA
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
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
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    zwidxpos: Position of weight indices for covariance Z|WA
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
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
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    widxpos: Position of weight indices for covariance Z|WA
    zidxpos: Position of next hidden state indices for covariance Z|Z+
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
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    widxpos: Position of weight indices for covariance Z|WA
    zidxpos: Position of next hidden state indices for covariance Z|Z+
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
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
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
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
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
    Shat: Statistical variance for the normalization layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
    Shat: Statistical variance for the normalization layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
    Shat: Statistical variance for the normalization layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
    Shat: Statistical variance for the normalization layers
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    deltaMz: Updated quantities for the mean of the hidden states
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
    aidx: Activation indices for covariance Z|WA i.e. FCzwa_2
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaMz: Updated quantities for the mean of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    widxpos: Position of weight indices for covariance Z|WA
    aidxpos: Position of activation indices for covariance Z|WA
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
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
    aidx: Activation indices for covariance Z|WA i.e. FCzwa_2
    zudidx: Next hidden state indices for covariance Z|Z+ i.e. Szz_ud
    deltaSz: Updated quantities for the variance of the hidden states
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    widxpos: Position of weight indices for covariance Z|WA
    aidxpos: Position of activation indices for covariance Z|WA
    zudidxpos: Position of next hidden state indices for covariance Z|Z+
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
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    jposIn: Positionos the Jacobian vector for this layer
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
/// NOISE INFERENCE
///////////////////////////////////////////////////////////////////////////
__global__ void compute_obs_noise_variance(float const *V, int n, float *Sa) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        Sa[col] += V[col];
    }
}

__global__ void get_obs_noise_variance_with_idx(float const *Sa,
                                                int const *ud_idx, int ny,
                                                int nye, int B, float *Sv)
/*Get observation noise variance from the last output layer

Args:
    Sa: Variance predicted using network
    udIdx: Selected indiced to update
    ny: Number of hidden states of the output layer without hidden states
        for noise observation
    nye: Number of observation to be updated for an observation
    B: Batch size
    Sv: Observation variance i.e., V = [nye x 1]
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx;
    if (col < nye * B) {
        // minus 1 due to matlab's indexing
        idx = ud_idx[col] + (col / nye) * ny - 1;
        Sv[col] += Sa[idx];
    }
}

__global__ void join_output_hidden_states(float const *z_mu, float const *z_v2,
                                          int ny, int B, float *z)
/*Join the updated values of output's hidden states and geteroscedastic
   observation noise's hidden states*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int h = ny / 2;
    int m, k;
    if (col < h * B) {
        m = (col / h) * ny + col % h;
        k = (col / h) * ny + col % h + h;
        z[m] = z_mu[col];
        z[k] = z_v2[col];
    }
}

__global__ void transfer_updated_values(float const *d_z_mu, int n, float *d_z)
/*Transfer the updated values from noise state to delta state. This is required
   for the case of the homoscedastic nosie in order to update the hidden state
   of the output layer*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        d_z[col] = d_z_mu[col];
    }
}

__global__ void delta_mz_Sz_backward(float const *ma_prior,
                                     float const *Sa_prior, float const *J,
                                     float const *Cza_prior,
                                     float const *ma_post, float const *Sa_post,
                                     int n, float *delta_mz, float *delta_Sz)
/*Compute the updated quantities for hidden states using the backward update
  i.e. smoother algorithm

Args:
    ma_prior: Prior mean of activation unit
    Sa_prior: Prior variance of activation unit
    J: Jacobian matrix
    Cza_prior: Covariance between hidden state and activation units
    ma_post: Posterior mean of activation units
    Sa_post: Posterior variance of activation units
    n: Number of activation units
    delta_mz: Updated quantities of mean for the hidden states
    delta_Sz: Updated quantities of variance for the hidden states
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Jz;
    if (col < n) {
        Jz = J[col] * Cza_prior[col] / Sa_prior[col];
        delta_mz[col] = Jz * (ma_post[col] - ma_prior[col]);
        delta_Sz[col] = Jz * (Sa_post[col] - Sa_prior[col]) * Jz;
    }
}

__global__ void delta_mz_Sz_with_indices_backward(
    float const *ma_prior, float const *Sa_prior, float const *J,
    float const *Cza_prior, float const *ma_post, float const *Sa_post,
    int const *ud_idx, int ny, int nye, int B, float *delta_mz, float *delta_Sz)
/*Compute the updated quantities for specified hidden states using the backward
  update i.e. smoother algorithm

Args:
    ma_prior: Prior mean of activation units
    Sa_prior: Prior variance of activation units
    J: Jacobian matrix
    Cza_prior: Covariance between hidden state and activation units
    ma_post: Posterior mean of activation units
    Sa_post: Posterior variance of activation units
    up_idx: Indices for the hidden states to be updated
    ny: Total number of hidden states for the output layer
    nye: Totoal number of hidden states to be updated for the output layer
    B: Batch size
    delta_mz: Updated values of mean for the hidden states
    delta_Sz: Updated values of variance for the hidden states
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Jz;
    int idx;
    if (col < nye * B) {
        idx = ud_idx[col] + (col / nye) * ny - 1;
        Jz = J[idx] * Cza_prior[idx] / Sa_prior[idx];
        delta_mz[idx] = Jz * (ma_post[idx] - ma_prior[idx]);
        delta_Sz[idx] = Jz * (Sa_post[idx] - Sa_prior[idx]) * Jz;
    }
}

__global__ void compute_posterior_for_v_squared(float const *delta_mv,
                                                float const *delta_Sv,
                                                float const *ma_v2, int n,
                                                float *mz_v2, float *Sz_v2)
/* Compute the posterior distribution for the v squared.

Args:
    delta_mv: Updated value of the mean for the observation noise (v)
    delta_Sv: Updated value of the variance of the observation noise
    ma_v2: Mean of activation units for the observation noise squared (v^2)
    Sa_v2: Variance of activation units for the observation noise squared
    n: Number of hidden states
    mz_v2: Mean of hidden states for the observation noise squared
    Sz_v2: Variance of hidden states for the observation noise squared
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float Sv_p;
    if (col < n) {
        Sv_p = ma_v2[col] + delta_Sv[col];
        mz_v2[col] = powf(delta_mv[col], 2) + Sv_p;
        Sz_v2[col] = 2 * powf(Sv_p, 2) + 4 * powf(delta_mv[col], 2) * Sv_p;
    }
}

__global__ void compute_prior_for_v_squared(float const *ma_v2b, float *Sa_v2b,
                                            int n, float *Sa_v2)
/* Compute the posterior distribition for observation noise v.

Args:
    ma_v2b: Mean of activation units for the observation noise squared
        (\overline{v}^2)
    Sa_v2b: Variance of activation units for the observation noise squared
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        Sa_v2[col] = 3 * Sa_v2b[col] + 2 * powf(ma_v2b[col], 2);
    }
}

void delta_mz_Sz_output_dist(ObsGPU &obs, Network &net,
                             NoiseStateGPU &noise_state)
/*Compute the updated quantities for the output distribution. The
   observation is defined following
                        y = x + v, v ~ N(0, \sigma_v^2),
   where y is the observation and x is the output distribution i.e.,
   x ~ N(\mu_x, Sx).

Args:
    obs: Observation object i.e., observation and its noises
    net: Network
    noise_state: Noise's hidden state
*/
{
    int z_pos = 0;
    int n = net.n_y * net.batch_size;
    unsigned int BLOCKS = (n + net.num_gpu_threads - 1) / net.num_gpu_threads;

    // NOTE: To use the deltaMzSz function, we assume that ma_v2b_prior is
    // equivalent to \Sigma_v.
    // Update hidden states for the mean
    deltaMzSz<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_mu, noise_state.d_Sa_mu, noise_state.d_Sz_mu,
        noise_state.d_J_mu, obs.d_y_batch, noise_state.d_ma_v2b_prior,
        noise_state.d_delta_mz_mu, noise_state.d_delta_Sz_mu, z_pos, n);

    // Update hidden states for observation noise's hidden states
    deltaMzSz<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_mu, noise_state.d_Sa_mu, noise_state.d_ma_v2b_prior,
        noise_state.d_J_v, obs.d_y_batch, noise_state.d_ma_v2b_prior,
        noise_state.d_delta_mv, noise_state.d_delta_Sv, z_pos, n);
}

void delta_mz_Sz_noise_dist(Network &net, NoiseStateGPU &noise_state)
/*Compute the updated quantities for the heteroscedastic & homoscedastic noise
   distribution for the observation noise squared (v^2). The observation is
   defined following
                    y = x + v, v ~ N(0, \sigma_v^2)

Args:
    net: Network
    noise_state: Noise state for the output layer

*/
{
    // Update hidden states for the observation noise squared
    int n = net.n_y * net.batch_size;
    unsigned int BLOCKS = (n + net.num_gpu_threads - 1) / net.num_gpu_threads;
    compute_posterior_for_v_squared<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_delta_mv, noise_state.d_delta_Sv,
        noise_state.d_ma_v2b_prior, n, noise_state.d_ma_v2_post,
        noise_state.d_Sa_v2_post);

    compute_prior_for_v_squared<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_v2b_prior, noise_state.d_Sa_v2b_prior, n,
        noise_state.d_Sa_v2_prior);

    // NOTE: We do not apply the activatation function i.e., exponential
    // function for the hidden states representing the observation noise for the
    // homoscedastic case so that we have to handle both following cases
    // differently.
    // Heteroscedastic case
    if (net.noise_type.compare("heteros") == 0) {
        delta_mz_Sz_backward<<<BLOCKS, net.num_gpu_threads>>>(
            noise_state.d_ma_v2b_prior, noise_state.d_Sa_v2_prior,
            noise_state.d_J_v2, noise_state.d_Cza_v2, noise_state.d_ma_v2_post,
            noise_state.d_Sa_v2_post, n, noise_state.d_delta_mz_v2b,
            noise_state.d_delta_Sz_v2b);

    } else if (net.noise_type.compare("homosce") == 0) {
        delta_mz_Sz_backward<<<BLOCKS, net.num_gpu_threads>>>(
            noise_state.d_ma_v2b_prior, noise_state.d_Sa_v2_prior,
            noise_state.d_J_v, noise_state.d_Sa_v2b_prior,
            noise_state.d_ma_v2_post, noise_state.d_Sa_v2_post, n,
            noise_state.d_delta_mz_v2b, noise_state.d_delta_Sz_v2b);

    } else {
        throw std::invalid_argument(
            "Noise inference type is invalid - delta_mz_Sz_noise_dist");
    }
}

void delta_mz_Sz_with_idx_output_dist(ObsGPU &obs, Network &net,
                                      NoiseStateGPU &noise_state)
/*Compute the updated quantities for the output distribution specified by
 indices
*/
{
    // Get number of hidden states for the output layer without the hidden
    // states for the observation noise
    int z_pos = 0;
    int n = net.nye * net.batch_size;
    unsigned int BLOCKS = (n + net.num_gpu_threads - 1) / net.num_gpu_threads;

    // Compute the observation noise variance
    get_obs_noise_variance_with_idx<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_v2b_prior, obs.d_idx_ud_batch, net.n_y, net.nye,
        net.batch_size, obs.d_V_batch);

    // Update hidden states for the mean
    deltaMzSzWithIndices<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_mu, noise_state.d_Sa_mu, noise_state.d_Sz_mu,
        noise_state.d_J_mu, obs.d_y_batch, obs.d_V_batch, obs.d_idx_ud_batch,
        noise_state.d_delta_mz_mu, noise_state.d_delta_Sz_mu, z_pos, net.n_y,
        net.nye, n);

    // Update hidden states for observation noise (v)
    deltaMzSzWithIndices<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_mu, noise_state.d_Sa_mu, noise_state.d_ma_v2b_prior,
        noise_state.d_J_v, obs.d_y_batch, obs.d_V_batch, obs.d_idx_ud_batch,
        noise_state.d_delta_mv, noise_state.d_delta_Sv, z_pos, net.n_y, net.nye,
        n);
}

void delta_mz_Sz_with_idx_noise_dist(ObsGPU &obs, Network &net,
                                     NoiseStateGPU &noise_state)
/*Compute the updated quantities for the heteroscedastic & homoscedastic noise
   distribution for the specified observation noise.
*/
{
    // Update hidden states for the observation noise squared
    int n = net.n_y * net.batch_size;
    unsigned int BLOCKS = (n + net.num_gpu_threads - 1) / net.num_gpu_threads;
    compute_posterior_for_v_squared<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_delta_mv, noise_state.d_delta_Sv,
        noise_state.d_ma_v2b_prior, n, noise_state.d_ma_v2_post,
        noise_state.d_Sa_v2_post);

    compute_prior_for_v_squared<<<BLOCKS, net.num_gpu_threads>>>(
        noise_state.d_ma_v2b_prior, noise_state.d_Sa_v2b_prior, n,
        noise_state.d_Sa_v2_prior);

    // Heteroscedastic case
    unsigned int BLOCK_B =
        (net.nye * net.batch_size + net.num_gpu_threads - 1) /
        net.num_gpu_threads;
    if (net.noise_type.compare("heteros") == 0) {
        delta_mz_Sz_with_indices_backward<<<BLOCK_B, net.num_gpu_threads>>>(
            noise_state.d_ma_v2b_prior, noise_state.d_Sa_v2_prior,
            noise_state.d_J_v2, noise_state.d_Cza_v2, noise_state.d_ma_v2_post,
            noise_state.d_Sa_v2_post, obs.d_idx_ud_batch, net.n_y, net.nye,
            net.batch_size, noise_state.d_delta_mz_v2b,
            noise_state.d_delta_Sz_v2b);
    }
    // Homoscedastic case
    else if (net.noise_type.compare("homosce") == 0) {
        delta_mz_Sz_with_indices_backward<<<BLOCK_B, net.num_gpu_threads>>>(
            noise_state.d_ma_v2b_prior, noise_state.d_Sa_v2_prior,
            noise_state.d_J_v, noise_state.d_Sa_v2b_prior,
            noise_state.d_ma_v2_post, noise_state.d_Sa_v2_post,
            obs.d_idx_ud_batch, net.n_y, net.nye, net.batch_size,
            noise_state.d_delta_mz_v2b, noise_state.d_delta_Sz_v2b);
    } else {
        throw std::invalid_argument(
            "Noise inference type is invalid - "
            "delta_mz_Sz_with_idx_noise_dist");
    }
}

///////////////////////////////////////////////////////////////////////////
/// UPDATED VALUES OF HIDDEN STATES FOR OUTPUT LAYER
///////////////////////////////////////////////////////////////////////////
__global__ void reset_updated_values(int n, float *z) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        z[col] = 0.0f;
    }
}
__global__ void update_homosce_noise(float const *delta_mz_v2b,
                                     float const *delta_Sz_v2b, int ny, int B,
                                     float *ma_v2b_prior, float *Sa_v2b_prior)
/* Compute the updated values for homoscedastic noise squared by summing up the
   mini-batches of updated values of each noise observation squared

Args:
    delta_mz_v2b: Updated values for the mean of the observation noise squared
    delta_Sz_v2B: Updatred values for the variance of the observation noise
        squared
    ny: Number of hidden states for the output layer
    B: Batch size
    ma_v2b_prior: Mean of the observation noise squared
    Sa_v2b_prior: Variance of the observation noise squared
 */
// TODO: Need to fixed the sum of each batch
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ny * B) {
        float tmp_m = 0.0f;
        float tmp_S = 0.0f;
        for (int j = 0; j < B; j++) {
            tmp_m += delta_mz_v2b[(j % B) * ny + col % ny];
            tmp_S += delta_Sz_v2b[(j % B) * ny + col % ny];
        }

        ma_v2b_prior[col] += tmp_m;
        Sa_v2b_prior[col] += tmp_S;
    }
}

void output_delta_mz_Sz_with_noise_inference(ObsGPU &obs, Network &net,
                                             StateGPU &state,
                                             DeltaStateGPU &d_state)
/* Compute the updated value for the output layer including the noise
   observation's hidden states
 */
{
    int z_pos = net.z_pos.back();
    if (net.is_idx_ud) {
        // Reset noise state's updated values to zeros
        int ny_B = net.n_y * net.batch_size;
        unsigned int BLOCK_RS =
            (ny_B + net.num_gpu_threads - 1) / net.num_gpu_threads;
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, state.noise_state.d_delta_mz_mu);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, state.noise_state.d_delta_Sz_mu);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, state.noise_state.d_delta_mv);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, state.noise_state.d_delta_Sv);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, state.noise_state.d_delta_mz_v2b);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, state.noise_state.d_delta_Sz_v2b);

        // Reset state's updated values to zeros
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, d_state.d_delta_mz);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, d_state.d_delta_Sz);

        // Compute updated values for the output distribution
        delta_mz_Sz_with_idx_output_dist(obs, net, state.noise_state);

        // Compute updated values for the noise observation of the output
        // distribution
        delta_mz_Sz_with_idx_noise_dist(obs, net, state.noise_state);

    } else {
        delta_mz_Sz_output_dist(obs, net, state.noise_state);
        delta_mz_Sz_noise_dist(net, state.noise_state);
    }

    // Join updated values (outputs + its observation noise)
    unsigned int BLOCKS;
    if (net.noise_type.compare("heteros") == 0) {
        BLOCKS = (net.n_y * net.batch_size + net.num_gpu_threads - 1) /
                 net.num_gpu_threads;
        join_output_hidden_states<<<BLOCKS, net.num_gpu_threads>>>(
            state.noise_state.d_delta_mz_mu, state.noise_state.d_delta_mz_v2b,
            net.nodes.back(), net.batch_size, d_state.d_delta_mz);

        join_output_hidden_states<<<BLOCKS, net.num_gpu_threads>>>(
            state.noise_state.d_delta_Sz_mu, state.noise_state.d_delta_Sz_v2b,
            net.nodes.back(), net.batch_size, d_state.d_delta_Sz);

    } else if (net.noise_type.compare("homosce") == 0) {
        int ny_B = net.n_y * net.batch_size;
        unsigned int BLOCK_TUD =
            (ny_B + net.num_gpu_threads - 1) / net.num_gpu_threads;
        transfer_updated_values<<<BLOCK_TUD, net.num_gpu_threads>>>(
            state.noise_state.d_delta_mz_mu, ny_B, d_state.d_delta_mz);
        transfer_updated_values<<<BLOCK_TUD, net.num_gpu_threads>>>(
            state.noise_state.d_delta_Sz_mu, ny_B, d_state.d_delta_Sz);

        update_homosce_noise<<<BLOCK_TUD, net.num_gpu_threads>>>(
            state.noise_state.d_delta_mz_v2b, state.noise_state.d_delta_Sz_v2b,
            net.nodes.back(), net.batch_size, state.noise_state.d_ma_v2b_prior,
            state.noise_state.d_Sa_v2b_prior);

    } else {
        throw std::invalid_argument(
            "Noise inference type is invalid - "
            "output_delta_mz_Sz_with_noise_inference");
    }
}

void output_delta_mz_Sz(ObsGPU &obs, Network &net, StateGPU &state,
                        DeltaStateGPU &d_state)
/* Compute the updated value for the hidden states of the output layer

 Args:
    obs: Observations
    net: Network architecture
    state: Hidden state of network
    d_state: Updated quantities for network's hidden states
 */
{
    int nl_B = net.nodes.back() * net.batch_size;
    if (!net.is_idx_ud) {
        int BLOCKS_UD = (nl_B + net.num_gpu_threads - 1) / net.num_gpu_threads;
        deltaMzSz<<<BLOCKS_UD, net.num_gpu_threads>>>(
            state.d_ma, state.d_Sa, state.d_Sz, state.d_J, obs.d_y_batch,
            obs.d_V_batch, d_state.d_delta_mz, d_state.d_delta_Sz,
            net.z_pos.back(), nl_B);
    } else  // Only update the hidden states in the indices udIdx
    {
        // Reset the updated values to zeros
        int ny_B = net.n_y * net.batch_size;
        unsigned int BLOCK_RS =
            (ny_B + net.num_gpu_threads - 1) / net.num_gpu_threads;
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, d_state.d_delta_mz);
        reset_updated_values<<<BLOCK_RS, net.num_gpu_threads>>>(
            ny_B, d_state.d_delta_Sz);

        int nl = net.nye * net.batch_size;
        int BLOCKS_UD = (nl + net.num_gpu_threads - 1) / net.num_gpu_threads;
        deltaMzSzWithIndices<<<BLOCKS_UD, net.num_gpu_threads>>>(
            state.d_ma, state.d_Sa, state.d_Sz, state.d_J, obs.d_y_batch,
            obs.d_V_batch, obs.d_idx_ud_batch, d_state.d_delta_mz,
            d_state.d_delta_Sz, net.z_pos.back(), net.nodes.back(), net.nye,
            nl);
    }
}

void remax_output_delta_z(ObsGPU &obs, Network &net, StateGPU &state,
                          DeltaStateGPU &d_state) {
    int no = net.nodes.back();
    int B = net.batch_size;
    int z_pos = net.z_pos.back();
    int THREADS = net.num_gpu_threads;
    dim3 dim_block(THREADS, THREADS);

    // Covariance between m and \check{a}
    unsigned int grid_row = (B + THREADS - 1) / THREADS;
    unsigned int grid_col = (no + THREADS - 1) / THREADS;
    dim3 dim_grid(grid_col, grid_row);
    compute_cov_m_a_check<<<dim_grid, dim_block>>>(
        state.remax.d_var_log, state.remax.d_cov_log_logsum, state.remax.d_mu_m,
        no, B, state.remax.d_cov_m_a_check);

    // Covariance between m and a
    compute_cov_m_a<<<dim_grid, dim_block>>>(
        state.remax.d_cov_m_a_check, state.d_ma, state.remax.d_var_m,
        state.d_Sz, state.remax.d_J_m, z_pos, no, B, state.remax.d_cov_m_a);

    // Updating quantities for hidden states
    delta_z_y_check<<<dim_grid, dim_block>>>(
        state.d_ma, state.d_Sa, state.remax.d_cov_m_a, obs.d_y_batch,
        obs.d_V_batch, no, B, z_pos, d_state.d_delta_mz, d_state.d_delta_Sz);
    state.copy_device_to_host();
    d_state.copy_device_to_host();
}

void update_output_hidden_states(ObsGPU &obs, Network &net, StateGPU &state,
                                 DeltaStateGPU &d_state)
/*Compute updated quantities for the output layer's hidden state

 Args:
    obs: Observations
    net: Network architecture
    state: Hidden state of network
    d_state: Updated quantities for network's hidden states
 */
{
    if (net.is_output_ud) {
        if (net.noise_type.compare("homosce") != 0 &&
            net.noise_type.compare("heteros") != 0 &&
            (net.activations.back() != net.act_names.remax)) {
            output_delta_mz_Sz(obs, net, state, d_state);
        } else if (net.activations.back() == net.act_names.remax) {
            remax_output_delta_z(obs, net, state, d_state);
        } else {
            output_delta_mz_Sz_with_noise_inference(obs, net, state, d_state);
        }
    } else {
        int nl_B = net.nodes.back() * net.batch_size;
        int BLOCKS_UD = (nl_B + net.num_gpu_threads - 1) / net.num_gpu_threads;
        duplicateMeanVar<<<BLOCKS_UD, net.num_gpu_threads>>>(
            obs.d_y_batch, obs.d_V_batch, d_state.d_delta_mz,
            d_state.d_delta_Sz, nl_B);
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
    int THREADS = net.num_gpu_threads;
    dim3 dimBlock(THREADS, THREADS);

    // Declare variables
    int zposL, zposIn, zposOut, wposIn, no, ni, ki, fi, wi;
    int hi, fo, wo, ho;
    int ki2, wihi, woho, niB, padIdx, xsIn, nextSc2ud;
    int fisc, scposOut, scposIn, zscposIn, wisc, hisc;
    int wihisc, ki2sc, rwIn, czIn, czInB, rwInfo, wscposIn;
    int zwscidxposIn;
    int zascidxposIn, zudscidxposIn, nisc, niscB;
    int rowwIn, rowwfo, wihiB;
    int colzIn, kiwo, K, raposIn, wihifi, fiB, xposIn;
    int nl, padIdxIn, nl_B;
    unsigned int gridRow, gridCol;
    int B = net.batch_size;
    int numLayers = net.layers.size();

    // Output-layer update
    update_output_hidden_states(obs, net, state, d_state);

    // Inovation vector for output layer
    zposL = net.z_pos.back();
    nl = net.nodes.back();
    nl_B = nl * B;
    int BLOCKS_I = (nl_B + THREADS - 1) / THREADS;

    // Inovavation vector for Z
    inovationMean<<<BLOCKS_I, THREADS>>>(state.d_Sz, d_state.d_delta_mz,
                                         d_state.d_delta_m, zposL, zposL, nl_B);
    inovationVar<<<BLOCKS_I, THREADS>>>(state.d_Sz, d_state.d_delta_Sz,
                                        d_state.d_delta_S, zposL, zposL, nl_B);

    // Hidden-layer update
    nextSc2ud = -1;
    for (int k = numLayers - 1; k-- > net.last_backward_layer;) {
        // General hyperparameters
        zposIn = net.z_pos[k];       // location of actv. input
        zposOut = net.z_pos[k + 1];  // location of h.s. output
        wposIn = net.w_pos[k];       // location of weights in param. vector
        no = net.nodes[k + 1];       // num. of nodes for output
        ni = net.nodes[k];           // num. of nodes for input
        // Handle multiple input sequences from LSTM layer
        if (net.layers[k] == net.layer_names.lstm) {
            ni = net.nodes[k] * net.input_seq_len;
        }

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
        //**
        // 7: LSTM layer
        //
        else if (net.layers[k + 1] == net.layer_names.lstm) {
            lstm_state_update(net, state, theta, d_state, k);
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

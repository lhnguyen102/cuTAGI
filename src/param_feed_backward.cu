///////////////////////////////////////////////////////////////////////////
// File:         param_feed_backward.cu
// Description:  backward pass for parametes in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 12, 2021
// Updated:      September 09, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
///////////////////////////////////////////////////////////////////////////

#include "../include/param_feed_backward.cuh"

////////////////////////////////////////////////////////////////////////////////
/// FULL-CONNECTED
////////////////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMwz
__global__ void fcDeltaMw(float const *Sw, float const *ma, float const *deltaM,
                          int wpos, int zposIn, int zposOut, int m, int n,
                          int k, float *deltaMw)
/* Compute update quantities for the mean of weights for full-connected layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaMw: Updated quantities for the mean of weights
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += ma[m * i + row + zposIn] * deltaM[col + k * i + zposOut];
        }
        deltaMw[col * m + row + wpos] = sum * Sw[col * m + row + wpos];
    }
}
// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
__global__ void fcDeltaSw(float const *Sw, float const *ma, float const *deltaS,
                          int wpos, int zposIn, int zposOut, int m, int n,
                          int k, float *deltaSw)
/* Compute update quantities for the variance of weights for full-connected
layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaS: Inovation vector for variance i.e (S_observation - S_prediction)
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaSw: Updated quantities for the variance of weights
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += ma[m * i + row + zposIn] * ma[m * i + row + zposIn] *
                   deltaS[col + k * i + zposOut];
        }
        deltaSw[col * m + row + wpos] =
            sum * Sw[col * m + row + wpos] * Sw[col * m + row + wpos];
    }
}
// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
__global__ void fcDeltaMb(float const *Cbz, float const *deltaM, int bpos,
                          int zposOut, int m, int n, int k, float *deltaMb)
/* Compute update quantities for the mean of biases for full-connected layer.

Args:
    Cbz: Covariance b|Z+
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    bpos: Bias position for this layer in the bias vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaMb: Updated quantities for the mean of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += deltaM[m * i + row + zposOut];
        }
        deltaMb[col * m + row + bpos] = sum * Cbz[col * m + row + bpos];
    }
}

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
__global__ void fcDeltaSb(float const *Cbz, float const *deltaS, int bpos,
                          int zposOut, int m, int n, int k, float *deltaSb)
/* Compute update quantities for the variance of biases for full-connected
layer.

Args:
    Cbz: Covariance b|Z+
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    bpos: Bias position for this layer in the bias vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaSb: Updated quantities for the variance of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += deltaS[m * i + row + zposOut];
        }
        deltaSb[col * m + row + bpos] =
            sum * Cbz[col * m + row + bpos] * Cbz[col * m + row + bpos];
    }
}
///////////////////////////////////////////////////////////////////////////
/// CONVOLUTIONAL
///////////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
__global__ void convDeltaMw(float const *Sw, float const *ma,
                            float const *deltaM, int const *aidx, int wpos,
                            int apos, int aidxpos, int m, int n, int k,
                            int woho, int wihi, int fi, int ki2, int padIdx,
                            float *deltaMw)
/* Compute update quantities for the mean of weights for convolutional layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    aidx: Activation indices for computing the mean of the product WA
    wpos: Weight position for this layer in the weight vector of network
    apos: Input-hidden-state position for this layer in the weight vector
          of network
    aidxpos: Position of the activation indices in its vector of the network
    m: ki x ki x fi
    n: wo x ho xB
    k: fo
    woho: Width x height of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    ki2: ki x ki
    padIdx: Size of the hidden state vector for this layer + 1
    deltaMw: Updated quantities for the mean of weights
*/
// TODO: remove the duplicate in the input variables
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[ki2 * (i % woho) + row % ki2 + aidxpos] +
                       (row / ki2) * wihi + (i / woho) * wihi * fi;
            if (aidx_tmp < padIdx) {
                sum += ma[aidx_tmp - 1 + apos] * deltaM[col * n + i];
                // minus 1 due to matlab's indexing
            }
        }
        deltaMw[col * m + row + wpos] = sum * Sw[col * m + row + wpos];
    }
}
// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
__global__ void convDeltaSw(float const *Sw, float const *ma,
                            float const *deltaS, int const *aidx, int wpos,
                            int apos, int aidxpos, int m, int n, int k,
                            int woho, int wihi, int fi, int ki2, int padIdx,
                            float *deltaSw)
/* Compute update quantities for the variance of weights for convolutional
layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    aidx: Activation indices for computing the mean of the product WA
    wpos: Weight position for this layer in the weight vector of network
    apos: Input-hidden-state position for this layer in the weight vector
          of network
    aidxpos: Position of the activation indices in its vector of the network
    m: ki x ki x fi
    n: wo x ho xB
    k: fo
    woho: Width x height of the output image
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    ki2: ki x ki
    padIdx: Size of the hidden state vector for this layer + 1
    deltaSw: Updated quantities for the variance of weights
*/
// TODO: remove the duplicate in the input variables
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int aidx_tmp = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            aidx_tmp = aidx[ki2 * (i % woho) + row % ki2 + aidxpos] +
                       (row / ki2) * wihi + (i / woho) * wihi * fi;
            if (aidx_tmp < padIdx) {
                sum += ma[aidx_tmp - 1 + apos] * ma[aidx_tmp - 1 + apos] *
                       deltaS[col * n + i];
                // minus 1 due to matlab's indexing
            }
        }
        deltaSw[col * m + row + wpos] =
            sum * Sw[col * m + row + wpos] * Sw[col * m + row + wpos];
    }
}
// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
__global__ void convDeltaMb(float const *Cbz, float const *deltaM, int bpos,
                            int m, int n, int k, float *deltaMb)
/* Compute update quantities for the mean of biases for convolutional layer.

Args:
    Cbz: Covariance b|Z+
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    bpos: Bias position for this layer in the bias vector of network
    m: ki x ki x fi
    n: wo x ho xB
    k: fo
    deltaMb: Updated quantities for the mean of biases

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += deltaM[col * n + i];
        }
        deltaMb[col * m + row + bpos] = sum * Cbz[col * m + row + bpos];
    }
}

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
__global__ void convDeltaSb(float const *Cbz, float const *deltaS, int bpos,
                            int m, int n, int k, float *deltaSb)
/* Compute update quantities for the variance of biases for convolutional layer.

Args:
    Cbz: Covariance b|Z+
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    bpos: Bias position for this layer in the bias vector of network
    m: ki x ki x fi
    n: wo x ho xB
    k: fo
    deltaSb: Updated quantities for the variance of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += deltaS[col * n + i];
        }
        deltaSb[col * m + row + bpos] =
            sum * Cbz[col * m + row + bpos] * Cbz[col * m + row + bpos];
    }
}
__global__ void permuteMeanVar(float const *deltaMinit, float const *deltaSinit,
                               float *deltaM, float *deltaS, int zpos, int woho,
                               int kp, int B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < kp && row < B)  // kp = woho * fo
    {
        // Note that (col/(w * h)) equvalent to floorf((col/(w * h)))
        // because of interger division
        deltaM[woho * (col / woho) * B + col % woho + row * woho] =
            deltaMinit[row * kp + col + zpos];
        deltaS[woho * (col / woho) * B + col % woho + row * woho] =
            deltaSinit[row * kp + col + zpos];
    }
}
///////////////////////////////////////////////////////////////////////////
/// TRANSPOSE CONVOLUTIONAL
///////////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
__global__ void tconvDeltaMw(float const *Sw, float const *ma,
                             float const *deltaM, int const *aidx,
                             int const *zidx, int wpos, int zposIn, int zposOut,
                             int aidxpos, int zidxpos, int woho, int fo,
                             int wihi, int fi, int ki, int B, float *deltaMw)
/* Compute update quantities for the mean of weights for transpose convolutional
layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    aidx: Activation indices for covariance W|Z+ i.e. FCwz_2
    zidx: Hidden state (Z+) indices for covariance Z|Z+ i.e. Swz_ud
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    aidxpos: Position of adix in its index vector of the network
    zidxpos: Position of zidx in its vector of the network
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the output image
    fi: Number of filters of the input image
    B: Number of batches
    deltaMw: Updated quantities for the mean of weights
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    int K = ki * ki * fo;
    int zidx_tmp = 0;  // updated index
    int aidx_tmp = 0;
    if (col < K && row < fi)  // m = fi, k = ki2 * fo
    {
        for (int i = 0; i < wihi * B; i++)  // n = wihi * B
        {
            aidx_tmp = aidx[(col % (ki * ki)) * wihi + i % wihi + aidxpos] +
                       row * wihi + (i / wihi) * wihi * fi - 1;

            zidx_tmp = zidx[(col % (ki * ki)) * wihi + i % wihi + zidxpos] +
                       (col / (ki * ki)) * woho + (i / wihi) * woho * fo - 1;

            if (aidx_tmp < wihi * fi * B) {
                sum += ma[aidx_tmp + zposIn] *
                       deltaM[zidx_tmp +
                              zposOut];  // minus 1 due to matlab's indexing
            }
        }
        deltaMw[col + row * K + wpos] = sum * Sw[col + row * K + wpos];
    }
}
// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
__global__ void tconvDeltaSw(float const *Sw, float const *ma,
                             float const *deltaS, int const *aidx,
                             int const *zidx, int wpos, int zposIn, int zposOut,
                             int aidxpos, int zidxpos, int woho, int fo,
                             int wihi, int fi, int ki, int B, float *deltaSw)
/* Compute update quantities for the variance of weights for transpose
convolutional layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    deltaS: Inovation vector for mean i.e. (S_observation - S_prediction)
    aidx: Activation indices for covariance W|Z+ i.e. FCwz_2
    zidx: Hidden state (Z+) indices for covariance Z|Z+ i.e. Swz_ud
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    aidxpos: Position of adix in its index vector of the network
    zidxpos: Position of zidx in its vector of the network
    woho: Width x height of the output image
    fo: Number of filters of the output image
    wihi: Width x height of the output image
    fi: Number of filters of the input image
    B: Number of batches
    deltaSw: Updated quantities for the mean of weights
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int K = ki * ki * fo;
    float sum = 0;
    int zidx_tmp = 0;  // updated index
    int aidx_tmp = 0;
    if (col < K && row < fi)  // m = fi, k = ki2 * fo
    {
        for (int i = 0; i < wihi * B; i++)  // n = wihi * B
        {
            // minus 1 due to matlab's indexing
            aidx_tmp = aidx[(col % (ki * ki)) * wihi + i % wihi + aidxpos] +
                       row * wihi + (i / wihi) * wihi * fi - 1;

            zidx_tmp = zidx[(col % (ki * ki)) * wihi + i % wihi + zidxpos] +
                       (col / (ki * ki)) * woho + (i / wihi) * woho * fo - 1;

            if (aidx_tmp < wihi * fi * B) {
                sum += ma[aidx_tmp + zposIn] * ma[aidx_tmp + zposIn] *
                       deltaS[zidx_tmp + zposOut];
            }
        }
        deltaSw[col + row * K + wpos] =
            sum * Sw[col + row * K + wpos] * Sw[col + row * K + wpos];
    }
}

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
__global__ void tconvDeltaMb(float const *Cbz, float const *deltaM, int bpos,
                             int zposOut, int woho, int fo, int B,
                             float *deltaMb)
/* Compute update quantities for the mean of biases for transpose convolutional
layer.

Args:
    Cbz: Covariance b|Z+
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    bpos: Bias position for this layer in the bias vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    woho: Width x height of the output image
    fo: Number of filters of the output image
    B: Number of batches
    deltaMb: Updated quantities for the mean of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < fo && row < 1)  // k = fo, m = 1
    {
        for (int i = 0; i < woho * B; i++)  // n = woho * B
        {
            sum += deltaM[col * woho + (i % woho) + (i / woho) * woho * fo +
                          zposOut];
        }
        deltaMb[col + bpos] = sum * Cbz[col + bpos];
    }
}

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
__global__ void tconvDeltaSb(float const *Cbz, float const *deltaS, int bpos,
                             int zposOut, int woho, int fo, int B,
                             float *deltaSb)
/* Compute update quantities for the variance of biases for transpose
convolutional layer.

Args:
    Cbz: Covariance b|Z+
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    bpos: Bias position for this layer in the bias vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    woho: Width x height of the output image
    fo: Number of filters of the output image
    B: Number of batches
    deltaSb: Updated quantities for the variance of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < fo && row < 1)  // k = fo, m = 1
    {
        for (int i = 0; i < woho * B; i++)  // n = woho * B
        {
            sum += deltaS[col * woho + (i % woho) + (i / woho) * woho * fo +
                          zposOut];
        }
        deltaSb[col + bpos] = sum * Cbz[col + bpos] * Cbz[col + bpos];
    }
}

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
                                int k, float *deltaMw, float *deltaSw)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to convolutional layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    mhat: Statistical mean for the normalization layers i.e. mra
    Shat: Statistical variance for the normalization layers i.e. Sra
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    ra_pos: Statistical mean and variance position for the
                normalization layer
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    m: fi x B
    k: wihi
    deltaMw: Updated quantities for the mean of weights
    deltaSw: Updated quantities for the variance of weights
*/
// TODO: remove the duplicates
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        A = (1 / sqrtf(Shat[row % fi + rapos] + epsilon)) *
            (ma[col + row * k + zposIn] - mhat[row % fi + rapos]) *
            Sw[row % fi + wpos];
        deltaMw[col + row * k] = A * deltaM[col + row * k + zposOut];
        deltaSw[col + row * k] = A * deltaS[col + row * k + zposOut] * A;
    }
}
// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void convbnDeltaMbSb(float const *Sb, float const *deltaM,
                                float const *deltaS, float epsilon, int bpos,
                                int zposOut, int wihi, int fi, int m, int k,
                                float *deltaMb, float *deltaSb)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to convolutional layer.

Args:
    Sb: Variance of biases
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    bpos: biases position for this layer in the weight vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    wihi: Width x height of the input image
    fi: Number of filters of the input image
    m: fi x B
    k: wihi
    deltaMb: Updated quantities for the mean of biases
    deltaSb: Updated quantities for the variance of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        A = Sb[row % fi + bpos];
        deltaMb[col + row * k] = A * deltaM[col + row * k + zposOut];
        deltaSb[col + row * k] = A * deltaS[col + row * k + zposOut] * A;
    }
}
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void fcbnDeltaMwSw(float const *Sw, float const *ma,
                              float const *mhat, float const *Shat,
                              float const *deltaM, float const *deltaS,
                              float epsilon, int wpos, int zposIn, int zposOut,
                              int rapos, int ni, int B, float *deltaMw,
                              float *deltaSw)
/* Compute update quantities for the mean & variance of weights for
batch-normalization layer applied to full-connected layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    mhat: Statistical mean for the normalization layers i.e. mra
    Shat: Statistical variance for the normalization layers i.e. Sra
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    ra_pos: Statistical mean and variance position for the
            normalization layer
    ni: Number of hidden units for input
    B: Number of batches
    deltaMw: Updated quantities for the mean of weights
    deltaSw: Updated quantities for the variance of weights
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    float sumM = 0;
    float sumS = 0;
    if (col < ni && row < 1) {
        for (int i = 0; i < B; i++) {
            A = (1 / sqrtf(Shat[col + rapos] + epsilon)) *
                (ma[col + i * ni + zposIn] - mhat[col + rapos]) *
                Sw[col + wpos];
            sumM += A * deltaM[col + i * ni + zposOut];
            sumS += A * deltaS[col + i * ni + zposOut] * A;
        }
        deltaMw[col + wpos] = sumM;
        deltaSw[col + wpos] = sumS;
    }
}
// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void fcbnDeltaMbSb(float const *Sb, float const *deltaM,
                              float const *deltaS, float epsilon, int bpos,
                              int zposOut, int ni, int B, float *deltaMb,
                              float *deltaSb)
/* Compute update quantities for the mean & variance of biases for
batch-normalization layer applied to full-connected layer.

Args:
    Sb: Variance of biases
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    bpos: Biases position for this layer in the weight vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    ni: Number of hidden units for input
    B: Number of batches
    deltaMb: Updated quantities for the mean of biases
    deltaSb: Updated quantities for the variance of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    float sumM = 0;
    float sumS = 0;
    if (col < ni && row < 1) {
        for (int i = 0; i < B; i++) {
            A = Sb[col + bpos];
            sumM += A * deltaM[col + i * ni + zposOut];
            sumS += A * deltaS[col + i * ni + zposOut] * A;
        }
        deltaMb[col + bpos] = sumM;
        deltaSb[col + bpos] = sumS;
    }
}
// Layer Normalization
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void convlnDeltaMwSw(float const *Sw, float const *ma,
                                float const *mhat, float const *Shat,
                                float const *deltaM, float const *deltaS,
                                float epsilon, int wpos, int zposIn,
                                int zposOut, int rapos, int wihi, int m, int k,
                                float *deltaMw, float *deltaSw)
/* Compute update quantities for the mean & variance of weights for
LAYER-NORMALIZATION layer applied to convolutional layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    mhat: Statistical mean for the normalization layers i.e. mra
    Shat: Statistical variance for the normalization layers i.e. Sra
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    ra_pos: Statistical mean and variance position for the
    normalization layer
    wihi: Width x height of the input image
    m: B
    k: wihi x fi
    deltaMw: Updated quantities for the mean of weights
    deltaSw: Updated quantities for the variance of weights
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        A = (1 / sqrtf(Shat[row + rapos] + epsilon)) *
            (ma[col + row * k + zposIn] - mhat[row + rapos]) *
            Sw[col / wihi + wpos];
        deltaMw[col + row * k] = A * deltaM[col + row * k + zposOut];
        deltaSw[col + row * k] = A * deltaS[col + row * k + zposOut] * A;
    }
}
// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void convlnDeltaMbSb(float const *Sb, float const *deltaM,
                                float const *deltaS, float epsilon, int bpos,
                                int zposOut, int wihi, int m, int k,
                                float *deltaMb, float *deltaSb)
/* Compute update quantities for the mean & variance of biases for
LAYER-NORMALIZATION layer applied to convolutional layer.

Args:
    Sb: Variance of biases
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    bpos: biases position for this layer in the weight vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    wihi: Width x height of the input image
    m: B
    k: wihi x fi
    deltaMb: Updated quantities for the mean of biases
    deltaSb: Updated quantities for the variance of biases
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    if (col < k && row < m)  // k = wihi, m = fi*B
    {
        A = Sb[col / wihi + bpos];
        deltaMb[col + row * k] = A * deltaM[col + row * k + zposOut];
        deltaSb[col + row * k] = A * deltaS[col + row * k + zposOut] * A;
    }
}

// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMw
// SW_new = SW_old + deltaSw
__global__ void fclnDeltaMwSw(float const *Sw, float const *ma,
                              float const *mhat, float const *Shat,
                              float const *deltaM, float const *deltaS,
                              float epsilon, int wpos, int zposIn, int zposOut,
                              int rapos, int ni, int B, float *deltaMw,
                              float *deltaSw)
/* Compute update quantities for the mean & variance of weights for
LAYER-NORMALIZATION layer applied to full-connected layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    mhat: Statistical mean for the normalization layers i.e. mra
    Shat: Statistical variance for the normalization layers i.e. Sra
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    wpos: Weight position for this layer in the weight vector of network
    zposIn: Input-hidden-state position for this layer in the hidden-state
        vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    ra_pos: Statistical mean and variance position for the
            normalization layer
    ni: Number of hidden units for input
    B: Number of batches
    deltaMw: Updated quantities for the mean of weights
    deltaSw: Updated quantities for the variance of weights
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    float sumM = 0;
    float sumS = 0;
    if (col < ni && row < 1) {
        for (int i = 0; i < B; i++) {
            A = (1 / sqrtf(Shat[i + rapos] + epsilon)) *
                (ma[col + i * ni + zposIn] - mhat[i + rapos]) * Sw[col + wpos];
            sumM += A * deltaM[col + i * ni + zposOut];
            sumS += A * deltaS[col + i * ni + zposOut] * A;
        }
        deltaMw[col + wpos] = sumM;
        deltaSw[col + wpos] = sumS;
    }
}
// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
// Sb_new = Sb_old + deltaSb
__global__ void fclnDeltaMbSb(float const *Sb, float const *deltaM,
                              float const *deltaS, float epsilon, int bpos,
                              int zposOut, int ni, int B, float *deltaMb,
                              float *deltaSb)
/* Compute update quantities for the mean & variance of biases for
LAYER-NORMALIZATION layer applied to full-connected layer.

Args:
    Sb: Variance of biases
    deltaM: Inovation vector for mean i.e. (M_observation - M_prediction)
    deltaS: Inovation vector for variance i.e. (S_observation - S_prediction)
    epsilon: Constant for normalization layer to avoid zero-division
    bpos: Biases position for this layer in the weight vector of network
    zposOut: Output-hidden-state position for this layer in the hidden-state
        vector of network
    ni: Number of hidden units for input
    B: Number of batches
    deltaMb: Updated quantities for the mean of biases
    deltaSb: Updated quantities for the variance of biases

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float A = 0;
    float sumM = 0;
    float sumS = 0;
    if (col < ni && row < 1) {
        for (int i = 0; i < B; i++) {
            A = Sb[col + bpos];
            sumM += A * deltaM[col + i * ni + zposOut];
            sumS += A * deltaS[col + i * ni + zposOut] * A;
        }
        deltaMb[col + bpos] = sumM;
        deltaSb[col + bpos] = sumS;
    }
}
__global__ void deltaParamSum(float const *deltaMe, float const *deltaSe,
                              int startpos, int wihi, int fi, int n,
                              float *deltaM, float *deltaS) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sumDeltaM = 0;
    float sumDeltaS = 0;
    if (col < fi && row < 1) {
        for (int i = 0; i < n; i++)  // n = wihi * fi
        {
            sumDeltaM +=
                deltaMe[(i / wihi) * wihi * fi + i % wihi + col * wihi];
            sumDeltaS +=
                deltaSe[(i / wihi) * wihi * fi + i % wihi + col * wihi];
        }
        deltaM[col + startpos] = sumDeltaM;
        deltaS[col + startpos] = sumDeltaS;
    }
}

///////////////////////////////////////////////////////////////////
/// PARAMETER BACKWARD PASS
///////////////////////////////////////////////////////////////////
void paramBackward(Network &net, ParamGPU &theta, StateGPU &state,
                   DeltaStateGPU &d_state, IndexGPU &idx,
                   DeltaParamGPU &d_theta)
/*Compute updated quantities for weights and biases using TAGI.

  Args:
    net: Network architecture
    theta: Network's weights and biases
    state: Hidden state of Network
    d_state: Difference between prediction and observation
    idx: Indices for network e.g. see indices.cpp

  Returns:
    d_theta: Updated quantities for weights and biases.
 */
{
    // Launch kernel
    int THREADS = net.num_gpu_threads;
    dim3 dimBlock(THREADS, THREADS);

    // Declare variable
    int zposIn, zposOut, wposIn, bposIn, no, ni, ki, fi, wi, hi;
    int fo, wo, ho, ki2, wihi, woho, wohofo, fiB, ki2fi;
    int wohoB, wihiB, wihifi, padIdx, padIdxsc, xsOut, fisc, wisc;
    int hisc, wihisc, ascidxposIn, zscposIn, wscposIn, bscposIn;
    int aidxposIn, raposIn;
    int B = net.batch_size;
    int numLayers = net.layers.size();

    for (int k = numLayers - 1; k-- > 0;) {
        // General hyperparameters
        zposIn = net.z_pos[k];       // location of actv. input
        zposOut = net.z_pos[k + 1];  // location of h.s. output
        wposIn = net.w_pos[k];       // location of weights in param. vector
        bposIn = net.b_pos[k];       // location of bias in param. vector
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
        wohofo = woho * fo;
        fiB = fi * B;
        ki2fi = ki2 * fi;
        wohoB = woho * B;
        wihiB = wihi * B;
        wihifi = wihi * fi;
        padIdx = wihi * fi * B + 1;  // padding index (following TAGI matlab)

        // Hyperparameters for residual networks. Note that current version
        // works only with CNN layer. Future version will include other layers.
        xsOut = net.shortcuts[k + 1];  // index for the shorcut layer
        // xsIn = (int) xslayer[k];
        fisc = net.filters[xsOut];
        wisc = net.widths[xsOut];
        hisc = net.heights[xsOut];
        wihisc = wisc * hisc;

        //**
        // Residual connection
        //
        if ((xsOut != -1) && (fisc != fo || wisc != wo)) {
            ascidxposIn = net.Fmwa_2_sc_pos[xsOut];
            int scposOut = net.sc_pos[k + 1];
            zscposIn = net.z_pos[xsOut];
            wscposIn = net.w_sc_pos[xsOut - 1];
            bscposIn = net.b_sc_pos[xsOut - 1];
            padIdxsc = wihisc * fisc * B + 1;

            unsigned int gridRow = (B + THREADS - 1) / THREADS;
            unsigned int gridCol = (wohofo + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCol, gridRow);

            permuteMeanVar<<<dimGrid, dimBlock>>>(
                d_state.d_delta_mx, d_state.d_delta_Sx, d_state.d_dummy_m,
                d_state.d_dummy_S, scposOut, woho, wohofo, B);

            unsigned int gridRow2 = (fisc + THREADS - 1) / THREADS;
            unsigned int gridCol2 = (fo + THREADS - 1) / THREADS;
            dim3 dimGrid2(gridCol2, gridRow2);

            convDeltaMw<<<dimGrid2, dimBlock>>>(
                theta.d_Sw_sc, state.d_ma, d_state.d_dummy_m, idx.d_Fmwa_2_sc,
                wscposIn, zscposIn, ascidxposIn, fisc, wohoB, fo, woho, wihisc,
                fisc, 1, padIdxsc, d_theta.d_delta_mw_sc);

            convDeltaSw<<<dimGrid2, dimBlock>>>(
                theta.d_Sw_sc, state.d_ma, d_state.d_dummy_S, idx.d_Fmwa_2_sc,
                wscposIn, zscposIn, ascidxposIn, fisc, wohoB, fo, woho, wihisc,
                fisc, 1, padIdxsc, d_theta.d_delta_Sw_sc);

            unsigned int gridCol3 = (fo + THREADS - 1) / THREADS;
            dim3 dimGrid3(gridCol3, 1);

            convDeltaMb<<<dimGrid3, dimBlock>>>(
                theta.d_Sb_sc, d_state.d_dummy_m, bscposIn, 1, wohoB, fo,
                d_theta.d_delta_mb_sc);

            convDeltaSb<<<dimGrid3, dimBlock>>>(
                theta.d_Sb_sc, d_state.d_dummy_S, bscposIn, 1, wohoB, fo,
                d_theta.d_delta_Sb_sc);
        }

        //**
        // 1: Full connected
        //
        if (net.layers[k + 1] == net.layer_names.fc) {
            unsigned int gridRow = (ni + THREADS - 1) / THREADS;
            unsigned int gridCol = (no + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCol, gridRow);
            fcDeltaMw<<<dimGrid, dimBlock>>>(
                theta.d_Sw, state.d_ma, d_state.d_delta_m, wposIn, zposIn,
                zposOut, ni, B, no, d_theta.d_delta_mw);

            fcDeltaSw<<<dimGrid, dimBlock>>>(
                theta.d_Sw, state.d_ma, d_state.d_delta_S, wposIn, zposIn,
                zposOut, ni, B, no, d_theta.d_delta_Sw);

            unsigned int gridRow2 = (no + THREADS - 1) / THREADS;
            dim3 dimGrid2(1, gridRow2);
            fcDeltaMb<<<dimGrid2, dimBlock>>>(theta.d_Sb, d_state.d_delta_m,
                                              bposIn, zposOut, no, B, 1,
                                              d_theta.d_delta_mb);
            fcDeltaSb<<<dimGrid2, dimBlock>>>(theta.d_Sb, d_state.d_delta_S,
                                              bposIn, zposOut, no, B, 1,
                                              d_theta.d_delta_Sb);
        }

        //**
        // 2: Convolutional
        //
        else if (net.layers[k + 1] == net.layer_names.conv) {
            aidxposIn = net.Fmwa_2_pos[k];
            unsigned int gridRow = (B + THREADS - 1) / THREADS;
            unsigned int gridCol = (wohofo + THREADS - 1) / THREADS;
            dim3 dimGrid(gridCol, gridRow);

            permuteMeanVar<<<dimGrid, dimBlock>>>(
                d_state.d_delta_m, d_state.d_delta_S, d_state.d_dummy_m,
                d_state.d_dummy_S, zposOut, woho, wohofo, B);

            unsigned int gridRow2 = (ki2fi + THREADS - 1) / THREADS;
            unsigned int gridCol2 = (fo + THREADS - 1) / THREADS;
            dim3 dimGrid2(gridCol2, gridRow2);

            convDeltaMw<<<dimGrid2, dimBlock>>>(
                theta.d_Sw, state.d_ma, d_state.d_dummy_m, idx.d_Fmwa_2, wposIn,
                zposIn, aidxposIn, ki2fi, wohoB, fo, woho, wihi, fi, ki2,
                padIdx, d_theta.d_delta_mw);

            convDeltaSw<<<dimGrid2, dimBlock>>>(
                theta.d_Sw, state.d_ma, d_state.d_dummy_S, idx.d_Fmwa_2, wposIn,
                zposIn, aidxposIn, ki2fi, wohoB, fo, woho, wihi, fi, ki2,
                padIdx, d_theta.d_delta_Sw);

            if (net.num_biases[k + 1] > 0) {
                unsigned int gridCol = (fo + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, 1);
                convDeltaMb<<<dimGrid, dimBlock>>>(
                    theta.d_Sb, d_state.d_dummy_m, bposIn, 1, wohoB, fo,
                    d_theta.d_delta_mb);
                convDeltaSb<<<dimGrid, dimBlock>>>(
                    theta.d_Sb, d_state.d_dummy_S, bposIn, 1, wohoB, fo,
                    d_theta.d_delta_Sb);
            }
        }

        //*
        // 5: Layer normalization
        //
        else if (net.layers[k + 1] == net.layer_names.ln) {
            raposIn = net.ra_pos[k];
            if (net.layers[k] == net.layer_names.conv)  // Convolutional
            {
                unsigned int gridRow = (B + THREADS - 1) / THREADS;
                unsigned int gridCol = (wihifi + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);

                convlnDeltaMwSw<<<dimGrid, dimBlock>>>(
                    theta.d_Sw, state.d_ma, state.d_mra, state.d_Sra,
                    d_state.d_delta_m, d_state.d_delta_S, net.epsilon, wposIn,
                    zposIn, zposOut, raposIn, wihi, B, wihifi,
                    d_state.d_dummy_m, d_state.d_dummy_S);

                unsigned int gridColS = (fi + THREADS - 1) / THREADS;
                dim3 dimGridS(gridColS, 1);

                // REMOVE dimGridS
                deltaParamSum<<<dimGridS, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, wposIn, wihi, fi,
                    wihiB, d_theta.d_delta_mw, d_theta.d_delta_Sw);

                convlnDeltaMbSb<<<dimGrid, dimBlock>>>(
                    theta.d_Sb, d_state.d_delta_m, d_state.d_delta_S,
                    net.epsilon, bposIn, zposOut, wihi, B, wihifi,
                    d_state.d_dummy_m, d_state.d_dummy_S);

                deltaParamSum<<<dimGridS, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, bposIn, wihi, fi,
                    wihiB, d_theta.d_delta_mb, d_theta.d_delta_Sb);
            } else if (net.layers[k] == net.layer_names.fc)  // Full-connected
            {
                // *TODO TO BE TESTED
                unsigned int gridCol = (ni + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, 1);

                fclnDeltaMwSw<<<dimGrid, dimBlock>>>(
                    theta.d_Sw, state.d_ma, state.d_mra, state.d_Sra,
                    d_state.d_delta_m, d_state.d_delta_S, net.epsilon, wposIn,
                    zposIn, zposOut, raposIn, ni, B, d_theta.d_delta_mw,
                    d_theta.d_delta_Sw);

                fclnDeltaMbSb<<<dimGrid, dimBlock>>>(
                    theta.d_Sb, d_state.d_delta_m, d_state.d_delta_S,
                    net.epsilon, bposIn, zposOut, ni, B, d_theta.d_delta_mb,
                    d_theta.d_delta_Sb);
            }
        }

        //*
        // 6: Batch normalization
        //
        else if (net.layers[k + 1] == net.layer_names.bn) {
            raposIn = net.ra_pos[k];
            if (net.layers[k] == net.layer_names.conv)  // Convolutional
            {
                unsigned int gridRow = (fiB + THREADS - 1) / THREADS;
                unsigned int gridCol = (wihi + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, gridRow);

                convbnDeltaMwSw<<<dimGrid, dimBlock>>>(
                    theta.d_Sw, state.d_ma, state.d_mra, state.d_Sra,
                    d_state.d_delta_m, d_state.d_delta_S, net.epsilon, wposIn,
                    zposIn, zposOut, raposIn, wihi, fi, fiB, wihi,
                    d_state.d_dummy_m, d_state.d_dummy_S);

                unsigned gridColS = (fi + THREADS - 1) / THREADS;
                dim3 dimGridS(gridColS, 1);

                deltaParamSum<<<dimGridS, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, wposIn, wihi, fi,
                    wihiB, d_theta.d_delta_mw, d_theta.d_delta_Sw);

                convbnDeltaMbSb<<<dimGrid, dimBlock>>>(
                    theta.d_Sb, d_state.d_delta_m, d_state.d_delta_S,
                    net.epsilon, bposIn, zposOut, wihi, fi, fiB, wihi,
                    d_state.d_dummy_m, d_state.d_dummy_S);

                deltaParamSum<<<dimGridS, dimBlock>>>(
                    d_state.d_dummy_m, d_state.d_dummy_S, bposIn, wihi, fi,
                    wihiB, d_theta.d_delta_mb, d_theta.d_delta_Sb);
            } else if (net.layers[k] == net.layer_names.fc)  // Full-connected
            {
                // TODO TO BE TESTED
                unsigned int gridCol = (ni + THREADS - 1) / THREADS;
                dim3 dimGrid(gridCol, 1);

                fcbnDeltaMwSw<<<dimGrid, dimBlock>>>(
                    theta.d_Sw, state.d_ma, state.d_mra, state.d_Sra,
                    d_state.d_delta_m, d_state.d_delta_S, net.epsilon, wposIn,
                    zposIn, zposOut, raposIn, ni, B, d_theta.d_delta_mw,
                    d_theta.d_delta_Sw);

                fcbnDeltaMbSb<<<dimGrid, dimBlock>>>(
                    theta.d_Sb, d_state.d_delta_m, d_state.d_delta_S,
                    net.epsilon, bposIn, zposOut, ni, B, d_theta.d_delta_mb,
                    d_theta.d_delta_Sb);
            }
        }
        //*
        // 21: Transpose convolutional
        //
        else if (net.layers[k + 1] == net.layer_names.tconv) {
            // Launch kernel
            unsigned int gridRowW = (fi + THREADS - 1) / THREADS;
            unsigned int gridColW = (ki * ki * fo + THREADS - 1) / THREADS;
            unsigned int gridRowB = 1;
            unsigned int gridColB = (fo + THREADS - 1) / THREADS;

            dim3 dimGridW(gridColW, gridRowW);
            dim3 dimGridB(gridColB, gridRowB);

            tconvDeltaMw<<<dimGridW, dimBlock>>>(
                theta.d_Sw, state.d_ma, d_state.d_delta_m, idx.d_FCwz_2,
                idx.d_Swz_ud, wposIn, zposIn, zposOut, net.FCwz_2_pos[k],
                net.Swz_ud_pos[k], woho, fo, wihi, fi, ki, B,
                d_theta.d_delta_mw);

            tconvDeltaSw<<<dimGridW, dimBlock>>>(
                theta.d_Sw, state.d_ma, d_state.d_delta_S, idx.d_FCwz_2,
                idx.d_Swz_ud, wposIn, zposIn, zposOut, net.FCwz_2_pos[k],
                net.Swz_ud_pos[k], woho, fo, wihi, fi, ki, B,
                d_theta.d_delta_Sw);

            tconvDeltaMb<<<dimGridB, dimBlock>>>(theta.d_Sb, d_state.d_delta_m,
                                                 bposIn, zposOut, woho, fo, B,
                                                 d_theta.d_delta_mb);

            tconvDeltaSb<<<dimGridB, dimBlock>>>(theta.d_Sb, d_state.d_delta_S,
                                                 bposIn, zposOut, woho, fo, B,
                                                 d_theta.d_delta_Sb);
        }
        // 7: LSTM
        //
        else if (net.layers[k + 1] == net.layer_names.lstm) {
            lstm_parameter_update(net, state, theta, d_state, d_theta, k);
        }
    }
}

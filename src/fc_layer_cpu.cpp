///////////////////////////////////////////////////////////////////////////////
// File:         fc_layer_cpu.cpp
// Description:  CPU version for fully-connected layer
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      April 12, 2023
// Updated:      April 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/fc_layer_cpu.h"

////////////////////////////////////////////////////////////////////////////////
/// STATE FEED FORWARD
////////////////////////////////////////////////////////////////////////////////
void fc_mean_cpu(std::vector<float> &mw, std::vector<float> &mb,
                 std::vector<float> &ma, int w_pos, int b_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k, std::vector<float> &mz)
/*Compute mean of product WA for full connected layer

Args:
    mw: Mean of weights
    mb: Mean of the biases
    ma: Mean of activation units
    mz: Mean of hidden states
    w_pos: Weight position for this layer in the weight vector of network
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_in: Input-hidden-state position for this layer in the hidden-state
        vector of network
    z_pos_out: Output-hidden-state position for this layer in the hidden-state
        vector of network
    n: Input node
    m: Output node
    k: Number of batches
*/
{
    float sum = 0;
    float ma_tmp = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                ma_tmp = ma[n * col + i + z_pos_in];
                sum += mw[row * n + i + w_pos] * ma_tmp;
            }
            mz[col * m + row + z_pos_out] = sum + mb[row + b_pos];
        }
    }
}

void fc_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                std::vector<float> &Sb, std::vector<float> &ma,
                std::vector<float> &Sa, int w_pos, int b_pos, int z_pos_in,
                int z_pos_out, int m, int n, int k, std::vector<float> &Sz)
/*Compute variance of product WA for full connected layer

Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Variance of the biases
    ma: Mean of activation units
    Sa: Variance of activation units
    Sz: Variance of hidden states
    w_pos: Weight position for this layer in the weight vector of network
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_in: Input-hidden-state position for this layer in the hidden-state
        vector of network
    z_pos_out: Output-hidden-state position for this layer in the hidden-state
        vector of network
    n: Input node
    m: Output node
    k: Number of batches
*/
{
    float sum = 0;
    float ma_tmp = 0;
    float Sa_tmp = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                ma_tmp = ma[n * col + i + z_pos_in];
                Sa_tmp = Sa[n * col + i + z_pos_in];
                sum += (mw[row * n + i + w_pos] * mw[row * n + i + w_pos] +
                        Sw[row * n + i + w_pos]) *
                           Sa_tmp +
                       Sw[row * n + i + w_pos] * ma_tmp * ma_tmp;
            }
            Sz[col * m + row + z_pos_out] = sum + Sb[row + b_pos];
        }
    }
}
//////////////////////////////////////////////////////////////////////////
/// MULTI-THREADING VERSION
void fc_mean_var_worker(std::vector<float> &mw, std::vector<float> &Sw,
                        std::vector<float> &mb, std::vector<float> &Sb,
                        std::vector<float> &ma, std::vector<float> &Sa,
                        int w_pos, int b_pos, int z_pos_in, int z_pos_out,
                        int m, int n, int k, int start_idx, int end_idx,
                        std::vector<float> &mz, std::vector<float> &Sz) {
    float ma_tmp;
    float Sa_tmp;
    for (int i = start_idx; i < end_idx; i++) {
        int row = i / k;
        int col = i % k;
        float sum_mz = 0.0f;
        float sum_Sz = 0.0f;
        for (int j = 0; j < n; j++) {
            ma_tmp = ma[n * col + j + z_pos_in];
            Sa_tmp = Sa[n * col + j + z_pos_in];
            sum_mz += mw[row * n + j + w_pos] * ma_tmp;
            sum_Sz += (mw[row * n + j + w_pos] * mw[row * n + j + w_pos] +
                       Sw[row * n + j + w_pos]) *
                          Sa_tmp +
                      Sw[row * n + j + w_pos] * ma_tmp * ma_tmp;
        }
        mz[col * m + row + z_pos_out] = sum_mz + mb[row + b_pos];
        Sz[col * m + row + z_pos_out] = sum_Sz + Sb[row + b_pos];
    }
}

void fc_mean_var_multithreading(std::vector<float> &mw, std::vector<float> &Sw,
                                std::vector<float> &mb, std::vector<float> &Sb,
                                std::vector<float> &ma, std::vector<float> &Sa,
                                int w_pos, int b_pos, int z_pos_in,
                                int z_pos_out, int m, int n, int k,
                                unsigned int NUM_THREADS,
                                std::vector<float> &mz, std::vector<float> &Sz)

{
    const int tot_ops = m * k;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            fc_mean_var_worker, std::ref(mw), std::ref(Sw), std::ref(mb),
            std::ref(Sb), std::ref(ma), std::ref(Sa), w_pos, b_pos, z_pos_in,
            z_pos_out, m, n, k, start_idx, end_idx, std::ref(mz), std::ref(Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// FULL COVARIANCE FOR FULLY CONNECTED LAYER
////////////////////////////////////////////////////////////////////////////////
void fc_full_cov_cpu(std::vector<float> &mw, std::vector<float> &Sa_f,
                     int w_pos, int no, int ni, int B,
                     std::vector<float> &Sz_fp)
/* Compute full covariance matrix for fully-connected layer.

Args:
    mw: Mean of weights
    Sa_f: Full-covariance matrix of activation units for the previous layer
    w_pos: Weight position for this layer in the weight vector of network
    no: Output node
    ni: Input node
    B: Number of batches
    Sz_fp: Partial full-covariance matrix of hidden states of current
        layer
 */
{
    int tu, col, row, k;
    float sum, Sa_in;
    for (int row = 0; row < no * B; row++) {
        for (int col = 0; col < no; col++) {
            if (col <= (row % no)) {
                sum = 0;
                for (int i = 0; i < ni * ni; i++) {
                    int row_in = i / ni;
                    int col_in = i % ni;
                    if (row_in > col_in)  // lower triangle
                    {
                        tu = (ni * col_in - ((col_in * (col_in + 1)) / 2) +
                              row_in);
                    } else {
                        tu = (ni * row_in - ((row_in * (row_in + 1)) / 2) +
                              col_in);
                    }
                    Sa_in = Sa_f[tu + (row / no) * (ni * (ni + 1)) / 2];

                    sum += mw[i % ni + (row % no) * ni + w_pos] * Sa_in *
                           mw[i / ni + (col % no) * ni + w_pos];
                }
                k = no * col - ((col * (col + 1)) / 2) + row % no +
                    (row / no) * (((no + 1) * no) / 2);
                Sz_fp[k] = sum;
            }
        }
    }
}

void fc_full_var_cpu(std::vector<float> &mw, std::vector<float> &Sw,
                     std::vector<float> &Sb, std::vector<float> &ma,
                     std::vector<float> &Sa, std::vector<float> &Sz_fp,
                     int w_pos, int b_pos, int z_pos_in, int z_pos_out, int no,
                     int ni, int B, std::vector<float> &Sz,
                     std::vector<float> &Sz_f)
/* Add diagonal terms to the full covariance matrix.

Args:
    mw: Mean of weights
    Sw: Variance of weights
    Sb: Variance of biases
    Sz_fp: Partial full-covariance matrix of hidden states of current
                layer
    w_pos: Weight position for this layer in the weight vector of network
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
        of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
        of network
    no: Output node
    ni: Input node
    B: Number of batches
    Sz: Diagonal covariance matrix for hidden states
    Sz_f: Full-covariance matrix for hidden states
 */
{
    int col, row, i, k;
    float sum, final_sum;
    for (int j = 0; j < (no * (no + 1) / 2) * B; j++) {
        Sz_f[j] = Sz_fp[j];
    }
    for (row = 0; row < no; row++) {
        for (col = 0; col < B; col++) {
            sum = 0;
            for (i = 0; i < ni; i++) {
                sum += Sw[row * ni + i + w_pos] * Sa[ni * col + i + z_pos_in] +
                       Sw[row * ni + i + w_pos] * ma[ni * col + i + z_pos_in] *
                           ma[ni * col + i + z_pos_in];
            }
            k = no * row - (row * (row - 1)) / 2 + col * (no * (no + 1)) / 2;
            final_sum = sum + Sb[row + b_pos] + Sz_fp[k];
            Sz[col * no + row + z_pos_out] = final_sum;
            Sz_f[k] = final_sum;
        }
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void full_cov_worker(std::vector<float> &mw, std::vector<float> &Sa_f,
                     int w_pos, int no, int ni, int B, int start_idx,
                     int end_idx, std::vector<float> &Sz_fp) {
    int tu, col, row, k;
    float Sa_in;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / no;
        col = j % no;
        if (col <= (row % no)) {
            float sum = 0.0f;
            for (int i = 0; i < ni * ni; i++) {
                if ((i / ni) > (i % ni))  // Upper triangle
                {
                    tu = (ni * (i % ni) - (((i % ni) * (i % ni + 1)) / 2) +
                          i / ni);
                } else {
                    tu = (ni * (i / ni) - (((i / ni) * (i / ni + 1)) / 2) +
                          i % ni);
                }
                Sa_in = Sa_f[tu + (row / no) * (ni * (ni + 1)) / 2];
                sum += mw[i % ni + (row % no) * ni + w_pos] * Sa_in *
                       mw[i / ni + (col % no) * ni + w_pos];
            }
            k = no * col - ((col * (col + 1)) / 2) + row % no +
                (row / no) * (((no + 1) * no) / 2);
            Sz_fp[k] = sum;
        }
    }
}

void fc_full_cov_multithreading(std::vector<float> &mw,
                                std::vector<float> &Sa_f, int w_pos, int no,
                                int ni, int B, unsigned int NUM_THREADS,
                                std::vector<float> &Sz_fp) {
    const int tot_ops = no * B * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(full_cov_worker, std::ref(mw), std::ref(Sa_f), w_pos,
                        no, ni, B, start_idx, end_idx, std::ref(Sz_fp));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void fc_full_var_worker(std::vector<float> &mw, std::vector<float> &Sw,
                        std::vector<float> &Sb, std::vector<float> &ma,
                        std::vector<float> &Sa, std::vector<float> &Sz_fp,
                        int w_pos, int b_pos, int z_pos_in, int z_pos_out,
                        int no, int ni, int B, int start_idx, int end_idx,
                        std::vector<float> &Sz, std::vector<float> &Sz_f) {
    int col, row, i, k;
    float final_sum;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / B;
        col = j % B;
        float sum = 0.0f;
        for (i = 0; i < ni; i++) {
            sum += Sw[row * ni + i + w_pos] * Sa[ni * col + i + z_pos_in] +
                   Sw[row * ni + i + w_pos] * ma[ni * col + i + z_pos_in] *
                       ma[ni * col + i + z_pos_in];
        }
        k = no * row - (row * (row - 1)) / 2 + col * (no * (no + 1)) / 2;
        final_sum = sum + Sb[row + b_pos] + Sz_fp[k];
        Sz[col * no + row + z_pos_out] = final_sum;
        Sz_f[k] = final_sum;
    }
}

void fc_full_var_multithreading(std::vector<float> &mw, std::vector<float> &Sw,
                                std::vector<float> &Sb, std::vector<float> &ma,
                                std::vector<float> &Sa,
                                std::vector<float> &Sz_fp, int w_pos, int b_pos,
                                int z_pos_in, int z_pos_out, int no, int ni,
                                int B, unsigned int NUM_THREADS,
                                std::vector<float> &Sz,
                                std::vector<float> &Sz_f) {
    const int tot_ops = no * B;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;

    for (int j = 0; j < (no * (no + 1) / 2) * B; j++) {
        Sz_f[j] = Sz_fp[j];
    }
    std::thread threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(fc_full_var_worker, std::ref(mw), std::ref(Sw),
                                 std::ref(Sb), std::ref(ma), std::ref(Sa),
                                 std::ref(Sz_fp), w_pos, b_pos, z_pos_in,
                                 z_pos_out, no, ni, B, start_idx, end_idx,
                                 std::ref(Sz), std::ref(Sz_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
/// STATE FEED BACKWARD
////////////////////////////////////////////////////////////////////////////////
void fc_delta_mz(std::vector<float> &mw, std::vector<float> &Sz,
                 std::vector<float> &J, std::vector<float> &delta_m, int w_pos,
                 int z_pos_in, int z_pos_out, int ni, int no, int B,
                 std::vector<float> &delta_mz)
/* Compute the updated quatitites of the mean of the hidden states.

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    J: Jacobian vector
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
    w_pos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    ni: Number of hidden units for input
    B: Number of batches
    no: Number of hidden units for output
    delta_mz: Updated quantities for the mean of output's hidden states
*/
{
    float sum = 0;
    for (int row = 0; row < ni; row++) {
        for (int col = 0; col < B; col++) {
            sum = 0;
            for (int i = 0; i < no; i++) {
                sum += mw[ni * i + row + w_pos] *
                       delta_m[col * no + i + z_pos_out];
            }
            // TODO: could we compine the inovation function with this one in
            // order to reduce number of operation because Sz / Sz = no ops
            delta_mz[col * ni + row] = sum * Sz[col * ni + row + z_pos_in] *
                                       J[col * ni + row + z_pos_in];
        }
    }
}

void fc_delta_Sz(std::vector<float> &mw, std::vector<float> &Sz,
                 std::vector<float> &J, std::vector<float> &delta_S, int w_pos,
                 int z_pos_in, int z_pos_out, int ni, int no, int B,
                 std::vector<float> &delta_Sz)
/* Compute the updated quatitites for the variance of the hidden states.

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    J: Jacobian vector
    delta_S: Inovation vector for variance i.e. (S_observation - S_prediction)
    wpos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    ni: Number of hidden units for input
    B: Number of batches
    no: Number of hidden units for output
    delta_Sz: Updated quantities for the varaince of output's hidden states
*/
{
    float sum = 0;
    for (int row = 0; row < ni; row++) {
        for (int col = 0; col < B; col++) {
            sum = 0;
            for (int i = 0; i < no; i++) {
                sum += mw[ni * i + row + w_pos] *
                       delta_S[col * no + i + z_pos_out] *
                       mw[ni * i + row + w_pos];
            }
            delta_Sz[col * ni + row] = sum * Sz[col * ni + row + z_pos_in] *
                                       Sz[col * ni + row + z_pos_in] *
                                       J[col * ni + row + z_pos_in] *
                                       J[col * ni + row + z_pos_in];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
/// MULTI-THREADING VERSION
void fc_delta_mzSz_worker(std::vector<float> &mw, std::vector<float> &Sz,
                          std::vector<float> &J, std::vector<float> &delta_m,
                          std::vector<float> &delta_S, int w_pos, int z_pos_in,
                          int z_pos_out, int ni, int no, int B, int start_idx,
                          int end_idx, std::vector<float> &delta_mz,
                          std::vector<float> &delta_Sz)

{
    for (int j = start_idx; j < end_idx; j++) {
        int row = j / B;
        int col = j % B;
        float sum_mz = 0.0f;
        float sum_Sz = 0.0f;
        for (int i = 0; i < no; i++) {
            sum_mz +=
                mw[ni * i + row + w_pos] * delta_m[col * no + i + z_pos_out];

            sum_Sz += mw[ni * i + row + w_pos] *
                      delta_S[col * no + i + z_pos_out] *
                      mw[ni * i + row + w_pos];
        }
        delta_mz[col * ni + row] = sum_mz * Sz[col * ni + row + z_pos_in] *
                                   J[col * ni + row + z_pos_in];
        delta_Sz[col * ni + row] = sum_Sz * Sz[col * ni + row + z_pos_in] *
                                   Sz[col * ni + row + z_pos_in] *
                                   J[col * ni + row + z_pos_in] *
                                   J[col * ni + row + z_pos_in];
    }
}

void fc_delta_mzSz_multithreading(std::vector<float> &mw,
                                  std::vector<float> &Sz, std::vector<float> &J,
                                  std::vector<float> &delta_m,
                                  std::vector<float> &delta_S, int w_pos,
                                  int z_pos_in, int z_pos_out, int ni, int no,
                                  int B, unsigned int NUM_THREADS,
                                  std::vector<float> &delta_mz,
                                  std::vector<float> &delta_Sz)

{
    const int tot_ops = ni * B;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(fc_delta_mzSz_worker, std::ref(mw), std::ref(Sz),
                        std::ref(J), std::ref(delta_m), std::ref(delta_S),
                        w_pos, z_pos_in, z_pos_out, ni, no, B, start_idx,
                        end_idx, std::ref(delta_mz), std::ref(delta_Sz));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

//////////////////////////////////////////////////////////////////////
/// PARAMETERS FEED BACKWARD
//////////////////////////////////////////////////////////////////////
// This function computes the update amount for weight mean
// mW_new = mW_old + deltaMwz
void fc_delta_mw(std::vector<float> &Sw, std::vector<float> &ma,
                 std::vector<float> &delta_m, int w_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_mw)
/* Compute update quantities for the mean of weights for full-connected layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
    w_pos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    delta_mw: Updated quantities for the mean of weights
 */
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += ma[m * i + row + z_pos_in] *
                       delta_m[col + k * i + z_pos_out];
            }
            delta_mw[col * m + row + w_pos] = sum * Sw[col * m + row + w_pos];
        }
    }
}

// This function computes the update amount for weight variance
// SW_new = SW_old + deltaSw
void fc_delta_Sw(std::vector<float> &Sw, std::vector<float> &ma,
                 std::vector<float> &delta_S, int w_pos, int z_pos_in,
                 int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_Sw)
/* Compute update quantities for the variance of weights for full-connected
layer.

Args:
    Sw: Variance of weights
    ma: Mean of activation units
    delta_S: Inovation vector for variance i.e (S_observation - S_prediction)
    w_pos: Weight position for this layer in the weight vector of network
    z_pos_in: Input-hidden-state position for this layer in the weight vector
            of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    m: Number of hidden units for input
    n: Number of batches
    k: Number of hidden units for output
    deltaSw: Updated quantities for the variance of weights
*/
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += ma[m * i + row + z_pos_in] * ma[m * i + row + z_pos_in] *
                       delta_S[col + k * i + z_pos_out];
            }
            delta_Sw[col * m + row + w_pos] =
                sum * Sw[col * m + row + w_pos] * Sw[col * m + row + w_pos];
        }
    }
}

// This function computes the update amount for bias mean
// mb_new = mb_old + deltaMb
void fc_delta_mb(std::vector<float> &C_bz, std::vector<float> &delta_m,
                 int b_pos, int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_mb)
/* Compute update quantities for the mean of biases for full-connected layer.

Args:
    C_bz: Covariance b|Z+
    delta_m: Inovation vector for mean i.e. (M_observation - M_prediction)
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
            of network
    m: Number of hidden units for outputs
    n: Number of batches
    k: 1
    deltaMb: Updated quantities for the mean of biases
*/
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += delta_m[m * i + row + z_pos_out];
            }
            delta_mb[col * m + row + b_pos] = sum * C_bz[col * m + row + b_pos];
        }
    }
}

// This function computes the update amount for bias variance
// Sb_new = Sb_old + deltaSb
void fc_delta_Sb(std::vector<float> &C_bz, std::vector<float> &delta_S,
                 int b_pos, int z_pos_out, int m, int n, int k,
                 std::vector<float> &delta_Sb)
/* Compute update quantities for the variance of biases for full-connected
layer.

Args:
    C_bz: Covariance b|Z+
    delta_S: Inovation vector for variance i.e. (S_observation - S_prediction)
    b_pos: Bias position for this layer in the bias vector of network
    z_pos_out: Output-hidden-state position for this layer in the weight vector
    of network
    m: Number of hidden units for outputs
    n: Number of batches
    k: 1
    deltaSb: Updated quantities for the variance of biases
*/
{
    float sum = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            sum = 0;
            for (int i = 0; i < n; i++) {
                sum += delta_S[m * i + row + z_pos_out];
            }
            delta_Sb[col * m + row + b_pos] =
                sum * C_bz[col * m + row + b_pos] * C_bz[col * m + row + b_pos];
        }
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTI-THREADING VERSION
void fc_delta_w_worker(std::vector<float> &Sw, std::vector<float> &ma,
                       std::vector<float> &delta_m, std::vector<float> &delta_S,
                       int w_pos, int z_pos_in, int z_pos_out, int m, int n,
                       int k, int start_idx, int end_idx,
                       std::vector<float> &delta_mw,
                       std::vector<float> &delta_Sw) {
    for (int j = start_idx; j < end_idx; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mw = 0.0f;
        float sum_Sw = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_mw +=
                ma[m * i + row + z_pos_in] * delta_m[col + k * i + z_pos_out];
            sum_Sw += ma[m * i + row + z_pos_in] * ma[m * i + row + z_pos_in] *
                      delta_S[col + k * i + z_pos_out];
        }
        delta_mw[col * m + row + w_pos] = sum_mw * Sw[col * m + row + w_pos];
        delta_Sw[col * m + row + w_pos] =
            sum_Sw * Sw[col * m + row + w_pos] * Sw[col * m + row + w_pos];
    }
}

void fc_delta_w_multithreading(std::vector<float> &Sw, std::vector<float> &ma,
                               std::vector<float> &delta_m,
                               std::vector<float> &delta_S, int w_pos,
                               int z_pos_in, int z_pos_out, int m, int n, int k,
                               unsigned int NUM_THREADS,
                               std::vector<float> &delta_mw,
                               std::vector<float> &delta_Sw)

{
    const int tot_ops = m * k;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            fc_delta_w_worker, std::ref(Sw), std::ref(ma), std::ref(delta_m),
            std::ref(delta_S), w_pos, z_pos_in, z_pos_out, m, n, k, start_idx,
            end_idx, std::ref(delta_mw), std::ref(delta_Sw));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void fc_delta_b_worker(std::vector<float> &C_bz, std::vector<float> &delta_m,
                       std::vector<float> &delta_S, int b_pos, int z_pos_out,
                       int m, int n, int k, int start_idx, int end_idx,
                       std::vector<float> &delta_mb,
                       std::vector<float> &delta_Sb)

{
    for (int j = start_idx; j < end_idx; j++) {
        int row = j / k;
        int col = j % k;
        float sum_mb = 0.0f;
        float sum_Sb = 0.0f;
        for (int i = 0; i < n; i++) {
            sum_mb += delta_m[m * i + row + z_pos_out];
            sum_Sb += delta_S[m * i + row + z_pos_out];
        }
        delta_mb[col * m + row + b_pos] = sum_mb * C_bz[col * m + row + b_pos];
        delta_Sb[col * m + row + b_pos] =
            sum_Sb * C_bz[col * m + row + b_pos] * C_bz[col * m + row + b_pos];
    }
}

void fc_delta_b_multithreading(std::vector<float> &C_bz,
                               std::vector<float> &delta_m,
                               std::vector<float> &delta_S, int b_pos,
                               int z_pos_out, int m, int n, int k,
                               unsigned int NUM_THREADS,
                               std::vector<float> &delta_mb,
                               std::vector<float> &delta_Sb)

{
    const int tot_ops = m * k;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::thread threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] =
            std::thread(fc_delta_b_worker, std::ref(C_bz), std::ref(delta_m),
                        std::ref(delta_S), b_pos, z_pos_out, m, n, k, start_idx,
                        end_idx, std::ref(delta_mb), std::ref(delta_Sb));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}
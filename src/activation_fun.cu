///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cu
// Description:  Activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      March 12, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
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
                             int zpos, int apos, int n, float *ma, float *J,
                             float *Sa) {
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
        if (omega * mz_til > omega_tol) {
            ma[apos + col] = omega * mz_til;
            Sa[apos + col] =
                omega * Sz_til + omega * (1.0f - omega) * powf(mz_til, 2);
            // J[apos + col] = powf(omega * kappa, 0.5); // Approx. formulation
            J[apos + col] = (((powf(mz[zpos + col], 2) + Sz[zpos + col]) *
                                  normcdff(-alpha) +
                              mz[zpos + col] * powf(Sz[zpos + col], 0.5) *
                                  (1.0f / powf(2.0f * pi, 0.5)) *
                                  expf(-powf(-alpha, 2) / 2.0f)) -
                             (ma[apos + col] * mz[zpos + col])) /
                            Sz[zpos + col];
        } else {
            ma[apos + col] = omega_tol;
            Sa[apos + col] =
                omega * Sz_til + omega * (1.0f - omega) * powf(omega_tol, 2);
            J[apos + col] = 0.0f;  // TODO replace by 1.0f
        }
    }
}

__global__ void mixture_tanh(float const *mz, float const *Sz, float omega_tol,
                             int zpos, int n, float *ma, float *J, float *Sa) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number
    if (col < n) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + col]) / powf(Sz[zpos + col], 0.5);
        alpha_upper = (1.0f - mz[zpos + col]) / powf(Sz[zpos + col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
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
        J[zpos + col] = omega;
    }
}

__global__ void mixture_sigmoid(float const *mz, float const *Sz,
                                float omega_tol, int zpos, int n, float *ma,
                                float *J, float *Sa) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number
    if (col < n) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + col]) / powf(Sz[zpos + col], 0.5);
        alpha_upper = (1.0f - mz[zpos + col]) / powf(Sz[zpos + col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                (pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[zpos + col] - beta * powf(Sz[zpos + col], 0.5);
        Sz_til = kappa * Sz[zpos + col];

        // Activation distribution
        ma[zpos + col] =
            (omega * mz_til - cdf_lower + (1 - cdf_upper)) / 2.0f + 0.5f;
        Sa[zpos + col] =
            (omega * Sz_til + omega * powf(mz_til - ma[zpos + col], 2) +
             cdf_lower * powf(1 + ma[zpos + col], 2) +
             (1 - cdf_upper) * powf(1 - ma[zpos + col], 2)) /
            4.0f;
        J[zpos + col] = omega;
    }
}

__global__ void stable_softmax(float const *mu_z, float *var_z, int no, int B,
                               int z_pos, float *mu_a, float *J, float *var_a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float max_mu = mu_z[0];
    float max_var = var_z[0];
    for (int j = 1; j < no; j++) {
        if (mu_z[j + i * no + z_pos] > max_mu) {
            max_mu = mu_z[j + i * no + z_pos];
            max_var = var_z[j + i * no + z_pos];
        }
    }

    float sum_mu = 0.0f;
    for (int j = 0; j < no; j++) {
        sum_mu += expf(mu_z[j + i * no + z_pos] - max_mu);
    }
    float tmp_mu;
    for (int j = 0; j < no; j++) {
        tmp_mu = expf(mu_z[j + no * i + z_pos] - max_mu) / sum_mu;
        mu_a[j + i * no + z_pos] = tmp_mu;
        J[j + no * i + z_pos] = tmp_mu * (1 - tmp_mu);
        var_a[j + no * i + z_pos] = J[j + no * i + z_pos] *
                                    (var_z[j + no * i + z_pos] + max_var) *
                                    J[j + no * i + z_pos];
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

__global__ void exp_fn(float const *mu_z, float const *var_z, int no, int B,
                       int z_pos, float *mu_e, float *var_e, float *cov_e_z)
/* Compute the mean, variance, and cov(e, z) for the exponential function e =
exp(x).

Args:
    mu_z: Mean of hidden states
    var_z: Variance of hidden states
    mu_e: Mean of activation units
    var_e: Variance of activation units
    cov_e_z: Covariance between hidden states and activation units
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp_mu, tmp_var;
    if (col < no * B) {
        tmp_mu = mu_z[col + z_pos];
        tmp_var = var_z[col + z_pos];
        mu_e[col] = expf(mu_z[col + z_pos] + 0.5 * var_z[col + z_pos]);
        var_e[col] = expf(2 * tmp_mu + tmp_var) * (expf(tmp_var) - 1);
        cov_e_z[col] = tmp_var * expf(tmp_mu + 0.5 * tmp_var);
    }
}

//////////////////////////////////////////////////////////////////////////////
//// REMAX
//////////////////////////////////////////////////////////////////////////////
__global__ void to_log(float const *mu_m, float const *var_m, int no, int B,
                       float *mu_log, float *var_log) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp_mu, tmp_var;
    if (col >= no || row >= B) return;

    tmp_var =
        logf(1.0f + (var_m[row * no + col] / powf(mu_m[row * no + col], 2)));
    tmp_mu = logf(mu_m[row * no + col]) - 0.5 * tmp_var;
    mu_log[row * no + col] = tmp_mu;
    var_log[row * no + col] = tmp_var;
}

__global__ void sum_class_hidden_states(float const *mu_m, float const *var_m,
                                        int no, int B, float *mu_sum,
                                        float *var_sum) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f, sum_var = 0.0f;
    if (col >= B) return;
    for (int j = 0; j < no; j++) {
        sum_mu += mu_m[col * no + j];
        sum_var += var_m[col * no + j];
    }
    mu_sum[col] = sum_mu;
    var_sum[col] = sum_var;
}

__global__ void compute_cov_log_logsum(float const *mu_m, float const *var_m,
                                       float const *mu_sum, int no, int B,
                                       float *cov_log_logsum) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= no || row >= B) return;
    cov_log_logsum[row * no + col] =
        logf(1.0f + var_m[row * no + col] * (1.0f / mu_sum[row]) *
                        (1.0f / mu_m[row * no + col]));
}

__global__ void compute_cov_m_a_check(float const *var_log,
                                      float const *cov_log_logsum,
                                      float const *mu_m, int no, int B,
                                      float *cov_m_a_check) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= no || row >= B) return;
    cov_m_a_check[row * no + col] =
        (var_log[row * no + col] - cov_log_logsum[row * no + col]) *
        mu_m[row * no + col];
}

__global__ void compute_cov_m_a(float const *cov_m_a_check, float const *mu_a,
                                float const *var_m, float const *var_z,
                                float const *J_m, int z_pos, int no, int B,
                                float *cov_m_a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= no || row >= B) return;
    cov_m_a[row * no + col] =
        mu_a[row * no + col + z_pos] * cov_m_a_check[row * no + col] *
        J_m[row * no + col] * var_z[row * no + col + z_pos] /
        var_m[row * no + col];
}

__global__ void compute_remax_prob(float const *mu_log, float const *var_log,
                                   float const *mu_logsum,
                                   float const *var_logsum,
                                   float const *cov_log_logsum, int z_pos,
                                   int no, int B, float *mu_a, float *var_a) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= no || row >= B) return;
    float tmp_mu, tmp_var;
    tmp_mu = mu_log[row * no + col] - mu_logsum[row];
    tmp_var = var_log[row * no + col] + var_logsum[row] -
              2.0f * cov_log_logsum[row * no + col];
    mu_a[row * no + col + z_pos] = expf(tmp_mu + 0.5 * tmp_var);
    var_a[row * no + col + z_pos] =
        expf(tmp_mu + 0.5 * tmp_var) * (expf(tmp_var) - 1.0f);
}

void remax(Network &net, StateGPU &state, int l) {
    int z_pos = net.z_pos[l];
    int no = net.nodes[l];
    int B = net.batch_size;
    int THREADS = net.num_gpu_threads;
    unsigned int BLOCKS = (no * B + THREADS - 1) / THREADS;
    unsigned int BATCH_BLOCKS = (B + THREADS - 1) / THREADS;
    unsigned int grid_row = (B + THREADS - 1) / THREADS;
    unsigned int grid_col = (no + THREADS - 1) / THREADS;
    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(THREADS, THREADS);
    dim3 dim_grid_1(1, grid_row);

    // mrelu
    mixture_relu<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, net.omega_tol,
                                      z_pos, 0, no * B, state.remax.d_mu_m,
                                      state.remax.d_J_m, state.remax.d_var_m);

    // log of mrelu
    to_log<<<dim_grid, dim_block>>>(state.remax.d_mu_m, state.remax.d_var_m, no,
                                    B, state.remax.d_mu_log,
                                    state.remax.d_var_log);

    // sum of mrelu
    sum_class_hidden_states<<<BATCH_BLOCKS, THREADS>>>(
        state.remax.d_mu_m, state.remax.d_var_m, no, B, state.remax.d_mu_sum,
        state.remax.d_var_sum);
    to_log<<<dim_grid_1, dim_block>>>(
        state.remax.d_mu_sum, state.remax.d_var_sum, 1, B,
        state.remax.d_mu_logsum, state.remax.d_var_logsum);

    // Covariance between log of mrelu and log of sum of mrelu
    compute_cov_log_logsum<<<dim_grid, dim_block>>>(
        state.remax.d_mu_m, state.remax.d_var_m, state.remax.d_mu_sum, no, B,
        state.remax.d_cov_log_logsum);

    // Compute remax probabilities
    compute_remax_prob<<<dim_grid, dim_block>>>(
        state.remax.d_mu_log, state.remax.d_var_log, state.remax.d_mu_logsum,
        state.remax.d_var_logsum, state.remax.d_cov_log_logsum, z_pos, no, B,
        state.d_ma, state.d_Sa);
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

void activate_hidden_states(Network &net, StateGPU &state, int j) {
    int THREADS = net.num_gpu_threads;
    int B = net.batch_size;
    int no = net.nodes[j];
    int MB = net.nodes[j] * net.batch_size;
    if (net.layers[j] == net.layer_names.lstm) {
        MB = net.nodes[j] * net.batch_size * net.input_seq_len;
    }
    int z_pos = net.z_pos[j];
    unsigned int BLOCKS = (MB + THREADS - 1) / THREADS;

    // Compute mean, variance, and Jacobian matrix
    if (net.activations[j] == net.act_names.tanh)  // tanh
    {
        tanhMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                         state.d_J, state.d_Sa, z_pos, MB);
    } else if (net.activations[j] == net.act_names.sigmoid)  // sigmoid
    {
        sigmoidMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                            state.d_J, state.d_Sa, z_pos, MB);
    } else if (net.activations[j] == net.act_names.relu)  // ReLU
    {
        reluMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                         state.d_J, state.d_Sa, z_pos, MB);
    } else if (net.activations[j] == net.act_names.softplus)  // softplus
    {
        softplusMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                             state.d_J, state.d_Sa, z_pos, MB);
    } else if (net.activations[j] == net.act_names.leakyrelu)  // leaky ReLU
    {
        leakyreluMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, net.alpha,
                                              state.d_ma, state.d_J, state.d_Sa,
                                              z_pos, MB);

    } else if (net.activations[j] == net.act_names.mrelu)  // mReLU
    {
        mixture_relu<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, net.omega_tol,
                                          z_pos, z_pos, MB, state.d_ma,
                                          state.d_J, state.d_Sa);

    } else if (net.activations[j] == net.act_names.mtanh)  // mtanh
    {
        mixture_tanh<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, net.omega_tol,
                                          z_pos, MB, state.d_ma, state.d_J,
                                          state.d_Sa);

    } else if (net.activations[j] == net.act_names.msigmoid)  // msigmoid
    {
        mixture_sigmoid<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz,
                                             net.omega_tol, z_pos, MB,
                                             state.d_ma, state.d_J, state.d_Sa);

    } else if (net.activations[j] == net.act_names.softmax) {
        unsigned int softmax_blocks = (net.batch_size + THREADS - 1) / THREADS;
        stable_softmax<<<softmax_blocks, THREADS>>>(state.d_mz, state.d_Sz, no,
                                                    B, z_pos, state.d_ma,
                                                    state.d_J, state.d_Sa);
    } else if (net.activations[j] == net.act_names.remax)  // cf softmax
    {
        remax(net, state, j);
    } else  // no activation
    {
        noActMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                          state.d_J, state.d_Sa, z_pos, MB);
    }
}
///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun.cu
// Description:  Activation function
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 07, 2022
// Updated:      February 05, 2023
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

__global__ void mixture_tanh(float const *mz, float const *Sz, float omega_tol,
                             int zpos, int n, float *ma, float *J, float *Sa) {
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
        J[zpos + col] = powf(omega * kappa, 0.5);
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
        J[zpos + col] = powf(omega * kappa, 0.5);
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

__global__ void compute_sum_exp(float const *mu_e, float const *var_e, int no,
                                int B, float *mu_e_tilde, float *var_e_tilde) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_m, sum_v;
    if (i < B) {
        sum_m = 0;
        sum_v = 0;
        for (int j = 0; j < no; j++) {
            sum_m += mu_e[i * no + j];
            sum_v += var_e[i * no + j];
        }
        mu_e_tilde[i] = sum_m;
        var_e_tilde[i] = sum_v;
    }
}

__global__ void compute_cov_coeff_z_e_tilde(float const *var_e_tilde,
                                            float const *var_z, int no, int B,
                                            int z_pos, float const *mu_e,
                                            float *rho_z_e_tilde)
/*Covariance between the hidden states (Z) and the sim of exp(Z)
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < B && col < no) {
        rho_z_e_tilde[row * no + col] =
            (powf(var_z[row * no + z_pos], 0.5) * mu_e[row * no + col]) /
            powf(var_e_tilde[col], 0.5);
    }
}

__global__ void compute_cov_coeff_e_e_tilde(float const *var_e_tilde,
                                            float const *var_z, int no, int B,
                                            int z_pos, float const *mu_e,
                                            float const *var_e,
                                            float *rho_e_e_tilde)
/*Covariance between exp(Z) and the sum of exp(Z)*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < B && col < no) {
        rho_e_e_tilde[row * no + col] =
            ((powf(var_z[row * no + z_pos], 0.5) * mu_e[row * no + col]) /
             powf(var_e_tilde[row], 0.5)) *
            ((powf(var_z[row * no + z_pos], 0.5) * mu_e[row * no + col]) /
             powf(var_e[row * no + col], 0.5));
    }
}

__global__ void compute_log_sum_exp(float const *mu_e_tilde,
                                    float const *var_e_tilde, int B,
                                    float *mu_e_check, float *var_e_check)
/*Mean and variance of log(sum(exp(Z)))*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp;
    if (col < B) {
        tmp = logf(1 + var_e_tilde[col] / powf(mu_e_tilde[col], 2));
        mu_e_check[col] = logf(mu_e_tilde[col]) - 0.5 * tmp;
        var_e_check[col] = tmp;
    }
}

__global__ void compute_cov_z_e_check(float const *rho_e_e_tilde,
                                      float const *mu_e, float const *var_e,
                                      float const *mu_e_tilde,
                                      float const *var_e_tilde, int no, int B,
                                      float *cov_z_e_check)
/*Covariance between hidden states (Z) and log(sum(exp(Z)))*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < B && col < no) {
        cov_z_e_check[row * no + col] = logf(
            1 + rho_e_e_tilde[row] *
                    (powf(var_e[row * no + col], 0.5) / mu_e[row * no + col]) *
                    (powf(var_e_tilde[row], 0.5) / mu_e_tilde[row]));
    }
}

__global__ void exp_log_softmax(float const *mu_z, float const *var_z,
                                float const *mu_e_check,
                                float const *var_e_check,
                                float const *cov_z_e_check, int no, int B,
                                int z_pos, float *mu_a, float *var_a)
/*Convert log of softmax to softmax space*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp_mu, tmp_var;
    if (row < B && col < no) {
        tmp_mu = mu_z[z_pos + row * no + col] - mu_e_check[row];
        tmp_var = var_z[z_pos + row * no + col] + var_e_check[row] -
                  2 * cov_z_e_check[row * no + col];
        mu_a[z_pos + row * no + col] = expf(tmp_mu + 0.5 * tmp_var);
        var_a[z_pos + row * no + col] = powf(tmp_mu, 2) * (expf(tmp_var) - 1);
    }
}

__global__ void compute_y_check(float const *mu_z, float const *var_z,
                                float const *mu_e_check,
                                float const *var_e_check,
                                float const *cov_z_e_check, int no, int B,
                                int z_pos, float *mu_y_check,
                                float *var_y_check)
/*Compute the \check{y} mean and variance
    \check{y} = Z - \check{E},
where \check{E} = log(sum(exp(z)))
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp_mu, tmp_var;
    if (row < B && col < no) {
        tmp_mu = mu_z[z_pos + row * no + col] - mu_e_check[row];
        tmp_var = var_z[z_pos + row * no + col] + var_e_check[row] -
                  2 * cov_z_e_check[row * no + col];
        mu_y_check[row * no + col] = tmp_mu;
        var_y_check[row * no + col] = tmp_var;
    }
}

__global__ void compute_cov_y_y_check(float const *mu_z, float const *var_z,
                                      float const *mu_e_check,
                                      float const *var_e_check,
                                      float const *cov_z_e_check, int no, int B,
                                      int z_pos, float *cov_y_y_check)
/*Covariance betwee y and \check{y}. The observation equation is defined
following
            y = exp(\check{y}) + V, v~N(0, \sigma_{2}^{2}),
where \hat{y} = exp(\check{y}).
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float tmp_mu, tmp_var;
    if (row < B && col < no) {
        tmp_mu = mu_z[z_pos + row * no + col] - mu_e_check[row];
        tmp_var = var_z[z_pos + row * no + col] + var_e_check[row] -
                  2 * cov_z_e_check[row * no + col];
        cov_y_y_check[row * no + col] = expf(tmp_mu + 0.5 * tmp_var) * tmp_var;
    }
}

__global__ void compute_cov_z_y_check(float const *var_z,
                                      float const *cov_z_e_check, int no, int B,
                                      int z_pos, float *cov_z_y_check)
/* Covariance between hidden state z and \check{y}. See function
   `compute_cov_y_y_check_cpu`*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < B && col < no) {
        cov_z_y_check[row * no + col] =
            var_z[z_pos + row * no + col] - cov_z_e_check[row * no + col];
    }
}

__global__ void compute_cov_z_y(float const *mu_a, float const *cov_z_y_check,
                                int no, int B, int z_pos, float *cov_z_y) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= no * B) return;
    cov_z_y[col] = mu_a[col + z_pos] * cov_z_y_check[col];
}

void closed_form_softmax(Network &net, StateGPU &state, int l)
/*Closed-form softmax function*/
{
    int z_pos = net.z_pos[l];
    int no = net.nodes[l];
    int B = net.batch_size;
    int THREADS = net.num_gpu_threads;

    // Transform to exponential space
    int blocks = (no * B + THREADS - 1) / THREADS;
    exp_fn<<<blocks, THREADS>>>(
        state.d_mz, state.d_Sz, no, B, z_pos, state.cf_softmax.d_mu_e,
        state.cf_softmax.d_var_e, state.cf_softmax.d_cov_z_e);

    // Compute sum of the exponential of all hidden states
    int batch_blocks = (B + THREADS - 1) / THREADS;
    compute_sum_exp<<<batch_blocks, THREADS>>>(
        state.cf_softmax.d_mu_e, state.cf_softmax.d_var_e, no, B,
        state.cf_softmax.d_mu_e_tilde, state.cf_softmax.d_var_e_tilde);

    // Compute covariance coefficient between exp(z) and sum(exp(z))
    unsigned int grid_row = (B + THREADS - 1) / THREADS;
    unsigned int grid_col = (no + THREADS - 1) / THREADS;
    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(THREADS, THREADS);
    compute_cov_coeff_e_e_tilde<<<dim_grid, dim_block>>>(
        state.cf_softmax.d_var_e_tilde, state.d_Sz, no, B, z_pos,
        state.cf_softmax.d_mu_e, state.cf_softmax.d_var_e,
        state.cf_softmax.d_rho_e_e_tilde);

    // Transform sum(exp(z)) in log space
    compute_log_sum_exp<<<batch_blocks, THREADS>>>(
        state.cf_softmax.d_mu_e_tilde, state.cf_softmax.d_var_e_tilde, B,
        state.cf_softmax.d_mu_e_check, state.cf_softmax.d_var_e_check);

    // Covariance between z and log(sum(exp(z)))
    compute_cov_z_e_check<<<dim_grid, dim_block>>>(
        state.cf_softmax.d_rho_e_e_tilde, state.cf_softmax.d_mu_e,
        state.cf_softmax.d_var_e, state.cf_softmax.d_mu_e_tilde,
        state.cf_softmax.d_var_e_tilde, no, B,
        state.cf_softmax.d_cov_z_e_check);

    // Convert to softmax probability
    exp_log_softmax<<<dim_grid, dim_block>>>(
        state.d_mz, state.d_Sz, state.cf_softmax.d_mu_e_check,
        state.cf_softmax.d_var_e_check, state.cf_softmax.d_cov_z_e_check, no, B,
        z_pos, state.d_ma, state.d_Sa);
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
                                          z_pos, MB, state.d_ma, state.d_J,
                                          state.d_Sa);

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

    } else if (net.activations[j] == net.act_names.cf_softmax)  // cf softmax
    {
        closed_form_softmax(net, state, j);
    } else  // no activation
    {
        noActMeanVar<<<BLOCKS, THREADS>>>(state.d_mz, state.d_Sz, state.d_ma,
                                          state.d_J, state.d_Sa, z_pos, MB);
    }
}
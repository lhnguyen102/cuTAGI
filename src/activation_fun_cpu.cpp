///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun_cpu.cpp
// Description:  Activation function (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 11, 2022
// Updated:      September 18, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/activation_fun_cpu.h"

////////////////////////////////////////////////////////////////////////
//// REMAX
////////////////////////////////////////////////////////////////////////
void to_log_cpu(std::vector<float> &mu_m, std::vector<float> &var_m, int z_pos,
                int no, int B, std::vector<float> &mu_log,
                std::vector<float> &var_log) {
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_var = logf(1.0f + (var_m[i * no + j + z_pos] /
                                   powf(mu_m[i * no + j + z_pos], 2)));
            tmp_mu = logf(mu_m[i * no + j + z_pos]) - 0.5 * tmp_var;
            mu_log[i * no + j + z_pos] = tmp_mu;
            var_log[i * no + j + z_pos] = tmp_var;
        }
    }
}

void sum_class_hidden_states_cpu(std::vector<float> &mu_m,
                                 std::vector<float> &var_m, int z_pos,
                                 int z_sum_pos, int no, int B,
                                 std::vector<float> &mu_sum,
                                 std::vector<float> &var_sum) {
    float sum_mu, sum_var;
    for (int i = 0; i < B; i++) {
        sum_mu = 0.0f;
        sum_var = 0.0f;
        for (int j = 0; j < no; j++) {
            sum_mu += mu_m[i * no + j + z_pos];
            sum_var += var_m[i * no + j + z_pos];
        }
        mu_sum[i + z_sum_pos] = sum_mu;
        var_sum[i + z_sum_pos] = sum_var;
    }
}

void compute_cov_log_logsum_cpu(std::vector<float> &mu_m,
                                std::vector<float> &var_m,
                                std::vector<float> &mu_sum, int z_pos,
                                int z_sum_pos, int no, int B,
                                std::vector<float> &cov_log_logsum) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            cov_log_logsum[i * no + j + z_pos] =
                logf(1.0f + var_m[i * no + j + z_pos] *
                                (1.0f / mu_sum[i + z_sum_pos]) *
                                (1.0f / mu_m[i * no + j + z_pos]));
        }
    }
}

void compute_cov_m_a_check_cpu(std::vector<float> &var_log,
                               std::vector<float> &cov_log_logsum,
                               std::vector<float> &mu_m, int z_pos,
                               int z_sum_pos, int no, int B,
                               std::vector<float> &cov_m_a_check) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            cov_m_a_check[i * no + j] =
                (var_log[i * no + j + z_pos] -
                 cov_log_logsum[i * no + j + z_sum_pos]) *
                mu_m[i * no + j + z_pos];
        }
    }
}

void compute_cov_m_a_cpu(std::vector<float> &cov_m_a_check,
                         std::vector<float> &mu_a, std::vector<float> &var_m,
                         std::vector<float> &var_z, std::vector<float> &J_m,
                         int z_pos, int a_pos, int no, int B,
                         std::vector<float> &cov_a_m) {
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            cov_a_m[i * no + j] =
                mu_a[i * no + j + a_pos] * cov_m_a_check[i * no + j] *
                J_m[i * no + j + z_pos] * var_z[i * no + j + a_pos] /
                var_m[i * no + j + z_pos];
        }
    }
}

void compute_remax_prob_cpu(std::vector<float> &mu_log,
                            std::vector<float> &var_log,
                            std::vector<float> &mu_logsum,
                            std::vector<float> &var_logsum,
                            std::vector<float> &cov_log_logsum, int z_pos,
                            int z_remax_pos, int z_sum_remax_pos, int no, int B,
                            std::vector<float> &mu_a,
                            std::vector<float> &var_a) {
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_mu = mu_log[i * no + j + z_remax_pos] -
                     mu_logsum[i + z_sum_remax_pos];
            tmp_var = var_log[i * no + j + z_remax_pos] +
                      var_logsum[i + z_sum_remax_pos] -
                      2 * cov_log_logsum[i * no + j + z_remax_pos];
            mu_a[i * no + j + z_pos] = expf(tmp_mu + 0.5 * tmp_var);
            var_a[i * no + j + z_pos] =
                expf(tmp_mu + 0.5 * tmp_var) * (expf(tmp_var) - 1.0f);
        }
    }
}

////////////////////////////////////////////////////////////////////////
//// CLASSIC ACTIVATION
////////////////////////////////////////////////////////////////////////
void no_act_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                         int zpos, int n, std::vector<float> &ma,
                         std::vector<float> &J, std::vector<float> &Sa)
/* No activation function

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    zpos: Input-hidden-state position for this layer in the weight vector
          of network
    n: Number of hidden units for this layer
    ma: Mean of activation units
    Sa: Variance of activation units
    J: Jacobian matrix
*/
{
    float onePad = 1;
    for (int col = 0; col < n; col++) {
        ma[col + zpos] = mz[col + zpos];
        J[col + zpos] = onePad;
        Sa[col + zpos] = Sz[col + zpos];
    }
}

void tanh_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa) {
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = tanhf(mz[col + zpos]);
        ma[col + zpos] = tmp;
        J[col + zpos] = (1 - powf(tmp, 2));
        Sa[col + zpos] =
            (1 - powf(tmp, 2)) * Sz[col + zpos] * (1 - powf(tmp, 2));
    }
}

void sigmoid_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int n, std::vector<float> &ma,
                          std::vector<float> &J, std::vector<float> &Sa) {
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        ma[col + zpos] = tmp;
        J[col + zpos] = tmp * (1 - tmp);
        Sa[col + zpos] = tmp * (1 - tmp) * Sz[col + zpos] * tmp * (1 - tmp);
    }
}

void relu_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                       int n, std::vector<float> &ma, std::vector<float> &J,
                       std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
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

void softplus_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                           int zpos, int n, std::vector<float> &ma,
                           std::vector<float> &J, std::vector<float> &Sa) {
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        ma[col + zpos] = logf(1 + expf(mz[col + zpos]));
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        J[col + zpos] = tmp;
        Sa[col + zpos] = tmp * Sz[col + zpos] * tmp;
    }
}

void leakyrelu_mean_var_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                            float alpha, int zpos, int n,
                            std::vector<float> &ma, std::vector<float> &J,
                            std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp = 0;
    for (int col = 0; col < n; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
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

// TO BE replace the first one
void mixture_relu_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                      float omega_tol, int z_pos, int a_pos, int start_idx,
                      int end_idx, std::vector<float> &ma,
                      std::vector<float> &J, std::vector<float> &Sa) {
    float alpha, beta, omega, kappa, mz_til, Sz_til;
    for (int i = start_idx; i < end_idx; i++) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mz[z_pos + i] / powf(Sz[z_pos + i], 0.5);
        omega = std::max(1 - normcdf_cpu(alpha), omega_tol);
        beta = normpdf_cpu(alpha, 0.0f, 1.0f) / omega;
        kappa = 1 + alpha * beta - powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[z_pos + i] + beta * powf(Sz[z_pos + i], 0.5);
        Sz_til = kappa * Sz[z_pos + i];

        // Activation distribution
        if (omega * mz_til > omega_tol) {
            ma[i + a_pos] = omega * mz_til;
            Sa[i + a_pos] =
                omega * Sz_til + omega * (1 - omega) * powf(mz_til, 2);
            // J[i + a_pos] = powf(omega * kappa, 0.5); // Approximate
            // formulation
            J[i + a_pos] =  // Exact(Huber, 2020)
                (((pow(mz[z_pos + i], 2) + Sz[z_pos + i]) *
                      normcdf_cpu(mz[z_pos + i] / pow(Sz[z_pos + i], 0.5)) +
                  mz[z_pos + i] * Sz[z_pos + i] *
                      normpdf_cpu(0.0f, mz[z_pos + i],
                                  pow(Sz[z_pos + i], 0.5))) -
                 (ma[i + a_pos] * mz[z_pos + i])) /
                Sz[z_pos + i];
        } else {
            ma[i + a_pos] = omega_tol;
            Sa[i + a_pos] =
                omega * Sz_til + omega * (1 - omega) * powf(omega_tol, 2);
            J[i + a_pos] = 0.0f;
        }
    }
}

void mixture_tanh_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                      float omega_tol, int zpos, int start_idx, int end_idx,
                      std::vector<float> &ma, std::vector<float> &J,
                      std::vector<float> &Sa) {
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = start_idx; i < end_idx; i++) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
        alpha_upper = (1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
        cdf_lower = normcdf_cpu(alpha_lower);
        cdf_upper = normcdf_cpu(alpha_upper);
        pdf_lower = normpdf_cpu(alpha_lower, 0.0f, 1.0f);
        pdf_upper = normpdf_cpu(alpha_upper, 0.0f, 1.0f);

        // Truncated distribution's parameters
        omega = std::max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                ((pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega) -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[zpos + i] - beta * pow(Sz[zpos + i], 0.5);
        Sz_til = kappa * Sz[zpos + i];

        // Activation distribution
        ma[zpos + i] = omega * mz_til - cdf_lower + (1 - cdf_upper);
        Sa[zpos + i] = omega * Sz_til + omega * powf(mz_til - ma[zpos + i], 2) +
                       cdf_lower * powf(1 + ma[zpos + i], 2) +
                       (1 - cdf_upper) * powf(1 - ma[zpos + i], 2);
        J[zpos + i] = omega;
    }
}

void mixture_sigmoid_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                         float omega_tol, int zpos, int start_idx, int end_idx,
                         std::vector<float> &ma, std::vector<float> &J,
                         std::vector<float> &Sa) {
    float alpha_lower, alpha_upper, omega, beta, kappa, mz_til, Sz_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    for (int i = start_idx; i < end_idx; i++) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
        alpha_upper = (1.0f - mz[zpos + i]) / powf(Sz[zpos + i], 0.5);
        cdf_lower = normcdf_cpu(alpha_lower);
        cdf_upper = normcdf_cpu(alpha_upper);
        pdf_lower = normpdf_cpu(alpha_lower, 0.0f, 1.0f);
        pdf_upper = normpdf_cpu(alpha_upper, 0.0f, 1.0f);

        // Truncated distribution's parameters
        omega = std::max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                ((pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega) -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[zpos + i] - beta * pow(Sz[zpos + i], 0.5);
        Sz_til = kappa * Sz[zpos + i];

        // Activation distribution
        ma[zpos + i] =
            (omega * mz_til - cdf_lower + (1 - cdf_upper)) / 2.0f + 0.5f;
        Sa[zpos + i] =
            (omega * Sz_til + omega * powf(mz_til - ma[zpos + i], 2) +
             cdf_lower * powf(1 + ma[zpos + i], 2) +
             (1 - cdf_upper) * powf(1 - ma[zpos + i], 2)) /
            4.0f;
        J[zpos + i] = omega * 0.5;
    }
}

void silu(std::vector<float> &mu_z, std::vector<float> &var_z, float omega_tol,
          int z_pos, int n, std::vector<float> &mu_a, std::vector<float> &J,
          std::vector<float> &var_a)
/*Sigmoid Linear Unit (silu)
Observation equation: y   = x * sigmoid(x) where sigmoid function is replaced
a mixture bound relu.
*/
{
    // Pass through inputs mixture of sigmoid
    mixture_sigmoid_cpu(mu_z, var_z, omega_tol, z_pos, 0, n, mu_a, J, var_a);

    // GLU operation
    for (int col = 0; col < n; col++) {
        float tmp_mu_a = mu_a[col + z_pos];
        float tmp_var_a = var_a[col + z_pos];
        float cov_zm = J[col + z_pos] * var_z[col + z_pos];

        var_a[col + z_pos] = var_z[col + z_pos] * tmp_var_a + powf(cov_zm, 2) +
                             2 * cov_zm * mu_z[col + z_pos] * tmp_mu_a +
                             powf(mu_z[col + z_pos], 2) * tmp_var_a +
                             powf(tmp_mu_a, 2) * var_z[col + z_pos];

        mu_a[col + z_pos] = mu_z[col + z_pos] * tmp_mu_a + cov_zm;
    }
}

void softmax_cpu(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                 int no, int B, std::vector<float> &ma, std::vector<float> &J,
                 std::vector<float> &Sa) {
    float sum;
    int idx;
    for (int i = 0; i < B; i++) {
        sum = 0.0f;
        idx = zpos + i * no;
        for (int j = 0; j < no; j++) {
            ma[idx + j] = exp(mz[idx + j]);
            sum += ma[idx + j];
        }
        for (int j = 0; j < no; j++) {
            ma[idx + j] = ma[idx + j] / sum;
            J[idx + j] = ma[idx + j] * (1 - ma[idx + j]);
            Sa[idx + j] = J[idx + j] * Sz[idx + j] * J[idx + j];
        }
    }
}

void stable_softmax_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                        int zpos, int no, int B, std::vector<float> &ma,
                        std::vector<float> &J, std::vector<float> &Sa) {
    float sum, max_m, max_v;
    int idx;
    for (int i = 0; i < B; i++) {
        sum = 0.0f;
        idx = zpos + i * no;
        auto max_idx =
            std::max_element(mz.begin() + idx, mz.begin() + idx + no) -
            mz.begin();
        max_m = mz[max_idx];
        max_v = Sz[max_idx];
        for (int j = 0; j < no; j++) {
            ma[idx + j] = expf(mz[idx + j] - max_m);
            sum += ma[idx + j];
        }
        for (int j = 0; j < no; j++) {
            ma[idx + j] = ma[idx + j] / sum;
            J[idx + j] = ma[idx + j] * (1 - ma[idx + j]);
            // TODO: double check on covatiance formulation
            Sa[idx + j] = J[idx + j] * (Sz[idx + j] + max_v) * J[idx + j];
        }
    }
}

void exp_fn_cpu(std::vector<float> &mu_z, std::vector<float> &var_z, int no,
                int B, int z_pos, std::vector<float> &mu_e,
                std::vector<float> &var_e, std::vector<float> &cov_e_z)
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
    float tmp_m, tmp_S;
    for (int i = 0; i < no * B; i++) {
        tmp_m = mu_z[i + z_pos];
        tmp_S = var_z[i + z_pos];
        mu_e[i] = expf(mu_z[i + z_pos] + 0.5 * var_z[i + z_pos]);
        var_e[i] = expf(2 * tmp_m + tmp_S) * (expf(tmp_S) - 1);
        cov_e_z[i] = tmp_S * expf(tmp_m + 0.5 * tmp_S);
    }
}

void remax_cpu(Network &net, NetState &state, int l)
/*Remax activation function*/
{
    int z_pos = net.z_pos[l];
    int z_remax_pos = state.remax.z_pos[l];
    int z_sum_remax_pos = state.remax.z_sum_pos[l];
    int no = net.nodes[l];
    int B = net.batch_size;

    // mrelu
    mixture_relu_cpu(state.mz, state.Sz, net.omega_tol, z_pos, z_remax_pos, 0,
                     no * B, state.remax.mu_m, state.remax.J_m,
                     state.remax.var_m);

    // log of mrelu
    to_log_cpu(state.remax.mu_m, state.remax.var_m, z_remax_pos, no, B,
               state.remax.mu_log, state.remax.var_log);

    // sum of mrelu
    sum_class_hidden_states_cpu(state.remax.mu_m, state.remax.var_m,
                                z_remax_pos, z_sum_remax_pos, no, B,
                                state.remax.mu_sum, state.remax.var_sum);
    to_log_cpu(state.remax.mu_sum, state.remax.var_sum, z_sum_remax_pos, 1, B,
               state.remax.mu_logsum, state.remax.var_logsum);

    // covariance between log of mrelu and log of sum of mrelu
    compute_cov_log_logsum_cpu(state.remax.mu_m, state.remax.var_m,
                               state.remax.mu_sum, z_remax_pos, z_sum_remax_pos,
                               no, B, state.remax.cov_log_logsum);

    // Compute remax probabilities
    compute_remax_prob_cpu(state.remax.mu_log, state.remax.var_log,
                           state.remax.mu_logsum, state.remax.var_logsum,
                           state.remax.cov_log_logsum, z_pos, z_remax_pos,
                           z_sum_remax_pos, no, B, state.ma, state.Sa);
}

void remax_cpu_v2(std::vector<float> &mz, std::vector<float> &Sz,
                  std::vector<float> &mu_m, std::vector<float> &var_m,
                  std::vector<float> &J_m, std::vector<float> &mu_log,
                  std::vector<float> &var_log, std::vector<float> &mu_sum,
                  std::vector<float> &var_sum, std::vector<float> &mu_logsum,
                  std::vector<float> &var_logsum,
                  std::vector<float> &cov_log_logsum, std::vector<float> &ma,
                  std::vector<float> &Sa, int z_pos, int z_remax_pos,
                  int z_sum_remax_pos, int no, int B, float omega_tol)
/*Remax is an activation function used to calculate the probability for each
   class as softmax*/
{
    int no_sum = 1;
    // mrelu
    mixture_relu_cpu(mz, Sz, omega_tol, z_pos, z_remax_pos, 0, no * B, mu_m,
                     J_m, var_m);

    // log of mrelu
    to_log_cpu(mu_m, var_m, z_remax_pos, no, B, mu_log, var_log);

    // sum of relu
    sum_class_hidden_states_cpu(mu_m, var_m, z_remax_pos, z_sum_remax_pos, no,
                                B, mu_sum, var_sum);
    to_log_cpu(mu_sum, var_sum, z_sum_remax_pos, no_sum, B, mu_logsum,
               var_logsum);

    // Covariance between log of mrelu and log of sum of relu
    compute_cov_log_logsum_cpu(mu_m, var_m, mu_sum, z_remax_pos,
                               z_sum_remax_pos, no, B, cov_log_logsum);

    // Compute remax probabilities
    compute_remax_prob_cpu(mu_log, var_log, mu_logsum, var_logsum,
                           cov_log_logsum, z_pos, z_remax_pos, z_sum_remax_pos,
                           no, B, ma, Sa);
}

void exp_fun_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                 std::vector<float> &ma, std::vector<float> &Sa,
                 std::vector<float> &Cza)
/* Exponential function y = exp(x)

Args:
    mz: Mean of hidden states
    Sz: Variance of hidden states
    ma: Mean of activation units
    Sa: Variance of activation units
    Cza: Covariance between hidden states and activation units
*/
{
    float tmp_m, tmp_S;
    for (int i = 0; i < mz.size(); i++) {
        tmp_m = mz[i];
        tmp_S = Sz[i];
        ma[i] = exp(mz[i] + 0.5 * Sz[i]);
        Sa[i] = exp(2 * tmp_m + tmp_S) * (exp(tmp_S) - 1);
        Cza[i] = tmp_S * exp(tmp_m + 0.5 * tmp_S);
    }
}

void act_full_cov(std::vector<float> &Sz_f, std::vector<float> &J, int no,
                  int B, int z_pos_out, std::vector<float> &Sa_f)
/*Activate the full covariance.

Args:
    Sz_f: Full-covariance matrix for hidden states
    J: Jacobian matrix
    no: Output node
    B: Number of batches
    z_pos_out: Output-hidden-state position for this layer in the weight vector
        of network
    Sa_f: Full-covariance matrix for activation units

*/
{
    int col, row, idx;
    for (row = 0; row < no * B; row++) {
        for (col = 0; col < no; col++) {
            if (col <= (row % no)) {
                idx = no * col - ((col * (col + 1)) / 2) + row % no +
                      (row / no) * (((no + 1) * no) / 2);
                Sa_f[idx] = Sz_f[idx] *
                            J[row % no + (row / no) * no + z_pos_out] *
                            J[col + (row / no) * no + z_pos_out];
            }
        }
    }
}

void no_act_full_cov(std::vector<float> &Sz_f, int no, int B,
                     std::vector<float> &Sa_f)
/* No activation layer*/
{
    int col;
    for (col = 0; col < (no * (no + 1)) / 2 * B; col++) {
        Sa_f[col] = Sz_f[col];
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void no_act_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                            int zpos, int start_idx, int end_idx,
                            std::vector<float> &ma, std::vector<float> &J,
                            std::vector<float> &Sa)

{
    int col;
    float onePad = 1;
    for (col = start_idx; col < end_idx; col++) {
        ma[col + zpos] = mz[col + zpos];
        J[col + zpos] = onePad;
        Sa[col + zpos] = Sz[col + zpos];
    }
}

void no_act_mean_var_multithreading(std::vector<float> &mz,
                                    std::vector<float> &Sz, int z_pos, int n,
                                    unsigned int NUM_THREADS,
                                    std::vector<float> &ma,
                                    std::vector<float> &J,
                                    std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
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
        threads[i] = std::thread(no_act_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void tanh_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int start_idx, int end_idx,
                          std::vector<float> &ma, std::vector<float> &J,
                          std::vector<float> &Sa) {
    int col;
    float tmp = 0;
    for (col = start_idx; col < end_idx; col++) {
        tmp = tanhf(mz[col + zpos]);
        ma[col + zpos] = tmp;
        J[col + zpos] = (1 - tmp * tmp);
        Sa[col + zpos] = (1 - tmp * tmp) * Sz[col + zpos] * (1 - tmp * tmp);
    }
}

void tanh_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  unsigned int NUM_THREADS,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
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
        threads[i] = std::thread(tanh_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void sigmoid_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                             int zpos, int start_idx, int end_idx,
                             std::vector<float> &ma, std::vector<float> &J,
                             std::vector<float> &Sa) {
    int col;
    float tmp;
    for (col = start_idx; col < end_idx; col++) {
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        ma[col + zpos] = tmp;
        J[col + zpos] = tmp * (1 - tmp);
        Sa[col + zpos] = tmp * (1 - tmp) * Sz[col + zpos] * tmp * (1 - tmp);
    }
}

void sigmoid_mean_var_multithreading(std::vector<float> &mz,
                                     std::vector<float> &Sz, int z_pos, int n,
                                     unsigned int NUM_THREADS,
                                     std::vector<float> &ma,
                                     std::vector<float> &J,
                                     std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
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
        threads[i] = std::thread(sigmoid_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void no_act_full_cov_worker(std::vector<float> &Sz_f, int start_idx,
                            int end_idx, std::vector<float> &Sa_f) {
    int col;
    for (col = start_idx; col < end_idx; col++) {
        Sa_f[col] = Sz_f[col];
    }
}

void no_act_full_cov_multithreading(std::vector<float> &Sz_f, int no, int B,
                                    unsigned int NUM_THREADS,
                                    std::vector<float> &Sa_f) {
    const int tot_ops = (no * (no + 1)) / 2 * B;
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
        threads[i] = std::thread(no_act_full_cov_worker, std::ref(Sz_f),
                                 start_idx, end_idx, std::ref(Sa_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void relu_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                          int zpos, int start_idx, int end_idx,
                          std::vector<float> &ma, std::vector<float> &J,
                          std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp;
    int col;
    for (col = start_idx; col < end_idx; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
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

void relu_mean_var_multithreading(std::vector<float> &mz,
                                  std::vector<float> &Sz, int z_pos, int n,
                                  unsigned int NUM_THREADS,
                                  std::vector<float> &ma, std::vector<float> &J,
                                  std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
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
        threads[i] = std::thread(relu_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void softplus_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                              int zpos, int start_idx, int end_idx,
                              std::vector<float> &ma, std::vector<float> &J,
                              std::vector<float> &Sa) {
    float tmp;
    int col;
    for (col = start_idx; col < end_idx; col++) {
        ma[col + zpos] = logf(1 + expf(mz[col + zpos]));
        tmp = 1 / (1 + expf(-mz[col + zpos]));
        J[col + zpos] = tmp;
        Sa[col + zpos] = tmp * Sz[col + zpos] * tmp;
    }
}

void softplus_mean_var_multithreading(std::vector<float> &mz,
                                      std::vector<float> &Sz, int z_pos, int n,
                                      unsigned int NUM_THREADS,
                                      std::vector<float> &ma,
                                      std::vector<float> &J,
                                      std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
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
        threads[i] = std::thread(softplus_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void leakyrelu_mean_var_worker(std::vector<float> &mz, std::vector<float> &Sz,
                               float alpha, int zpos, int start_idx,
                               int end_idx, std::vector<float> &ma,
                               std::vector<float> &J, std::vector<float> &Sa) {
    float zeroPad = 0;
    float onePad = 1;
    float tmp;
    int col;
    for (col = start_idx; col < end_idx; col++) {
        tmp = std::max(mz[col + zpos], zeroPad);
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

void leakyrelu_mean_var_multithreading(
    std::vector<float> &mz, std::vector<float> &Sz, float alpha, int z_pos,
    int n, unsigned int NUM_THREADS, std::vector<float> &ma,
    std::vector<float> &J, std::vector<float> &Sa)

{
    const int n_batch = n / NUM_THREADS;
    const int rem_batch = n % NUM_THREADS;
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
        threads[i] = std::thread(leakyrelu_mean_var_worker, std::ref(mz),
                                 std::ref(Sz), alpha, z_pos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void mixture_relu_multithreading(std::vector<float> &mz, std::vector<float> &Sz,
                                 float omega_tol, int zpos, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &ma, std::vector<float> &J,
                                 std::vector<float> &Sa) {
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(mixture_relu_cpu, std::ref(mz), std::ref(Sz),
                                 omega_tol, zpos, zpos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void mixture_tanh_multithreading(std::vector<float> &mz, std::vector<float> &Sz,
                                 float omega_tol, int zpos, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &ma, std::vector<float> &J,
                                 std::vector<float> &Sa) {
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::thread threads[num_threads];

    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(mixture_tanh_cpu, std::ref(mz), std::ref(Sz),
                                 omega_tol, zpos, start_idx, end_idx,
                                 std::ref(ma), std::ref(J), std::ref(Sa));
    }

    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void mixture_sigmoid_multithreading(std::vector<float> &mz,
                                    std::vector<float> &Sz, float omega_tol,
                                    int zpos, int n, unsigned int num_threads,
                                    std::vector<float> &ma,
                                    std::vector<float> &J,
                                    std::vector<float> &Sa) {
    const int n_batch = n / num_threads;
    const int rem_batch = n % num_threads;
    int start_idx, end_idx;
    std::thread threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        if (i == 0) {
            start_idx = n_batch * i;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        } else {
            start_idx = n_batch * i + rem_batch;
            end_idx = (n_batch * (i + 1)) + rem_batch;
        }
        threads[i] = std::thread(
            mixture_sigmoid_cpu, std::ref(mz), std::ref(Sz), omega_tol, zpos,
            start_idx, end_idx, std::ref(ma), std::ref(J), std::ref(Sa));
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
}

void softmax_worker(std::vector<float> &mz, std::vector<float> &Sz, int zpos,
                    int no, int B, int start_idx, int end_idx,
                    std::vector<float> &ma, std::vector<float> &J,
                    std::vector<float> &Sa) {
    int row, col;
    for (int i = start_idx; i < end_idx; i++) {
        row = i / no;
        col = i % no;
    }
}

void act_full_cov_worker(std::vector<float> &Sz_f, std::vector<float> &J,
                         int no, int B, int z_pos_out, int start_idx,
                         int end_idx, std::vector<float> &Sa_f) {
    int col, row, idx;
    for (int j = start_idx; j < end_idx; j++) {
        row = j / no;
        col = j % no;
        if (col <= (row % no)) {
            idx = no * col - ((col * (col + 1)) / 2) + row % no +
                  (row / no) * (((no + 1) * no) / 2);

            Sa_f[idx] = Sz_f[idx] * J[row % no + (row / no) * no + z_pos_out] *
                        J[col + (row / no) * no + z_pos_out];
        }
    }
}

void act_full_cov_multithreading(std::vector<float> &Sz_f,
                                 std::vector<float> &J, int no, int B,
                                 int z_pos_out, unsigned int NUM_THREADS,
                                 std::vector<float> &Sa_f) {
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
            std::thread(act_full_cov_worker, std::ref(Sz_f), std::ref(J), no, B,
                        z_pos_out, start_idx, end_idx, std::ref(Sa_f));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void activate_hidden_states_cpu(Network &net, NetState &state, int j) {
    int B = net.batch_size;
    int no = net.nodes[j];
    int ni = net.nodes[j - 1];
    int z_pos_out = net.z_pos[j];
    int z_pos_in = net.z_pos[j - 1];
    int w_pos_in = net.w_pos[j - 1];
    int b_pos_in = net.b_pos[j - 1];
    int no_B = no * B;

    // Handle multiple input sequences from LSTM layer
    if (net.layers[j - 1] == net.layer_names.lstm) {
        ni = net.nodes[j - 1] * net.input_seq_len;
    }
    if (net.layers[j] == net.layer_names.lstm) {
        no_B = no * B * net.input_seq_len;
    }
    if (net.activations[j] == net.act_names.tanh)  // tanh
    {
        if (no * B > net.min_operations && net.multithreading) {
            tanh_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                         net.num_cpu_threads, state.ma, state.J,
                                         state.Sa);
        } else {
            tanh_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                              state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.sigmoid)  // sigmoid
    {
        if (no * B > net.min_operations && net.multithreading) {
            sigmoid_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                            net.num_cpu_threads, state.ma,
                                            state.J, state.Sa);

        } else {
            sigmoid_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                 state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.relu)  // ReLU
    {
        if (no * B > net.min_operations && net.multithreading) {
            relu_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                         net.num_cpu_threads, state.ma, state.J,
                                         state.Sa);
        } else {
            relu_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                              state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.mrelu)  // mReLU
    {
        if (no * B > net.min_operations && net.multithreading) {
            mixture_relu_multithreading(state.mz, state.Sz, net.omega_tol,
                                        z_pos_out, no_B, net.num_cpu_threads,
                                        state.ma, state.J, state.Sa);
        } else {
            mixture_relu_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out,
                             z_pos_out, 0, no_B, state.ma, state.J, state.Sa);
        }

    } else if (net.activations[j] == net.act_names.mtanh)  // mtanh
    {
        if (no * B > net.min_operations && net.multithreading) {
            mixture_tanh_multithreading(state.mz, state.Sz, net.omega_tol,
                                        z_pos_out, no_B, net.num_cpu_threads,
                                        state.ma, state.J, state.Sa);
        } else {
            mixture_tanh_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out, 0,
                             no_B, state.ma, state.J, state.Sa);
        }

    } else if (net.activations[j] == net.act_names.msigmoid)  // msigmoid
    {
        if (no * B > net.min_operations && net.multithreading) {
            mixture_sigmoid_multithreading(state.mz, state.Sz, net.omega_tol,
                                           z_pos_out, no_B, net.num_cpu_threads,
                                           state.ma, state.J, state.Sa);
        } else {
            mixture_sigmoid_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out, 0,
                                no_B, state.ma, state.J, state.Sa);
        }

    } else if (net.activations[j] == net.act_names.softplus)  // softplus
    {
        if (no * B > net.min_operations && net.multithreading) {
            softplus_mean_var_multithreading(state.mz, state.Sz, z_pos_out,
                                             no_B, net.num_cpu_threads,
                                             state.ma, state.J, state.Sa);

        } else {
            softplus_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                  state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.leakyrelu)  // leaky ReLU
    {
        if (no * B > net.min_operations && net.multithreading) {
            leakyrelu_mean_var_multithreading(
                state.mz, state.Sz, net.alpha, z_pos_out, no_B,
                net.num_cpu_threads, state.ma, state.J, state.Sa);
        } else {
            leakyrelu_mean_var_cpu(state.mz, state.Sz, net.alpha, z_pos_out,
                                   no_B, state.ma, state.J, state.Sa);
        }
    } else if (net.activations[j] == net.act_names.softmax)  // softmax
    {
        // softmax_cpu(state.mz, state.Sz, z_pos_out, no, B, state.ma, state.J,
        //             state.Sa);
        stable_softmax_cpu(state.mz, state.Sz, z_pos_out, no, B, state.ma,
                           state.J, state.Sa);
    } else if (net.activations[j] == net.act_names.remax)  // cf softmax
    {
        remax_cpu(net, state, j);
    } else  // no activation
    {
        if (no * B > net.min_operations && net.multithreading) {
            no_act_mean_var_multithreading(state.mz, state.Sz, z_pos_out, no_B,
                                           net.num_cpu_threads, state.ma,
                                           state.J, state.Sa);
        } else {
            no_act_mean_var_cpu(state.mz, state.Sz, z_pos_out, no_B, state.ma,
                                state.J, state.Sa);
        }
    }

    // Full-covariance mode
    if (net.is_full_cov) {
        if (net.activations[j] == 0) {
            if (no * B * no > net.min_operations && net.multithreading) {
                no_act_full_cov_multithreading(state.Sz_f, no, B,
                                               net.num_cpu_threads, state.Sa_f);
            } else {
                no_act_full_cov(state.Sz_f, no, B, state.Sa_f);
            }
        } else {
            if (((no * (no + 1) / 2) * B) > net.min_operations &&
                net.multithreading) {
                act_full_cov_multithreading(state.Sz_f, state.J, no, B,
                                            z_pos_out, net.num_cpu_threads,
                                            state.Sa_f);
            } else {
                act_full_cov(state.Sz_f, state.J, no, B, z_pos_out, state.Sa_f);
            }
        }
    }
}

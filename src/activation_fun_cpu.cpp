///////////////////////////////////////////////////////////////////////////////
// File:         activation_fun_cpu.cpp
// Description:  Activation function (CPU version)
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 11, 2022
// Updated:      January 05, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "../include/activation_fun_cpu.h"

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

void mixture_relu_cpu(std::vector<float> &mz, std::vector<float> &Sz,
                      float omega_tol, int zpos, int start_idx, int end_idx,
                      std::vector<float> &ma, std::vector<float> &J,
                      std::vector<float> &Sa) {
    float alpha, beta, omega, kappa, mz_til, Sz_til;
    for (int i = start_idx; i < end_idx; i++) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mz[zpos + i] / powf(Sz[zpos + i], 0.5);
        omega = std::max(1 - normcdf_cpu(alpha), omega_tol);
        beta = normpdf_cpu(alpha, 0.0f, 1.0f) / omega;
        kappa = 1 + alpha * beta - powf(beta, 2);

        // Gaussian mixture's paramters
        mz_til = mz[zpos + i] + beta * powf(Sz[zpos + i], 0.5);
        Sz_til = kappa * Sz[zpos + i];

        // Activation distribution
        ma[zpos + i] = omega * mz_til;
        Sa[zpos + i] = omega * Sz_til + omega * (1 - omega) * powf(mz_til, 2);
        J[zpos + i] = powf(omega * kappa, 0.5);
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
        J[zpos + i] = powf(omega * kappa, 0.5);
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
        J[zpos + i] = powf(omega * kappa, 0.5);
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
            Sa[idx + j] = J[idx + j] * (Sz[idx + j] + max_v) * J[idx + j];
        }
    }
}

void max_norm(std::vector<float> &mz, std::vector<float> &Sz, int zpos, int no,
              int B, std::vector<float> &mz_norm, std::vector<float> &Sz_norm) {
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
            mz_norm[idx + j] = mz[idx + j] - max_m;
            Sz_norm[idx + j] = Sz[idx + j] + max_v;
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

void compute_sum_exp_cpu(std::vector<float> &me, std::vector<float> &ve, int no,
                         int B, std::vector<float> &me_tilde,
                         std::vector<float> &ve_tilde) {
    float sum_m, sum_v;
    for (int i = 0; i < B; i++) {
        sum_m = 0;
        sum_v = 0;
        for (int j = 0; j < no; j++) {
            sum_m += me[i * no + j];
            sum_v += ve[i * no + j];
        }
        me_tilde[i] = sum_m;
        ve_tilde[i] = sum_v;
    }
}

void compute_cov_coeff_z_e_tilde_cpu(std::vector<float> &ve_tilde,
                                     std::vector<float> &vz, int no, int B,
                                     int z_pos, std::vector<float> &me,
                                     std::vector<float> &rho_z_e_tilde)
/*Covariance between the hidden states (Z) and the sim of exp(Z)
 */
{
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            rho_z_e_tilde[i * no + j] =
                (powf(vz[i * no + z_pos], 0.5) * me[i * no + j]) /
                powf(ve_tilde[i], 0.5);
        }
    }
}

void compute_cov_coeff_e_e_tilde_cpu(std::vector<float> &ve_tilde,
                                     std::vector<float> &vz, int no, int B,
                                     int z_pos, std::vector<float> &me,
                                     std::vector<float> &ve,
                                     std::vector<float> &rho_e_e_tilde)
/*Covariance between exp(Z) and the sum of exp(Z)*/
{
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            rho_e_e_tilde[i * no + j] =
                ((powf(vz[i * no + z_pos], 0.5) * me[i * no + j]) /
                 powf(ve_tilde[i], 0.5)) *
                ((powf(vz[i * no + z_pos], 0.5) * me[i * no + j]) /
                 powf(ve[i * no + j], 0.5));
        }
    }
}

void compute_log_sum_exp_cpu(std::vector<float> &me_tilde,
                             std::vector<float> &ve_tilde, int B,
                             std::vector<float> &me_check,
                             std::vector<float> &ve_check)
/*Mean and variance of log(sum(exp(Z)))*/
{
    float tmp;
    for (int i = 0; i < B; i++) {
        tmp = logf(1 + ve_tilde[i] / powf(me_tilde[i], 2));
        me_check[i] = logf(me_tilde[i]) - 0.5 * tmp;
        ve_check[i] = tmp;
    }
}

void compute_cov_z_e_check_cpu(std::vector<float> &rho_e_e_tilde,
                               std::vector<float> &me, std::vector<float> &ve,
                               std::vector<float> &me_tilde,
                               std::vector<float> &ve_tilde, int no, int B,
                               std::vector<float> &cov_z_e_check)
/*Covariance between hidden states (Z) and log(sum(exp(Z)))*/
{
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            cov_z_e_check[i * no + j] =
                logf(1 + rho_e_e_tilde[i * no + j] *
                             (powf(ve[i * no + j], 0.5) / me[i * no + j]) *
                             (powf(ve_tilde[i], 0.5) / me_tilde[i]));
        }
    }
}

void exp_log_softmax_cpu(std::vector<float> &mz, std::vector<float> &vz,
                         std::vector<float> &me_check,
                         std::vector<float> &ve_check,
                         std::vector<float> &cov_z_e_check, float sigma_v,
                         int no, int B, int z_pos, std::vector<float> &ma,
                         std::vector<float> &va)
/*Convert log of softmax to softmax space*/
{
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_mu = mz[z_pos + i * no + j] - me_check[i];
            tmp_var = vz[z_pos + i * no + j] + ve_check[i] -
                      2 * cov_z_e_check[i * no + j];
            ma[z_pos + i * no + j] = expf(tmp_mu + 0.5 * tmp_var);
            va[z_pos + i * no + j] =
                expf(2 * tmp_mu + tmp_var) * (expf(tmp_var) - 1);
        }
    }
}

void exp_log_softmax_cpu_v2(std::vector<float> &mz, std::vector<float> &vz,
                            std::vector<float> &me_check,
                            std::vector<float> &ve_check,
                            std::vector<float> &cov_z_e_check, float sigma_v,
                            int no, int B, int z_pos, std::vector<float> &ma,
                            std::vector<float> &va)
/*Convert log of softmax to softmax space*/
{
    float tmp_mu, tmp_var, max_m, max_v;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_mu = mz[z_pos + i * no + j] - me_check[i];
            tmp_var = vz[z_pos + i * no + j] + ve_check[i] -
                      2 * cov_z_e_check[i * no + j];
            ma[z_pos + i * no + j] = tmp_mu + 0.5 * tmp_var;
            va[z_pos + i * no + j] = tmp_var;
        }
        int idx = z_pos + i * no;
        auto max_idx =
            std::max_element(ma.begin() + idx, ma.begin() + idx + no) -
            ma.begin();
        max_m = ma[max_idx];
        max_v = va[max_idx];
        for (int j = 0; j < no; j++) {
            ma[z_pos + i * no + j] =
                expf(ma[z_pos + i * no + j] - max_m +
                     0.5 * (va[z_pos + i * no + j] + max_v));
            va[z_pos + i * no + j] = powf(ma[z_pos + i * no + j], 2) *
                                     (expf(va[z_pos + i * no + j] + max_v) - 1);
        }
    }
}

void compute_y_check_cpu(std::vector<float> &mz, std::vector<float> &vz,
                         std::vector<float> &me_check,
                         std::vector<float> &ve_check,
                         std::vector<float> &cov_z_e_check,
                         std::vector<float> &var_noise, int no, int B,
                         int z_pos, std::vector<float> &mu_y_check,
                         std::vector<float> &var_y_check)
/*Compute the \check{y} mean and variance
    \check{y} = Z - \check{E},
where \check{E} = log(sum(exp(z)))
*/
{
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_mu = mz[z_pos + i * no + j] - me_check[i];
            tmp_var = vz[z_pos + i * no + j] + ve_check[i] -
                      2 * cov_z_e_check[i * no + j] + var_noise[i * no + j];
            mu_y_check[i * no + j] = tmp_mu;
            var_y_check[i * no + j] = tmp_var;
        }
    }
}

void compute_cov_y_y_check_cpu(std::vector<float> &mz, std::vector<float> &vz,
                               std::vector<float> &me_check,
                               std::vector<float> &ve_check,
                               std::vector<float> &cov_z_e_check, int no, int B,
                               int z_pos, std::vector<float> &cov_y_y_check)
/*Covariance betwee y and \check{y}. The observation equation is defined
following
            y = exp(\check{y}) + V, v~N(0, \sigma_{2}^{2}),
where \hat{y} = exp(\check{y}).
*/
{
    float tmp_mu, tmp_var;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            tmp_mu = mz[z_pos + i * no + j] - me_check[i];
            tmp_var = vz[z_pos + i * no + j] + ve_check[i] -
                      2 * cov_z_e_check[i * no + j];
            cov_y_y_check[i * no + j] = expf(tmp_mu + 0.5 * tmp_var) * tmp_var;
        }
    }
}

void compute_cov_z_y_check_cpu(std::vector<float> &var_z,
                               std::vector<float> &cov_z_e_check, int no, int B,
                               int z_pos, std::vector<float> &cov_z_y_check)
/* Covariance between hidden state z and \check{y}. See function
   `compute_cov_y_y_check_cpu`*/
{
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < no; j++) {
            cov_z_y_check[i * no + j] =
                var_z[z_pos + i * no + j] - cov_z_e_check[i * no + j];
        }
    }
}

void compute_cov_z_y_cpu(std::vector<float> &mu_a,
                         std::vector<float> &cov_z_y_check, int no, int B,
                         int z_pos, std::vector<float> &cov_z_y) {
    for (int i = 0; i < no * B; i++) {
        cov_z_y[i] = mu_a[i + z_pos] * cov_z_y_check[i];
    }
}

void closed_form_softmax_cpu(Network &net, NetState &state, int l)
/*Closed-form softmax function*/
{
    int z_pos = net.z_pos[l];
    int no = net.nodes[l];
    int B = net.batch_size;

    max_norm(state.mz, state.Sz, z_pos, no, B, state.mz, state.Sz);

    // Transform to exponential space
    exp_fn_cpu(state.mz, state.Sz, no, B, z_pos, state.cf_softmax.mu_e,
               state.cf_softmax.var_e, state.cf_softmax.cov_z_e);

    // Compute sum of the exponential of all hidden states
    compute_sum_exp_cpu(state.cf_softmax.mu_e, state.cf_softmax.var_e, no, B,
                        state.cf_softmax.mu_e_tilde,
                        state.cf_softmax.var_e_tilde);

    // Compute covariance coefficient between epx(z) and sum(exp(z))
    compute_cov_coeff_e_e_tilde_cpu(state.cf_softmax.var_e_tilde, state.Sz, no,
                                    B, z_pos, state.cf_softmax.mu_e,
                                    state.cf_softmax.var_e,
                                    state.cf_softmax.rho_e_e_tilde);

    // Transform sum(exp(z)) in log space
    compute_log_sum_exp_cpu(
        state.cf_softmax.mu_e_tilde, state.cf_softmax.var_e_tilde, B,
        state.cf_softmax.mu_e_check, state.cf_softmax.var_e_check);

    // Covariance between z and log(sum(exp(z)))
    compute_cov_z_e_check_cpu(
        state.cf_softmax.rho_e_e_tilde, state.cf_softmax.mu_e,
        state.cf_softmax.var_e, state.cf_softmax.mu_e_tilde,
        state.cf_softmax.var_e_tilde, no, B, state.cf_softmax.cov_z_e_check);

    // Convert to softmax probability
    exp_log_softmax_cpu(state.mz, state.Sz, state.cf_softmax.mu_e_check,
                        state.cf_softmax.var_e_check,
                        state.cf_softmax.cov_z_e_check, 0.0f, no, B, z_pos,
                        state.ma, state.Sa);
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
                                 omega_tol, zpos, start_idx, end_idx,
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
            mixture_relu_cpu(state.mz, state.Sz, net.omega_tol, z_pos_out, 0,
                             no_B, state.ma, state.J, state.Sa);
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
    } else if (net.activations[j] == net.act_names.cf_softmax)  // cf softmax
    {
        closed_form_softmax_cpu(net, state, j);
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

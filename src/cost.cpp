///////////////////////////////////////////////////////////////////////////////
// File:         cost.cpp
// Description:  Cost functions for TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 01, 2022
// Updated:      April 02, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/cost.h"

void fliplr(std::vector<int> &v)
/* Flip an array from left to right
 *
 * Args:
 *    v: An array
 *
 * Returns:
 *    v: Flipped array
 *    */
{
    for (int i = 0; i < v.size() / 2; i++) {
        int tmp = v[i];
        v[i] = v[v.size() - i - 1];
        v[v.size() - i - 1] = tmp;
    }
}

std::vector<int> dec_to_bi(int base, int num, int n_c)
/*
 * Convert decimal to binary.
 *
 * Args:
 *    base: base number e.g. binary base = 2
 *    num: Number to be converted
 *    n_c: Number of columns to store base
 *
 * Returns:
 *    res: Base number
 *    */
{
    // Initialize pointers
    std::vector<int> res(n_c, 0);
    int index = 0;
    while (num > 0) {
        res[index++] = num % base;
        num /= base;
    }

    // Flip the left to right the base number
    fliplr(res);

    return res;
}

int bi_to_dec(std::vector<int> &v, int base)
/*
 * Convert binary to decimal.
 *
 * Args:
 *    v: vector of binary number
 *    base: base number
 *
 * Returns:
 *   num: Decimal number
 *   */
{
    int num = 0;
    int mlp = 1;
    for (int i = v.size() - 1; i >= 0; i--) {
        if (v[i] >= base) {
            printf("Invalid number");
            return -1;
        }

        num += v[i] * mlp;
        mlp = mlp * base;
    }

    return num;
}

HRCSoftmax class_to_obs(int n_classes)
/*
 * Convert class to hierachical softmax for classificaiton task.
 *
 * Args:
 *    n_classes: Number of classes.
 *
 * Returns:
 *    obs: Observation matrix for each class i.e. -1 and 1
 *    idx: Indices for each observation
 *    n_obs: Number of observation
 *    len: Length of the observation vector
 **/
{
    int base = 2, L;
    float tmp;

    // Compute the length of binary vector
    L = std::ceil(std::log2(n_classes));

    // Get observations including -1 and 1
    std::vector<int> C(L * n_classes);
    std::vector<float> obs(L * n_classes);
    for (int r = 0; r < n_classes; r++) {
        std::vector<int> tmp = dec_to_bi(base, r, L);
        for (int c = 0; c < L; c++) {
            C[r * L + c] = tmp[c];
            obs[r * L + c] = pow(-1.0f, tmp[c]);
        }
    }

    // Compute C_sum
    std::vector<int> idx(L * n_classes, 1);
    std::vector<int> C_sum(L + 1, 0);
    C_sum[L] = n_classes;
    for (int l = L - 1; l >= 0; l--) {
        tmp = std::ceil(C_sum[l + 1] / 2.0f);
        C_sum[l] = tmp;
    }

    // Compute cumulative sum for C_sum
    for (int l = 1; l < L + 1; l++) {
        C_sum[l] = C_sum[l - 1] + C_sum[l];
    }
    for (int l = 0; l < L + 1; l++) {
        C_sum[l] = C_sum[l] + 1;
    }

    // Get indices for observations
    for (int r = 0; r < n_classes; r++) {
        for (int c = 0; c < L - 1; c++) {
            std::vector<int> tmp(c + 1);
            for (int t = 0; t < c + 1; t++) {
                tmp[t] = C[r * L + t];
            }
            idx[r * L + c + 1] = bi_to_dec(tmp, base) + C_sum[c];
        }
    }

    // Compute the lenght of the observation vector
    int len = *std::max_element(idx.begin(), idx.end());

    return {obs, idx, L, len};
}

//////////////////////////////////////////////////////////////////////////////
// CONVERT OBSERVATION TO CLASS
/////////////////////////////////////////////////////////////////////////////
// float normalCDF(float x)
// /* Normal cumulative distribution */
// {
//     return std::erfc(-x / std::sqrt(2)) / 2;
// }

std::vector<float> obs_to_class(std::vector<float> &mz, std::vector<float> &Sz,
                                HRCSoftmax &hs, int n_classes)
/*
 * Convert observation to classes.
 *
 * Args:
 *    mz: Mean of hidden states of the ouput layer
 *    Sz: Variance of hidden states of the output layer
 *    hs: Hierarchical softmax output
 *    n_classes: Number of classes
 *
 * Returns:
 *    P: Probability of the class
 **/
{
    // Initialization
    std::vector<float> P(n_classes);
    std::vector<float> P_z(hs.len);
    std::vector<float> P_obs(hs.n_obs * n_classes);
    float alpha = 3;

    // Compute probability for each observation
    for (int i = 0; i < hs.len; i++) {
        P_z[i] = normcdf_cpu(mz[i] / pow(pow(1 / alpha, 2) + Sz[i], 0.5));
    }

    // Compute probability for the class
    for (int r = 0; r < n_classes; r++) {
        float tmp = 1.0f;
        for (int c = 0; c < hs.n_obs; c++) {
            if (hs.obs[r * hs.n_obs + c] == -1.0f) {
                tmp *= std::abs(P_z[hs.idx[r * hs.n_obs + c] - 1] - 1.0f);
            } else {
                tmp *= P_z[hs.idx[r * hs.n_obs + c] - 1];
            }
        }
        P[r] = tmp;
    }

    return P;
}

////////////////////////////////////////////////////////////////////////////
// ERROR RATE
////////////////////////////////////////////////////////////////////////////
std::tuple<std::vector<int>, std::vector<float>> get_error(
    std::vector<float> &mz, std::vector<float> &Sz, std::vector<int> &labels,
    int n_classes, int B)
/*
 * Compute error given an input image
 *
 * Args:
 *    mz: Mean of hidden states of the output layer
 *    Sz: Variance of hidden states of the output layer
 *    labels: Real label
 *    hs: Hierarchical softmax output
 *    n_classes: Number of classes
 *
 * Returns:
 *    er: error 1: wrong prediciton and 0: right one
 *    P: Probability for each class
 * */
{
    // Initialization
    auto hs = class_to_obs(n_classes);
    std::vector<int> er(B, 0);
    std::vector<float> P(B * n_classes);
    std::vector<float> mz_tmp(hs.len);
    std::vector<float> Sz_tmp(hs.len);

    // Compute probability for each class
    for (int r = 0; r < B; r++) {
        // Get sample
        for (int i = 0; i < hs.len; i++) {
            mz_tmp[i] = mz[r * hs.len + i];
            Sz_tmp[i] = Sz[r * hs.len + i];
        }

        // Compute probability
        std::vector<float> tmp(n_classes, 0);
        tmp = obs_to_class(mz_tmp, Sz_tmp, hs, n_classes);

        // Store in P matrix
        for (int c = 0; c < n_classes; c++) {
            P[r * n_classes + c] = tmp[c];
        }

        // Prediction
        int pred = std::distance(tmp.begin(),
                                 std::max_element(tmp.begin(), tmp.end()));

        // Get error
        if (pred != labels[r]) {
            er[r] = 1;
        }
    }

    return {er, P};
}

std::tuple<std::vector<int>, std::vector<float>, std::vector<int>> get_error_v2(
    std::vector<float> &mz, std::vector<float> &Sz, std::vector<int> &labels,
    int n_classes, int B)
/*
 * Compute error given an input image
 *
 * Args:
 *    mz: Mean of hidden states of the output layer
 *    Sz: Variance of hidden states of the output layer
 *    labels: Real label
 *    hs: Hierarchical softmax output
 *    n_classes: Number of classes
 *
 * Returns:
 *    er: error 1: wrong prediciton and 0: right one
 *    P: Probability for each class
 * */
{
    // Initialization
    auto hs = class_to_obs(n_classes);
    std::vector<int> er(B, 0);
    std::vector<int> preds(B);
    std::vector<float> P(B * n_classes);
    std::vector<float> mz_tmp(hs.len);
    std::vector<float> Sz_tmp(hs.len);

    // Compute probability for each class
    for (int r = 0; r < B; r++) {
        // Get sample
        for (int i = 0; i < hs.len; i++) {
            mz_tmp[i] = mz[r * hs.len + i];
            Sz_tmp[i] = Sz[r * hs.len + i];
        }

        // Compute probability
        auto tmp = obs_to_class(mz_tmp, Sz_tmp, hs, n_classes);

        // Store in P matrix
        for (int c = 0; c < n_classes; c++) {
            P[r * n_classes + c] = tmp[c];
        }

        // Prediction
        preds[r] = std::distance(tmp.begin(),
                                 std::max_element(tmp.begin(), tmp.end()));
        // Get error
        if (preds[r] != labels[r]) {
            er[r] = 1;
        }
    }

    return {er, P, preds};
}

std::vector<int> get_class_error(std::vector<float> &ma,
                                 std::vector<int> &labels, int n_classes,
                                 int B) {
    std::vector<int> er(B, 0);
    int idx;
    for (int i = 0; i < B; i++) {
        idx = i * n_classes;
        int pred =
            std::max_element(ma.begin() + idx, ma.begin() + idx + n_classes) -
            ma.begin() - idx;

        if (pred != labels[i]) {
            er[i] = 1;
        }
    }
    return er;
}

float mean_squared_error(std::vector<float> &pred, std::vector<float> &obs)
/* Compute mean squared error.
Args:
    pred: Prediction
    obs: Observation

Returns:
    mse: Mean squared error
*/
{
    if (pred.size() != obs.size()) {
        throw std::invalid_argument(
            "Prediciton and observation does not have the same lenght - "
            "cost.cpp");
    }
    float sum = 0;
    for (int i = 0; i < pred.size(); i++) {
        sum += pow((obs[i] - pred[i]), 2);
    }

    return sum / obs.size();
}

float avg_univar_log_lik(std::vector<float> &x, std::vector<float> &mu,
                         std::vector<float> &sigma)
/* Compute the average of univariate log-likelihood.

Args:
    x: Prediction
    mu: Observation's mean
    sigma: Observation's standard deviation

Returns:
    avg_log_lik: Averaged log-likelihood

*NOTE: We assume that pred ~ Normal(mu, sigma).
*/
{
    if (x.size() == 0 || mu.size() == 0 || sigma.size() == 0) {
        throw std::invalid_argument(
            "Invalid inputs for normal density - cost.cpp");
    }
    float sum = 0;
    float PI_C = 3.141592653f;

    for (int i = 0; i < x.size(); i++) {
        sum += -0.5 * log(2 * PI_C * pow(sigma[i], 2)) -
               0.5 * pow((x[i] - mu[i]) / sigma[i], 2);
    }

    return sum / x.size();
}

float compute_average_error_rate(std::vector<int> &error_rate, int curr_idx,
                                 int n_past_data)
/*Compute running error rate.

  Args:
    error_rate: Vector of error rate
    curr_idx: Index of the current error rate
    n_past_data: Number of past data from the current index
*/
{
    int end_idx = curr_idx - n_past_data;
    if (end_idx < 0) {
        end_idx = 0;
        n_past_data = curr_idx;
    }

    float tmp = 0;
    for (int i = 0; i < n_past_data; i++) {
        tmp += error_rate[end_idx + i];
    }

    float avg_error = tmp / n_past_data;

    return avg_error;
}

/////////////////////////////////
// TEST UNI
////////////////////////////////
void test_class_to_obs() {
    int n_classes = 10;
    HRCSoftmax hs = class_to_obs(n_classes);

    std::cout << "Observation = "
              << "\n";
    print_matrix(hs.obs, hs.n_obs, n_classes);
    std::cout << "Index = "
              << "\n";
    print_matrix(hs.idx, hs.n_obs, n_classes);
}

void test_obs_to_class() {
    // Get obs
    int n_classes = 10, B = 2;
    std::vector<int> labels = {2, 3};
    HRCSoftmax hs = class_to_obs(n_classes);

    // Get prob
    std::vector<float> mz = {1, 1, 0, -1, 0, 0, 0, 1,  0, 0, 0,
                             1, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0};

    std::vector<float> Sz(hs.len * 2, 0.02f);
    std::vector<int> er;
    std::vector<float> P;
    std::tie(er, P) = get_error(mz, Sz, labels, n_classes, B);

    std::cout << "Prob = "
              << "\n";
    print_matrix(P, n_classes, B);

    std::cout << "Error"
              << "\n";
    for (int j = 0; j < B; j++) {
        std::cout << er[j] << "\n";
    }
}

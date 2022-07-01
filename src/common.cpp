///////////////////////////////////////////////////////////////////////////////
// File:         common.cpp
// Description:  Common function used for computing indices for TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 15, 2022
// Updated:      July 01, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// Copyright (c) 2022 Luong-Ha Nguyen & James-A. Goulet. Some rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "../include/common.h"

std::string get_current_dir() {
    char buff[FILENAME_MAX];  // create string buffer to hold path
    GetCurrentDir(buff, FILENAME_MAX);
    std::string current_working_dir(buff);

    return current_working_dir;
}

int sum(std::vector<int> &v)
/*
 * Compute summation of a vector.
 *
 * Args:
 *    v: Vector
 *
 * Returns:
 *    s: Summation of the vector v
 **/
{
    int s = 0;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
    }

    return s;
}

std::vector<int> transpose_matrix(std::vector<int> &M, int w, int h)
/*
 * Transpose a matrix.
 *
 * Args:
 *    M: Matrix
 *
 * Returns:
 *    w: Number of columns
 *    h: Number of rows
 **/
{
    std::vector<int> tM(M.size());
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            tM[r + c * h] = M[c + r * w];
        }
    }
    return tM;
}

void create_directory(std::string &path) {
    /* Check if the directory exists if not create the folder
     */
    struct stat st = {0};
    const char *res_path_c = path.c_str();
#if defined(__linux__) || defined(__APPLE__)
    if (stat(res_path_c, &st) == -1) {
        mkdir(res_path_c, 0700);
    }
#endif

#ifdef _WIN32
    if (stat(res_path_c, &st) == -1) {
        _mkdir(res_path_c);
    }
#endif
}

void decay_obs_noise(float &sigma_v, float &decay_factor, float &sigma_v_min)
/* Decrease the value of observation noise after each epoch

Args:
    sigma_v: Observation noise
    decay_factor: Decreasing percentage (default value: 0.99)
    sigma_v_min: Minimum value of observation noise (default value: 0.3)

*/
{
    sigma_v = decay_factor * sigma_v;
    if (sigma_v < sigma_v_min) {
        sigma_v = sigma_v_min;
    }
}

//////////////////////////////////////////////////////////////////////
/// OUTPUT HIDDEN STATES
//////////////////////////////////////////////////////////////////////
void get_output_hidden_states_cpu(std::vector<float> &z, int z_pos,
                                  std::vector<float> &z_mu)
/*Get output's distrinution

Args:
    z: Mean of activation units of the entire network
    z_pos: Position of hidden state for the output layer
        in hidden-state vector of network
    z_mu: Hidden states for the output
*/
{
    for (int i = 0; i < z_mu.size(); i++) {
        z_mu[i] = z[z_pos + i];
    }
}

void get_output_hidden_states_ni_cpu(std::vector<float> &z, int ny, int z_pos,
                                     std::vector<float> &z_mu)
/* Get hidden states of the output layer for the noise-inference case

Args:
    z: Output hidden states of the entire network
    ny: Number of hidden states of the output layer including hidden states
        for noise observation
    z_pos: Position of hidden state for the output layer
        in hidden-state vector of network
    z_mu: Hidden states for the output
 */
{
    int n = z_mu.size();
    int h = ny / 2;
    int m;
    for (int i = 0; i < n; i++) {
        m = (i / h) * ny + i % h;
        z_mu[i] = z[z_pos + m];
    }
}

void get_noise_hidden_states_cpu(std::vector<float> &z, int ny, int z_pos,
                                 std::vector<float> &z_v2)
/* Get hidden states of the output layer
 */
{
    int n = z_v2.size();
    int h = ny / 2;
    int k;
    for (int i = 0; i < n; i++) {
        k = (i / h) * ny + i % h + h;
        z_v2[i] = z[z_pos + k];
    }
}

void compute_output_variance(std::vector<float> &Sa, std::vector<float> &V,
                             std::vector<float> &S)
/* Add observation noise (V) to the model noise (Sa)

Args:
    Sa: Model variance from last layer of the network
    V: Observation variance i.e., \sigma_V_2
    S: Output's total variance
 */
{
    for (int i = 0; i < Sa.size(); i++) {
        S[i] = Sa[i] + V[i];
    }
}

void compute_output_variance_with_idx(std::vector<float> &Sa,
                                      std::vector<float> &V,
                                      std::vector<int> ud_idx, int ny, int nye,
                                      std::vector<float> &S)
/* Add observation noise (V) to the model noise (Sa) for the given outputs

Args:
    Sa: Model variance from last layer of the network
    V: Observation variance i.e., \sigma_V_2
    up_idx: Indices for the hidden states to be updated
    ny: Total number of hidden states for the output layer
    nye: Totoal number of hidden states to be updated for the output layer
    S: Output's total variance

*/
// NOTE: We might only need to compute the output variance of the given indices
{
    int idx;
    for (int i = 0; i < ud_idx.size(); i++) {
        idx = ud_idx[i] + (i / nye) * ny - 1;
        S[idx] = Sa[idx] + V[idx];
    }
}

void output_hidden_states(NetState &state, Network &net,
                          std::vector<float> &ma_output,
                          std::vector<float> &Sa_output)
/* Get output's activation units

Args:
net: Network architecture
state: Hidden state of network
ma_output: Mean of output's activation units
Sa_output: Variance of output's activation units

*/
{
    if (net.noise_type.compare("heteros") == 0) {
        for (int i = 0; i < net.n_y * net.batch_size; i++) {
            ma_output[i] = state.noise_state.ma_mu[i];
        }
        compute_output_variance(state.noise_state.Sa_mu,
                                state.noise_state.ma_v2b_prior, Sa_output);
    } else {
        get_output_hidden_states_cpu(state.ma, net.z_pos.back(), ma_output);
        get_output_hidden_states_cpu(state.Sa, net.z_pos.back(), Sa_output);
        if (net.noise_type.compare("homosce") == 0) {
            compute_output_variance(state.noise_state.Sa_mu,
                                    state.noise_state.ma_v2b_prior, Sa_output);
        }
    }
}
void get_output_states(std::vector<float> &ma, std::vector<float> Sa,
                       std::vector<float> &ma_output,
                       std::vector<float> &Sa_output, int idx)
/*Get output's distrinution

Args:
    ma: Mean of activation units of the entire network
    ma: Variance of activation units of the entire network
    ma_output: mean of activation units of the output layer
    Sa_output: Variance of activation units of the output layer
    idx: Starting index of the output layer
*/
{
    for (int i = 0; i < ma_output.size(); i++) {
        ma_output[i] = ma[idx + i];
        Sa_output[i] = Sa[idx + i];
    }
}

//////////////////////////////////////////////////////////////////////
/// NOISE INFERENCE
//////////////////////////////////////////////////////////////////////
void set_homosce_noise_param(std::vector<float> &mu_v2b,
                             std::vector<float> &sigma_v2b,
                             std::vector<float> &ma_v2b_prior,
                             std::vector<float> &Sa_v2b_prior)
/* Set user-specified parameter values for the prior of homoscedastic noise

Args:
    mu_v2b: User-specified mean of the homoscedastic observatio noise's
        distribution for each observation
    sigma_v2b: Standard deviation of the homoscedastic observation noise's
        distribution for each observation
    ma_v2b_prior: Mean of the homoscedastic observation noise's distribution in
        batches
    Sa_v2b_prior: Variance of the homoscedastic observation noise's distribution
        in batches
*/
{
    int ny = mu_v2b.size();
    for (int i = 0; i < ma_v2b_prior.size(); i++) {
        ma_v2b_prior[i] = mu_v2b[(i % ny)];
        Sa_v2b_prior[i] = pow(sigma_v2b[(i % ny)], 2);
    }
}

void get_homosce_noise_param(std::vector<float> &ma_v2b_prior,
                             std::vector<float> &Sa_v2b_prior,
                             std::vector<float> &mu_v2b,
                             std::vector<float> &sigma_v2b)
/* Get the mean and standard deviation of the observation noise's
 distribution after training.
 */
{
    int ny = mu_v2b.size();
    for (int i = 0; i < ny; i++) {
        mu_v2b[i] = ma_v2b_prior[i * ny];
        sigma_v2b[i] = pow(Sa_v2b_prior[i * ny], 0.5);
    }
}

//////////////////////////////////////////////////////////////////////
/// FULL COVARIANCE
//////////////////////////////////////////////////////////////////////
std::vector<float> initialize_upper_triu(float &Sx, int n)
/* Initialize the covariance matrix where only the elements of the triangle
upper matrix are stored in a vector.

Args:
    Sx: Initial value of the diagonal term of the covariance matrix
    n: Size of the covariance matrix

Returns:
    Sx_tu: Vector of the triangle upper matrix
*/
{
    std::vector<float> Sx_tu;
    int tu;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            if (row <= col) {
                tu = n * row + col;
                if (row == col) {
                    Sx_tu.push_back(Sx);
                } else {
                    Sx_tu.push_back(0.0f);
                }
            }
        }
    }
    return Sx_tu;
}

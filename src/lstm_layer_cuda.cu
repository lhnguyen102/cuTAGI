///////////////////////////////////////////////////////////////////////////////
// File:         lstm_layer_cuda.cu
// Description:  Header file for Long-Short Term Memory (LSTM) forward pass
//               in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 22, 2024
// Updated:      March 31, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/activation_cuda.cuh"
#include "../include/lstm_layer.h"
#include "../include/lstm_layer_cuda.cuh"
#include "../include/param_init.h"

__global__ void lstm_linear_fwd_mean_var_cuda(
    float const *mu_w, float const *var_w, float const *mu_b,
    float const *var_b, const float *mu_a, const float *var_a,
    size_t input_size, size_t output_size, int batch_size, bool bias, int w_pos,
    int b_pos, float *mu_z, float *var_z)
/*
 */
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mu = 0.0f;
    float sum_var = 0.0f;

    if (col < batch_size && row < output_size) {
        for (int i = 0; i < input_size; i++) {
            float mu_a_tmp = mu_a[input_size * col + i];
            float var_a_tmp = var_a[input_size * col + i];
            float mu_w_tmp = mu_w[row * input_size + i + w_pos];
            float var_w_tmp = var_w[row * input_size + i + w_pos];

            sum_mu += mu_w_tmp * mu_a_tmp;
            sum_var += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                       var_w_tmp * mu_a_tmp * mu_a_tmp;
        }

        if (bias) {
            mu_z[col * output_size + row] = sum_mu + mu_b[row + b_pos];
            var_z[col * output_size + row] = sum_var + var_b[row + b_pos];
        } else {
            mu_z[col * output_size + row] = sum_mu;
            var_z[col * output_size + row] = sum_var;
        }
    }
}

__global__ void lstm_cov_input_cell_states_cuda(
    float const *Sha, float const *mw, float const *Ji_ga, float const *Jc_ga,
    int w_pos_i, int w_pos_c, int ni, int no, int seq_len, int B, float *Ci_c)
/*Compute covariance between input gates and cell states. Note that we store the
   hidden state vector as follows: z = [seq1, seq2, ..., seq n] where seq's
   shape = [1, no * B]

Args:
    Sha: Variance of the activations + previous hidden states of lstm layer
    mw: Mean of weights
    Ji_ga: Jacobian matrix (diagonal) of input gate
    Jc_ga: Jacobian matrix (diagonal) of cell state gate
    w_pos_i: Weight position for input gate in the weight vector of network
    w_pos_c: Weight position for cell state gate in the weight vector of network
    ni: Input node
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    Ci_c: Convariance between input and cell state gates
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    int k, i, m, x, y;
    if (col < no && row < B * seq_len) {
        sum = 0;
        x = row / seq_len;
        y = row % seq_len;
        for (int j = 0; j < ni + no; j++) {
            k = j + col * (ni + no);
            m = j + y * (ni + no) + x * (seq_len * (ni + no));
            sum += mw[w_pos_i + k] * Sha[m] * mw[w_pos_c + k];
        }
        i = col + y * no + x * seq_len * no;
        Ci_c[i] = Ji_ga[i] * sum * Jc_ga[i];
    }
}

__global__ void lstm_cell_state_mean_var_cuda(
    float const *mf_ga, float const *Sf_ga, float const *mi_ga,
    float const *Si_ga, float const *mc_ga, float const *Sc_ga,
    float const *mc_prev, float const *Sc_prev, float const *Ci_c, int no,
    int seq_len, int B, float *mc, float *Sc)
/*Compute cell states for the current state

Args:
    mf_ga: Mean of the forget gate
    Sf_ga: Variance of the forget gate
    mi_ga: Mean of the input gate
    Si_ga: Variance of the input gate
    mc_ga: Mean of the cell state gate
    Sc_ga: Variance of the cell state gate
    mc_prev: Mean of the cell state of the previous states
    Sc_prev: Variance of the cell state of the previous states
    Ci_c: Covariance of input and cell state gates
    no: Output node
    seq_len: Input sequence length
    B: Batch siz
    mc: Mean of the cell state
    Sc: Variance of the cell state
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k, x, y;
    if (col < no && row < B * seq_len) {
        x = row / seq_len;
        y = row % seq_len;
        k = col + y * no + x * no * seq_len;

        mc[k] = mf_ga[k] * mc_prev[k] + mi_ga[k] * mc_ga[k] + Ci_c[k];
        Sc[k] = Sc_prev[k] * mf_ga[k] * mf_ga[k] + Sc_prev[k] * Sf_ga[k] +
                Sf_ga[k] * mc_prev[k] * mc_prev[k] +
                Sc_ga[k] * mi_ga[k] * mi_ga[k] + Si_ga[k] * Sc_ga[k] +
                Si_ga[k] * mc_ga[k] * mc_ga[k] + Ci_c[k] * Ci_c[k] +
                2 * Ci_c[k] * mi_ga[k] * mc_ga[k];
    }
}

__global__ void lstm_cov_output_tanh_cell_states_cuda(
    float const *mw, float const *Sha, float const *mc_prev, float const *Jc_a,
    float const *Jf_ga, float const *mi_ga, float const *Ji_ga,
    float const *mc_ga, float const *Jc_ga, float const *Jo_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len, int B,
    float *Co_tanh_c)
/*Compute convariance between output gates & tanh(cell states)

Args:
    mw: Mean of weights
    Sha: Variance of the activations + previous hidden states of lstm layer
    mc_prev: Mean of cell state (i.e., hidden state) of the previous step
    Jca: Jacobian matrix (diagonal) of cell states
    Jf_ga: Jacobian matrix (diagonal) of forget gates
    mi_ga: Mean of the input gate
    Ji_ga: Jacobian matrix (diagonal) of input gates
    mc_ga: Mean of the cell state gate
    Jc_ga: Jacobian matrix (diagonal) of cell state gates
    Jo_ga: Jacobian matrix (diagonal) of output gates
    w_pos_f: Weight position for forget gate in the weight vector of network
    w_pos_i: Weight position for input gate in the weight vector of network
    w_pos_c: Weight position for cell state gate in the weight vector of network
    w_pos_o: Weight position for output gate in the weight vector of network
    ni: Input node
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    Co_tanh_c: Covariance between outputs and tanh of cell states
 */
// TODO: DOUBLE CHECK if prev_mc is hidden state or activation unit
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_fo, sum_io, sum_oc;
    int k, m, i, x, y;
    if (col < no && row < B * seq_len) {
        x = row / seq_len;
        y = row % seq_len;
        k = col + y * no + x * no * seq_len;
        sum_fo = 0;
        sum_io = 0;
        sum_oc = 0;
        for (int j = 0; j < ni; j++) {
            k = j + col * (ni + no);
            m = j + y * (ni + no) + x * (seq_len * (ni + no));
            sum_fo += mw[w_pos_f + k] * Sha[m] * mw[w_pos_o + k];
            sum_io += mw[w_pos_i + k] * Sha[m] * mw[w_pos_o + k];
            sum_oc += mw[w_pos_c + k] * Sha[m] * mw[w_pos_o + k];
        }
        i = col + y * no + x * seq_len * no;
        Co_tanh_c[i] = Jc_a[i] * (Jo_ga[i] * sum_fo * Jf_ga[i] * mc_prev[i] +
                                  Jo_ga[i] * sum_io * Ji_ga[i] * mc_ga[i] +
                                  Jo_ga[i] * sum_oc * Jc_ga[i] * mi_ga[i]);
    }
}

__global__ void lstm_hidden_state_mean_var_cuda(
    float const *mo_ga, float const *So_ga, float const *mc_a,
    float const *Sc_a, float const *Co_tanh_c, int no, int seq_len, int B,
    float *mz, float *Sz)
/*Compute mean and variance for hidden states of the LSTM layer

Args:
    mo_ga: Mean of the output gate
    So_ga: Variance of the output gate
    mca: Mean of the activated cell states
    Sca: Variance of the activated cell states
    Co_tanh_c: Covariance between outputs and tanh of cell states
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    mz: Mean of hidden states
    Sz: Variance of hidden states
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int k, j, x, y;
    if (col < no && row < B * seq_len) {
        x = row / seq_len;
        y = row % seq_len;
        j = col + y * no + x * no * seq_len;
        k = col + y * no + x * no * seq_len;

        mz[k] = mo_ga[j] * mc_a[j] + Co_tanh_c[j];
        Sz[k] = Sc_a[j] * mo_ga[j] * mo_ga[j] + Sc_a[j] * So_ga[j] +
                So_ga[j] * mc_a[j] * mc_a[j] + Co_tanh_c[j] * Co_tanh_c[j] +
                2 * Co_tanh_c[j] * mo_ga[j] * mc_a[j];
    }
}

__global__ void lstm_cat_act_and_prev_states_cuda(float const *a,
                                                  float const *b, int n, int m,
                                                  int seq_len, int B, float *c)
/*Concatenate two vectors*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < B && col < seq_len) {
        for (int i = 0; i < n; i++) {
            c[i + col * (n + m) + row * (n + m) * seq_len] =
                a[i + col * n + row * seq_len * n];
        }

        for (int j = 0; j < m; j++) {
            c[j + n + col * (n + m) + row * (n + m) * seq_len] =
                b[j + col * m + row * m * seq_len];
        }
    }
}

__global__ void to_prev_states(float const *curr, int n, float *prev)
/*Transfer data from current cell & hidden to previous cell & hidden states
   which are used for the next step*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        prev[col] = curr[col];
    }
}

////////////////////////////////////////////////////////////////////////////////
// BACKWARD PASS
////////////////////////////////////////////////////////////////////////////////
__global__ void lstm_delta_mean_var_z(
    float const *mw, float const *Jf_ga, float const *mi_ga, float const *Ji_ga,
    float const *mc_ga, float const *Jc_ga, float const *mo_ga,
    float const *Jo_ga, float const *mc_prev, float const *mca,
    float const *Jca, float const *delta_m_out, float const *delta_S_out,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int no, int ni,
    int seq_len, int B, float *delta_m, float *delta_S)
/*Compute the updated quatitites of the mean of the hidden states for lstm
   layer

Args:
    mw: Mean of weights
    Jf_ga: Jacobian matrix (diagonal) of forget gates
    mi_ga: Mean of the input gate
    Ji_ga: Jacobian matrix (diagonal) of input gates
    mc_ga: Mean of the cell state gate
    Jc_ga: Jacobian matrix (diagonal) of cell state gates
    mo_ga: Mean of the output gate
    Jo_ga: Jacobian matrix (diagonal) of output gates
    mc_prev: Mean of cell state (i.e., hidden state) of the previous step
    mca: Mean of the activated cell states
    Sca: Variance of the activated cell states
    delta_m_out: Inovation vector for mean i.e. (M_observation - M_prediction)
    delta_S_out: Inovation vector for variance i.e. (M_observation -
        M_prediction) w_pos_f: Weight position for forget gate in the weight
        vector ofnetwork
    w_pos_i: Weight position for input gate in the weight vector of network
    w_pos_c: Weight position for cell state gate in the weight vector of network
    w_pos_o: Weight position for output gate in the weight vector of network
    ni: Input node
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    delta_m: Updated quantities for the mean of output's hidden states
    delta_S: Updated quantities for the varaince of output's hidden states

NOTE: All LSTM states excepted mc_prev are from the next layer e.g., mi_ga(l+1)
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_mf, sum_mi, sum_mc, sum_mo, sum_Sz;
    float Czz_f, Czz_i, Czz_c, Czz_o;
    int k, m, i, x, y;
    if (row < B * seq_len && col < ni) {
        x = row / seq_len;
        y = row % seq_len;

        sum_mf = 0.0f;
        sum_mi = 0.0f;
        sum_mc = 0.0f;
        sum_mo = 0.0f;
        sum_Sz = 0.0f;
        for (int j = 0; j < no; j++) {
            k = j + y * no + x * no * seq_len;
            i = j + y * no + x * no * seq_len;

            // Forget gate
            Czz_f = Jca[k] * mo_ga[k] * Jf_ga[k] *
                    mw[(ni + no) * j + col + w_pos_f] * mc_prev[k];
            sum_mf += Czz_f * delta_m_out[i];

            // Input gate
            Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                    mw[(ni + no) * j + col + w_pos_i] * mc_ga[k];
            sum_mi += Czz_i * delta_m_out[i];

            // Cell state gate
            Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                    mw[(ni + no) * j + col + w_pos_c] * mi_ga[k];
            sum_mc += Czz_c * delta_m_out[i];

            // Output gate
            Czz_o = Jo_ga[k] * mw[(ni + no) * j + col + w_pos_o] * mca[k];
            sum_mo += Czz_o * delta_m_out[i];
            float tmp_sum_cov = Czz_f + Czz_i + Czz_c + Czz_o;
            sum_Sz += tmp_sum_cov * tmp_sum_cov * delta_S_out[i];
        }

        // Updating quantities
        m = x * ni * seq_len + y * ni + col;
        delta_m[m] = (sum_mf + sum_mi + sum_mc + sum_mo);
        delta_S[m] = sum_Sz;
    }
}

__global__ void lstm_update_prev_hidden_states(
    const float *mu_h_prior, const float *var_h_prior, const float *delta_mu,
    const float *delta_var, int num_states, float *mu_h_prev, float *var_h_prev)
/*
 */
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_states) {
        mu_h_prev[i] = mu_h_prior[i] + delta_mu[i] * var_h_prior[i];
        var_h_prev[i] = (1.0f + delta_mu[i] * var_h_prior[i]) * var_h_prior[i];
    }
}

__global__ void lstm_update_prev_cell_states(
    const float *mu_c_prior, const float *var_c_prior, const float *jcb_ca,
    const float *mu_o_ga, const float *delta_mu, const float *delta_var,
    int num_states, float *mu_c_prev, float *var_c_prev)
/*
 */
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_states) {
        float tmp = var_c_prior[i] * jcb_ca[i] * mu_o_ga[i];
        mu_c_prev[i] = mu_c_prior[i] + tmp * delta_mu[i];
        var_c_prev[i] = var_c_prior[i] + tmp * delta_var[i] * tmp;
    }
}

__global__ void lstm_delta_mean_var_w(
    float const *Sw, float const *mha, float const *Jf_ga, float const *mi_ga,
    float const *Ji_ga, float const *mc_ga, float const *Jc_ga,
    float const *mo_ga, float const *Jo_ga, float const *mc_prev,
    float const *mca, float const *Jc, float const *delta_m_out,
    float const *delta_S_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, float *delta_mw,
    float *delta_Sw)
/*Compute updating quantities of the weight parameters for lstm layer

Args:
    Sw: Variance of weights
    mha: Mean of the activations + previous hidden states of lstm layer
    Jf_ga: Jacobian matrix (diagonal) of forget gates
    mi_ga: Mean of the input gate
    Ji_ga: Jacobian matrix (diagonal) of input gates
    mc_ga: Mean of the cell state gate
    Jc_ga: Jacobian matrix (diagonal) of cell state gates
    mo_ga: Mean of the output gate
    Jo_ga: Jacobian matrix (diagonal) of output gates
    mc_prev: Mean of cell state (i.e., hidden state) of the previous step
    mca: Mean of the activated cell states
    Jca: Jacobian matrix (diagonal) of cell states
    delta_m_out: Inovation vector for mean i.e. (M_observation - M_prediction)
    delta_S_out: Inovation vector for variance i.e. (M_observation -
        M_prediction)
    w_pos_f: Weight position for forget gate in the weight vector of network
    w_pos_i: Weight position for input gate in the weight vector of network
    w_pos_c: Weight position for cell state gate in the weight vector of network
    w_pos_o: Weight position for output gate in the weight vector of network
    ni: Input node
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    delta_mw: Updated quantities for the mean of weights
    deltaSw: Updated quantities for the variance of weights

NOTE: All LSTM states are from the next layer e.g., mi_ga(l+1)

*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, m, l, i, x, y;
    if (row < (ni + no) && col < no) {
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int t = 0; t < B * seq_len; t++) {
            x = t / seq_len;
            y = t % seq_len;

            k = col + y * no + no * seq_len * x;
            i = col + y * no + no * seq_len * x;
            l = row + y * (ni + no) + (ni + no) * seq_len * x;

            // Forget gate
            Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k] * mha[l];
            sum_mf += Cwa_f * delta_m_out[i];
            sum_Sf += Cwa_f * delta_S_out[i] * Cwa_f;

            // Input gate
            Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k] * mha[l];
            sum_mi += Cwa_i * delta_m_out[i];
            sum_Si += Cwa_i * delta_S_out[i] * Cwa_i;

            // Cell state gate
            Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k] * mha[l];
            sum_mc += Cwa_c * delta_m_out[i];
            sum_Sc += Cwa_c * delta_S_out[i] * Cwa_c;

            // Output gate
            Cwa_o = Jo_ga[k] * mca[k] * mha[l];
            sum_mo += Cwa_o * delta_m_out[i];
            sum_So += Cwa_o * delta_S_out[i] * Cwa_o;
        }
        // Updating quantities for weights
        m = col * (ni + no) + row;
        delta_mw[m + w_pos_f] = sum_mf * Sw[m + w_pos_f];
        delta_Sw[m + w_pos_f] = Sw[m + w_pos_f] * sum_Sf * Sw[m + w_pos_f];

        delta_mw[m + w_pos_i] = sum_mi * Sw[m + w_pos_i];
        delta_Sw[m + w_pos_i] = Sw[m + w_pos_i] * sum_Si * Sw[m + w_pos_i];

        delta_mw[m + w_pos_c] = sum_mc * Sw[m + w_pos_c];
        delta_Sw[m + w_pos_c] = Sw[m + w_pos_c] * sum_Sc * Sw[m + w_pos_c];

        delta_mw[m + w_pos_o] = sum_mo * Sw[m + w_pos_o];
        delta_Sw[m + w_pos_o] = Sw[m + w_pos_o] * sum_So * Sw[m + w_pos_o];
    }
}

__global__ void lstm_delta_mean_var_b(
    float const *Sb, float const *Jf_ga, float const *mi_ga, float const *Ji_ga,
    float const *mc_ga, float const *Jc_ga, float const *mo_ga,
    float const *Jo_ga, float const *mc_prev, float const *mca, float const *Jc,
    float const *delta_m_out, float const *delta_S_out, int b_pos_f,
    int b_pos_i, int b_pos_c, int b_pos_o, int no, int seq_len, int B,
    float *delta_mb, float *delta_Sb)
/*Compute updating quantities of the bias for the lstm layer

Args:
    Sb: Variance of biases
    Jf_ga: Jacobian matrix (diagonal) of forget gates
    mi_ga: Mean of the input gate
    Ji_ga: Jacobian matrix (diagonal) of input gates
    mc_ga: Mean of the cell state gate
    Jc_ga: Jacobian matrix (diagonal) of cell state gates
    mo_ga: Mean of the output gate
    Jo_ga: Jacobian matrix (diagonal) of output gates
    mc_prev: Mean of cell state (i.e., hidden state) of the previous step
    mca: Mean of the activated cell states
    Jca: Jacobian matrix (diagonal) of cell states
    delta_m_out: Inovation vector for mean i.e. (M_observation - M_prediction)
    delta_S_out: Inovation vector for variance i.e. (M_observation -
        M_prediction)
    b_pos_f: Bias position for forget gate in the bias vector of network
    b_pos_i: Bias position for input gate in the weight vector of network
    b_pos_c: Bias position for cell state gate in the bias vector of network
    b_pos_o: Bias position for output gate in the bias vector of network
    ni: Input node
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    deltaMb: Updated quantities for the mean of biases
    deltaSb: Updated quantities for the variance of biases

NOTE: All LSTM states are from the next layer e.g., mi_ga(l+1)
*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, i, x, y;
    if (col < no) {
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int t = 0; t < B * seq_len; t++) {
            x = t / seq_len;
            y = t % seq_len;

            k = col + y * no + no * seq_len * x;
            i = col + y * no + no * seq_len * x;

            // Forget gate
            Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k];
            sum_mf += Cwa_f * delta_m_out[i];
            sum_Sf += Cwa_f * delta_S_out[i] * Cwa_f;

            // Input gate
            Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k];
            sum_mi += Cwa_i * delta_m_out[i];
            sum_Si += Cwa_i * delta_S_out[i] * Cwa_i;

            // Cell state gate
            Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k];
            sum_mc += Cwa_c * delta_m_out[i];
            sum_Sc += Cwa_c * delta_S_out[i] * Cwa_c;

            // Output gate
            Cwa_o = Jo_ga[k] * mca[k];
            sum_mo += Cwa_o * delta_m_out[i];
            sum_So += Cwa_o * delta_S_out[i] * Cwa_o;
        }
        // Updating quantities for biases
        delta_mb[col + b_pos_f] = sum_mf * Sb[col + b_pos_f];
        delta_Sb[col + b_pos_f] =
            Sb[col + b_pos_f] * sum_Sf * Sb[col + b_pos_f];

        delta_mb[col + b_pos_i] = sum_mi * Sb[col + b_pos_i];
        delta_Sb[col + b_pos_i] =
            Sb[col + b_pos_i] * sum_Si * Sb[col + b_pos_i];

        delta_mb[col + b_pos_c] = sum_mc * Sb[col + b_pos_c];
        delta_Sb[col + b_pos_c] =
            Sb[col + b_pos_c] * sum_Sc * Sb[col + b_pos_c];

        delta_mb[col + b_pos_o] = sum_mo * Sb[col + b_pos_o];
        delta_Sb[col + b_pos_o] =
            Sb[col + b_pos_o] * sum_So * Sb[col + b_pos_o];
    }
}
////////////////////////////////////////////////////////////////////////////////
// LSTM
////////////////////////////////////////////////////////////////////////////////
LSTMCuda::LSTMCuda(size_t input_size, size_t output_size, int seq_len,
                   bool bias, float gain_w, float gain_b,
                   std::string init_method)
    : seq_len(seq_len),
      gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method)
/**/
{
    this->input_size = input_size;
    this->output_size = output_size;
    this->bias = bias;

    this->get_number_param();
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
}

LSTMCuda::~LSTMCuda()
/*
 */
{}

std::string LSTMCuda::get_layer_info() const
/*
 */
{
    return "LSTM(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LSTMCuda::get_layer_name() const
/*
 */
{
    return "LSTMCuda";
}

LayerType LSTMCuda::get_layer_type() const
/*
 */
{
    return LayerType::LSTM;
}

int LSTMCuda::get_input_size()
/*
 */
{
    return this->input_size * this->seq_len;
}

int LSTMCuda::get_output_size()
/*
 */
{
    return this->output_size * this->seq_len;
}

void LSTMCuda::get_number_param()
/*
 */
{
    // We stack the weights of 4 gates in the same vector
    this->num_weights =
        4 * this->output_size * (this->input_size + this->output_size);
    this->num_biases = 0;
    if (this->bias) {
        this->num_biases = 4 * this->output_size;
        this->b_pos_f = 0;
        this->b_pos_i = this->output_size;
        this->b_pos_c = 2 * this->output_size;
        this->b_pos_o = 3 * this->output_size;
    }

    this->w_pos_f = 0;
    this->w_pos_i = this->output_size * (this->input_size + this->output_size);
    this->w_pos_c =
        2 * this->output_size * (this->input_size + this->output_size);
    this->w_pos_o =
        3 * this->output_size * (this->input_size + this->output_size);
}

void LSTMCuda::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_lstm(this->init_method, this->gain_w, this->gain_b,
                              this->input_size, this->output_size,
                              this->num_weights, this->num_biases);

    this->allocate_param_memory();
    this->params_to_device();
}

void LSTMCuda::prepare_input(BaseHiddenStates &input_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    int batch_size = cu_input_states->block_size;

    unsigned int grid_row =
        (batch_size + this->num_cuda_threads - 1) / (this->num_cuda_threads);
    unsigned int grid_col =
        (this->seq_len + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(this->num_cuda_threads, this->num_cuda_threads);

    lstm_cat_act_and_prev_states_cuda<<<dim_grid, dim_block>>>(
        cu_input_states->d_mu_a, this->lstm_state.d_mu_h_prev, this->input_size,
        this->output_size, this->seq_len, batch_size, this->lstm_state.d_mu_ha);
    lstm_cat_act_and_prev_states_cuda<<<dim_grid, dim_block>>>(
        cu_input_states->d_var_a, this->lstm_state.d_var_h_prev,
        this->input_size, this->output_size, this->seq_len, batch_size,
        this->lstm_state.d_var_ha);
}

void LSTMCuda::forget_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int num_act = b_seq * this->output_size;
    unsigned int grid_col =
        (b_seq + this->num_cuda_threads - 1) / this->num_cuda_threads;
    unsigned int grid_row = (this->output_size + this->num_cuda_threads - 1) /
                            this->num_cuda_threads;
    unsigned int act_block =
        (num_act + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(this->num_cuda_threads, this->num_cuda_threads);

    lstm_linear_fwd_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        this->lstm_state.d_mu_ha, this->lstm_state.d_var_ha, ni_c,
        this->output_size, b_seq, this->bias, this->w_pos_f, this->b_pos_f,
        this->lstm_state.d_mu_f_ga, this->lstm_state.d_var_f_ga);

    sigmoid_mean_var_cuda<<<act_block, this->num_cuda_threads>>>(
        this->lstm_state.d_mu_f_ga, this->lstm_state.d_var_f_ga, num_act,
        this->lstm_state.d_mu_f_ga, this->lstm_state.d_jcb_f_ga,
        this->lstm_state.d_var_f_ga);
}

void LSTMCuda::input_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int num_act = b_seq * this->output_size;
    unsigned int grid_col =
        (b_seq + this->num_cuda_threads - 1) / this->num_cuda_threads;
    unsigned int grid_row = (this->output_size + this->num_cuda_threads - 1) /
                            this->num_cuda_threads;
    unsigned int act_block =
        (num_act + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(this->num_cuda_threads, this->num_cuda_threads);

    lstm_linear_fwd_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        this->lstm_state.d_mu_ha, this->lstm_state.d_var_ha, ni_c,
        this->output_size, b_seq, this->bias, this->w_pos_i, this->b_pos_i,
        this->lstm_state.d_mu_i_ga, this->lstm_state.d_var_i_ga);

    sigmoid_mean_var_cuda<<<act_block, this->num_cuda_threads>>>(
        this->lstm_state.d_mu_i_ga, this->lstm_state.d_var_i_ga, num_act,
        this->lstm_state.d_mu_i_ga, this->lstm_state.d_jcb_i_ga,
        this->lstm_state.d_var_i_ga);
}

void LSTMCuda::cell_state_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int num_act = b_seq * this->output_size;
    unsigned int grid_col =
        (b_seq + this->num_cuda_threads - 1) / this->num_cuda_threads;
    unsigned int grid_row = (this->output_size + this->num_cuda_threads - 1) /
                            this->num_cuda_threads;
    unsigned int act_block =
        (num_act + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(this->num_cuda_threads, this->num_cuda_threads);

    lstm_linear_fwd_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        this->lstm_state.d_mu_ha, this->lstm_state.d_var_ha, ni_c,
        this->output_size, b_seq, this->bias, this->w_pos_c, this->b_pos_c,
        this->lstm_state.d_mu_c_ga, this->lstm_state.d_var_c_ga);

    tanh_mean_var_cuda<<<act_block, this->num_cuda_threads>>>(
        this->lstm_state.d_mu_c_ga, this->lstm_state.d_var_c_ga, num_act,
        this->lstm_state.d_mu_c_ga, this->lstm_state.d_jcb_c_ga,
        this->lstm_state.d_var_c_ga);
}

void LSTMCuda::output_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int num_act = b_seq * this->output_size;
    unsigned int grid_col =
        (b_seq + this->num_cuda_threads - 1) / this->num_cuda_threads;
    unsigned int grid_row = (this->output_size + this->num_cuda_threads - 1) /
                            this->num_cuda_threads;
    unsigned int act_block =
        (num_act + this->num_cuda_threads - 1) / this->num_cuda_threads;

    dim3 dim_grid(grid_col, grid_row);
    dim3 dim_block(this->num_cuda_threads, this->num_cuda_threads);

    lstm_linear_fwd_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
        this->lstm_state.d_mu_ha, this->lstm_state.d_var_ha, ni_c,
        this->output_size, b_seq, this->bias, this->w_pos_o, this->b_pos_o,
        this->lstm_state.d_mu_o_ga, this->lstm_state.d_var_o_ga);

    sigmoid_mean_var_cuda<<<act_block, this->num_cuda_threads>>>(
        this->lstm_state.d_mu_o_ga, this->lstm_state.d_var_o_ga, num_act,
        this->lstm_state.d_mu_o_ga, this->lstm_state.d_jcb_o_ga,
        this->lstm_state.d_var_o_ga);
}

void LSTMCuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_state.set_num_states(
            batch_size * this->seq_len * this->output_size,
            batch_size * this->seq_len * this->input_size);
    }
    // Update number of actual states.
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size * this->seq_len;

    if (this->seq_len == 1 && batch_size == 1) {
        cudaMemcpy(this->lstm_state.d_mu_h_prev, this->lstm_state.d_mu_h_prior,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->lstm_state.d_var_h_prev,
                   this->lstm_state.d_var_h_prior,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->lstm_state.d_mu_c_prev, this->lstm_state.d_mu_c_prior,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->lstm_state.d_var_c_prev,
                   this->lstm_state.d_var_c_prior,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    this->prepare_input(input_states);
    this->forget_gate(batch_size);
    this->input_gate(batch_size);
    this->cell_state_gate(batch_size);
    this->output_gate(batch_size);

    int no_b_seq = batch_size * this->seq_len * this->output_size;
    unsigned int act_blocks =
        (no_b_seq + this->num_cuda_threads - 1) / this->num_cuda_threads;
    unsigned int gridRow_cov =
        (batch_size * this->seq_len + this->num_cuda_threads - 1) /
        this->num_cuda_threads;
    unsigned int gridCol_cov =
        (this->output_size + this->num_cuda_threads - 1) /
        this->num_cuda_threads;
    dim3 dim_grid(gridCol_cov, gridRow_cov);
    dim3 dim_block(this->num_cuda_threads, this->num_cuda_threads);

    // Cov(input gate, cell state gate)
    lstm_cov_input_cell_states_cuda<<<dim_grid, dim_block>>>(
        this->lstm_state.d_var_ha, this->d_mu_w, this->lstm_state.d_jcb_i_ga,
        this->lstm_state.d_jcb_c_ga, this->w_pos_i, this->w_pos_c,
        this->input_size, this->output_size, this->seq_len, batch_size,
        this->lstm_state.d_cov_i_c);

    // Mean and variance for the current cell states
    lstm_cell_state_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->lstm_state.d_mu_f_ga, this->lstm_state.d_var_f_ga,
        this->lstm_state.d_mu_i_ga, this->lstm_state.d_var_i_ga,
        this->lstm_state.d_mu_c_ga, this->lstm_state.d_var_c_ga,
        this->lstm_state.d_mu_c_prev, this->lstm_state.d_var_c_prev,
        this->lstm_state.d_cov_i_c, this->output_size, this->seq_len,
        batch_size, this->lstm_state.d_mu_c, this->lstm_state.d_var_c);

    tanh_mean_var_cuda<<<act_blocks, this->num_cuda_threads>>>(
        this->lstm_state.d_mu_c, this->lstm_state.d_var_c, no_b_seq,
        this->lstm_state.d_mu_ca, this->lstm_state.d_jcb_ca,
        this->lstm_state.d_var_ca);

    // Cov(output gate, tanh(cell states))
    lstm_cov_output_tanh_cell_states_cuda<<<dim_grid, dim_block>>>(
        this->d_mu_w, this->lstm_state.d_var_ha, this->lstm_state.d_mu_c_prev,
        this->lstm_state.d_jcb_ca, this->lstm_state.d_jcb_f_ga,
        this->lstm_state.d_mu_i_ga, this->lstm_state.d_jcb_i_ga,
        this->lstm_state.d_mu_c_ga, this->lstm_state.d_jcb_c_ga,
        this->lstm_state.d_jcb_o_ga, this->w_pos_f, this->w_pos_i,
        this->w_pos_c, this->w_pos_o, this->input_size, this->output_size,
        this->seq_len, batch_size, this->lstm_state.d_cov_o_tanh_c);

    // Mean and variance for hidden states
    lstm_hidden_state_mean_var_cuda<<<dim_grid, dim_block>>>(
        this->lstm_state.d_mu_o_ga, this->lstm_state.d_var_o_ga,
        this->lstm_state.d_mu_ca, this->lstm_state.d_var_ca,
        this->lstm_state.d_cov_o_tanh_c, this->output_size, this->seq_len,
        batch_size, cu_output_states->d_mu_a, cu_output_states->d_var_a);

    // Update backward state for inferring parameters
    if (this->training) {
        this->store_states_for_training_cuda(*cu_input_states,
                                             *cu_output_states);
    }

    // Saved the previous hidden states
    if (this->seq_len == 1 && batch_size == 1) {
        cudaMemcpy(this->lstm_state.d_mu_h_prior, cu_output_states->d_mu_a,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->lstm_state.d_var_h_prior, cu_output_states->d_var_a,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->lstm_state.d_mu_c_prior, this->lstm_state.d_mu_c,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->lstm_state.d_var_c_prior, this->lstm_state.d_var_c,
                   this->lstm_state.num_states * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
}

void LSTMCuda::backward(BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_delta_states,
                        BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    // New poitner will point to the same memory location when casting
    DeltaStateCuda *cu_input_delta_states =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_output_delta_states =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);
    BackwardStateCuda *cu_next_bwd_states =
        dynamic_cast<BackwardStateCuda *>(this->bwd_states.get());

    // Initialization
    int batch_size = input_delta_states.block_size;
    int threads = this->num_cuda_threads;

    unsigned int gridRow_cov =
        (batch_size * this->seq_len + this->num_cuda_threads - 1) /
        this->num_cuda_threads;
    unsigned int gridCol_cov = (this->input_size + this->num_cuda_threads - 1) /
                               this->num_cuda_threads;
    dim3 dim_grid(gridCol_cov, gridRow_cov);
    dim3 dim_block(threads, threads);

    if (state_udapte) {
        lstm_delta_mean_var_z<<<dim_grid, dim_block>>>(
            this->d_mu_w, this->lstm_state.d_jcb_f_ga,
            this->lstm_state.d_mu_i_ga, this->lstm_state.d_jcb_i_ga,
            this->lstm_state.d_mu_c_ga, this->lstm_state.d_jcb_c_ga,
            this->lstm_state.d_mu_o_ga, this->lstm_state.d_jcb_o_ga,
            this->lstm_state.d_mu_c_prev, this->lstm_state.d_mu_ca,
            this->lstm_state.d_jcb_ca, cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->w_pos_f, this->w_pos_i,
            this->w_pos_c, this->w_pos_o, this->output_size, this->input_size,
            this->seq_len, batch_size, cu_output_delta_states->d_delta_mu,
            cu_output_delta_states->d_delta_var);
    }

    if (param_update) {
        // Launch kernel
        unsigned int b_blocks = (this->output_size + threads - 1) / threads;
        unsigned int grid_row_p =
            (this->input_size + this->output_size + threads - 1) / threads;
        unsigned int grid_col_p = (this->output_size + threads - 1) / threads;
        dim3 dim_grid_p(grid_col_p, grid_row_p);

        lstm_delta_mean_var_w<<<dim_grid_p, dim_block>>>(
            this->d_var_w, this->lstm_state.d_mu_ha,
            this->lstm_state.d_jcb_f_ga, this->lstm_state.d_mu_i_ga,
            this->lstm_state.d_jcb_i_ga, this->lstm_state.d_mu_c_ga,
            this->lstm_state.d_jcb_c_ga, this->lstm_state.d_mu_o_ga,
            this->lstm_state.d_jcb_o_ga, this->lstm_state.d_mu_c_prev,
            this->lstm_state.d_mu_ca, this->lstm_state.d_jcb_ca,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->w_pos_f, this->w_pos_i,
            this->w_pos_c, this->w_pos_o, this->output_size, this->input_size,
            this->seq_len, batch_size, this->d_delta_mu_w, this->d_delta_var_w);

        if (this->bias) {
            lstm_delta_mean_var_b<<<b_blocks, threads>>>(
                this->d_var_b, this->lstm_state.d_jcb_f_ga,
                this->lstm_state.d_mu_i_ga, this->lstm_state.d_jcb_i_ga,
                this->lstm_state.d_mu_c_ga, this->lstm_state.d_jcb_c_ga,
                this->lstm_state.d_mu_o_ga, this->lstm_state.d_jcb_o_ga,
                this->lstm_state.d_mu_c_prev, this->lstm_state.d_mu_ca,
                this->lstm_state.d_jcb_ca, cu_input_delta_states->d_delta_mu,
                cu_input_delta_states->d_delta_var, this->b_pos_f,
                this->b_pos_i, this->b_pos_c, this->b_pos_o, this->output_size,
                this->seq_len, batch_size, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }

    if (this->seq_len == 1 && batch_size == 1) {
        const unsigned int ps_grid_size =
            (this->lstm_state.num_states + this->num_cuda_threads - 1) /
            this->num_cuda_threads;

        lstm_update_prev_hidden_states<<<ps_grid_size,
                                         this->num_cuda_threads>>>(
            this->lstm_state.d_mu_h_prior, this->lstm_state.d_var_h_prior,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->lstm_state.num_states,
            this->lstm_state.d_mu_h_prior, this->lstm_state.d_var_h_prior);
        lstm_update_prev_cell_states<<<ps_grid_size, this->num_cuda_threads>>>(
            this->lstm_state.d_mu_c_prior, this->lstm_state.d_var_c_prior,
            this->lstm_state.d_jcb_ca, this->lstm_state.d_mu_o_ga,
            cu_input_delta_states->d_delta_mu,
            cu_input_delta_states->d_delta_var, this->lstm_state.num_states,
            this->lstm_state.d_mu_c_prior, this->lstm_state.d_var_c_prior);
    }
}

std::unique_ptr<BaseLayer> LSTMCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_linear = std::make_unique<LSTM>(
        this->input_size, this->output_size, this->seq_len, this->bias,
        this->gain_w, this->gain_b, this->init_method);

    host_linear->mu_w = this->mu_w;
    host_linear->var_w = this->var_w;
    host_linear->mu_b = this->mu_b;
    host_linear->var_b = this->var_b;

    return host_linear;
}

void LSTMCuda::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
    }
    if (this->training) {
        this->allocate_param_delta();
    }
}

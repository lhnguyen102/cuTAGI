///////////////////////////////////////////////////////////////////////////////
// File:         lstm_layer_cuda.cu
// Description:  Header file for Long-Short Term Memory (LSTM) forward pass
//               in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 22, 2024
// Updated:      March 25, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/lstm_layer_cuda.cuh"
#include "../include/param_init.h"

__global__ void cov_input_cell_states(float const *Sha, float const *mw,
                                      float const *Ji_ga, float const *Jc_ga,
                                      int w_pos_i, int w_pos_c, int ni, int no,
                                      int seq_len, int B, float *Ci_c)
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

__global__ void cell_state_mean_var(float const *mf_ga, float const *Sf_ga,
                                    float const *mi_ga, float const *Si_ga,
                                    float const *mc_ga, float const *Sc_ga,
                                    float const *mc_prev, float const *Sc_prev,
                                    float const *Ci_c, int no, int seq_len,
                                    int B, float *mc, float *Sc)
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
                Si_ga[k] * mc_ga[k] * mc_ga[k] + powf(Ci_c[k], 2) +
                2 * Ci_c[k] * mi_ga[k] * mc_ga[k];
    }
}

__global__ void cov_output_tanh_cell_states(
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

__global__ void hidden_state_mean_var_lstm(float const *mo_ga,
                                           float const *So_ga,
                                           float const *mc_a, float const *Sc_a,
                                           float const *Co_tanh_c, int no,
                                           int seq_len, int B, float *mz,
                                           float *Sz)
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
                So_ga[j] * mc_a[j] * mc_a[j] + powf(Co_tanh_c[j], 2) +
                2 * Co_tanh_c[j] * mo_ga[j] * mc_a[j];
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
            sum_mf += Czz_f * delta_m[i];

            // Input gate
            Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                    mw[(ni + no) * j + col + w_pos_i] * mc_ga[k];
            sum_mi += Czz_i * delta_m[i];

            // Cell state gate
            Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                    mw[(ni + no) * j + col + w_pos_c] * mi_ga[k];
            sum_mc += Czz_c * delta_m[i];

            // Output gate
            Czz_o = Jo_ga[k] * mw[(ni + no) * j + col + w_pos_o] * mca[k];
            sum_mo += Czz_o * delta_m_out[i];
            sum_Sz += powf(Czz_f + Czz_i + Czz_c + Czz_o, 2) * delta_S_out[i];
        }

        // Updating quantities
        m = x * ni * seq_len + y * ni + col;
        delta_m[m] = (sum_mf + sum_mi + sum_mc + sum_mo);
        delta_S[m] = sum_Sz;
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
        this->bwd_states = std::make_unique<BaseBackwardStates>();
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
        this->b_pos_i = 2 * this->output_size;
        this->b_pos_i = 3 * this->output_size;
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
}

void LSTMCuda::prepare_input(BaseHiddenStates &input_state)
/*
 */
{}

void LSTMCuda::forget_gate(int batch_size)
/*
 */
{}

void LSTMCuda::input_gate(int batch_size)
/*
 */
{}

void LSTMCuda::cell_state_gate(int batch_size)
/*
 */
{}

void LSTMCuda::output_gate(int batch_size)
/*
 */
{}

///////////////////////////////////////////////////////////////////////////////
// File:         lstm_feed_forward.cu
// Description:  Long-Short Term Memory (LSTM) forward pass in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      September 05, 2022
// Updated:      December 14, 2022
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/lstm_feed_forward.cuh"

__global__ void cov_input_cell_states(float const *Sha, float const *mw,
                                      float const *Ji_ga, float const *Jc_ga,
                                      int z_pos_o, int w_pos_i, int w_pos_c,
                                      int ni, int no, int seq_len, int B,
                                      float *Ci_c)
/*Compute covariance between input gates and cell states. Note that we store the
   hidden state vector as follows: z = [seq1, seq2, ..., seq n] where seq's
   shape = [1, no * B]

Args:
    Sha: Variance of the activations + previous hidden states of lstm layer
    mw: Mean of weights
    Ji_ga: Jacobian matrix (diagonal) of input gate
    Jc_ga: Jacobian matrix (diagonal) of cell state gate
    z_pos_o: Output-hidden-state position for this layer in the hidden-state
        vector of network
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
        Ci_c[i] = Ji_ga[i + z_pos_o] * sum * Jc_ga[i + z_pos_o];
    }
}

__global__ void cell_state_mean_var(float const *mf_ga, float const *Sf_ga,
                                    float const *mi_ga, float const *Si_ga,
                                    float const *mc_ga, float const *Sc_ga,
                                    float const *mc_prev, float const *Sc_prev,
                                    float const *Ci_c, int z_pos_o, int no,
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
    z_pos_o: Output-hidden-state position for this layer in the hidden-state
        vector of network
    no: Output node
    seq_len: Input sequence length
    B: Batch siz
    mc: Mean of the cell state
    Sc: Variance of the cell state
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int m, k, x, y;
    if (col < no && row < B * seq_len) {
        x = row / seq_len;
        y = row % seq_len;
        k = col + y * no + x * no * seq_len;

        m = k + z_pos_o;
        mc[m] = mf_ga[m] * mc_prev[m] + mi_ga[m] * mc_ga[m] + Ci_c[k];
        Sc[m] = Sc_prev[m] * mf_ga[m] * mf_ga[m] + Sc_prev[m] * Sf_ga[m] +
                Sf_ga[m] * mc_prev[m] * mc_prev[m] +
                Sc_ga[m] * mi_ga[m] * mi_ga[m] + Si_ga[m] * Sc_ga[m] +
                Si_ga[m] * mc_ga[m] * mc_ga[m] + powf(Ci_c[k], 2) +
                2 * Ci_c[k] * mi_ga[m] * mc_ga[m];
    }
}

__global__ void cov_output_tanh_cell_states(
    float const *mw, float const *Sha, float const *mc_prev, float const *Jc_a,
    float const *Jf_ga, float const *mi_ga, float const *Ji_ga,
    float const *mc_ga, float const *Jc_ga, float const *Jo_ga,
    int z_pos_o_lstm, int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o,
    int ni, int no, int seq_len, int B, float *Co_tanh_c)
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
    z_pos_o_lstm: Output-hidden-state position for LSTM layer in the
        LSTM hidden-state vector of network
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
        i = col + y * no + x * seq_len * no + z_pos_o_lstm;
        Co_tanh_c[i - z_pos_o_lstm] =
            Jc_a[i] * (Jo_ga[i] * sum_fo * Jf_ga[i] * mc_prev[i] +
                       Jo_ga[i] * sum_io * Ji_ga[i] * mc_ga[i] +
                       Jo_ga[i] * sum_oc * Jc_ga[i] * mi_ga[i]);
    }
}

__global__ void hidden_state_mean_var_lstm(
    float const *mo_ga, float const *So_ga, float const *mc_a,
    float const *Sc_a, float const *Co_tanh_c, int z_pos_o, int z_pos_o_lstm,
    int no, int seq_len, int B, float *mz, float *Sz)
/*Compute mean and variance for hidden states of the LSTM layer

Args:
    mo_ga: Mean of the output gate
    So_ga: Variance of the output gate
    mca: Mean of the activated cell states
    Sca: Variance of the activated cell states
    Co_tanh_c: Covariance between outputs and tanh of cell states
    z_pos_o: Output-hidden-state position for this layer in the hidden-state
        vector of network
    z_pos_o_lstm: Output-hidden-state position for LSTM layer in the
        LSTM hidden-state vector of network
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    mz: Mean of hidden states
    Sz: Variance of hidden states
*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int m, k, j, x, y;
    if (col < no && row < B * seq_len) {
        x = row / seq_len;
        y = row % seq_len;
        j = col + y * no + x * no * seq_len;
        m = j + z_pos_o_lstm;
        k = col + y * no + x * no * seq_len + z_pos_o;

        mz[k] = mo_ga[m] * mc_a[m] + Co_tanh_c[j];
        Sz[k] = Sc_a[m] * mo_ga[m] * mo_ga[m] + Sc_a[m] * So_ga[m] +
                So_ga[m] * mc_a[m] * mc_a[m] + powf(Co_tanh_c[j], 2) +
                2 * Co_tanh_c[j] * mo_ga[m] * mc_a[m];
    }
}

__global__ void to_prev_states(float const *curr, int n, int z_pos,
                               int z_pos_lstm, float *prev)
/*Transfer data from current cell & hidden to previous cell & hidden states
   which are used for the next step*/
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        prev[col + z_pos_lstm] = curr[col + z_pos];
    }
}

void save_prev_states(Network &net, StateGPU &state)
/*Save the hidden and cell states of the previous time step*/
{
    int THREADS = net.num_gpu_threads;
    for (int j = 1; j < net.layers.size(); j++) {
        if (net.layers[j] == net.layer_names.lstm) {
            int no_b_seq = net.nodes[j] * net.batch_size * net.input_seq_len;
            int BLOCK_PH = (no_b_seq + THREADS - 1) / THREADS;
            int z_pos_o_lstm = net.z_pos_lstm[j];
            int z_pos_o = net.z_pos[j];
            to_prev_states<<<BLOCK_PH, THREADS>>>(state.lstm.d_mc, no_b_seq,
                                                  z_pos_o_lstm, z_pos_o_lstm,
                                                  state.lstm.d_mc_prev);
            to_prev_states<<<BLOCK_PH, THREADS>>>(state.lstm.d_Sc, no_b_seq,
                                                  z_pos_o_lstm, z_pos_o_lstm,
                                                  state.lstm.d_Sc_prev);

            to_prev_states<<<BLOCK_PH, THREADS>>>(state.d_mz, no_b_seq, z_pos_o,
                                                  z_pos_o_lstm,
                                                  state.lstm.d_mh_prev);
            to_prev_states<<<BLOCK_PH, THREADS>>>(state.d_Sz, no_b_seq, z_pos_o,
                                                  z_pos_o_lstm,
                                                  state.lstm.d_Sh_prev);
        }
    }
}

__global__ void cat_activations_and_prev_states(float const *a, float const *b,
                                                int n, int m, int seq_len,
                                                int B, int z_pos_a, int z_pos_b,
                                                float *c)
/*Concatenate two vectors*/
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row < B && col < seq_len) {
        for (int i = 0; i < n; i++) {
            c[i + col * (n + m) + row * (n + m) * seq_len] =
                a[i + z_pos_a + col * n + row * seq_len * n];
        }

        for (int j = 0; j < m; j++) {
            c[j + n + col * (n + m) + row * (n + m) * seq_len] =
                b[j + z_pos_b + col * m + row * m * seq_len];
        }
    }
}

void forget_gate(Network &net, StateGPU &state, ParamGPU &theta, int l) {
    int z_pos_i_lstm = 0;
    int ni_c = net.nodes[l - 1] + net.nodes[l];
    int b_seq = net.batch_size * net.input_seq_len;
    int no_b_seq = net.nodes[l] * b_seq;

    int THREADS = net.num_gpu_threads;
    unsigned int ACT_BLOCKS = (no_b_seq + THREADS - 1) / THREADS;
    unsigned int gridRow = (net.nodes[l] + THREADS - 1) / THREADS;
    unsigned int gridCol = (b_seq + THREADS - 1) / THREADS;

    dim3 dimGrid(gridCol, gridRow);
    dim3 dimBlock(THREADS, THREADS);

    fcMean<<<dimGrid, dimBlock>>>(theta.d_mw, theta.d_mb, state.lstm.d_mha,
                                  state.lstm.d_mf_ga, net.w_pos[l - 1],
                                  net.b_pos[l - 1], z_pos_i_lstm,
                                  net.z_pos_lstm[l], net.nodes[l], ni_c, b_seq);
    fcVar<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_Sw, theta.d_Sb, state.lstm.d_mha, state.lstm.d_Sha,
        state.lstm.d_Sf_ga, net.w_pos[l - 1], net.b_pos[l - 1], z_pos_i_lstm,
        net.z_pos_lstm[l], net.nodes[l], ni_c, b_seq);

    mixture_sigmoid<<<ACT_BLOCKS, THREADS>>>(
        state.lstm.d_mf_ga, state.lstm.d_Sf_ga, net.omega_tol,
        net.z_pos_lstm[l], no_b_seq, state.lstm.d_mf_ga, state.lstm.d_Jf_ga,
        state.lstm.d_Sf_ga);
}

void input_gate(Network &net, StateGPU &state, ParamGPU &theta, int l) {
    int z_pos_i_lstm = 0;
    int ni_c = net.nodes[l - 1] + net.nodes[l];
    int b_seq = net.batch_size * net.input_seq_len;
    int no_b_seq = net.nodes[l] * b_seq;

    int THREADS = net.num_gpu_threads;
    unsigned int ACT_BLOCKS = (no_b_seq + THREADS - 1) / THREADS;
    unsigned int gridRow = (net.nodes[l] + THREADS - 1) / THREADS;
    unsigned int gridCol = (b_seq + THREADS - 1) / THREADS;

    dim3 dimGrid(gridCol, gridRow);
    dim3 dimBlock(THREADS, THREADS);

    int w_pos_i = net.w_pos[l - 1] + ni_c * net.nodes[l];
    int b_pos_i = net.b_pos[l - 1] + net.nodes[l];

    fcMean<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_mb, state.lstm.d_mha, state.lstm.d_mi_ga, w_pos_i,
        b_pos_i, z_pos_i_lstm, net.z_pos_lstm[l], net.nodes[l], ni_c, b_seq);
    fcVar<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_Sw, theta.d_Sb, state.lstm.d_mha, state.lstm.d_Sha,
        state.lstm.d_Si_ga, w_pos_i, b_pos_i, z_pos_i_lstm, net.z_pos_lstm[l],
        net.nodes[l], ni_c, b_seq);
    mixture_sigmoid<<<ACT_BLOCKS, THREADS>>>(
        state.lstm.d_mi_ga, state.lstm.d_Si_ga, net.omega_tol,
        net.z_pos_lstm[l], no_b_seq, state.lstm.d_mi_ga, state.lstm.d_Ji_ga,
        state.lstm.d_Si_ga);
}

void cell_state_gate(Network &net, StateGPU &state, ParamGPU &theta, int l) {
    int z_pos_i_lstm = 0;
    int ni_c = net.nodes[l - 1] + net.nodes[l];
    int b_seq = net.batch_size * net.input_seq_len;
    int no_b_seq = net.nodes[l] * b_seq;

    int THREADS = net.num_gpu_threads;
    unsigned int ACT_BLOCKS = (no_b_seq + THREADS - 1) / THREADS;
    unsigned int gridRow = (net.nodes[l] + THREADS - 1) / THREADS;
    unsigned int gridCol = (b_seq + THREADS - 1) / THREADS;

    dim3 dimGrid(gridCol, gridRow);
    dim3 dimBlock(THREADS, THREADS);

    int w_pos_c = net.w_pos[l - 1] + 2 * ni_c * net.nodes[l];
    int b_pos_c = net.b_pos[l - 1] + 2 * net.nodes[l];

    fcMean<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_mb, state.lstm.d_mha, state.lstm.d_mc_ga, w_pos_c,
        b_pos_c, z_pos_i_lstm, net.z_pos_lstm[l], net.nodes[l], ni_c, b_seq);
    fcVar<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_Sw, theta.d_Sb, state.lstm.d_mha, state.lstm.d_Sha,
        state.lstm.d_Sc_ga, w_pos_c, b_pos_c, z_pos_i_lstm, net.z_pos_lstm[l],
        net.nodes[l], ni_c, b_seq);
    mixture_tanh<<<ACT_BLOCKS, THREADS>>>(
        state.lstm.d_mc_ga, state.lstm.d_Sc_ga, net.omega_tol,
        net.z_pos_lstm[l], no_b_seq, state.lstm.d_mc_ga, state.lstm.d_Jc_ga,
        state.lstm.d_Sc_ga);
}

void output_gate(Network &net, StateGPU &state, ParamGPU &theta, int l) {
    int z_pos_i_lstm = 0;
    int ni_c = net.nodes[l - 1] + net.nodes[l];
    int b_seq = net.batch_size * net.input_seq_len;
    int no_b_seq = net.nodes[l] * b_seq;

    int THREADS = net.num_gpu_threads;
    unsigned int ACT_BLOCKS = (no_b_seq + THREADS - 1) / THREADS;
    unsigned int gridRow = (net.nodes[l] + THREADS - 1) / THREADS;
    unsigned int gridCol = (b_seq + THREADS - 1) / THREADS;

    dim3 dimGrid(gridCol, gridRow);
    dim3 dimBlock(THREADS, THREADS);

    int w_pos_o = net.w_pos[l - 1] + 3 * ni_c * net.nodes[l];
    int b_pos_o = net.b_pos[l - 1] + 3 * net.nodes[l];
    fcMean<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_mb, state.lstm.d_mha, state.lstm.d_mo_ga, w_pos_o,
        b_pos_o, z_pos_i_lstm, net.z_pos_lstm[l], net.nodes[l], ni_c, b_seq);
    fcVar<<<dimGrid, dimBlock>>>(
        theta.d_mw, theta.d_Sw, theta.d_Sb, state.lstm.d_mha, state.lstm.d_Sha,
        state.lstm.d_So_ga, w_pos_o, b_pos_o, z_pos_i_lstm, net.z_pos_lstm[l],
        net.nodes[l], ni_c, b_seq);
    mixture_sigmoid<<<ACT_BLOCKS, THREADS>>>(
        state.lstm.d_mo_ga, state.lstm.d_So_ga, net.omega_tol,
        net.z_pos_lstm[l], no_b_seq, state.lstm.d_mo_ga, state.lstm.d_Jo_ga,
        state.lstm.d_So_ga);
}

void lstm_state_forward(Network &net, StateGPU &state, ParamGPU &theta, int l)
/*Steps for computing hidden states mean and covariance for the lstm layer

Args:
    net: Network architecture and properties
    state: Network states
    theta: Network parameters
    l: Index for current layer

NOTE: Weight & bias vector for lstm is defined following
            w = [w_f, w_i, w_c, w_o] & b = [b_f, b_i, b_c, b_o]
*/
// TODO: Fix cycle import between feed_forward_cpu and lstm_feed_forward_cpu
{
    // Initialization
    int ni = net.nodes[l - 1];
    int no = net.nodes[l];
    int z_pos_i = net.z_pos[l - 1];
    int z_pos_o = net.z_pos[l];
    int w_pos_f, w_pos_i, w_pos_c, w_pos_o;
    int z_pos_o_lstm = net.z_pos_lstm[l];
    int no_b_seq = no * net.batch_size * net.input_seq_len;
    int ni_c = ni + no;
    int b_seq = net.batch_size * net.input_seq_len;

    // Launch kernel
    int THREADS = net.num_gpu_threads;
    unsigned int ACT_BLOCKS = (no_b_seq + THREADS - 1) / THREADS;
    unsigned int gridRow_cov = (b_seq + THREADS - 1) / THREADS;
    unsigned int gridCol_cov = (no + THREADS - 1) / THREADS;
    dim3 dimGrid_cov(gridCol_cov, gridRow_cov);
    dim3 dimBlock(THREADS, THREADS);

    // Concatenate the hidden states from the previous time step and
    // activations from the previous layer
    unsigned int gridRow_cat = (net.batch_size + THREADS - 1) / THREADS;
    unsigned int gridCol_cat = (net.input_seq_len + THREADS - 1) / THREADS;
    dim3 dimGrid_cat(gridCol_cat, gridRow_cat);
    cat_activations_and_prev_states<<<dimGrid_cat, dimBlock>>>(
        state.d_ma, state.lstm.d_mh_prev, ni, no, net.input_seq_len,
        net.batch_size, z_pos_i, z_pos_o_lstm, state.lstm.d_mha);
    cat_activations_and_prev_states<<<dimGrid_cat, dimBlock>>>(
        state.d_Sa, state.lstm.d_Sh_prev, ni, no, net.input_seq_len,
        net.batch_size, z_pos_i, z_pos_o_lstm, state.lstm.d_Sha);

    // Forget gate
    w_pos_f = net.w_pos[l - 1];
    forget_gate(net, state, theta, l);

    // Input gate
    w_pos_i = net.w_pos[l - 1] + ni_c * no;
    input_gate(net, state, theta, l);

    // Cell state gate
    w_pos_c = net.w_pos[l - 1] + 2 * ni_c * no;
    cell_state_gate(net, state, theta, l);

    // Output gate
    w_pos_o = net.w_pos[l - 1] + 3 * ni_c * no;
    output_gate(net, state, theta, l);

    // Cov(input gate, cell state gate)
    cov_input_cell_states<<<dimGrid_cov, dimBlock>>>(
        state.lstm.d_Sha, theta.d_mw, state.lstm.d_Ji_ga, state.lstm.d_Jc_ga,
        z_pos_o_lstm, w_pos_i, w_pos_c, ni, no, net.input_seq_len,
        net.batch_size, state.lstm.d_Ci_c);

    // Mean and variance for the current cell states
    cell_state_mean_var<<<dimGrid_cov, dimBlock>>>(
        state.lstm.d_mf_ga, state.lstm.d_Sf_ga, state.lstm.d_mi_ga,
        state.lstm.d_Si_ga, state.lstm.d_mc_ga, state.lstm.d_Sc_ga,
        state.lstm.d_mc_prev, state.lstm.d_Sc_prev, state.lstm.d_Ci_c,
        z_pos_o_lstm, no, net.input_seq_len, net.batch_size, state.lstm.d_mc,
        state.lstm.d_Sc);
    mixture_tanh<<<ACT_BLOCKS, THREADS>>>(
        state.lstm.d_mc, state.lstm.d_Sc, net.omega_tol, z_pos_o_lstm, no_b_seq,
        state.lstm.d_mca, state.lstm.d_Jca, state.lstm.d_Sca);

    // Cov(output gate, tanh(cell states))
    cov_output_tanh_cell_states<<<dimGrid_cov, dimBlock>>>(
        theta.d_mw, state.lstm.d_Sha, state.lstm.d_mc_prev, state.lstm.d_Jca,
        state.lstm.d_Jf_ga, state.lstm.d_mi_ga, state.lstm.d_Ji_ga,
        state.lstm.d_mc_ga, state.lstm.d_Jc_ga, state.lstm.d_Jo_ga,
        z_pos_o_lstm, w_pos_f, w_pos_i, w_pos_c, w_pos_o, ni, no,
        net.input_seq_len, net.batch_size, state.lstm.d_Co_tanh_c);

    // Mean and variance for hidden states
    hidden_state_mean_var_lstm<<<dimGrid_cov, dimBlock>>>(
        state.lstm.d_mo_ga, state.lstm.d_So_ga, state.lstm.d_mca,
        state.lstm.d_Sca, state.lstm.d_Co_tanh_c, z_pos_o, z_pos_o_lstm, no,
        net.input_seq_len, net.batch_size, state.d_mz, state.d_Sz);
}
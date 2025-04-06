
#include "../include/lstm_layer.h"

#include <cmath>
#include <thread>
#include <tuple>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/param_init.h"

#ifdef USE_CUDA
#include "../include/lstm_layer_cuda.cuh"
#endif

// TODO: merge this two following functions with linear layer
void lstm_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                       std::vector<float> &mu_b, std::vector<float> &var_b,
                       std::vector<float> &mu_a, std::vector<float> &var_a,
                       int start_chunk, int end_chunk, size_t input_size,
                       size_t output_size, int batch_size, bool bias, int w_pos,
                       int b_pos, std::vector<float> &mu_z,
                       std::vector<float> &var_z)
/*Compute mean of product WA for full connected layer

Args:
  mu_w: Mean of weights
  mu_b: Mean of the biases
  mu_a: Mean of activation units
  mu_z: Mean of hidden states
  start_chunk: Start index of the chunk
  end_chunk: End index of the chunk
  n: Input node
  m: Output node
  k: Number of batches
*/
{
    int n = input_size;
    for (int i = start_chunk; i < end_chunk; i++) {
        int row = i / batch_size;
        int col = i % batch_size;
        float sum_mu_z = 0.0f;
        float sum_var_z = 0.0f;
        for (int j = 0; j < input_size; j++) {
            float mu_a_tmp = mu_a[n * col + j];
            float var_a_tmp = var_a[n * col + j];
            float mu_w_tmp = mu_w[row * n + j + w_pos];
            float var_w_tmp = var_w[row * n + j + w_pos];

            sum_mu_z += mu_w_tmp * mu_a_tmp;
            sum_var_z += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                         var_w_tmp * mu_a_tmp * mu_a_tmp;
        }
        if (bias) {
            mu_z[col * output_size + row] = sum_mu_z + mu_b[row + b_pos];
            var_z[col * output_size + row] = sum_var_z + var_b[row + b_pos];
        } else {
            mu_z[col * output_size + row] = sum_mu_z;
            var_z[col * output_size + row] = sum_var_z;
        }
    }
}

void lstm_fwd_mean_var_mp(std::vector<float> &mu_w, std::vector<float> &var_w,
                          std::vector<float> &mu_b, std::vector<float> &var_b,
                          std::vector<float> &mu_a, std::vector<float> &var_a,
                          size_t input_size, size_t output_size, int batch_size,
                          bool bias, int w_pos, int b_pos,
                          unsigned int num_threads, std::vector<float> &mu_z,
                          std::vector<float> &var_z)
/*Multi-processing verion of forward pass for fc layer
 */
{
    const int tot_ops = output_size * batch_size;

    int start_chunk, end_chunk;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    int n_per_thread = tot_ops / num_threads;
    int extra = tot_ops % num_threads;

    for (int i = 0; i < num_threads; i++) {
        int start_chunk = i * n_per_thread + std::min(i, extra);
        int end_chunk = start_chunk + n_per_thread + (i < extra ? 1 : 0);

        threads.emplace_back([=, &mu_w, &var_w, &mu_b, &var_b, &mu_a, &var_a,
                              &input_size, &output_size, &batch_size, &bias,
                              &mu_z, &var_z] {
            lstm_fwd_mean_var(mu_w, var_w, mu_b, var_b, mu_a, var_a,
                              start_chunk, end_chunk, input_size, output_size,
                              batch_size, bias, w_pos, b_pos, mu_z, var_z);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

void lstm_cov_input_cell_states(std::vector<float> &var_ha,
                                std::vector<float> &mu_w,
                                std::vector<float> &jcb_i_ga,
                                std::vector<float> &jcb_c_ga, int w_pos_i,
                                int w_pos_c, int ni, int no, int seq_len, int B,
                                std::vector<float> &cov_i_c)
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
    float sum;
    int k, i, m;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                sum = 0;
                for (int j = 0; j < ni + no; j++) {
                    k = j + z * (ni + no);
                    m = j + y * (ni + no) + x * (seq_len * (ni + no));
                    sum += mu_w[w_pos_i + k] * var_ha[m] * mu_w[w_pos_c + k];
                }
                i = z + y * no + x * seq_len * no;
                cov_i_c[i] = jcb_i_ga[i] * sum * jcb_c_ga[i];
            }
        }
    }
}

void lstm_cell_state_mean_var(
    std::vector<float> &mu_f_ga, std::vector<float> &var_f_ga,
    std::vector<float> &mu_i_ga, std::vector<float> &var_i_ga,
    std::vector<float> &mu_c_ga, std::vector<float> &var_c_ga,
    std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev,
    std::vector<float> &cov_i_c, int no, int seq_len, int B,
    std::vector<float> &mu_c, std::vector<float> &var_c)
/*Compute mean and variance cell states

Args:
    mu_f_ga: Mean of the forget gate
    var_f_ga: Variance of the forget gate
    mu_i_ga: Mean of the input gate
    var_i_ga: Variance of the input gate
    mu_c_ga: Mean of the cell state gate
    var_c_ga: Variance of the cell state gate
    mu_c_prev: Mean of the cell state of the previous states
    var_c_prev: Variance of the cell state of the previous states
    cov_i_c: Covariance of input and cell state gates
    no: Output node
    seq_len: Input sequence length
    B: Batch siz
    mu_c: Mean of the cell state
    var_c: Variance of the cell state

*/
{
    int k;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                k = z + y * no + x * no * seq_len;
                mu_c[k] = mu_f_ga[k] * mu_c_prev[k] + mu_i_ga[k] * mu_c_ga[k] +
                          cov_i_c[k];
                var_c[k] = var_c_prev[k] * mu_f_ga[k] * mu_f_ga[k] +
                           var_c_prev[k] * var_f_ga[k] +
                           var_f_ga[k] * mu_c_prev[k] * mu_c_prev[k] +
                           var_c_ga[k] * mu_i_ga[k] * mu_i_ga[k] +
                           var_i_ga[k] * var_c_ga[k] +
                           var_i_ga[k] * mu_c_ga[k] * mu_c_ga[k] +
                           powf(cov_i_c[k], 2) +
                           2 * cov_i_c[k] * mu_i_ga[k] * mu_c_ga[k];
            }
        }
    }
}

void lstm_cov_output_tanh_cell_states(
    std::vector<float> &mu_w, std::vector<float> &var_ha,
    std::vector<float> &mu_c_prev, std::vector<float> &jcb_ca,
    std::vector<float> &jcb_f_ga, std::vector<float> &mu_i_ga,
    std::vector<float> &jcb_i_ga, std::vector<float> &mu_c_ga,
    std::vector<float> &jcb_c_ga, std::vector<float> &jcb_o_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len,
    int batch_size, std::vector<float> &cov_tanh_c)
/*Compute convariance between output gates & tanh(cell states)

Args:
    mu_w: Mean of weights
    var_ha: Variance of the activations + previous hidden states of lstm layer
    mu_c_prev: Mean of cell state (i.e., hidden state) of the previous step
    jcb_ca: Jacobian matrix (diagonal) of cell states
    jcb_f_ga: Jacobian matrix (diagonal) of forget gates
    mu_i_ga: Mean of the input gate
    jcb_i_ga: Jacobian matrix (diagonal) of input gates
    mu_c_ga: Mean of the cell state gate
    jcb_c_ga: Jacobian matrix (diagonal) of cell state gates
    jcb_o_ga: Jacobian matrix (diagonal) of output gates
    w_pos_f: Weight position for forget gate in the weight vector of network
    w_pos_i: Weight position for input gate in the weight vector of network
    w_pos_c: Weight position for cell state gate in the weight vector of network
    w_pos_o: Weight position for output gate in the weight vector of network
    ni: Input node
    no: Output node
    seq_len: Input sequence length
    batch_size: Batch size
    cov_tanh_c: Covariance between outputs and tanh of cell states

 */
{
    float sum_fo, sum_io, sum_oc;
    int k, m, i;
    for (int x = 0; x < batch_size; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                sum_fo = 0.0f;
                sum_io = 0.0f;
                sum_oc = 0.0f;
                for (int j = 0; j < ni; j++) {
                    k = j + z * (ni + no);
                    m = j + y * (ni + no) + x * (seq_len * (ni + no));
                    sum_fo += mu_w[w_pos_f + k] * var_ha[m] * mu_w[w_pos_o + k];
                    sum_io += mu_w[w_pos_i + k] * var_ha[m] * mu_w[w_pos_o + k];
                    sum_oc += mu_w[w_pos_c + k] * var_ha[m] * mu_w[w_pos_o + k];
                }
                i = z + y * no + x * seq_len * no;
                cov_tanh_c[i] =
                    jcb_ca[i] *
                    (jcb_o_ga[i] * sum_fo * jcb_f_ga[i] * mu_c_prev[i] +
                     jcb_o_ga[i] * sum_io * jcb_i_ga[i] * mu_c_ga[i] +
                     jcb_o_ga[i] * sum_oc * jcb_c_ga[i] * mu_i_ga[i]);
            }
        }
    }
}

void lstm_hidden_state_mean_var(std::vector<float> &mu_o_ga,
                                std::vector<float> &var_o_ga,
                                std::vector<float> &mu_ca,
                                std::vector<float> &var_ca,
                                std::vector<float> &cov_o_tanh_c, int no,
                                int seq_len, int B, std::vector<float> &mu_z,
                                std::vector<float> &var_z)
/*Compute mean and variance for hidden states of the LSTM layer

Args:
    mu_o_ga: Mean of the output gate
    var_o_ga: Variance of the output gate
    mu_ca: Mean of the activated cell states
    var_ca: Variance of the activated cell states
    cov_o_tanh_c: Covariance between outputs and tanh of cell states
    no: Output node
    seq_len: Input sequence length
    B: Batch size
    mu_z: Mean of hidden states
    var_z: Variance of hidden states

*/
{
    int k, j;
    for (int x = 0; x < B; x++) {
        for (int y = 0; y < seq_len; y++) {
            for (int z = 0; z < no; z++) {
                j = z + y * no + x * no * seq_len;
                k = z + y * no + x * no * seq_len;

                mu_z[k] = mu_o_ga[j] * mu_ca[j] + cov_o_tanh_c[j];

                var_z[k] = var_ca[j] * mu_o_ga[j] * mu_o_ga[j] +
                           var_ca[j] * var_o_ga[j] +
                           var_o_ga[j] * mu_ca[j] * mu_ca[j] +
                           powf(cov_o_tanh_c[j], 2) +
                           2 * cov_o_tanh_c[j] * mu_o_ga[j] * mu_ca[j];
            }
        }
    }
}

void lstm_to_prev_states(std::vector<float> &curr, int n,
                         std::vector<float> &prev)
/*Transfer data from current cell & hidden to previous cell & hidden states
   which are used for the next step*/
{
    for (int i = 0; i < n; i++) {
        prev[i] = curr[i];
    }
}

void lstm_cat_activations_and_prev_states(std::vector<float> &a,
                                          std::vector<float> &b, int n, int m,
                                          int seq_len, int B,
                                          std::vector<float> &c)
/*Concatenate two vectors a and b*/
{
    for (int k = 0; k < B; k++) {
        for (int s = 0; s < seq_len; s++) {
            for (int i = 0; i < n; i++) {
                c[i + s * (n + m) + k * (n + m) * seq_len] =
                    a[i + s * n + k * seq_len * n];
            }

            for (int j = 0; j < m; j++) {
                c[j + n + s * (n + m) + k * (n + m) * seq_len] =
                    b[j + s * m + k * m * seq_len];
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////
/// MULTITHREAD VERSION
//////////////////////////////////////////////////////////////////////
void lstm_cov_input_cell_states_worker(
    std::vector<float> &Sha, std::vector<float> &mw, std::vector<float> &Ji_ga,
    std::vector<float> &Jc_ga, int w_pos_i, int w_pos_c, int ni, int no,
    int seq_len, int B, int start_idx, int end_idx, std::vector<float> &Ci_c)
/*
 */
{
    float sum;
    int k, i, m, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;
        sum = 0;
        for (int j = 0; j < ni + no; j++) {
            k = j + z * (ni + no);
            m = j + y * (ni + no) + x * (seq_len * (ni + no));
            sum += mw[w_pos_i + k] * Sha[m] * mw[w_pos_c + k];
        }
        i = z + y * no + x * seq_len * no;
        Ci_c[i] = Ji_ga[i] * sum * Jc_ga[i];
    }
}

void lstm_cov_input_cell_states_mp(
    std::vector<float> &Sha, std::vector<float> &mw, std::vector<float> &Ji_ga,
    std::vector<float> &Jc_ga, int w_pos_i, int w_pos_c, int ni, int no,
    int seq_len, int B, int NUM_THREADS, std::vector<float> &Ci_c)
/*
 */
{
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_cov_input_cell_states_worker, std::ref(Sha), std::ref(mw),
            std::ref(Ji_ga), std::ref(Jc_ga), w_pos_i, w_pos_c, ni, no, seq_len,
            B, start_idx, end_idx, std::ref(Ci_c));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void lstm_cell_state_mean_var_worker(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int no, int seq_len, int start_idx, int end_idx,
    std::vector<float> &mc, std::vector<float> &Sc)
/*
 */
{
    int k, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;

        k = z + y * no + x * no * seq_len;
        mc[k] = mf_ga[k] * mc_prev[k] + mi_ga[k] * mc_ga[k] + Ci_c[k];
        Sc[k] = Sc_prev[k] * mf_ga[k] * mf_ga[k] + Sc_prev[k] * Sf_ga[k] +
                Sf_ga[k] * mc_prev[k] * mc_prev[k] +
                Sc_ga[k] * mi_ga[k] * mi_ga[k] + Si_ga[k] * Sc_ga[k] +
                Si_ga[k] * mc_ga[k] * mc_ga[k] + powf(Ci_c[k], 2) +
                2 * Ci_c[k] * mi_ga[k] * mc_ga[k];
    }
}

void lstm_cell_state_mean_var_mp(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int no, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &mc, std::vector<float> &Sc)
/*
 */
{
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_cell_state_mean_var_worker, std::ref(mf_ga), std::ref(Sf_ga),
            std::ref(mi_ga), std::ref(Si_ga), std::ref(mc_ga), std::ref(Sc_ga),
            std::ref(mc_prev), std::ref(Sc_prev), std::ref(Ci_c), no, seq_len,
            start_idx, end_idx, std::ref(mc), std::ref(Sc));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void lstm_cov_output_tanh_cell_states_worker(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jca,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len,
    int start_idx, int end_idx, std::vector<float> &Co_tanh_c)
/*
 */
{
    float sum_fo, sum_io, sum_oc;
    int k, m, i, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;
        sum_fo = 0;
        sum_io = 0;
        sum_oc = 0;
        for (int j = 0; j < ni; j++) {
            k = j + z * (ni + no);
            m = j + y * (ni + no) + x * (seq_len * (ni + no));
            sum_fo += mw[w_pos_f + k] * Sha[m] * mw[w_pos_o + k];
            sum_io += mw[w_pos_i + k] * Sha[m] * mw[w_pos_o + k];
            sum_oc += mw[w_pos_c + k] * Sha[m] * mw[w_pos_o + k];
        }
        i = z + y * no + x * seq_len * no;
        Co_tanh_c[i] = Jca[i] * (Jo_ga[i] * sum_fo * Jf_ga[i] * mc_prev[i] +
                                 Jo_ga[i] * sum_io * Ji_ga[i] * mc_ga[i] +
                                 Jo_ga[i] * sum_oc * Jc_ga[i] * mi_ga[i]);
    }
}

void lstm_cov_output_tanh_cell_states_mp(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jca,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len, int B,
    int NUM_THREADS, std::vector<float> &Co_tanh_c)
/*
 */
{
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_cov_output_tanh_cell_states_worker, std::ref(mw),
            std::ref(Sha), std::ref(mc_prev), std::ref(Jca), std::ref(Jf_ga),
            std::ref(mi_ga), std::ref(Ji_ga), std::ref(mc_ga), std::ref(Jc_ga),
            std::ref(Jo_ga), w_pos_f, w_pos_i, w_pos_c, w_pos_o, ni, no,
            seq_len, start_idx, end_idx, std::ref(Co_tanh_c));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void lstm_hidden_state_mean_var_worker(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int no, int seq_len, int start_idx,
    int end_idx, std::vector<float> &mz, std::vector<float> &Sz)
/*
 */
{
    int k, j, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (no * seq_len);
        y = (t % (no * seq_len)) / no;
        z = t % no;

        j = z + y * no + x * no * seq_len;
        k = z + y * no + x * no * seq_len;
        mz[k] = mo_ga[j] * mc_a[j] + Co_tanh_c[j];
        Sz[k] = Sc_a[j] * mo_ga[j] * mo_ga[j] + Sc_a[j] * So_ga[j] +
                So_ga[j] * mc_a[j] * mc_a[j] + powf(Co_tanh_c[j], 2) +
                2 * Co_tanh_c[j] * mo_ga[j] * mc_a[j];
    }
}

void lstm_hidden_state_mean_var_mp(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int no, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &mz, std::vector<float> &Sz)
/*
 */
{
    const int tot_ops = B * seq_len * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_hidden_state_mean_var_worker, std::ref(mo_ga), std::ref(So_ga),
            std::ref(mc_a), std::ref(Sc_a), std::ref(Co_tanh_c), no, seq_len,
            start_idx, end_idx, std::ref(mz), std::ref(Sz));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void lstm_cat_activations_and_prev_states_worker(std::vector<float> &a,
                                                 std::vector<float> &b, int n,
                                                 int m, int seq_len, int B,
                                                 int start_idx, int end_idx,
                                                 std::vector<float> &c)
/*Concatenate two vectors*/
{
    int k, s;
    for (int t = start_idx; t < end_idx; t++) {
        k = t / seq_len;
        s = t % seq_len;

        for (int i = 0; i < n; i++) {
            c[i + s * (n + m) + k * (n + m) * seq_len] =
                a[i + s * n + k * seq_len * n];
        }

        for (int j = 0; j < m; j++) {
            c[j + n + s * (n + m) + k * (n + m) * seq_len] =
                b[j + s * m + k * m * seq_len];
        }
    }
}

void lstm_cat_activations_and_prev_states_mp(std::vector<float> &a,
                                             std::vector<float> &b, int n,
                                             int m, int seq_len, int B,
                                             int NUM_THREADS,
                                             std::vector<float> &c)
/*
 */
{
    const int tot_ops = B * seq_len;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(lstm_cat_activations_and_prev_states_worker,
                                 std::ref(a), std::ref(b), n, m, seq_len, B,
                                 start_idx, end_idx, std::ref(c));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
// BACKWARD PASS
////////////////////////////////////////////////////////////////////////////////
void lstm_delta_mean_var_z_worker(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_m_out,
    std::vector<float> &delta_S_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int start_idx, int end_idx,
    std::vector<float> &delta_m, std::vector<float> &delta_S)
/*
 */
{
    float sum_mf, sum_mi, sum_mc, sum_mo, sum_Sz;
    float Czz_f, Czz_i, Czz_c, Czz_o;
    int k, m, i, x, y, z;
    for (int t = start_idx; t < end_idx; t++) {
        x = t / (ni * seq_len);
        y = (t % (ni * seq_len)) / ni;
        z = t % ni;

        sum_mf = 0;
        sum_mi = 0;
        sum_mc = 0;
        sum_mo = 0;
        sum_Sz = 0;
        for (int j = 0; j < no; j++) {
            k = j + x * no * seq_len + y * no;
            i = j + x * no * seq_len + y * no;
            // Forget gate
            Czz_f = Jca[k] * mo_ga[k] * Jf_ga[k] *
                    mw[(ni + no) * j + z + w_pos_f] * mc_prev[k];
            sum_mf += Czz_f * delta_m_out[i];

            // Input gate
            Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                    mw[(ni + no) * j + z + w_pos_i] * mc_ga[k];
            sum_mi += Czz_i * delta_m_out[i];

            // Cell state gate
            Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                    mw[(ni + no) * j + z + w_pos_c] * mi_ga[k];
            sum_mc += Czz_c * delta_m_out[i];

            // Output gate
            Czz_o = Jo_ga[k] * mw[(ni + no) * j + z + w_pos_o] * mca[k];
            sum_mo += Czz_o * delta_m_out[i];
            sum_Sz += powf(Czz_f + Czz_i + Czz_c + Czz_o, 2) * delta_S_out[i];
        }

        // Updating quantities
        m = x * ni * seq_len + y * ni + z;
        delta_m[m] = (sum_mf + sum_mi + sum_mc + sum_mo);
        delta_S[m] = sum_Sz;
    }
}

void lstm_delta_mean_var_z_mp(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_m_out,
    std::vector<float> &delta_S_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &delta_m, std::vector<float> &delta_S)
/*
 */
{
    const int tot_ops = B * seq_len * ni;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_delta_mean_var_z_worker, std::ref(mw), std::ref(Jf_ga),
            std::ref(mi_ga), std::ref(Ji_ga), std::ref(mc_ga), std::ref(Jc_ga),
            std::ref(mo_ga), std::ref(Jo_ga), std::ref(mc_prev), std::ref(mca),
            std::ref(Jca), std::ref(delta_m_out), std::ref(delta_S_out),
            w_pos_f, w_pos_i, w_pos_c, w_pos_o, no, ni, seq_len, start_idx,
            end_idx, std::ref(delta_m), std::ref(delta_S));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void lstm_update_prev_hidden_states_worker(
    std::vector<float> &mu_h_prior, std::vector<float> &var_h_prior,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int start_idx,
    int end_idx, std::vector<float> &mu_h_prev, std::vector<float> &var_h_prev)
/*
 */
{
    for (size_t i = start_idx; i < end_idx; i++) {
        mu_h_prev[i] = mu_h_prior[i] + delta_mu[i] * var_h_prior[i];
        var_h_prev[i] = (1.0f + delta_var[i] * var_h_prior[i]) * var_h_prior[i];
    }
}

void lstm_update_prev_hidden_states_mp(std::vector<float> &mu_h_prior,
                                       std::vector<float> &var_h_prior,
                                       std::vector<float> &delta_mu,
                                       std::vector<float> &delta_var,
                                       int num_states, unsigned NUM_THREADS,
                                       std::vector<float> &mu_h_prev,
                                       std::vector<float> &var_h_prev)
/*
 */
{
    const unsigned int num_per_threads = num_states / NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        start_idx = i * num_per_threads;
        end_idx =
            (i == NUM_THREADS - 1) ? num_states : (i + 1) * num_per_threads;

        threads.emplace_back([=, &mu_h_prior, &var_h_prior, &delta_mu,
                              &delta_var, &mu_h_prev, &var_h_prev] {
            lstm_update_prev_hidden_states_worker(
                mu_h_prior, var_h_prior, delta_mu, delta_var, start_idx,
                end_idx, mu_h_prev, var_h_prev);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads.clear();
}

void lstm_update_prev_cell_states_worker(
    std::vector<float> &mu_c_prior, std::vector<float> &var_c_prior,
    std::vector<float> &jcb_ca, std::vector<float> &mu_o_ga,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int start_idx,
    int end_idx, std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev)
/*
 */
{
    for (size_t i = start_idx; i < end_idx; i++) {
        float tmp = var_c_prior[i] * jcb_ca[i] * mu_o_ga[i];
        mu_c_prev[i] = mu_c_prior[i] + tmp * delta_mu[i];
        var_c_prev[i] = var_c_prior[i] + tmp * delta_var[i] * tmp;
    }
}

void lstm_update_prev_cell_states_mp(
    std::vector<float> &mu_c_prior, std::vector<float> &var_c_prior,
    std::vector<float> &jcb_ca, std::vector<float> &mu_o_ga,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int num_states,
    unsigned int NUM_THREADS, std::vector<float> &mu_c_prev,
    std::vector<float> &var_c_prev)
/*
 */
{
    const unsigned int num_per_threads = num_states / NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        start_idx = i * num_per_threads;
        end_idx =
            (i == NUM_THREADS - 1) ? num_states : (i + 1) * num_per_threads;
        threads.emplace_back([=, &mu_c_prior, &var_c_prior, &jcb_ca, &mu_o_ga,
                              &delta_mu, &delta_var, &mu_c_prev, &var_c_prev] {
            lstm_update_prev_cell_states_worker(
                mu_c_prior, var_c_prior, jcb_ca, mu_o_ga, delta_mu, delta_var,
                start_idx, end_idx, mu_c_prev, var_c_prev);
        });
    }

    for (auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads.clear();
}

void lstm_delta_mean_var_w_worker(
    std::vector<float> &Sw, std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, int start_idx, int end_idx,
    std::vector<float> &delta_mw, std::vector<float> &delta_Sw)
/*
 */
{
    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, m, l, i, row, col, x, y;
    for (int t = start_idx; t < end_idx; t++) {
        row = t / no;
        col = t % no;
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int j = 0; j < B * seq_len; j++) {
            x = j / seq_len;
            y = j % seq_len;

            k = col + y * no + no * seq_len * x;
            i = col + y * no + no * seq_len * x;
            l = row + y * (ni + no) + (ni + no) * seq_len * x;

            // Forget gate
            Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k] * mha[l];
            sum_mf += Cwa_f * delta_m[i];
            sum_Sf += Cwa_f * delta_S[i] * Cwa_f;

            // Input gate
            Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k] * mha[l];
            sum_mi += Cwa_i * delta_m[i];
            sum_Si += Cwa_i * delta_S[i] * Cwa_i;

            // Cell state gate
            Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k] * mha[l];
            sum_mc += Cwa_c * delta_m[i];
            sum_Sc += Cwa_c * delta_S[i] * Cwa_c;

            // Output gate
            Cwa_o = Jo_ga[k] * mca[k] * mha[l];
            sum_mo += Cwa_o * delta_m[i];
            sum_So += Cwa_o * delta_S[i] * Cwa_o;
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

void lstm_delta_mean_var_w_mp(
    std::vector<float> &Sw, std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &delta_mw, std::vector<float> &delta_Sw)
/*
 */
{
    const int tot_ops = (ni + no) * no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_delta_mean_var_w_worker, std::ref(Sw), std::ref(mha),
            std::ref(Jf_ga), std::ref(mi_ga), std::ref(Ji_ga), std::ref(mc_ga),
            std::ref(Jc_ga), std::ref(mo_ga), std::ref(Jo_ga),
            std::ref(mc_prev), std::ref(mca), std::ref(Jc), std::ref(delta_m),
            std::ref(delta_S), w_pos_f, w_pos_i, w_pos_c, w_pos_o, no, ni,
            seq_len, B, start_idx, end_idx, std::ref(delta_mw),
            std::ref(delta_Sw));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

void lstm_delta_mean_var_b_worker(
    std::vector<float> &Sb, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int b_pos_f, int b_pos_i, int b_pos_c,
    int b_pos_o, int no, int seq_len, int B, int start_idx, int end_idx,
    std::vector<float> &delta_mb, std::vector<float> &delta_Sb)
/*Compute updating quantities of the bias for the lstm layer
 */
{
    float sum_mf, sum_Sf, Cwa_f, sum_mi, sum_Si, Cwa_i, sum_mc, sum_Sc, Cwa_c,
        sum_mo, sum_So, Cwa_o;
    int k, l, i;
    for (int row = start_idx; row < end_idx; row++) {
        sum_mf = 0;
        sum_Sf = 0;
        sum_mi = 0;
        sum_Si = 0;
        sum_mc = 0;
        sum_Sc = 0;
        sum_mo = 0;
        sum_So = 0;
        for (int x = 0; x < B; x++) {
            for (int y = 0; y < seq_len; y++) {
                k = row + y * no + no * seq_len * x;
                i = row + y * no + no * seq_len * x;

                // Forget gate
                Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k];
                sum_mf += Cwa_f * delta_m[i];
                sum_Sf += Cwa_f * delta_S[i] * Cwa_f;

                // Input gate
                Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k];
                sum_mi += Cwa_i * delta_m[i];
                sum_Si += Cwa_i * delta_S[i] * Cwa_i;

                // Cell state gate
                Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k];
                sum_mc += Cwa_c * delta_m[i];
                sum_Sc += Cwa_c * delta_S[i] * Cwa_c;

                // Output gate
                Cwa_o = Jo_ga[k] * mca[k];
                sum_mo += Cwa_o * delta_m[i];
                sum_So += Cwa_o * delta_S[i] * Cwa_o;
            }
        }
        // Updating quantities for biases
        delta_mb[row + b_pos_f] = sum_mf * Sb[row + b_pos_f];
        delta_Sb[row + b_pos_f] =
            Sb[row + b_pos_f] * sum_Sf * Sb[row + b_pos_f];

        delta_mb[row + b_pos_i] = sum_mi * Sb[row + b_pos_i];
        delta_Sb[row + b_pos_i] =
            Sb[row + b_pos_i] * sum_Si * Sb[row + b_pos_i];

        delta_mb[row + b_pos_c] = sum_mc * Sb[row + b_pos_c];
        delta_Sb[row + b_pos_c] =
            Sb[row + b_pos_c] * sum_Sc * Sb[row + b_pos_c];

        delta_mb[row + b_pos_o] = sum_mo * Sb[row + b_pos_o];
        delta_Sb[row + b_pos_o] =
            Sb[row + b_pos_o] * sum_So * Sb[row + b_pos_o];
    }
}

void lstm_delta_mean_var_b_mp(
    std::vector<float> &Sb, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int b_pos_f, int b_pos_i, int b_pos_c,
    int b_pos_o, int no, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &delta_mb, std::vector<float> &delta_Sb)
/*
 */
{
    const int tot_ops = no;
    const int n_batch = tot_ops / NUM_THREADS;
    const int rem_batch = tot_ops % NUM_THREADS;
    int start_idx, end_idx;
    std::vector<std::thread> threads(NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        get_multithread_indices(i, n_batch, rem_batch, start_idx, end_idx);

        threads[i] = std::thread(
            lstm_delta_mean_var_b_worker, std::ref(Sb), std::ref(Jf_ga),
            std::ref(mi_ga), std::ref(Ji_ga), std::ref(mc_ga), std::ref(Jc_ga),
            std::ref(mo_ga), std::ref(Jo_ga), std::ref(mc_prev), std::ref(mca),
            std::ref(Jc), std::ref(delta_m), std::ref(delta_S), b_pos_f,
            b_pos_i, b_pos_c, b_pos_o, no, seq_len, B, start_idx, end_idx,
            std::ref(delta_mb), std::ref(delta_Sb));
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
}

////////////////////////////////////////////////////////////////////////////////
// LSTM
////////////////////////////////////////////////////////////////////////////////

LSTM::LSTM(size_t input_size, size_t output_size, int seq_len, bool bias,
           float gain_w, float gain_b, std::string init_method)
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

LSTM::~LSTM() {}

std::string LSTM::get_layer_info() const
/*
 */
{
    return "LSTM(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LSTM::get_layer_name() const
/*
 */
{
    return "LSTM";
}

LayerType LSTM::get_layer_type() const
/*
 */
{
    return LayerType::LSTM;
}

int LSTM::get_input_size()
/*
 */
{
    return this->input_size * this->seq_len;
}

int LSTM::get_output_size()
/*
 */
{
    return this->output_size * this->seq_len;
}

int LSTM::get_max_num_states()
/*
 */
{
    int in_size = static_cast<int>(this->input_size) * this->seq_len;
    int out_size = static_cast<int>(this->output_size) * this->seq_len;
    return std::max(in_size, out_size);
}

void LSTM::get_number_param()
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

void LSTM::init_weight_bias()
/*
 */
{
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_lstm(this->init_method, this->gain_w, this->gain_b,
                              this->input_size, this->output_size,
                              this->num_weights, this->num_biases);
}

void LSTM::prepare_input(BaseHiddenStates &input_state)
/*
 */
{
    int batch_size = input_state.block_size;
    if (this->num_threads > 1) {
        lstm_cat_activations_and_prev_states_mp(
            input_state.mu_a, lstm_states.mu_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, this->num_threads,
            lstm_states.mu_ha);
        lstm_cat_activations_and_prev_states_mp(
            input_state.var_a, lstm_states.var_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, this->num_threads,
            lstm_states.var_ha);
    } else {
        lstm_cat_activations_and_prev_states(
            input_state.mu_a, lstm_states.mu_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, lstm_states.mu_ha);
        lstm_cat_activations_and_prev_states(
            input_state.var_a, lstm_states.var_h_prev, this->input_size,
            this->output_size, this->seq_len, batch_size, lstm_states.var_ha);
    }
}

void LSTM::forget_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int end_chunk = this->output_size * batch_size * this->seq_len;
    if (this->num_threads > 1) {
        lstm_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                             lstm_states.mu_ha, lstm_states.var_ha, ni_c,
                             this->output_size, b_seq, this->bias,
                             this->w_pos_f, this->b_pos_f, this->num_threads,
                             lstm_states.mu_f_ga, lstm_states.var_f_ga);

        sigmoid_mean_var_mp(lstm_states.mu_f_ga, lstm_states.var_f_ga,
                            end_chunk, this->num_threads, lstm_states.mu_f_ga,
                            lstm_states.jcb_f_ga, lstm_states.var_f_ga);

    } else {
        lstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                          lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                          ni_c, this->output_size, b_seq, this->bias,
                          this->w_pos_f, this->b_pos_f, lstm_states.mu_f_ga,
                          lstm_states.var_f_ga);

        sigmoid_mean_var(lstm_states.mu_f_ga, lstm_states.var_f_ga, 0,
                         end_chunk, lstm_states.mu_f_ga, lstm_states.jcb_f_ga,
                         lstm_states.var_f_ga);
    }
}

void LSTM::input_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int end_chunk = this->output_size * batch_size * this->seq_len;
    if (this->num_threads > 1) {
        lstm_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                             lstm_states.mu_ha, lstm_states.var_ha, ni_c,
                             this->output_size, b_seq, this->bias,
                             this->w_pos_i, this->b_pos_i, this->num_threads,
                             lstm_states.mu_i_ga, lstm_states.var_i_ga);
        sigmoid_mean_var_mp(lstm_states.mu_i_ga, lstm_states.var_i_ga,
                            end_chunk, this->num_threads, lstm_states.mu_i_ga,
                            lstm_states.jcb_i_ga, lstm_states.var_i_ga);
    } else {
        lstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                          lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                          ni_c, this->output_size, b_seq, this->bias,
                          this->w_pos_i, this->b_pos_i, lstm_states.mu_i_ga,
                          lstm_states.var_i_ga);
        sigmoid_mean_var(lstm_states.mu_i_ga, lstm_states.var_i_ga, 0,
                         end_chunk, lstm_states.mu_i_ga, lstm_states.jcb_i_ga,
                         lstm_states.var_i_ga);
    }
}

void LSTM::cell_state_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int end_chunk = this->output_size * batch_size * this->seq_len;
    if (this->num_threads > 1) {
        lstm_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                             lstm_states.mu_ha, lstm_states.var_ha, ni_c,
                             this->output_size, b_seq, this->bias,
                             this->w_pos_c, this->b_pos_c, this->num_threads,
                             lstm_states.mu_c_ga, lstm_states.var_c_ga);
        tanh_mean_var_mp(lstm_states.mu_c_ga, lstm_states.var_c_ga, end_chunk,
                         this->num_threads, lstm_states.mu_c_ga,
                         lstm_states.jcb_c_ga, lstm_states.var_c_ga);
    } else {
        lstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                          lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                          ni_c, this->output_size, b_seq, this->bias,
                          this->w_pos_c, this->b_pos_c, lstm_states.mu_c_ga,
                          lstm_states.var_c_ga);
        tanh_mean_var(lstm_states.mu_c_ga, lstm_states.var_c_ga, 0, end_chunk,
                      lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                      lstm_states.var_c_ga);
    }
}

void LSTM::output_gate(int batch_size)
/*
 */
{
    int ni_c = this->input_size + this->output_size;
    int b_seq = batch_size * this->seq_len;
    int end_chunk = this->output_size * batch_size * this->seq_len;
    if (this->num_threads > 1) {
        lstm_fwd_mean_var_mp(this->mu_w, this->var_w, this->mu_b, this->var_b,
                             lstm_states.mu_ha, lstm_states.var_ha, ni_c,
                             this->output_size, b_seq, this->bias,
                             this->w_pos_o, this->b_pos_o, this->num_threads,
                             lstm_states.mu_o_ga, lstm_states.var_o_ga);
        sigmoid_mean_var_mp(lstm_states.mu_o_ga, lstm_states.var_o_ga,
                            end_chunk, this->num_threads, lstm_states.mu_o_ga,
                            lstm_states.jcb_o_ga, lstm_states.var_o_ga);
    } else {
        lstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                          lstm_states.mu_ha, lstm_states.var_ha, 0, end_chunk,
                          ni_c, this->output_size, b_seq, this->bias,
                          this->w_pos_o, this->b_pos_o, lstm_states.mu_o_ga,
                          lstm_states.var_o_ga);
        sigmoid_mean_var(lstm_states.mu_o_ga, lstm_states.var_o_ga, 0,
                         end_chunk, lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                         lstm_states.var_o_ga);
    }
}

void LSTM::forward(BaseHiddenStates &input_states,
                   BaseHiddenStates &output_states, BaseTempStates &temp_states)
/*
 */
{
    // Checkout input size
    if (this->input_size * this->seq_len != input_states.actual_size) {
        std::string message = "Input size mismatch: " +
                              std::to_string(this->input_size * this->seq_len) +
                              " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    int batch_size = input_states.block_size;
    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_states.set_num_states(
            batch_size * this->seq_len * this->output_size,
            batch_size * this->seq_len * this->input_size);
    }
    // Update number of actual states.
    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size * this->seq_len;

    // TODO: This is not efficient for memory and performance. Update the
    // previous states
    if (this->seq_len == 1 && batch_size == 1) {
        lstm_to_prev_states(this->lstm_states.mu_h_prior,
                            this->lstm_states.mu_h_prior.size(),
                            this->lstm_states.mu_h_prev);
        lstm_to_prev_states(this->lstm_states.var_h_prior,
                            this->lstm_states.var_h_prior.size(),
                            this->lstm_states.var_h_prev);
        lstm_to_prev_states(this->lstm_states.mu_c_prior,
                            this->lstm_states.mu_c_prior.size(),
                            this->lstm_states.mu_c_prev);
        lstm_to_prev_states(this->lstm_states.var_c_prior,
                            this->lstm_states.var_c_prior.size(),
                            this->lstm_states.var_c_prev);
    }

    this->prepare_input(input_states);
    this->forget_gate(batch_size);
    this->input_gate(batch_size);
    this->cell_state_gate(batch_size);
    this->output_gate(batch_size);

    int end_chunk = this->output_size * batch_size * this->seq_len;

    if (this->num_threads > 1) {
        lstm_cov_input_cell_states_mp(
            lstm_states.var_ha, this->mu_w, lstm_states.jcb_i_ga,
            lstm_states.jcb_c_ga, this->w_pos_i, this->w_pos_c,
            this->input_size, this->output_size, this->seq_len, batch_size,
            this->num_threads, lstm_states.cov_i_c);

        lstm_cell_state_mean_var_mp(
            lstm_states.mu_f_ga, lstm_states.var_f_ga, lstm_states.mu_i_ga,
            lstm_states.var_i_ga, lstm_states.mu_c_ga, lstm_states.var_c_ga,
            lstm_states.mu_c_prev, lstm_states.var_c_prev, lstm_states.cov_i_c,
            this->output_size, this->seq_len, batch_size, this->num_threads,
            lstm_states.mu_c, lstm_states.var_c);

        tanh_mean_var_mp(lstm_states.mu_c, lstm_states.var_c, end_chunk,
                         this->num_threads, lstm_states.mu_ca,
                         lstm_states.jcb_ca, lstm_states.var_ca);

        lstm_cov_output_tanh_cell_states_mp(
            this->mu_w, lstm_states.var_ha, lstm_states.mu_c_prev,
            lstm_states.jcb_ca, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
            lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
            lstm_states.jcb_o_ga, this->w_pos_f, this->w_pos_i, this->w_pos_c,
            this->w_pos_o, this->input_size, this->output_size, this->seq_len,
            batch_size, this->num_threads, lstm_states.cov_o_tanh_c);

        lstm_hidden_state_mean_var_mp(
            lstm_states.mu_o_ga, lstm_states.var_o_ga, lstm_states.mu_ca,
            lstm_states.var_ca, lstm_states.cov_o_tanh_c, this->output_size,
            this->seq_len, batch_size, this->num_threads, output_states.mu_a,
            output_states.var_a);

    } else {
        lstm_cov_input_cell_states(
            lstm_states.var_ha, this->mu_w, lstm_states.jcb_i_ga,
            lstm_states.jcb_c_ga, this->w_pos_i, this->w_pos_c,
            this->input_size, this->output_size, this->seq_len, batch_size,
            lstm_states.cov_i_c);

        lstm_cell_state_mean_var(
            lstm_states.mu_f_ga, lstm_states.var_f_ga, lstm_states.mu_i_ga,
            lstm_states.var_i_ga, lstm_states.mu_c_ga, lstm_states.var_c_ga,
            lstm_states.mu_c_prev, lstm_states.var_c_prev, lstm_states.cov_i_c,
            this->output_size, this->seq_len, batch_size, lstm_states.mu_c,
            lstm_states.var_c);

        tanh_mean_var(lstm_states.mu_c, lstm_states.var_c, 0, end_chunk,
                      lstm_states.mu_ca, lstm_states.jcb_ca,
                      lstm_states.var_ca);

        lstm_cov_output_tanh_cell_states(
            this->mu_w, lstm_states.var_ha, lstm_states.mu_c_prev,
            lstm_states.jcb_ca, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
            lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
            lstm_states.jcb_o_ga, this->w_pos_f, this->w_pos_i, this->w_pos_c,
            this->w_pos_o, this->input_size, this->output_size, this->seq_len,
            batch_size, lstm_states.cov_o_tanh_c);

        lstm_hidden_state_mean_var(
            lstm_states.mu_o_ga, lstm_states.var_o_ga, lstm_states.mu_ca,
            lstm_states.var_ca, lstm_states.cov_o_tanh_c, this->output_size,
            this->seq_len, batch_size, output_states.mu_a, output_states.var_a);
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
    // Save the previous states
    if (this->seq_len == 1 && batch_size == 1) {
        lstm_to_prev_states(output_states.mu_a,
                            this->lstm_states.mu_h_prior.size(),
                            this->lstm_states.mu_h_prior);
        lstm_to_prev_states(output_states.var_a,
                            this->lstm_states.var_h_prior.size(),
                            this->lstm_states.var_h_prior);
        lstm_to_prev_states(this->lstm_states.mu_c,
                            this->lstm_states.mu_c_prior.size(),
                            this->lstm_states.mu_c_prior);
        lstm_to_prev_states(this->lstm_states.var_c,
                            this->lstm_states.var_c_prior.size(),
                            this->lstm_states.var_c_prior);
    }
}

void LSTM::backward(BaseDeltaStates &input_delta_states,
                    BaseDeltaStates &output_delta_states,
                    BaseTempStates &temp_states, bool state_udapte)
/*
 */
{
    int batch_size = input_delta_states.block_size;
    if (state_udapte) {
        if (this->num_threads > 1) {
            lstm_delta_mean_var_z_mp(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, batch_size,
                this->num_threads, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        } else {
            int end_chunk = batch_size * this->seq_len * this->input_size;
            lstm_delta_mean_var_z_worker(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, 0,
                end_chunk, output_delta_states.delta_mu,
                output_delta_states.delta_var);
        }
    }

    if (param_update) {
        if (this->num_threads > 1) {
            lstm_delta_mean_var_w_mp(
                this->var_w, lstm_states.mu_ha, lstm_states.jcb_f_ga,
                lstm_states.mu_i_ga, lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                lstm_states.jcb_c_ga, lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, batch_size,
                this->num_threads, this->delta_mu_w, this->delta_var_w);

            if (this->bias) {
                lstm_delta_mean_var_b_mp(
                    this->var_b, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                    lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                    lstm_states.jcb_c_ga, lstm_states.mu_o_ga,
                    lstm_states.jcb_o_ga, lstm_states.mu_c_prev,
                    lstm_states.mu_ca, lstm_states.jcb_ca,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->b_pos_f, this->b_pos_i, this->b_pos_c, this->b_pos_o,
                    this->output_size, this->seq_len, batch_size,
                    this->num_threads, this->delta_mu_b, this->delta_var_b);
            }
        } else {
            int end_chunk_w =
                (this->input_size + this->output_size) * this->output_size;
            lstm_delta_mean_var_w_worker(
                this->var_w, lstm_states.mu_ha, lstm_states.jcb_f_ga,
                lstm_states.mu_i_ga, lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                lstm_states.jcb_c_ga, lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                this->output_size, this->input_size, this->seq_len, batch_size,
                0, end_chunk_w, this->delta_mu_w, this->delta_var_w);

            if (this->bias) {
                lstm_delta_mean_var_b_worker(
                    this->var_b, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                    lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                    lstm_states.jcb_c_ga, lstm_states.mu_o_ga,
                    lstm_states.jcb_o_ga, lstm_states.mu_c_prev,
                    lstm_states.mu_ca, lstm_states.jcb_ca,
                    input_delta_states.delta_mu, input_delta_states.delta_var,
                    this->b_pos_f, this->b_pos_i, this->b_pos_c, this->b_pos_o,
                    this->output_size, this->seq_len, batch_size, 0,
                    this->output_size, this->delta_mu_b, this->delta_var_b);
            }
        }
    }
    if (this->seq_len == 1 && batch_size == 1) {
        if (this->num_threads > 1) {
            lstm_update_prev_hidden_states_mp(
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->lstm_states.num_states, this->num_threads,
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior);
            lstm_update_prev_cell_states_mp(
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior,
                this->lstm_states.jcb_ca, this->lstm_states.mu_o_ga,
                input_delta_states.delta_mu, input_delta_states.delta_var,
                this->lstm_states.num_states, this->num_threads,
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior);
        } else {
            lstm_update_prev_hidden_states_worker(
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior,
                input_delta_states.delta_mu, input_delta_states.delta_var, 0,
                this->lstm_states.num_states, this->lstm_states.mu_h_prior,
                this->lstm_states.var_h_prior);
            lstm_update_prev_cell_states_worker(
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior,
                this->lstm_states.jcb_ca, this->lstm_states.mu_o_ga,
                input_delta_states.delta_mu, input_delta_states.delta_var, 0,
                this->lstm_states.num_states, this->lstm_states.mu_c_prior,
                this->lstm_states.var_c_prior);
        }
    }
}

// retrieve the cell state and hidden state
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
LSTM::get_LSTM_states() const {
    return std::make_tuple(lstm_states.mu_h_prior, lstm_states.var_h_prior,
                           lstm_states.mu_c_prior, lstm_states.var_c_prior);
}
// set the cell state and hidden state
void LSTM::set_LSTM_states(const std::vector<float> &mu_h,
                           const std::vector<float> &var_h,
                           const std::vector<float> &mu_c,
                           const std::vector<float> &var_c) {
    this->lstm_states.mu_h_prior = mu_h;
    this->lstm_states.var_h_prior = var_h;
    this->lstm_states.mu_c_prior = mu_c;
    this->lstm_states.var_c_prior = var_c;
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LSTM::to_cuda() {
    this->device = "cuda";
    auto cuda_layer = std::make_unique<LSTMCuda>(
        this->input_size, this->output_size, this->seq_len, this->bias,
        this->gain_w, this->gain_b, this->init_method);

    // Move params from this->layer to cuda_layer
    auto base_cuda = dynamic_cast<BaseLayerCuda *>(cuda_layer.get());
    base_cuda->copy_params_from(*this);

    return cuda_layer;
}
#endif

void LSTM::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
    }
    if (this->training) {
        this->allocate_param_delta();
    }
}

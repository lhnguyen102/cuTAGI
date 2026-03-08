
#include "../include/tlstm_layer.h"

#include <cmath>
#include <thread>
#include <tuple>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/custom_logger.h"
#include "../include/lstm_layer.h"
#include "../include/param_init.h"

#ifdef USE_CUDA
#include "../include/tlstm_layer_cuda.cuh"
#endif

template <typename Fn>
void parallel_for(int total, unsigned int num_threads, Fn &&fn) {
    if (num_threads <= 1) {
        fn(0, total);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int per_thread = total / num_threads;
    int extra = total % num_threads;
    for (int i = 0; i < static_cast<int>(num_threads); i++) {
        int start = i * per_thread + std::min(i, extra);
        int end = start + per_thread + (i < extra ? 1 : 0);
        threads.emplace_back(std::forward<Fn>(fn), start, end);
    }
    for (auto &t : threads) t.join();
}

////////////////////////////////////////////////////////////////////////////////
// FORWARD FREE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void tlstm_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        std::vector<float> &mu_a, std::vector<float> &var_a,
                        int start_chunk, int end_chunk, size_t input_size,
                        size_t output_size, int batch_size, int seq_len,
                        int time_step, bool bias, int w_pos, int b_pos,
                        std::vector<float> &mu_z, std::vector<float> &var_z) {
    for (int i = start_chunk; i < end_chunk; i++) {
        int row = i / batch_size;  // output node index
        int col = i % batch_size;  // batch size
        float sum_mu_z = 0.0f;
        float sum_var_z = 0.0f;
        int input_offset = col * seq_len * input_size + time_step * input_size;
        int output_offset =
            col * seq_len * output_size + time_step * output_size;

        for (int j = 0; j < input_size; j++) {
            float mu_a_tmp =
                mu_a[input_offset + j];  // [batch_size, input_size]
            float var_a_tmp = var_a[input_offset + j];
            float mu_w_tmp = mu_w[row * input_size + j +
                                  w_pos];  // [output_size, input_size]
            float var_w_tmp = var_w[row * input_size + j + w_pos];

            sum_mu_z += mu_w_tmp * mu_a_tmp;
            sum_var_z += (mu_w_tmp * mu_w_tmp + var_w_tmp) * var_a_tmp +
                         var_w_tmp * mu_a_tmp * mu_a_tmp;
        }
        if (bias) {
            mu_z[output_offset + row] = sum_mu_z + mu_b[row + b_pos];
            var_z[output_offset + row] = sum_var_z + var_b[row + b_pos];
        } else {
            mu_z[output_offset + row] = sum_mu_z;  // [batch_size, output_size]
            var_z[output_offset + row] = sum_var_z;
        }
    }
}

void tlstm_cat_activations_and_prev_states(std::vector<float> &vec_a,
                                           std::vector<float> &vec_b, int n,
                                           int m, int batch_size, int seq_len,
                                           int time_step,
                                           std::vector<float> &vec_c) {
    int ni_c = n + m;
    for (int b = 0; b < batch_size; b++) {
        int a_off = b * seq_len * n + time_step * n;
        int b_off = b * seq_len * m + time_step * m;
        int c_off = b * seq_len * ni_c + time_step * ni_c;
        for (int i = 0; i < n; i++) {
            vec_c[c_off + i] = vec_a[a_off + i];
        }
        for (int j = 0; j < m; j++) {
            vec_c[c_off + n + j] = vec_b[b_off + j];
        }
    }
}

void lstm_sigmoid_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_idx, int end_idx, int seq_len, int no,
                           int time_step, std::vector<float> &mu_a,
                           std::vector<float> &jcb, std::vector<float> &var_a) {
    for (int idx = start_idx; idx < end_idx; idx++) {
        int batch_idx = idx / no;
        int output_idx = idx % no;
        int off = output_idx + time_step * no + batch_idx * seq_len * no;
        float tmp = 1.0f / (1.0f + expf(-mu_z[off]));
        mu_a[off] = tmp;
        jcb[off] = tmp * (1.0f - tmp);
        var_a[off] = jcb[off] * var_z[off] * jcb[off];
    }
}

void lstm_tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                        int start_idx, int end_idx, int seq_len, int no,
                        int time_step, std::vector<float> &mu_a,
                        std::vector<float> &jcb, std::vector<float> &var_a) {
    for (int idx = start_idx; idx < end_idx; idx++) {
        int batch_idx = idx / no;
        int output_idx = idx % no;
        int off = output_idx + time_step * no + batch_idx * seq_len * no;
        float tmp = tanhf(mu_z[off]);
        mu_a[off] = tmp;
        jcb[off] = 1.0f - tmp * tmp;
        var_a[off] = jcb[off] * var_z[off] * jcb[off];
    }
}
void tlstm_cov_input_cell_states(std::vector<float> &var_ha,
                                 std::vector<float> &mu_w,
                                 std::vector<float> &jcb_i_ga,
                                 std::vector<float> &jcb_c_ga, int w_pos_i,
                                 int w_pos_c, int ni, int no, int start_idx,
                                 int end_idx, int seq_len, int time_step,
                                 std::vector<float> &cov_i_c) {
    int ni_c = ni + no;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int batch_idx = idx / no;
        int output_idx = idx % no;
        float sum = 0.0f;
        for (int j = 0; j < ni_c; j++) {
            int k = j + output_idx * ni_c;
            int m = j + time_step * ni_c + batch_idx * seq_len * ni_c;
            sum += mu_w[w_pos_i + k] * var_ha[m] * mu_w[w_pos_c + k];
        }
        int i = output_idx + time_step * no + batch_idx * seq_len * no;
        cov_i_c[i] = jcb_i_ga[i] * sum * jcb_c_ga[i];
    }
}

void tlstm_cell_state_mean_var(
    std::vector<float> &mu_f_ga, std::vector<float> &var_f_ga,
    std::vector<float> &mu_i_ga, std::vector<float> &var_i_ga,
    std::vector<float> &mu_c_ga, std::vector<float> &var_c_ga,
    std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev,
    std::vector<float> &cov_i_c, int no, int start_idx, int end_idx,
    int seq_len, int time_step, std::vector<float> &mu_c,
    std::vector<float> &var_c) {
    for (int idx = start_idx; idx < end_idx; idx++) {
        int batch_idx = idx / no;
        int output_idx = idx % no;
        int k = output_idx + time_step * no + batch_idx * no * seq_len;
        mu_c[k] =
            mu_f_ga[k] * mu_c_prev[k] + mu_i_ga[k] * mu_c_ga[k] + cov_i_c[k];
        var_c[k] = var_c_prev[k] * mu_f_ga[k] * mu_f_ga[k] +
                   var_c_prev[k] * var_f_ga[k] +
                   var_f_ga[k] * mu_c_prev[k] * mu_c_prev[k] +
                   var_c_ga[k] * mu_i_ga[k] * mu_i_ga[k] +
                   var_i_ga[k] * var_c_ga[k] +
                   var_i_ga[k] * mu_c_ga[k] * mu_c_ga[k] + powf(cov_i_c[k], 2) +
                   2 * cov_i_c[k] * mu_i_ga[k] * mu_c_ga[k];
    }
}

void tlstm_cov_output_tanh_cell_states(
    std::vector<float> &mu_w, std::vector<float> &var_ha,
    std::vector<float> &mu_c_prev, std::vector<float> &jcb_ca,
    std::vector<float> &jcb_f_ga, std::vector<float> &mu_i_ga,
    std::vector<float> &jcb_i_ga, std::vector<float> &mu_c_ga,
    std::vector<float> &jcb_c_ga, std::vector<float> &jcb_o_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int start_idx,
    int end_idx, int seq_len, int time_step, std::vector<float> &cov_tanh_c) {
    int ni_c = ni + no;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int batch_idx = idx / no;   // batch index
        int output_idx = idx % no;  // output index
        float sum_fo = 0.0f, sum_io = 0.0f, sum_oc = 0.0f;
        for (int j = 0; j < ni; j++) {
            int k = j + output_idx * ni_c;
            int m = j + time_step * ni_c + batch_idx * seq_len * ni_c;
            sum_fo += mu_w[w_pos_f + k] * var_ha[m] * mu_w[w_pos_o + k];
            sum_io += mu_w[w_pos_i + k] * var_ha[m] * mu_w[w_pos_o + k];
            sum_oc += mu_w[w_pos_c + k] * var_ha[m] * mu_w[w_pos_o + k];
        }
        int i = output_idx + time_step * no + batch_idx * no * seq_len;
        cov_tanh_c[i] =
            jcb_ca[i] * (jcb_o_ga[i] * sum_fo * jcb_f_ga[i] * mu_c_prev[i] +
                         jcb_o_ga[i] * sum_io * jcb_i_ga[i] * mu_c_ga[i] +
                         jcb_o_ga[i] * sum_oc * jcb_c_ga[i] * mu_i_ga[i]);
    }
}

void tlstm_hidden_state_mean_var(std::vector<float> &mu_o_ga,
                                 std::vector<float> &var_o_ga,
                                 std::vector<float> &mu_ca,
                                 std::vector<float> &var_ca,
                                 std::vector<float> &cov_o_tanh_c, int no,
                                 int start_idx, int end_idx, int seq_len,
                                 int time_step, std::vector<float> &mu_z,
                                 std::vector<float> &var_z) {
    for (int idx = start_idx; idx < end_idx; idx++) {
        int k = (idx % no) + time_step * no + (idx / no) * no * seq_len;
        mu_z[k] = mu_o_ga[k] * mu_ca[k] + cov_o_tanh_c[k];
        var_z[k] = var_ca[k] * mu_o_ga[k] * mu_o_ga[k] +
                   var_ca[k] * var_o_ga[k] + var_o_ga[k] * mu_ca[k] * mu_ca[k] +
                   powf(cov_o_tanh_c[k], 2) +
                   2 * cov_o_tanh_c[k] * mu_o_ga[k] * mu_ca[k];
    }
}

////////////////////////////////////////////////////////////////////////////////
// BACKWARD FREE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void tlstm_delta_mean_var_z(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_mu_out,
    std::vector<float> &delta_var_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int start_idx, int end_idx, int seq_len,
    int time_step, std::vector<float> &delta_mu,
    std::vector<float> &delta_var) {
    int ni_c = ni + no;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int batch_idx = idx / ni_c;   // batch index
        int output_idx = idx % ni_c;  // output index
        float sum_mf = 0, sum_mi = 0, sum_mc = 0, sum_mo = 0;
        float sum_var_z = 0;
        for (int j = 0; j < no; j++) {
            int k = j + batch_idx * seq_len * no + time_step * no;
            int delta_idx = batch_idx * no + j;

            float Czz_f = Jca[k] * mo_ga[k] * Jf_ga[k] *
                          mw[ni_c * j + output_idx + w_pos_f] * mc_prev[k];
            float Czz_i = Jca[k] * mo_ga[k] * Ji_ga[k] *
                          mw[ni_c * j + output_idx + w_pos_i] * mc_ga[k];
            float Czz_c = Jca[k] * mo_ga[k] * Jc_ga[k] *
                          mw[ni_c * j + output_idx + w_pos_c] * mi_ga[k];
            float Czz_o =
                Jo_ga[k] * mw[ni_c * j + output_idx + w_pos_o] * mca[k];

            sum_mf += Czz_f * delta_mu_out[delta_idx];
            sum_mi += Czz_i * delta_mu_out[delta_idx];
            sum_mc += Czz_c * delta_mu_out[delta_idx];
            sum_mo += Czz_o * delta_mu_out[delta_idx];
            sum_var_z += powf(Czz_f + Czz_i + Czz_c + Czz_o, 2) *
                         delta_var_out[delta_idx];
        }
        int m = batch_idx * ni_c + output_idx;
        delta_mu[m] = sum_mf + sum_mi + sum_mc + sum_mo;
        delta_var[m] = sum_var_z;
    }
}

void tlstm_delta_mean_var_w(
    std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_mu,
    std::vector<float> &delta_var, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int start_idx, int end_idx, int batch_size,
    int seq_len, int time_step, std::vector<float> &sum_mu_w_f,
    std::vector<float> &sum_var_w_f, std::vector<float> &sum_mu_w_i,
    std::vector<float> &sum_var_w_i, std::vector<float> &sum_mu_w_c,
    std::vector<float> &sum_var_w_c, std::vector<float> &sum_mu_w_o,
    std::vector<float> &sum_var_w_o) {
    int ni_c = ni + no;
    for (int idx = start_idx; idx < end_idx; idx++) {
        int row = idx / no;  // input index
        int col = idx % no;  // output index
        float s_mu_f = 0, s_var_f = 0, s_mu_i = 0, s_var_i = 0;
        float s_mu_c = 0, s_var_c = 0, s_mu_o = 0, s_var_o = 0;
        for (int x = 0; x < batch_size; x++) {
            int k = col + x * seq_len * no + time_step * no;
            int l = row + x * seq_len * ni_c + time_step * ni_c;
            int delta_idx = x * no + col;

            float Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k] * mha[l];
            s_mu_f += Cwa_f * delta_mu[delta_idx];
            s_var_f += Cwa_f * delta_var[delta_idx] * Cwa_f;

            float Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k] * mha[l];
            s_mu_i += Cwa_i * delta_mu[delta_idx];
            s_var_i += Cwa_i * delta_var[delta_idx] * Cwa_i;

            float Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k] * mha[l];
            s_mu_c += Cwa_c * delta_mu[delta_idx];
            s_var_c += Cwa_c * delta_var[delta_idx] * Cwa_c;

            float Cwa_o = Jo_ga[k] * mca[k] * mha[l];
            s_mu_o += Cwa_o * delta_mu[delta_idx];
            s_var_o += Cwa_o * delta_var[delta_idx] * Cwa_o;
        }
        int m = col * ni_c + row;
        sum_mu_w_f[m] += s_mu_f;
        sum_var_w_f[m] += s_var_f;
        sum_mu_w_i[m] += s_mu_i;
        sum_var_w_i[m] += s_var_i;
        sum_mu_w_c[m] += s_mu_c;
        sum_var_w_c[m] += s_var_c;
        sum_mu_w_o[m] += s_mu_o;
        sum_var_w_o[m] += s_var_o;
    }
}

void tlstm_delta_mean_var_b(
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
    std::vector<float> &Jo_ga, std::vector<float> &mc_prev,
    std::vector<float> &mca, std::vector<float> &Jc,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int no,
    int start_idx, int end_idx, int batch_size, int seq_len, int time_step,
    std::vector<float> &sum_mu_b_f, std::vector<float> &sum_var_b_f,
    std::vector<float> &sum_mu_b_i, std::vector<float> &sum_var_b_i,
    std::vector<float> &sum_mu_b_c, std::vector<float> &sum_var_b_c,
    std::vector<float> &sum_mu_b_o, std::vector<float> &sum_var_b_o) {
    for (int row = start_idx; row < end_idx; row++) {
        float s_mu_f = 0, s_var_f = 0, s_mu_i = 0, s_var_i = 0;
        float s_mu_c = 0, s_var_c = 0, s_mu_o = 0, s_var_o = 0;
        for (int x = 0; x < batch_size; x++) {
            int k = row + x * seq_len * no + time_step * no;
            int delta_idx = x * no + row;

            float Cwa_f = Jc[k] * Jf_ga[k] * mc_prev[k] * mo_ga[k];
            s_mu_f += Cwa_f * delta_mu[delta_idx];
            s_var_f += Cwa_f * delta_var[delta_idx] * Cwa_f;

            float Cwa_i = Jc[k] * Ji_ga[k] * mc_ga[k] * mo_ga[k];
            s_mu_i += Cwa_i * delta_mu[delta_idx];
            s_var_i += Cwa_i * delta_var[delta_idx] * Cwa_i;

            float Cwa_c = Jc[k] * Jc_ga[k] * mi_ga[k] * mo_ga[k];
            s_mu_c += Cwa_c * delta_mu[delta_idx];
            s_var_c += Cwa_c * delta_var[delta_idx] * Cwa_c;

            float Cwa_o = Jo_ga[k] * mca[k];
            s_mu_o += Cwa_o * delta_mu[delta_idx];
            s_var_o += Cwa_o * delta_var[delta_idx] * Cwa_o;
        }
        sum_mu_b_f[row] += s_mu_f;
        sum_var_b_f[row] += s_var_f;
        sum_mu_b_i[row] += s_mu_i;
        sum_var_b_i[row] += s_var_i;
        sum_mu_b_c[row] += s_mu_c;
        sum_var_b_c[row] += s_var_c;
        sum_mu_b_o[row] += s_mu_o;
        sum_var_b_o[row] += s_var_o;
    }
}

void tlstm_update_hidden_state_posterior(std::vector<float> &mu_h_prior,
                                         std::vector<float> &var_h_prior,
                                         std::vector<float> &delta_mu,
                                         std::vector<float> &delta_var,
                                         int start_idx, int end_idx,
                                         std::vector<float> &mu_h_prev,
                                         std::vector<float> &var_h_prev) {
    for (int i = start_idx; i < end_idx; i++) {
        mu_h_prev[i] = mu_h_prior[i] + delta_mu[i] * var_h_prior[i];
        var_h_prev[i] = (1.0f + delta_var[i] * var_h_prior[i]) * var_h_prior[i];
    }
}

void tlstm_update_cell_state_posterior(
    std::vector<float> &mu_c_prior, std::vector<float> &var_c_prior,
    std::vector<float> &jcb_ca, std::vector<float> &mu_o_ga,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int start_idx,
    int end_idx, std::vector<float> &mu_c_prev,
    std::vector<float> &var_c_prev) {
    for (int i = start_idx; i < end_idx; i++) {
        float tmp = var_c_prior[i] * jcb_ca[i] * mu_o_ga[i];
        mu_c_prev[i] = mu_c_prior[i] + tmp * delta_mu[i];
        var_c_prev[i] = var_c_prior[i] + tmp * delta_var[i] * tmp;
    }
}

////////////////////////////////////////////////////////////////////////////////
// TLSTM CLASS
////////////////////////////////////////////////////////////////////////////////

TLSTM::TLSTM(size_t input_size, size_t output_size, bool last_timestep,
             int seq_len, bool bias, float gain_w, float gain_b,
             std::string init_method, int device_idx)
    : gain_w(gain_w),
      gain_b(gain_b),
      init_method(init_method),
      last_timestep(last_timestep) {
    this->input_size = input_size;
    this->output_size = output_size;
    this->seq_len = seq_len;
    this->bias = bias;
    this->device_idx = device_idx;

    this->get_number_param();
    this->init_weight_bias();
    if (this->training) {
        this->allocate_param_delta();
    }
}

TLSTM::~TLSTM() {}

std::string TLSTM::get_layer_info() const {
    return "TLSTM(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string TLSTM::get_layer_name() const { return "TLSTM"; }

LayerType TLSTM::get_layer_type() const { return LayerType::TLSTM; }

int TLSTM::get_input_size() { return this->input_size * this->seq_len; }

int TLSTM::get_output_size() {
    if (this->last_timestep) {
        return this->output_size;
    }
    return this->output_size * this->seq_len;
}

int TLSTM::get_max_num_states() {
    int in_size = static_cast<int>(this->input_size) * this->seq_len;
    int out_size = static_cast<int>(this->output_size) * this->seq_len;
    return std::max(in_size, out_size);
}

void TLSTM::get_number_param() {
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

void TLSTM::init_weight_bias() {
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_lstm(this->init_method, this->gain_w, this->gain_b,
                              this->input_size, this->output_size,
                              this->num_weights, this->num_biases);
}

void TLSTM::forward(BaseHiddenStates &input_states,
                    BaseHiddenStates &output_states,
                    BaseTempStates &temp_states) {
    if (this->input_size != input_states.actual_size) {
        std::string message =
            "Input size mismatch: " + std::to_string(this->input_size) +
            " vs " + std::to_string(input_states.actual_size);
        LOG(LogLevel::ERROR, message);
    }

    int batch_size = input_states.block_size;
    int seq_len = this->seq_len;
    int ni = this->input_size;
    int no = this->output_size;
    int ni_c = ni + no;

    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_states.set_num_states(batch_size * seq_len * no,
                                         batch_size * seq_len * ni);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.seq_len = this->seq_len;
    output_states.actual_size = this->output_size;

    int end_chunk = no * batch_size;
    if (seq_len == 1 && batch_size == 1) {
        lstm_states.mu_h_prev = lstm_states.mu_h_prior;
        lstm_states.var_h_prev = lstm_states.var_h_prior;
        lstm_states.mu_c_prev = lstm_states.mu_c_prior;
        lstm_states.var_c_prev = lstm_states.var_c_prior;

    } else {
        lstm_states.reset_prev_states();
    }

    for (int t = 0; t < seq_len; t++) {
        tlstm_cat_activations_and_prev_states(
            input_states.mu_a, lstm_states.mu_h_prev, ni, no, batch_size,
            seq_len, t, lstm_states.mu_ha);
        tlstm_cat_activations_and_prev_states(
            input_states.var_a, lstm_states.var_h_prev, ni, no, batch_size,
            seq_len, t, lstm_states.var_ha);

        // Forget gate
        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               lstm_states.mu_ha, lstm_states.var_ha, s, e,
                               ni_c, no, batch_size, seq_len, t, this->bias,
                               this->w_pos_f, this->b_pos_f,
                               lstm_states.mu_f_ga, lstm_states.var_f_ga);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            lstm_sigmoid_mean_var(lstm_states.mu_f_ga, lstm_states.var_f_ga, s,
                                  e, seq_len, no, t, lstm_states.mu_f_ga,
                                  lstm_states.jcb_f_ga, lstm_states.var_f_ga);
        });

        // Input gate
        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               lstm_states.mu_ha, lstm_states.var_ha, s, e,
                               ni_c, no, batch_size, seq_len, t, this->bias,
                               this->w_pos_i, this->b_pos_i,
                               lstm_states.mu_i_ga, lstm_states.var_i_ga);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            lstm_sigmoid_mean_var(lstm_states.mu_i_ga, lstm_states.var_i_ga, s,
                                  e, seq_len, no, t, lstm_states.mu_i_ga,
                                  lstm_states.jcb_i_ga, lstm_states.var_i_ga);
        });

        // Cell state gate
        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               lstm_states.mu_ha, lstm_states.var_ha, s, e,
                               ni_c, no, batch_size, seq_len, t, this->bias,
                               this->w_pos_c, this->b_pos_c,
                               lstm_states.mu_c_ga, lstm_states.var_c_ga);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            lstm_tanh_mean_var(lstm_states.mu_c_ga, lstm_states.var_c_ga, s, e,
                               seq_len, no, t, lstm_states.mu_c_ga,
                               lstm_states.jcb_c_ga, lstm_states.var_c_ga);
        });

        // Output gate
        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_fwd_mean_var(this->mu_w, this->var_w, this->mu_b, this->var_b,
                               lstm_states.mu_ha, lstm_states.var_ha, s, e,
                               ni_c, no, batch_size, seq_len, t, this->bias,
                               this->w_pos_o, this->b_pos_o,
                               lstm_states.mu_o_ga, lstm_states.var_o_ga);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            lstm_sigmoid_mean_var(lstm_states.mu_o_ga, lstm_states.var_o_ga, s,
                                  e, seq_len, no, t, lstm_states.mu_o_ga,
                                  lstm_states.jcb_o_ga, lstm_states.var_o_ga);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_cov_input_cell_states(
                lstm_states.var_ha, this->mu_w, lstm_states.jcb_i_ga,
                lstm_states.jcb_c_ga, this->w_pos_i, this->w_pos_c, ni, no, s,
                e, seq_len, t, lstm_states.cov_i_c);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_cell_state_mean_var(
                lstm_states.mu_f_ga, lstm_states.var_f_ga, lstm_states.mu_i_ga,
                lstm_states.var_i_ga, lstm_states.mu_c_ga, lstm_states.var_c_ga,
                lstm_states.mu_c_prev, lstm_states.var_c_prev,
                lstm_states.cov_i_c, no, s, e, seq_len, t, lstm_states.mu_c,
                lstm_states.var_c);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            lstm_tanh_mean_var(lstm_states.mu_c, lstm_states.var_c, s, e,
                               seq_len, no, t, lstm_states.mu_ca,
                               lstm_states.jcb_ca, lstm_states.var_ca);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_cov_output_tanh_cell_states(
                this->mu_w, lstm_states.var_ha, lstm_states.mu_c_prev,
                lstm_states.jcb_ca, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.jcb_o_ga, this->w_pos_f, this->w_pos_i,
                this->w_pos_c, this->w_pos_o, ni, no, s, e, seq_len, t,
                lstm_states.cov_o_tanh_c);
        });

        parallel_for(end_chunk, this->num_threads, [&](int s, int e) {
            tlstm_hidden_state_mean_var(
                lstm_states.mu_o_ga, lstm_states.var_o_ga, lstm_states.mu_ca,
                lstm_states.var_ca, lstm_states.cov_o_tanh_c, no, s, e, seq_len,
                t, output_states.mu_a, output_states.var_a);
        });
        if (t < seq_len - 1) {
            for (int b = 0; b < batch_size; b++) {
                for (int z = 0; z < no; z++) {
                    int prev_idx = b * seq_len * no + (t + 1) * no + z;
                    int curr_idx = b * seq_len * no + t * no + z;
                    lstm_states.mu_h_prev[prev_idx] =
                        output_states.mu_a[curr_idx];
                    lstm_states.var_h_prev[prev_idx] =
                        output_states.var_a[curr_idx];
                    lstm_states.mu_c_prev[prev_idx] =
                        lstm_states.mu_c[curr_idx];
                    lstm_states.var_c_prev[prev_idx] =
                        lstm_states.var_c[curr_idx];
                }
            }
        }
    }

    if (seq_len == 1 && batch_size == 1) {
        for (int b = 0; b < batch_size; b++) {
            int src = b * seq_len * no + (seq_len - 1) * no;
            int dst = b * no;
            for (int z = 0; z < no; z++) {
                lstm_states.mu_h_prior[dst + z] = output_states.mu_a[src + z];
                lstm_states.var_h_prior[dst + z] = output_states.var_a[src + z];
                lstm_states.mu_c_prior[dst + z] = lstm_states.mu_c[src + z];
                lstm_states.var_c_prior[dst + z] = lstm_states.var_c[src + z];
            }
        }
    }

    if (this->last_timestep) {
        for (int b = 0; b < batch_size; b++) {
            int src = b * seq_len * no + (seq_len - 1) * no;
            int dst = b * no;
            for (int z = 0; z < no; z++) {
                output_states.mu_a[dst + z] = output_states.mu_a[src + z];
                output_states.var_a[dst + z] = output_states.var_a[src + z];
            }
        }
        output_states.seq_len = 1;
    }

    if (this->training) {
        this->storing_states_for_training(input_states, output_states);
    }
}

void TLSTM::backward(BaseDeltaStates &input_delta_states,
                     BaseDeltaStates &output_delta_states,
                     BaseTempStates &temp_states, bool state_udapte) {
    int batch_size = input_delta_states.block_size;
    int seq_len = this->seq_len;
    int ni = this->input_size;
    int no = this->output_size;
    int ni_c = ni + no;

    // Recurrent deltas from t+1
    std::vector<float> delta_rec_mu(batch_size * no, 0.0f);
    std::vector<float> delta_rec_var(batch_size * no, 0.0f);
    std::vector<float> combined_delta_mu(batch_size * no, 0.0f);
    std::vector<float> combined_delta_var(batch_size * no, 0.0f);

    // Temp buffer for full [x(t), h(t-1)]
    std::vector<float> delta_xh_mu(batch_size * ni_c, 0.0f);
    std::vector<float> delta_xh_var(batch_size * ni_c, 0.0f);

    std::vector<float> &delta_mu_buf = input_delta_states.delta_mu;
    std::vector<float> &delta_var_buf = input_delta_states.delta_var;

    // Accumulators for weight and bias deltas (raw sums, no Sw/Sb yet)
    int w_size = ni_c * no;
    std::vector<float> sum_mu_w_f(w_size, 0.0f), sum_var_w_f(w_size, 0.0f);
    std::vector<float> sum_mu_w_i(w_size, 0.0f), sum_var_w_i(w_size, 0.0f);
    std::vector<float> sum_mu_w_c(w_size, 0.0f), sum_var_w_c(w_size, 0.0f);
    std::vector<float> sum_mu_w_o(w_size, 0.0f), sum_var_w_o(w_size, 0.0f);

    std::vector<float> sum_mu_b_f(no, 0.0f), sum_var_b_f(no, 0.0f);
    std::vector<float> sum_mu_b_i(no, 0.0f), sum_var_b_i(no, 0.0f);
    std::vector<float> sum_mu_b_c(no, 0.0f), sum_var_b_c(no, 0.0f);
    std::vector<float> sum_mu_b_o(no, 0.0f), sum_var_b_o(no, 0.0f);

    if (seq_len == 1 && batch_size == 1) {
        int total_prior = batch_size * no;
        parallel_for(total_prior, this->num_threads, [&](int s, int e) {
            tlstm_update_hidden_state_posterior(
                this->lstm_states.mu_h_prior, this->lstm_states.var_h_prior,
                delta_rec_mu, delta_rec_var, s, e, this->lstm_states.mu_h_prior,
                this->lstm_states.var_h_prior);
        });
        parallel_for(total_prior, this->num_threads, [&](int s, int e) {
            tlstm_update_cell_state_posterior(
                this->lstm_states.mu_c_prior, this->lstm_states.var_c_prior,
                this->lstm_states.jcb_ca, this->lstm_states.mu_o_ga,
                delta_rec_mu, delta_rec_var, s, e, this->lstm_states.mu_c_prior,
                this->lstm_states.var_c_prior);
        });
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        // Combine incoming + recurrent deltas
        bool has_direct = !this->last_timestep || (t == seq_len - 1);
        for (int b = 0; b < batch_size; b++) {
            int flat = b * no;
            int buf_off =
                this->last_timestep ? flat : b * seq_len * no + t * no;
            for (int j = 0; j < no; j++) {
                combined_delta_mu[flat + j] = delta_rec_mu[flat + j];
                combined_delta_var[flat + j] = delta_rec_var[flat + j];
                if (has_direct) {
                    combined_delta_mu[flat + j] += delta_mu_buf[buf_off + j];
                    combined_delta_var[flat + j] += delta_var_buf[buf_off + j];
                }
            }
        }

        if (param_update) {
            parallel_for(ni_c * no, this->num_threads, [&](int s, int e) {
                tlstm_delta_mean_var_w(
                    lstm_states.mu_ha, lstm_states.jcb_f_ga,
                    lstm_states.mu_i_ga, lstm_states.jcb_i_ga,
                    lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                    lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                    lstm_states.mu_c_prev, lstm_states.mu_ca,
                    lstm_states.jcb_ca, combined_delta_mu, combined_delta_var,
                    this->w_pos_f, this->w_pos_i, this->w_pos_c, this->w_pos_o,
                    no, ni, s, e, batch_size, seq_len, t, sum_mu_w_f,
                    sum_var_w_f, sum_mu_w_i, sum_var_w_i, sum_mu_w_c,
                    sum_var_w_c, sum_mu_w_o, sum_var_w_o);
            });

            if (this->bias) {
                parallel_for(no, this->num_threads, [&](int s, int e) {
                    tlstm_delta_mean_var_b(
                        lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                        lstm_states.jcb_i_ga, lstm_states.mu_c_ga,
                        lstm_states.jcb_c_ga, lstm_states.mu_o_ga,
                        lstm_states.jcb_o_ga, lstm_states.mu_c_prev,
                        lstm_states.mu_ca, lstm_states.jcb_ca,
                        combined_delta_mu, combined_delta_var, no, s, e,
                        batch_size, seq_len, t, sum_mu_b_f, sum_var_b_f,
                        sum_mu_b_i, sum_var_b_i, sum_mu_b_c, sum_var_b_c,
                        sum_mu_b_o, sum_var_b_o);
                });
            }
        }

        std::fill(delta_xh_mu.begin(), delta_xh_mu.end(), 0.0f);
        std::fill(delta_xh_var.begin(), delta_xh_var.end(), 0.0f);

        parallel_for(batch_size * ni_c, this->num_threads, [&](int s, int e) {
            tlstm_delta_mean_var_z(
                this->mu_w, lstm_states.jcb_f_ga, lstm_states.mu_i_ga,
                lstm_states.jcb_i_ga, lstm_states.mu_c_ga, lstm_states.jcb_c_ga,
                lstm_states.mu_o_ga, lstm_states.jcb_o_ga,
                lstm_states.mu_c_prev, lstm_states.mu_ca, lstm_states.jcb_ca,
                combined_delta_mu, combined_delta_var, this->w_pos_f,
                this->w_pos_i, this->w_pos_c, this->w_pos_o, no, ni, s, e,
                seq_len, t, delta_xh_mu, delta_xh_var);
        });

        for (int b = 0; b < batch_size; b++) {
            for (int j = 0; j < ni; j++) {
                int idx = b * seq_len * ni + t * ni + j;
                output_delta_states.delta_mu[idx] = delta_xh_mu[b * ni_c + j];
                output_delta_states.delta_var[idx] = delta_xh_var[b * ni_c + j];
            }
            for (int j = 0; j < no; j++) {
                delta_rec_mu[b * no + j] = delta_xh_mu[b * ni_c + ni + j];
                delta_rec_var[b * no + j] = delta_xh_var[b * ni_c + ni + j];
            }
        }
    }

    // Multiply accumulated sums by var_w/var_b to get final deltas
    if (param_update) {
        parallel_for(w_size, this->num_threads, [&](int s, int e) {
            for (int m = s; m < e; m++) {
                this->delta_mu_w[m + this->w_pos_f] =
                    sum_mu_w_f[m] * this->var_w[m + this->w_pos_f];
                this->delta_var_w[m + this->w_pos_f] =
                    this->var_w[m + this->w_pos_f] * sum_var_w_f[m] *
                    this->var_w[m + this->w_pos_f];

                this->delta_mu_w[m + this->w_pos_i] =
                    sum_mu_w_i[m] * this->var_w[m + this->w_pos_i];
                this->delta_var_w[m + this->w_pos_i] =
                    this->var_w[m + this->w_pos_i] * sum_var_w_i[m] *
                    this->var_w[m + this->w_pos_i];

                this->delta_mu_w[m + this->w_pos_c] =
                    sum_mu_w_c[m] * this->var_w[m + this->w_pos_c];
                this->delta_var_w[m + this->w_pos_c] =
                    this->var_w[m + this->w_pos_c] * sum_var_w_c[m] *
                    this->var_w[m + this->w_pos_c];

                this->delta_mu_w[m + this->w_pos_o] =
                    sum_mu_w_o[m] * this->var_w[m + this->w_pos_o];
                this->delta_var_w[m + this->w_pos_o] =
                    this->var_w[m + this->w_pos_o] * sum_var_w_o[m] *
                    this->var_w[m + this->w_pos_o];
            }
        });

        if (this->bias) {
            parallel_for(no, this->num_threads, [&](int s, int e) {
                for (int r = s; r < e; r++) {
                    this->delta_mu_b[r + this->b_pos_f] =
                        sum_mu_b_f[r] * this->var_b[r + this->b_pos_f];
                    this->delta_var_b[r + this->b_pos_f] =
                        this->var_b[r + this->b_pos_f] * sum_var_b_f[r] *
                        this->var_b[r + this->b_pos_f];

                    this->delta_mu_b[r + this->b_pos_i] =
                        sum_mu_b_i[r] * this->var_b[r + this->b_pos_i];
                    this->delta_var_b[r + this->b_pos_i] =
                        this->var_b[r + this->b_pos_i] * sum_var_b_i[r] *
                        this->var_b[r + this->b_pos_i];

                    this->delta_mu_b[r + this->b_pos_c] =
                        sum_mu_b_c[r] * this->var_b[r + this->b_pos_c];
                    this->delta_var_b[r + this->b_pos_c] =
                        this->var_b[r + this->b_pos_c] * sum_var_b_c[r] *
                        this->var_b[r + this->b_pos_c];

                    this->delta_mu_b[r + this->b_pos_o] =
                        sum_mu_b_o[r] * this->var_b[r + this->b_pos_o];
                    this->delta_var_b[r + this->b_pos_o] =
                        this->var_b[r + this->b_pos_o] * sum_var_b_o[r] *
                        this->var_b[r + this->b_pos_o];
                }
            });
        }
    }
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>>
TLSTM::get_LSTM_states() const {
    return std::make_tuple(lstm_states.mu_h_prior, lstm_states.var_h_prior,
                           lstm_states.mu_c_prior, lstm_states.var_c_prior);
}

void TLSTM::set_LSTM_states(const std::vector<float> &mu_h,
                            const std::vector<float> &var_h,
                            const std::vector<float> &mu_c,
                            const std::vector<float> &var_c) {
    this->lstm_states.mu_h_prior = mu_h;
    this->lstm_states.var_h_prior = var_h;
    this->lstm_states.mu_c_prior = mu_c;
    this->lstm_states.var_c_prior = var_c;

    this->lstm_states.mu_h_prev = mu_h;
    this->lstm_states.var_h_prev = var_h;
    this->lstm_states.mu_c_prev = mu_c;
    this->lstm_states.var_c_prev = var_c;
}

void TLSTM::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
    }
    if (this->training) {
        this->allocate_param_delta();
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> TLSTM::to_cuda(int device_idx) {
    this->device = "cuda";
    this->device_idx = device_idx;
    auto cuda_layer = std::make_unique<TLSTMCuda>(
        this->input_size, this->output_size, this->last_timestep, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method,
        this->device_idx);

    auto base_cuda = dynamic_cast<BaseLayerCuda *>(cuda_layer.get());
    base_cuda->copy_params_from(*this);

    return cuda_layer;
}
#endif


#include "../include/custom_logger.h"
#include "../include/lstm_layer.h"
#include "../include/param_init.h"
#include "../include/tlstm_layer.h"
#include "../include/tlstm_layer_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
// FORWARD KERNELS
////////////////////////////////////////////////////////////////////////////////

__global__ void tlstm_cat_cuda(const float *a, const float *b, int n, int m,
                               int batch_size, int seq_len, int time_step,
                               float *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    int ni_c = n + m;
    int a_off = idx * seq_len * n + time_step * n;
    int b_off = idx * seq_len * m + time_step * m;
    int c_off = idx * seq_len * ni_c + time_step * ni_c;
    for (int i = 0; i < n; i++) {
        c[c_off + i] = a[a_off + i];
    }
    for (int j = 0; j < m; j++) {
        c[c_off + n + j] = b[b_off + j];
    }
}

__global__ void tlstm_fwd_mean_var_cuda(const float *mu_w, const float *var_w,
                                        const float *mu_b, const float *var_b,
                                        const float *mu_a, const float *var_a,
                                        int input_size, int output_size,
                                        int batch_size, int seq_len,
                                        int time_step, bool bias, int w_pos,
                                        int b_pos, float *mu_z, float *var_z) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output_size || col >= batch_size) return;

    int in_off = col * seq_len * input_size + time_step * input_size;
    int out_off = col * seq_len * output_size + time_step * output_size;

    float sum_mu = 0.0f, sum_var = 0.0f;
    for (int j = 0; j < input_size; j++) {
        float ma = mu_a[in_off + j];
        float va = var_a[in_off + j];
        float mw = mu_w[row * input_size + j + w_pos];
        float vw = var_w[row * input_size + j + w_pos];
        sum_mu += mw * ma;
        sum_var += (mw * mw + vw) * va + vw * ma * ma;
    }
    if (bias) {
        mu_z[out_off + row] = sum_mu + mu_b[row + b_pos];
        var_z[out_off + row] = sum_var + var_b[row + b_pos];
    } else {
        mu_z[out_off + row] = sum_mu;
        var_z[out_off + row] = sum_var;
    }
}

__global__ void tlstm_sigmoid_cuda(float *mu_z, float *var_z, int no,
                                   int seq_len, int time_step, int total,
                                   float *mu_a, float *jcb, float *var_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int off = (idx / no) * seq_len * no + time_step * no + (idx % no);
    float tmp = 1.0f / (1.0f + expf(-mu_z[off]));
    mu_a[off] = tmp;
    jcb[off] = tmp * (1.0f - tmp);
    var_a[off] = jcb[off] * var_z[off] * jcb[off];
}

__global__ void tlstm_tanh_cuda(float *mu_z, float *var_z, int no, int seq_len,
                                int time_step, int total, float *mu_a,
                                float *jcb, float *var_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int off = (idx / no) * seq_len * no + time_step * no + (idx % no);
    float tmp = tanhf(mu_z[off]);
    mu_a[off] = tmp;
    jcb[off] = 1.0f - tmp * tmp;
    var_a[off] = jcb[off] * var_z[off] * jcb[off];
}

__global__ void tlstm_cov_input_cell_cuda(
    const float *var_ha, const float *mu_w, const float *jcb_i_ga,
    const float *jcb_c_ga, int w_pos_i, int w_pos_c, int ni, int no,
    int seq_len, int time_step, int total, float *cov_i_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ni_c = ni + no;
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

__global__ void tlstm_cell_state_cuda(
    const float *mu_f_ga, const float *var_f_ga, const float *mu_i_ga,
    const float *var_i_ga, const float *mu_c_ga, const float *var_c_ga,
    const float *mu_c_prev, const float *var_c_prev, const float *cov_i_c,
    int no, int seq_len, int time_step, int total, float *mu_c, float *var_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int batch_idx = idx / no;
    int output_idx = idx % no;
    int k = output_idx + time_step * no + batch_idx * no * seq_len;

    mu_c[k] = mu_f_ga[k] * mu_c_prev[k] + mu_i_ga[k] * mu_c_ga[k] + cov_i_c[k];
    var_c[k] =
        var_c_prev[k] * mu_f_ga[k] * mu_f_ga[k] + var_c_prev[k] * var_f_ga[k] +
        var_f_ga[k] * mu_c_prev[k] * mu_c_prev[k] +
        var_c_ga[k] * mu_i_ga[k] * mu_i_ga[k] + var_i_ga[k] * var_c_ga[k] +
        var_i_ga[k] * mu_c_ga[k] * mu_c_ga[k] + cov_i_c[k] * cov_i_c[k] +
        2.0f * cov_i_c[k] * mu_i_ga[k] * mu_c_ga[k];
}

__global__ void tlstm_cov_output_tanh_cell_states_cuda(
    const float *mu_w, const float *var_ha, const float *mu_c_prev,
    const float *jcb_ca, const float *jcb_f_ga, const float *mu_i_ga,
    const float *jcb_i_ga, const float *mu_c_ga, const float *jcb_c_ga,
    const float *jcb_o_ga, int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o,
    int ni, int no, int seq_len, int time_step, int total, float *cov_tanh_c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ni_c = ni + no;
    int batch_idx = idx / no;
    int output_idx = idx % no;
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

__global__ void tlstm_hidden_state_mean_var_cuda(
    const float *mu_o_ga, const float *var_o_ga, const float *mu_ca,
    const float *var_ca, const float *cov_o_tanh_c, int no, int seq_len,
    int time_step, int total, float *mu_z, float *var_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int k = (idx % no) + time_step * no + (idx / no) * no * seq_len;
    mu_z[k] = mu_o_ga[k] * mu_ca[k] + cov_o_tanh_c[k];
    var_z[k] = var_ca[k] * mu_o_ga[k] * mu_o_ga[k] + var_ca[k] * var_o_ga[k] +
               var_o_ga[k] * mu_ca[k] * mu_ca[k] +
               cov_o_tanh_c[k] * cov_o_tanh_c[k] +
               2.0f * cov_o_tanh_c[k] * mu_o_ga[k] * mu_ca[k];
}

__global__ void tlstm_copy_prev_states_cuda(
    const float *mu_h, const float *var_h, const float *mu_c,
    const float *var_c, int no, int seq_len, int time_step, int total,
    float *mu_h_prev, float *var_h_prev, float *mu_c_prev, float *var_c_prev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int batch_idx = idx / no;
    int output_idx = idx % no;
    int curr = batch_idx * seq_len * no + time_step * no + output_idx;
    int next = batch_idx * seq_len * no + (time_step + 1) * no + output_idx;
    mu_h_prev[next] = mu_h[curr];
    var_h_prev[next] = var_h[curr];
    mu_c_prev[next] = mu_c[curr];
    var_c_prev[next] = var_c[curr];
}

__global__ void tlstm_extract_last_timestep_cuda(const float *src, int no,
                                                 int seq_len, int total,
                                                 float *dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int batch_idx = idx / no;
    int output_idx = idx % no;
    int src_idx = batch_idx * seq_len * no + (seq_len - 1) * no + output_idx;
    dst[idx] = src[src_idx];
}

////////////////////////////////////////////////////////////////////////////////
// BACKWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
__global__ void tlstm_combine_delta_cuda(
    const float *delta_rec_mu, const float *delta_rec_var,
    const float *delta_mu_buf, const float *delta_var_buf, int no, int seq_len,
    int time_step, bool has_direct, bool last_timestep, int total,
    float *combined_mu, float *combined_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int batch_idx = idx / no;
    int output_idx = idx % no;
    combined_mu[idx] = delta_rec_mu[idx];
    combined_var[idx] = delta_rec_var[idx];
    if (has_direct) {
        int buf_off = last_timestep ? idx
                                    : batch_idx * seq_len * no +
                                          time_step * no + output_idx;
        combined_mu[idx] += delta_mu_buf[buf_off];
        combined_var[idx] += delta_var_buf[buf_off];
    }
}

__global__ void tlstm_delta_z_cuda(
    const float *mw, const float *Jf_ga, const float *mi_ga, const float *Ji_ga,
    const float *mc_ga, const float *Jc_ga, const float *mo_ga,
    const float *Jo_ga, const float *mc_prev, const float *mca,
    const float *Jca, const float *delta_mu_out, const float *delta_var_out,
    int w_pos_f, int w_pos_i, int w_pos_c, int w_pos_o, int no, int ni,
    int seq_len, int time_step, int total, float *delta_mu, float *delta_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ni_c = ni + no;
    int batch_idx = idx / ni_c;
    int output_idx = idx % ni_c;
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
        float Czz_o = Jo_ga[k] * mw[ni_c * j + output_idx + w_pos_o] * mca[k];

        sum_mf += Czz_f * delta_mu_out[delta_idx];
        sum_mi += Czz_i * delta_mu_out[delta_idx];
        sum_mc += Czz_c * delta_mu_out[delta_idx];
        sum_mo += Czz_o * delta_mu_out[delta_idx];
        float tmp = Czz_f + Czz_i + Czz_c + Czz_o;
        sum_var_z += tmp * tmp * delta_var_out[delta_idx];
    }
    int m = batch_idx * ni_c + output_idx;
    delta_mu[m] = sum_mf + sum_mi + sum_mc + sum_mo;
    delta_var[m] = sum_var_z;
}

__global__ void tlstm_delta_w_cuda(const float *mha, const float *Jf_ga,
                                   const float *mi_ga, const float *Ji_ga,
                                   const float *mc_ga, const float *Jc_ga,
                                   const float *mo_ga, const float *Jo_ga,
                                   const float *mc_prev, const float *mca,
                                   const float *Jc, const float *delta_mu,
                                   const float *delta_var, int w_pos_f,
                                   int w_pos_i, int w_pos_c, int w_pos_o,
                                   int no, int ni, int batch_size, int seq_len,
                                   int time_step, int total, float *sum_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int ni_c = ni + no;
    int row = idx / no;  // output index
    int col = idx % no;  // input index
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

    // Layout: [gate][mu/var][ni_c * no]
    // gate order: f=0, i=1, c=2, o=3
    int m = col * ni_c + row;
    int stride = ni_c * no;
    sum_w[0 * 2 * stride + 0 * stride + m] += s_mu_f;
    sum_w[0 * 2 * stride + 1 * stride + m] += s_var_f;
    sum_w[1 * 2 * stride + 0 * stride + m] += s_mu_i;
    sum_w[1 * 2 * stride + 1 * stride + m] += s_var_i;
    sum_w[2 * 2 * stride + 0 * stride + m] += s_mu_c;
    sum_w[2 * 2 * stride + 1 * stride + m] += s_var_c;
    sum_w[3 * 2 * stride + 0 * stride + m] += s_mu_o;
    sum_w[3 * 2 * stride + 1 * stride + m] += s_var_o;
}

__global__ void tlstm_delta_b_cuda(
    const float *Jf_ga, const float *mi_ga, const float *Ji_ga,
    const float *mc_ga, const float *Jc_ga, const float *mo_ga,
    const float *Jo_ga, const float *mc_prev, const float *mca, const float *Jc,
    const float *delta_mu, const float *delta_var, int no, int batch_size,
    int seq_len, int time_step, float *sum_b) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= no) return;
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

    // Layout: [gate][mu/var][no]
    sum_b[0 * 2 * no + 0 * no + row] += s_mu_f;
    sum_b[0 * 2 * no + 1 * no + row] += s_var_f;
    sum_b[1 * 2 * no + 0 * no + row] += s_mu_i;
    sum_b[1 * 2 * no + 1 * no + row] += s_var_i;
    sum_b[2 * 2 * no + 0 * no + row] += s_mu_c;
    sum_b[2 * 2 * no + 1 * no + row] += s_var_c;
    sum_b[3 * 2 * no + 0 * no + row] += s_mu_o;
    sum_b[3 * 2 * no + 1 * no + row] += s_var_o;
}

__global__ void tlstm_split_delta_xh_cuda(
    const float *delta_xh_mu, const float *delta_xh_var, int ni, int no,
    int seq_len, int time_step, int batch_size, float *out_delta_mu,
    float *out_delta_var, float *rec_delta_mu, float *rec_delta_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ni_c = ni + no;
    if (idx >= batch_size * ni_c) return;
    int b = idx / ni_c;
    int j = idx % ni_c;
    if (j < ni) {
        int dst = b * seq_len * ni + time_step * ni + j;
        out_delta_mu[dst] = delta_xh_mu[b * ni_c + j];
        out_delta_var[dst] = delta_xh_var[b * ni_c + j];
    } else {
        int h_idx = j - ni;
        rec_delta_mu[b * no + h_idx] = delta_xh_mu[b * ni_c + j];
        rec_delta_var[b * no + h_idx] = delta_xh_var[b * ni_c + j];
    }
}

__global__ void tlstm_scale_delta_w_cuda(const float *sum_w, const float *var_w,
                                         int w_pos_f, int w_pos_i, int w_pos_c,
                                         int w_pos_o, int w_size,
                                         float *delta_mu_w,
                                         float *delta_var_w) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= w_size) return;
    int stride = w_size;
    for (int g = 0; g < 4; g++) {
        int w_pos;
        if (g == 0)
            w_pos = w_pos_f;
        else if (g == 1)
            w_pos = w_pos_i;
        else if (g == 2)
            w_pos = w_pos_c;
        else
            w_pos = w_pos_o;

        float sm = sum_w[g * 2 * stride + 0 * stride + m];
        float sv = sum_w[g * 2 * stride + 1 * stride + m];
        float vw = var_w[m + w_pos];
        delta_mu_w[m + w_pos] = sm * vw;
        delta_var_w[m + w_pos] = vw * sv * vw;
    }
}

__global__ void tlstm_scale_delta_b_cuda(const float *sum_b, const float *var_b,
                                         int b_pos_f, int b_pos_i, int b_pos_c,
                                         int b_pos_o, int no, float *delta_mu_b,
                                         float *delta_var_b) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= no) return;
    for (int g = 0; g < 4; g++) {
        int b_pos;
        if (g == 0)
            b_pos = b_pos_f;
        else if (g == 1)
            b_pos = b_pos_i;
        else if (g == 2)
            b_pos = b_pos_c;
        else
            b_pos = b_pos_o;

        float sm = sum_b[g * 2 * no + 0 * no + r];
        float sv = sum_b[g * 2 * no + 1 * no + r];
        float vb = var_b[r + b_pos];
        delta_mu_b[r + b_pos] = sm * vb;
        delta_var_b[r + b_pos] = vb * sv * vb;
    }
}

__global__ void tlstm_update_hidden_posterior_cuda(
    const float *mu_h_prior, const float *var_h_prior, const float *delta_mu,
    const float *delta_var, int total, float *mu_h_prev, float *var_h_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    mu_h_prev[i] = mu_h_prior[i] + delta_mu[i] * var_h_prior[i];
    var_h_prev[i] = (1.0f + delta_var[i] * var_h_prior[i]) * var_h_prior[i];
}

__global__ void tlstm_update_cell_posterior_cuda(
    const float *mu_c_prior, const float *var_c_prior, const float *jcb_ca,
    const float *mu_o_ga, const float *delta_mu, const float *delta_var,
    int total, float *mu_c_prev, float *var_c_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float tmp = var_c_prior[i] * jcb_ca[i] * mu_o_ga[i];
    mu_c_prev[i] = mu_c_prior[i] + tmp * delta_mu[i];
    var_c_prev[i] = var_c_prior[i] + tmp * delta_var[i] * tmp;
}

////////////////////////////////////////////////////////////////////////////////
// TLSTMCuda CLASS
////////////////////////////////////////////////////////////////////////////////

TLSTMCuda::TLSTMCuda(size_t input_size, size_t output_size, bool last_timestep,
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
    if (this->training) {
        this->allocate_param_delta();
    }
}

TLSTMCuda::~TLSTMCuda() { this->deallocate_bwd_buffers(); }

std::string TLSTMCuda::get_layer_info() const {
    return "TLSTM(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string TLSTMCuda::get_layer_name() const { return "TLSTMCuda"; }

LayerType TLSTMCuda::get_layer_type() const { return LayerType::TLSTM; }

int TLSTMCuda::get_input_size() { return this->input_size * this->seq_len; }

int TLSTMCuda::get_output_size() {
    if (this->last_timestep) return this->output_size;
    return this->output_size * this->seq_len;
}

int TLSTMCuda::get_max_num_states() {
    int in_size = static_cast<int>(this->input_size) * this->seq_len;
    int out_size = static_cast<int>(this->output_size) * this->seq_len;
    return std::max(in_size, out_size);
}

void TLSTMCuda::get_number_param() {
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

void TLSTMCuda::init_weight_bias() {
    std::tie(this->mu_w, this->var_w, this->mu_b, this->var_b) =
        init_weight_bias_lstm(this->init_method, this->gain_w, this->gain_b,
                              this->input_size, this->output_size,
                              this->num_weights, this->num_biases);
    this->allocate_param_memory();
    this->params_to_device();
}

void TLSTMCuda::allocate_bwd_buffers(int batch_size) {
    this->deallocate_bwd_buffers();

    int no = this->output_size;
    int ni = this->input_size;
    int ni_c = ni + no;

    cudaSetDevice(this->device_idx);
    cudaMalloc(&d_buf_rec_mu, batch_size * no * sizeof(float));
    cudaMalloc(&d_buf_rec_var, batch_size * no * sizeof(float));
    cudaMalloc(&d_buf_combined_mu, batch_size * no * sizeof(float));
    cudaMalloc(&d_buf_combined_var, batch_size * no * sizeof(float));
    cudaMalloc(&d_buf_xh_mu, batch_size * ni_c * sizeof(float));
    cudaMalloc(&d_buf_xh_var, batch_size * ni_c * sizeof(float));

    // 4 gates * 2 (mu,var) * ni_c * no
    cudaMalloc(&d_buf_sum_w, 8 * ni_c * no * sizeof(float));
    // 4 gates * 2 (mu,var) * no
    cudaMalloc(&d_buf_sum_b, 8 * no * sizeof(float));
}

void TLSTMCuda::deallocate_bwd_buffers() {
    if (d_buf_rec_mu) cudaFree(d_buf_rec_mu);
    if (d_buf_rec_var) cudaFree(d_buf_rec_var);
    if (d_buf_combined_mu) cudaFree(d_buf_combined_mu);
    if (d_buf_combined_var) cudaFree(d_buf_combined_var);
    if (d_buf_xh_mu) cudaFree(d_buf_xh_mu);
    if (d_buf_xh_var) cudaFree(d_buf_xh_var);
    if (d_buf_sum_w) cudaFree(d_buf_sum_w);
    if (d_buf_sum_b) cudaFree(d_buf_sum_b);
    d_buf_rec_mu = d_buf_rec_var = nullptr;
    d_buf_combined_mu = d_buf_combined_var = nullptr;
    d_buf_xh_mu = d_buf_xh_var = nullptr;
    d_buf_sum_w = d_buf_sum_b = nullptr;
}

void TLSTMCuda::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states) {
    if (this->input_size != input_states.actual_size) {
        LOG(LogLevel::ERROR,
            "Input size mismatch: " + std::to_string(this->input_size) +
                " vs " + std::to_string(input_states.actual_size));
    }

    HiddenStateCuda *cu_in = dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_out = dynamic_cast<HiddenStateCuda *>(&output_states);

    int batch_size = input_states.block_size;
    int seq_len = this->seq_len;
    int ni = this->input_size;
    int no = this->output_size;
    int ni_c = ni + no;
    int end_chunk = no * batch_size;
    unsigned int threads = this->num_cuda_threads;

    this->set_cap_factor_udapte(batch_size);

    if (this->_batch_size != batch_size) {
        this->_batch_size = batch_size;
        this->lstm_state.set_num_states(batch_size * seq_len * no,
                                        batch_size * seq_len * ni,
                                        this->device_idx);
        if (this->training) {
            this->allocate_bwd_buffers(batch_size);
        }
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.seq_len = seq_len;
    output_states.actual_size = this->output_size;

    cudaSetDevice(this->device_idx);
    if (seq_len == 1 && batch_size == 1) {
        int n = lstm_state.num_states;
        cudaMemcpy(lstm_state.d_mu_h_prev, lstm_state.d_mu_h_prior,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(lstm_state.d_var_h_prev, lstm_state.d_var_h_prior,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(lstm_state.d_mu_c_prev, lstm_state.d_mu_c_prior,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(lstm_state.d_var_c_prev, lstm_state.d_var_c_prior,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        lstm_state.reset_prev_states();
    }

    unsigned int cat_blocks = (batch_size + threads - 1) / threads;
    unsigned int act_blocks = (end_chunk + threads - 1) / threads;
    unsigned int grid_row = (no + threads - 1) / threads;
    unsigned int grid_col = (batch_size + threads - 1) / threads;
    dim3 fwd_grid(grid_col, grid_row);
    dim3 fwd_block(threads, threads);

    for (int t = 0; t < seq_len; t++) {
        // Concatenate [x(t), h_prev(t)]
        tlstm_cat_cuda<<<cat_blocks, threads>>>(
            cu_in->d_mu_a, lstm_state.d_mu_h_prev, ni, no, batch_size, seq_len,
            t, lstm_state.d_mu_ha);
        tlstm_cat_cuda<<<cat_blocks, threads>>>(
            cu_in->d_var_a, lstm_state.d_var_h_prev, ni, no, batch_size,
            seq_len, t, lstm_state.d_var_ha);

        // Forget gate
        tlstm_fwd_mean_var_cuda<<<fwd_grid, fwd_block>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            lstm_state.d_mu_ha, lstm_state.d_var_ha, ni_c, no, batch_size,
            seq_len, t, this->bias, this->w_pos_f, this->b_pos_f,
            lstm_state.d_mu_f_ga, lstm_state.d_var_f_ga);
        tlstm_sigmoid_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_f_ga, lstm_state.d_var_f_ga, no, seq_len, t,
            end_chunk, lstm_state.d_mu_f_ga, lstm_state.d_jcb_f_ga,
            lstm_state.d_var_f_ga);

        // Input gate
        tlstm_fwd_mean_var_cuda<<<fwd_grid, fwd_block>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            lstm_state.d_mu_ha, lstm_state.d_var_ha, ni_c, no, batch_size,
            seq_len, t, this->bias, this->w_pos_i, this->b_pos_i,
            lstm_state.d_mu_i_ga, lstm_state.d_var_i_ga);
        tlstm_sigmoid_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_i_ga, lstm_state.d_var_i_ga, no, seq_len, t,
            end_chunk, lstm_state.d_mu_i_ga, lstm_state.d_jcb_i_ga,
            lstm_state.d_var_i_ga);

        // Cell state gate
        tlstm_fwd_mean_var_cuda<<<fwd_grid, fwd_block>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            lstm_state.d_mu_ha, lstm_state.d_var_ha, ni_c, no, batch_size,
            seq_len, t, this->bias, this->w_pos_c, this->b_pos_c,
            lstm_state.d_mu_c_ga, lstm_state.d_var_c_ga);
        tlstm_tanh_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_c_ga, lstm_state.d_var_c_ga, no, seq_len, t,
            end_chunk, lstm_state.d_mu_c_ga, lstm_state.d_jcb_c_ga,
            lstm_state.d_var_c_ga);

        // Output gate
        tlstm_fwd_mean_var_cuda<<<fwd_grid, fwd_block>>>(
            this->d_mu_w, this->d_var_w, this->d_mu_b, this->d_var_b,
            lstm_state.d_mu_ha, lstm_state.d_var_ha, ni_c, no, batch_size,
            seq_len, t, this->bias, this->w_pos_o, this->b_pos_o,
            lstm_state.d_mu_o_ga, lstm_state.d_var_o_ga);
        tlstm_sigmoid_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_o_ga, lstm_state.d_var_o_ga, no, seq_len, t,
            end_chunk, lstm_state.d_mu_o_ga, lstm_state.d_jcb_o_ga,
            lstm_state.d_var_o_ga);

        // Cov(input, cell)
        tlstm_cov_input_cell_cuda<<<act_blocks, threads>>>(
            lstm_state.d_var_ha, this->d_mu_w, lstm_state.d_jcb_i_ga,
            lstm_state.d_jcb_c_ga, this->w_pos_i, this->w_pos_c, ni, no,
            seq_len, t, end_chunk, lstm_state.d_cov_i_c);

        // Cell state
        tlstm_cell_state_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_f_ga, lstm_state.d_var_f_ga, lstm_state.d_mu_i_ga,
            lstm_state.d_var_i_ga, lstm_state.d_mu_c_ga, lstm_state.d_var_c_ga,
            lstm_state.d_mu_c_prev, lstm_state.d_var_c_prev,
            lstm_state.d_cov_i_c, no, seq_len, t, end_chunk, lstm_state.d_mu_c,
            lstm_state.d_var_c);

        // tanh(cell)
        tlstm_tanh_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_c, lstm_state.d_var_c, no, seq_len, t, end_chunk,
            lstm_state.d_mu_ca, lstm_state.d_jcb_ca, lstm_state.d_var_ca);

        // Cov(output, tanh(cell))
        tlstm_cov_output_tanh_cell_states_cuda<<<act_blocks, threads>>>(
            this->d_mu_w, lstm_state.d_var_ha, lstm_state.d_mu_c_prev,
            lstm_state.d_jcb_ca, lstm_state.d_jcb_f_ga, lstm_state.d_mu_i_ga,
            lstm_state.d_jcb_i_ga, lstm_state.d_mu_c_ga, lstm_state.d_jcb_c_ga,
            lstm_state.d_jcb_o_ga, this->w_pos_f, this->w_pos_i, this->w_pos_c,
            this->w_pos_o, ni, no, seq_len, t, end_chunk,
            lstm_state.d_cov_o_tanh_c);

        // Hidden state
        tlstm_hidden_state_mean_var_cuda<<<act_blocks, threads>>>(
            lstm_state.d_mu_o_ga, lstm_state.d_var_o_ga, lstm_state.d_mu_ca,
            lstm_state.d_var_ca, lstm_state.d_cov_o_tanh_c, no, seq_len, t,
            end_chunk, cu_out->d_mu_a, cu_out->d_var_a);

        // Copy to prev for next timestep
        if (t < seq_len - 1) {
            tlstm_copy_prev_states_cuda<<<act_blocks, threads>>>(
                cu_out->d_mu_a, cu_out->d_var_a, lstm_state.d_mu_c,
                lstm_state.d_var_c, no, seq_len, t, end_chunk,
                lstm_state.d_mu_h_prev, lstm_state.d_var_h_prev,
                lstm_state.d_mu_c_prev, lstm_state.d_var_c_prev);
        }
    }

    // Save priors from last timestep
    if (seq_len == 1 && batch_size == 1) {
        int n = lstm_state.num_states;
        cudaMemcpy(lstm_state.d_mu_h_prior, cu_out->d_mu_a, n * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(lstm_state.d_var_h_prior, cu_out->d_var_a, n * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(lstm_state.d_mu_c_prior, lstm_state.d_mu_c,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(lstm_state.d_var_c_prior, lstm_state.d_var_c,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // Extract last timestep output
    if (this->last_timestep) {
        tlstm_extract_last_timestep_cuda<<<act_blocks, threads>>>(
            cu_out->d_mu_a, no, seq_len, end_chunk, cu_out->d_mu_a);
        tlstm_extract_last_timestep_cuda<<<act_blocks, threads>>>(
            cu_out->d_var_a, no, seq_len, end_chunk, cu_out->d_var_a);
        output_states.seq_len = 1;
    }

    if (this->training) {
        this->store_states_for_training_cuda(*cu_in, *cu_out);
    }
}

void TLSTMCuda::backward(BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         BaseTempStates &temp_states, bool state_udapte) {
    DeltaStateCuda *cu_in_delta =
        dynamic_cast<DeltaStateCuda *>(&input_delta_states);
    DeltaStateCuda *cu_out_delta =
        dynamic_cast<DeltaStateCuda *>(&output_delta_states);

    int batch_size = input_delta_states.block_size;
    int seq_len = this->seq_len;
    int ni = this->input_size;
    int no = this->output_size;
    int ni_c = ni + no;
    constexpr unsigned int threads = 256;

    cudaSetDevice(this->device_idx);

    // Zero recurrent delta and accumulators
    cudaMemset(d_buf_rec_mu, 0, batch_size * no * sizeof(float));
    cudaMemset(d_buf_rec_var, 0, batch_size * no * sizeof(float));
    if (param_update) {
        cudaMemset(d_buf_sum_w, 0, 8 * ni_c * no * sizeof(float));
        cudaMemset(d_buf_sum_b, 0, 8 * no * sizeof(float));
    }

    unsigned int blocks_no_b = (batch_size * no + threads - 1) / threads;
    unsigned int blocks_nic_b = (batch_size * ni_c + threads - 1) / threads;
    unsigned int blocks_w = (ni_c * no + threads - 1) / threads;
    unsigned int blocks_no = (no + threads - 1) / threads;

    // Update priors (seq_len==1 && batch_size==1)
    if (seq_len == 1 && batch_size == 1) {
        int total = batch_size * no;
        tlstm_update_hidden_posterior_cuda<<<blocks_no_b, threads>>>(
            lstm_state.d_mu_h_prior, lstm_state.d_var_h_prior, d_buf_rec_mu,
            d_buf_rec_var, total, lstm_state.d_mu_h_prior,
            lstm_state.d_var_h_prior);
        tlstm_update_cell_posterior_cuda<<<blocks_no_b, threads>>>(
            lstm_state.d_mu_c_prior, lstm_state.d_var_c_prior,
            lstm_state.d_jcb_ca, lstm_state.d_mu_o_ga, d_buf_rec_mu,
            d_buf_rec_var, total, lstm_state.d_mu_c_prior,
            lstm_state.d_var_c_prior);
    }

    for (int t = seq_len - 1; t >= 0; t--) {
        // Combine incoming + recurrent deltas
        bool has_direct = !this->last_timestep || (t == seq_len - 1);
        tlstm_combine_delta_cuda<<<blocks_no_b, threads>>>(
            d_buf_rec_mu, d_buf_rec_var, cu_in_delta->d_delta_mu,
            cu_in_delta->d_delta_var, no, seq_len, t, has_direct,
            this->last_timestep, batch_size * no, d_buf_combined_mu,
            d_buf_combined_var);

        if (param_update) {
            tlstm_delta_w_cuda<<<blocks_w, threads>>>(
                lstm_state.d_mu_ha, lstm_state.d_jcb_f_ga, lstm_state.d_mu_i_ga,
                lstm_state.d_jcb_i_ga, lstm_state.d_mu_c_ga,
                lstm_state.d_jcb_c_ga, lstm_state.d_mu_o_ga,
                lstm_state.d_jcb_o_ga, lstm_state.d_mu_c_prev,
                lstm_state.d_mu_ca, lstm_state.d_jcb_ca, d_buf_combined_mu,
                d_buf_combined_var, this->w_pos_f, this->w_pos_i, this->w_pos_c,
                this->w_pos_o, no, ni, batch_size, seq_len, t, ni_c * no,
                d_buf_sum_w);

            if (this->bias) {
                tlstm_delta_b_cuda<<<blocks_no, threads>>>(
                    lstm_state.d_jcb_f_ga, lstm_state.d_mu_i_ga,
                    lstm_state.d_jcb_i_ga, lstm_state.d_mu_c_ga,
                    lstm_state.d_jcb_c_ga, lstm_state.d_mu_o_ga,
                    lstm_state.d_jcb_o_ga, lstm_state.d_mu_c_prev,
                    lstm_state.d_mu_ca, lstm_state.d_jcb_ca, d_buf_combined_mu,
                    d_buf_combined_var, no, batch_size, seq_len, t,
                    d_buf_sum_b);
            }
        }

        // Delta z
        cudaMemset(d_buf_xh_mu, 0, batch_size * ni_c * sizeof(float));
        cudaMemset(d_buf_xh_var, 0, batch_size * ni_c * sizeof(float));

        tlstm_delta_z_cuda<<<blocks_nic_b, threads>>>(
            this->d_mu_w, lstm_state.d_jcb_f_ga, lstm_state.d_mu_i_ga,
            lstm_state.d_jcb_i_ga, lstm_state.d_mu_c_ga, lstm_state.d_jcb_c_ga,
            lstm_state.d_mu_o_ga, lstm_state.d_jcb_o_ga, lstm_state.d_mu_c_prev,
            lstm_state.d_mu_ca, lstm_state.d_jcb_ca, d_buf_combined_mu,
            d_buf_combined_var, this->w_pos_f, this->w_pos_i, this->w_pos_c,
            this->w_pos_o, no, ni, seq_len, t, batch_size * ni_c, d_buf_xh_mu,
            d_buf_xh_var);

        // Split delta_xh -> output + recurrent
        tlstm_split_delta_xh_cuda<<<blocks_nic_b, threads>>>(
            d_buf_xh_mu, d_buf_xh_var, ni, no, seq_len, t, batch_size,
            cu_out_delta->d_delta_mu, cu_out_delta->d_delta_var, d_buf_rec_mu,
            d_buf_rec_var);
    }

    // Scale accumulated weight/bias deltas
    if (param_update) {
        int w_size = ni_c * no;
        unsigned int blocks_ws = (w_size + threads - 1) / threads;
        tlstm_scale_delta_w_cuda<<<blocks_ws, threads>>>(
            d_buf_sum_w, this->d_var_w, this->w_pos_f, this->w_pos_i,
            this->w_pos_c, this->w_pos_o, w_size, this->d_delta_mu_w,
            this->d_delta_var_w);

        if (this->bias) {
            tlstm_scale_delta_b_cuda<<<blocks_no, threads>>>(
                d_buf_sum_b, this->d_var_b, this->b_pos_f, this->b_pos_i,
                this->b_pos_c, this->b_pos_o, no, this->d_delta_mu_b,
                this->d_delta_var_b);
        }
    }
}

std::unique_ptr<BaseLayer> TLSTMCuda::to_host() {
    auto host_layer = std::make_unique<TLSTM>(
        this->input_size, this->output_size, this->last_timestep, this->seq_len,
        this->bias, this->gain_w, this->gain_b, this->init_method);

    host_layer->mu_w = this->mu_w;
    host_layer->var_w = this->var_w;
    host_layer->mu_b = this->mu_b;
    host_layer->var_b = this->var_b;

    return host_layer;
}

void TLSTMCuda::preinit_layer() {
    if (this->num_weights == 0) {
        this->get_number_param();
        this->init_weight_bias();
    }
    if (this->training) {
        this->allocate_param_delta();
    }
}

void TLSTMCuda::d_get_LSTM_states(std::vector<float> &mu_h,
                                  std::vector<float> &var_h,
                                  std::vector<float> &mu_c,
                                  std::vector<float> &var_c) const {
    int n = this->lstm_state.num_states;
    mu_h.resize(n);
    var_h.resize(n);
    mu_c.resize(n);
    var_c.resize(n);
    cudaMemcpy(mu_h.data(), lstm_state.d_mu_h_prior, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(var_h.data(), lstm_state.d_var_h_prior, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(mu_c.data(), lstm_state.d_mu_c_prior, n * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(var_c.data(), lstm_state.d_var_c_prior, n * sizeof(float),
               cudaMemcpyDeviceToHost);
}

void TLSTMCuda::d_set_LSTM_states(const std::vector<float> &mu_h,
                                  const std::vector<float> &var_h,
                                  const std::vector<float> &mu_c,
                                  const std::vector<float> &var_c) {
    int n = this->lstm_state.num_states;
    if (static_cast<int>(mu_h.size()) != n ||
        static_cast<int>(var_h.size()) != n ||
        static_cast<int>(mu_c.size()) != n ||
        static_cast<int>(var_c.size()) != n) {
        LOG(LogLevel::ERROR,
            "d_set_LSTM_states size mismatch. Expected " + std::to_string(n));
        return;
    }
    cudaMemcpy(lstm_state.d_mu_h_prior, mu_h.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lstm_state.d_var_h_prior, var_h.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lstm_state.d_mu_c_prior, mu_c.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(lstm_state.d_var_c_prior, var_c.data(), n * sizeof(float),
               cudaMemcpyHostToDevice);
}

void TLSTMCuda::to(int device_idx) { this->device_idx = device_idx; }

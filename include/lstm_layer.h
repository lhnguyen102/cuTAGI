#pragma once
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

void lstm_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                       std::vector<float> &mu_b, std::vector<float> &var_b,
                       std::vector<float> &mu_a, std::vector<float> &var_a,
                       int start_chunk, int end_chunk, size_t input_size,
                       size_t output_size, int batch_size, bool bias, int w_pos,
                       int b_pos, std::vector<float> &mu_z,
                       std::vector<float> &var_z);

void lstm_fwd_mean_var_mp(std::vector<float> &mu_w, std::vector<float> &var_w,
                          std::vector<float> &mu_b, std::vector<float> &var_b,
                          std::vector<float> &mu_a, std::vector<float> &var_a,
                          size_t input_size, size_t output_size, int batch_size,
                          bool bias, int w_pos, int b_pos,
                          unsigned int num_threads, std::vector<float> &mu_z,
                          std::vector<float> &var_z);

void lstm_cov_input_cell_states(std::vector<float> &var_ha,
                                std::vector<float> &mu_w,
                                std::vector<float> &jcb_i_ga,
                                std::vector<float> &jcb_c_ga, int w_pos_i,
                                int w_pos_c, int ni, int no, int seq_len, int B,
                                std::vector<float> &cov_i_c);

void lstm_cell_state_mean_var(
    std::vector<float> &mu_f_ga, std::vector<float> &var_f_ga,
    std::vector<float> &mu_i_ga, std::vector<float> &var_i_ga,
    std::vector<float> &mu_c_ga, std::vector<float> &var_c_ga,
    std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev,
    std::vector<float> &cov_i_c, int no, int seq_len, int B,
    std::vector<float> &mu_c, std::vector<float> &var_c);

void lstm_cov_output_tanh_cell_states(
    std::vector<float> &mu_w, std::vector<float> &var_ha,
    std::vector<float> &mu_c_prev, std::vector<float> &jcb_ca,
    std::vector<float> &jcb_f_ga, std::vector<float> &mu_i_ga,
    std::vector<float> &jcb_i_ga, std::vector<float> &mu_c_ga,
    std::vector<float> &jcb_c_ga, std::vector<float> &jcb_o_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len,
    int batch_size, std::vector<float> &cov_tanh_c);

void lstm_hidden_state_mean_var(std::vector<float> &mu_o_ga,
                                std::vector<float> &var_o_ga,
                                std::vector<float> &mu_ca,
                                std::vector<float> &var_ca,
                                std::vector<float> &cov_o_tanh_c, int no,
                                int seq_len, int B, std::vector<float> &mu_z,
                                std::vector<float> &var_z);

void lstm_to_prev_states(std::vector<float> &curr, int n,
                         std::vector<float> &prev);

void lstm_cat_activations_and_prev_states(std::vector<float> &a,
                                          std::vector<float> &b, int n, int m,
                                          int seq_len, int B,
                                          std::vector<float> &c);

void lstm_cov_input_cell_states_worker(
    std::vector<float> &Sha, std::vector<float> &mw, std::vector<float> &Ji_ga,
    std::vector<float> &Jc_ga, int w_pos_i, int w_pos_c, int ni, int no,
    int seq_len, int B, int start_idx, int end_idx, std::vector<float> &Ci_c);

void lstm_cov_input_cell_states_mp(
    std::vector<float> &Sha, std::vector<float> &mw, std::vector<float> &Ji_ga,
    std::vector<float> &Jc_ga, int w_pos_i, int w_pos_c, int ni, int no,
    int seq_len, int B, int NUM_THREADS, std::vector<float> &Ci_c);

void lstm_cell_state_mean_var_worker(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int no, int seq_len, int start_idx, int end_idx,
    std::vector<float> &mc, std::vector<float> &Sc);

void lstm_cell_state_mean_var_mp(
    std::vector<float> &mf_ga, std::vector<float> &Sf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Si_ga,
    std::vector<float> &mc_ga, std::vector<float> &Sc_ga,
    std::vector<float> &mc_prev, std::vector<float> &Sc_prev,
    std::vector<float> &Ci_c, int no, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &mc, std::vector<float> &Sc);

void lstm_cov_output_tanh_cell_states_worker(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jca,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len,
    int start_idx, int end_idx, std::vector<float> &Co_tanh_c);

void lstm_cov_output_tanh_cell_states_mp(
    std::vector<float> &mw, std::vector<float> &Sha,
    std::vector<float> &mc_prev, std::vector<float> &Jca,
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &Jo_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int seq_len, int B,
    int NUM_THREADS, std::vector<float> &Co_tanh_c);

void lstm_hidden_state_mean_var_worker(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int no, int seq_len, int start_idx,
    int end_idx, std::vector<float> &mz, std::vector<float> &Sz);

void lstm_hidden_state_mean_var_mp(
    std::vector<float> &mo_ga, std::vector<float> &So_ga,
    std::vector<float> &mc_a, std::vector<float> &Sc_a,
    std::vector<float> &Co_tanh_c, int no, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &mz, std::vector<float> &Sz);

void lstm_cat_activations_and_prev_states_worker(std::vector<float> &a,
                                                 std::vector<float> &b, int n,
                                                 int m, int seq_len, int B,
                                                 int start_idx, int end_idx,
                                                 std::vector<float> &c);

void lstm_cat_activations_and_prev_states_mp(std::vector<float> &a,
                                             std::vector<float> &b, int n,
                                             int m, int seq_len, int B,
                                             int NUM_THREADS,
                                             std::vector<float> &c);

void lstm_delta_mean_var_z_worker(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_m_out,
    std::vector<float> &delta_S_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int start_idx, int end_idx,
    std::vector<float> &delta_m, std::vector<float> &delta_S);

void lstm_delta_mean_var_z_mp(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_m_out,
    std::vector<float> &delta_S_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &delta_m, std::vector<float> &delta_S);

void lstm_update_prev_hidden_states_worker(
    std::vector<float> &mu_h_prior, std::vector<float> &var_h_prior,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int start_idx,
    int end_idx, std::vector<float> &mu_h_prev, std::vector<float> &var_h_prev);

void lstm_update_prev_hidden_states_mp(std::vector<float> &mu_h_prior,
                                       std::vector<float> &var_h_prior,
                                       std::vector<float> &delta_mu,
                                       std::vector<float> &delta_var,
                                       int num_states, unsigned NUM_THREADS,
                                       std::vector<float> &mu_h_prev,
                                       std::vector<float> &var_h_prev);

void lstm_update_prev_cell_states_worker(
    std::vector<float> &mu_c_prior, std::vector<float> &var_c_prior,
    std::vector<float> &jcb_ca, std::vector<float> &mu_o_ga,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int start_idx,
    int end_idx, std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev);

void lstm_update_prev_cell_states_mp(
    std::vector<float> &mu_c_prior, std::vector<float> &var_c_prior,
    std::vector<float> &jcb_ca, std::vector<float> &mu_o_ga,
    std::vector<float> &delta_mu, std::vector<float> &delta_var, int num_states,
    unsigned int NUM_THREADS, std::vector<float> &mu_c_prev,
    std::vector<float> &var_c_prev);

void lstm_delta_mean_var_w_worker(
    std::vector<float> &Sw, std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, int start_idx, int end_idx,
    std::vector<float> &delta_mw, std::vector<float> &delta_Sw);

void lstm_delta_mean_var_w_mp(
    std::vector<float> &Sw, std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &delta_mw, std::vector<float> &delta_Sw);

void lstm_delta_mean_var_b_worker(
    std::vector<float> &Sb, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int b_pos_f, int b_pos_i, int b_pos_c,
    int b_pos_o, int no, int seq_len, int B, int start_idx, int end_idx,
    std::vector<float> &delta_mb, std::vector<float> &delta_Sb);

void lstm_delta_mean_var_b_mp(
    std::vector<float> &Sb, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int b_pos_f, int b_pos_i, int b_pos_c,
    int b_pos_o, int no, int seq_len, int B, int NUM_THREADS,
    std::vector<float> &delta_mb, std::vector<float> &delta_Sb);

class LSTM : public BaseLayer {
   public:
    int seq_len = 1;
    int _batch_size = -1;
    float act_omega = 0.0000001f;
    float gain_w;
    float gain_b;
    std::string init_method;
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    BaseLSTMStates lstm_states;

    LSTM(size_t input_size, size_t output_size, int seq_len = 1,
         bool bias = true, float gain_w = 1.0f, float gain_b = 1.0f,
         std::string init_method = "Xavier");

    ~LSTM();

    // Delete copy constructor and copy assignment
    LSTM(const LSTM &) = delete;
    LSTM &operator=(const LSTM &) = delete;

    // Optionally implement move constructor and move assignment
    LSTM(LSTM &&) = default;
    LSTM &operator=(LSTM &&) = default;

    virtual std::string get_layer_info() const override;

    virtual std::string get_layer_name() const override;

    virtual LayerType get_layer_type() const override;

    int get_input_size() override;

    int get_output_size() override;

    int get_max_num_states() override;

    void get_number_param();

    void init_weight_bias() override;

    void prepare_input(BaseHiddenStates &input_state);

    void forget_gate(int batch_size);

    void input_gate(int batch_size);

    void cell_state_gate(int batch_size);

    void output_gate(int batch_size);

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states) override;

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte = true) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
    void preinit_layer() override;

   protected:
};

#pragma once
#include <tuple>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

// Offset-aware forward functions
void tlstm_fwd_mean_var(std::vector<float> &mu_w, std::vector<float> &var_w,
                        std::vector<float> &mu_b, std::vector<float> &var_b,
                        std::vector<float> &mu_a, std::vector<float> &var_a,
                        int start_chunk, int end_chunk, size_t input_size,
                        size_t output_size, int batch_size, int seq_len,
                        int time_step, bool bias, int w_pos, int b_pos,
                        std::vector<float> &mu_z, std::vector<float> &var_z);

void tlstm_cat_activations_and_prev_states(std::vector<float> &a,
                                           std::vector<float> &b, int n, int m,
                                           int batch_size, int seq_len,
                                           int time_step,
                                           std::vector<float> &c);

void tlstm_cov_input_cell_states(std::vector<float> &var_ha,
                                 std::vector<float> &mu_w,
                                 std::vector<float> &jcb_i_ga,
                                 std::vector<float> &jcb_c_ga, int w_pos_i,
                                 int w_pos_c, int ni, int no, int batch_size,
                                 int seq_len, int time_step,
                                 std::vector<float> &cov_i_c);

void tlstm_cell_state_mean_var(
    std::vector<float> &mu_f_ga, std::vector<float> &var_f_ga,
    std::vector<float> &mu_i_ga, std::vector<float> &var_i_ga,
    std::vector<float> &mu_c_ga, std::vector<float> &var_c_ga,
    std::vector<float> &mu_c_prev, std::vector<float> &var_c_prev,
    std::vector<float> &cov_i_c, int no, int batch_size, int seq_len,
    int time_step, std::vector<float> &mu_c, std::vector<float> &var_c);

void tlstm_cov_output_tanh_cell_states(
    std::vector<float> &mu_w, std::vector<float> &var_ha,
    std::vector<float> &mu_c_prev, std::vector<float> &jcb_ca,
    std::vector<float> &jcb_f_ga, std::vector<float> &mu_i_ga,
    std::vector<float> &jcb_i_ga, std::vector<float> &mu_c_ga,
    std::vector<float> &jcb_c_ga, std::vector<float> &jcb_o_ga, int w_pos_f,
    int w_pos_i, int w_pos_c, int w_pos_o, int ni, int no, int batch_size,
    int seq_len, int time_step, std::vector<float> &cov_tanh_c);

void tlstm_hidden_state_mean_var(
    std::vector<float> &mu_o_ga, std::vector<float> &var_o_ga,
    std::vector<float> &mu_ca, std::vector<float> &var_ca,
    std::vector<float> &cov_o_tanh_c, int no, int batch_size, int seq_len,
    int time_step, std::vector<float> &mu_z, std::vector<float> &var_z);

// Backward functions
void tlstm_delta_mean_var_z(
    std::vector<float> &mw, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jca, std::vector<float> &delta_mu_out,
    std::vector<float> &delta_var_out, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int batch_size, int seq_len, int time_step,
    std::vector<float> &delta_mu, std::vector<float> &delta_var);

void tlstm_delta_mean_var_w(
    std::vector<float> &mha, std::vector<float> &Jf_ga,
    std::vector<float> &mi_ga, std::vector<float> &Ji_ga,
    std::vector<float> &mc_ga, std::vector<float> &Jc_ga,
    std::vector<float> &mo_ga, std::vector<float> &Jo_ga,
    std::vector<float> &mc_prev, std::vector<float> &mca,
    std::vector<float> &Jc, std::vector<float> &delta_m,
    std::vector<float> &delta_S, int w_pos_f, int w_pos_i, int w_pos_c,
    int w_pos_o, int no, int ni, int batch_size, int seq_len, int time_step,
    std::vector<float> &sum_mu_w_f, std::vector<float> &sum_var_w_f,
    std::vector<float> &sum_mu_w_i, std::vector<float> &sum_var_w_i,
    std::vector<float> &sum_mu_w_c, std::vector<float> &sum_var_w_c,
    std::vector<float> &sum_mu_w_o, std::vector<float> &sum_var_w_o);

void tlstm_delta_mean_var_b(
    std::vector<float> &Jf_ga, std::vector<float> &mi_ga,
    std::vector<float> &Ji_ga, std::vector<float> &mc_ga,
    std::vector<float> &Jc_ga, std::vector<float> &mo_ga,
    std::vector<float> &Jo_ga, std::vector<float> &mc_prev,
    std::vector<float> &mca, std::vector<float> &Jc,
    std::vector<float> &delta_m, std::vector<float> &delta_S, int no,
    int batch_size, int seq_len, int time_step, std::vector<float> &sum_mu_b_f,
    std::vector<float> &sum_var_b_f, std::vector<float> &sum_mu_b_i,
    std::vector<float> &sum_var_b_i, std::vector<float> &sum_mu_b_c,
    std::vector<float> &sum_var_b_c, std::vector<float> &sum_mu_b_o,
    std::vector<float> &sum_var_b_o);

class TLSTM : public BaseLayer {
   public:
    int _batch_size = -1;
    float act_omega = 0.0000001f;
    float gain_w;
    float gain_b;
    std::string init_method;
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    bool last_timestep = false;
    BaseLSTMStates lstm_states;

    TLSTM(size_t input_size, size_t output_size, bool last_timestep = false,
          int seq_len = 1, bool bias = true, float gain_w = 1.0f,
          float gain_b = 1.0f, std::string init_method = "Xavier",
          int device_idx = 0);

    ~TLSTM();

    TLSTM(const TLSTM &) = delete;
    TLSTM &operator=(const TLSTM &) = delete;

    TLSTM(TLSTM &&) = default;
    TLSTM &operator=(TLSTM &&) = default;

    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>,
               std::vector<float>>
    get_LSTM_states() const;

    void set_LSTM_states(const std::vector<float> &new_mu_h_prior,
                         const std::vector<float> &new_var_h_prior,
                         const std::vector<float> &new_mu_c_prior,
                         const std::vector<float> &new_var_c_prior);

    virtual std::string get_layer_info() const override;
    virtual std::string get_layer_name() const override;
    virtual LayerType get_layer_type() const override;
    int get_input_size() override;
    int get_output_size() override;
    int get_max_num_states() override;

    void get_number_param();
    void init_weight_bias() override;

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states) override;

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte = true) override;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif

    void preinit_layer() override;

   protected:
};

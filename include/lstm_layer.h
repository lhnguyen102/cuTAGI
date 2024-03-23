///////////////////////////////////////////////////////////////////////////////
// File:         lstm_layer.h
// Description:  Header file for Long-Short Term Memory (LSTM) forward pass
//               in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 22, 2024
// Updated:      March 23, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

class LSTM : public BaseLayer {
   public:
    int seq_len = 1;
    int _batch_size = 1;
    float act_omega = 0.001f;
    float gain_w;
    float gain_b;
    std::string init_method;
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    std::vector<float> mu_ha, var_ha, mu_f_ga, var_f_ga, jcb_f_ga, mu_i_ga,
        var_i_ga, jcb_i_ga, mu_c_ga, var_c_ga, jcb_c_ga, mu_o_ga, var_o_ga,
        jcb_o_ga, mu_ca, var_ca, jcb_ca, mu_c, var_c, mu_c_prev, var_c_prev,
        mu_h_prev, var_h_prev, cov_i_c, cov_o_tanh_c;

    LSTM(size_t input_size, size_t output_size, int seq_len, bool bias = true,
         float gain_w = 1.0f, float gain_b = 1.0f,
         std::string init_method = "He");

    ~LSTM();

    // Delete copy constructor and copy assignment
    LSTM(const LSTM &) = delete;
    LSTM &operator=(const LSTM &) = delete;

    // Optionally implement move constructor and move assignment
    LSTM(LSTM &&) = default;
    LSTM &operator=(LSTM &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void get_number_param();

    void init_weight_bias();

    void prepare_input(BaseHiddenStates &input_state);

    void forget_gate(int batch_size);

    void input_gate(int batch_size);

    void cell_state_gate(int batch_size);

    void output_gate(int batch_size);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void state_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_hidden_states,
                        BaseTempStates &temp_states) override;

    void param_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &delta_states,
                        BaseTempStates &temp_states) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
    void preinit_layer() override;

   protected:
    void allocate_states(int batch_size);
};

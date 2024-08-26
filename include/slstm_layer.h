///////////////////////////////////////////////////////////////////////////////
// File:         lstm_layer.h
// Description:  Header file for Long-Short Term Memory (LSTM) forward pass
//               in TAGI
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 22, 2024
// Updated:      April 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
#include "lstm_layer.h"

class SLSTM : public LSTM {
   public:
    SmoothingSLSTM smoothing_states;
    SLSTM(size_t input_size, size_t output_size, int seq_len = 1,
          bool bias = true, float gain_w = 1.0f, float gain_b = 1.0f,
          std::string init_method = "Xavier")
        : LSTM(input_size, output_size, seq_len, bias, gain_w, gain_b,
               init_method) {}

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    void smoother(BaseTempStates &temp_states);
};

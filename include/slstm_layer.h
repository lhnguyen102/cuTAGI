#pragma once
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
#include "lstm_layer.h"

class SLSTM : public LSTM {
   public:
    SmoothSLSTM smooth_states;
    int time_step = 0;
    SLSTM(size_t input_size, size_t output_size, int seq_len = 1,
          bool bias = true, float gain_w = 1.0f, float gain_b = 1.0f,
          std::string init_method = "Xavier", int time_step = 0)
        : LSTM(input_size, output_size, seq_len, bias, gain_w, gain_b,
               init_method),
          time_step(time_step) {}

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void prepare_input_smooth(SmoothingHiddenStates &input_state);

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    void smoother();
};

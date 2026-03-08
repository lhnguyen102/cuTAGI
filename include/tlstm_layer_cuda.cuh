#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct_cuda.cuh"

class TLSTMCuda : public BaseLayerCuda {
   public:
    int _batch_size = -1;
    float gain_w;
    float gain_b;
    std::string init_method;
    int w_pos_f, b_pos_f, w_pos_i, b_pos_i, w_pos_c, b_pos_c, w_pos_o, b_pos_o;
    bool last_timestep = false;

    LSTMStateCuda lstm_state;

    // Backward temp buffers (cached across calls)
    float *d_buf_rec_mu = nullptr, *d_buf_rec_var = nullptr;
    float *d_buf_combined_mu = nullptr, *d_buf_combined_var = nullptr;
    float *d_buf_xh_mu = nullptr, *d_buf_xh_var = nullptr;
    float *d_buf_sum_w = nullptr, *d_buf_sum_b = nullptr;

    TLSTMCuda(size_t input_size, size_t output_size, bool last_timestep = false,
              int seq_len = 1, bool bias = true, float gain_w = 1.0f,
              float gain_b = 1.0f, std::string init_method = "Xavier",
              int device_idx = 0);

    ~TLSTMCuda();

    TLSTMCuda(const TLSTMCuda &) = delete;
    TLSTMCuda &operator=(const TLSTMCuda &) = delete;
    TLSTMCuda(TLSTMCuda &&) = default;
    TLSTMCuda &operator=(TLSTMCuda &&) = default;

    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;
    int get_input_size() override;
    int get_output_size() override;
    int get_max_num_states() override;

    void get_number_param();
    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;
    void to(int device_idx) override;
    void preinit_layer() override;

    void d_get_LSTM_states(std::vector<float> &mu_h, std::vector<float> &var_h,
                           std::vector<float> &mu_c,
                           std::vector<float> &var_c) const;

    void d_set_LSTM_states(const std::vector<float> &mu_h,
                           const std::vector<float> &var_h,
                           const std::vector<float> &mu_c,
                           const std::vector<float> &var_c);

   protected:
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;

   private:
    void allocate_bwd_buffers(int batch_size);
    void deallocate_bwd_buffers();
};

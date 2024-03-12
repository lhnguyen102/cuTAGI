///////////////////////////////////////////////////////////////////////////////
// File:         convtranspose2d_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      March 10, 2024
// Updated:      March 11, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "base_layer_cuda.cuh"

class ConvTranspose2dCuda : public BaseLayerCuda {
   public:
    std::string init_method;
    size_t kernel_size = 0;
    int padding = 0;
    int stride = 1;
    int padding_type = 1;
    float gain_w;
    float gain_b;

    int *d_idx_mwa_1;
    int *d_idx_mwa_2;
    int *d_idx_cov_wz_2;
    int *d_idx_var_wz_ud;
    int *d_idx_cov_z_wa_1;
    int *d_idx_var_z_ud;

    std::vector<int> idx_mwa_1;
    std::vector<int> idx_mwa_2;
    std::vector<int> idx_cov_wz_2;
    std::vector<int> idx_var_wz_ud;
    std::vector<int> idx_cov_z_wa_1;
    std::vector<int> idx_var_z_ud;

    ConvTranspose2dCuda(size_t in_channels, size_t out_channels,
                        size_t kernel_size, bool bias = true, int stride = 1,
                        int padding = 0, int padding_type = 0,
                        size_t in_width = 0, size_t in_height = 0,
                        float gain_w = 1.0f, float gain_b = 1.0f,
                        std::string init_method = "He");

    ~ConvTranspose2dCuda();

    // Delete copy constructor and copy assignment
    ConvTranspose2dCuda(const ConvTranspose2dCuda &) = delete;
    ConvTranspose2dCuda &operator=(const ConvTranspose2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    ConvTranspose2dCuda(ConvTranspose2dCuda &&) = default;
    ConvTranspose2dCuda &operator=(ConvTranspose2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

    void get_number_param();

    void init_weight_bias();

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void state_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &input_delta_states,
                        BaseDeltaStates &output_delta_states,
                        BaseTempStates &temp_states) override;

    void param_backward(BaseBackwardStates &next_bwd_states,
                        BaseDeltaStates &delta_states,
                        BaseTempStates &temp_states) override;

    std::unique_ptr<BaseLayer> to_host() override;

    void preinit_layer() override;

   protected:
    void allocate_param_delta();
    void allocate_conv_index();
    void conv_index_to_device();
    void lazy_index_init();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};
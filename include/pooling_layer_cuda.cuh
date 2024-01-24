///////////////////////////////////////////////////////////////////////////////
// File:         pooling_layer_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      January 08, 2024
// Updated:      January 23, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "base_layer_cuda.cuh"

class AvgPool2dCuda : public BaseLayerCuda {
   public:
    size_t kernel_size = 0;
    int stride = 0;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> pool_idx, z_ud_idx;
    size_t row_zw = 0, col_z_ud = 0;
    bool overlap = true;

    int *d_pool_idx, *d_z_ud_idx;

    AvgPool2dCuda(size_t kernel_size, int stride = -1, int padding = 0,
                  int padding_type = 0);

    virtual ~AvgPool2dCuda();

    // Delete copy constructor and copy assignment
    AvgPool2dCuda(const AvgPool2dCuda &) = delete;
    AvgPool2dCuda &operator=(const AvgPool2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    AvgPool2dCuda(AvgPool2dCuda &&) = default;
    AvgPool2dCuda &operator=(AvgPool2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

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

    void update_weights() override{};

    void update_biases() override{};

   protected:
    void lazy_index_init();
    void allocate_avgpool2d_index();
    void avgpool2d_index_to_device();
};

////////////////////////////////////////////////////////////////////////////////
// Pool2d Backward and Forward
////////////////////////////////////////////////////////////////////////////////
__global__ void avgpool2d_fwd_overlapped_mean_var_cuda(
    float const *mu_a, float const *var_a, int const *a_idx, int woho, int wihi,
    int ki, int k, int pad_idx, float *mu_z, float *var_z);

__global__ void avgpool2d_fwd_mean_var_cuda(float const *mu_a,
                                            float const *var_a,
                                            int const *a_idx, int woho,
                                            int wihi, int ki, int k,
                                            float *mu_z, float *var_z);

__global__ void avgpool2d_bwd_overlapped_delta_z_cuda(
    float const *jcb, float const *delta_mu_out, float const *delta_var_out,
    int const *z_ud_idx, int woho, int wihi, int ki, int n, int k, int pad_idx,
    float *delta_mu, float *delta_var);

__global__ void avgpool2d_bwd_delta_z_cuda(float const *jcb,
                                           float const *delta_mu_out,
                                           float const *delta_var_out, int wo,
                                           int ki, int k, float *delta_mu,
                                           float *delta_var);
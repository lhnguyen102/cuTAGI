#pragma once

#include "base_layer_cuda.cuh"

class Conv2dCuda : public BaseLayerCuda {
   public:
    float gain_w;
    float gain_b;
    std::string init_method;
    size_t kernel_size = 0;
    int padding = 0;
    int stride = 1;
    int padding_type = 1;

    int *d_idx_mwa_2;
    int *d_idx_cov_zwa_1;
    int *d_idx_var_z_ud;
    std::vector<int> idx_mwa_2;
    std::vector<int> idx_cov_zwa_1;
    std::vector<int> idx_var_z_ud;
    int row_zw = 0, col_z_ud = 0;

    Conv2dCuda(size_t in_channels, size_t out_channels, size_t kernel_size,
               bool bias = true, int stride = 1, int padding = 0,
               int padding_type = 1, size_t in_width = 0, size_t in_height = 0,
               float gain_w = 1.0f, float gain_b = 1.0f,
               std::string init_method = "He");

    virtual ~Conv2dCuda();

    // Delete copy constructor and copy assignment
    Conv2dCuda(const Conv2dCuda &) = delete;
    Conv2dCuda &operator=(const Conv2dCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Conv2dCuda(Conv2dCuda &&) = default;
    Conv2dCuda &operator=(Conv2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

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

    void preinit_layer() override;

   protected:
    void allocate_conv_index();
    void conv_index_to_device();
    void lazy_index_init();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};

////////////////////////////////////////////////////////////////////////////////
// Conv2d Backward and Forward
////////////////////////////////////////////////////////////////////////////////
__global__ void conv2d_fwd_mean_var_cuda(float const *mu_w, float const *var_w,
                                         float const *mu_b, float const *var_b,
                                         float const *mu_a, float const *var_a,
                                         int const *aidx, int woho, int fo,
                                         int wihi, int fi, int ki, int B,
                                         int pad_idx, bool bias, float *mu_z,
                                         float *var_z);

__global__ void conv2d_bwd_delta_z_cuda(
    float const *mu_w, float const *jcb, float const *delta_mu_out,
    const float *delta_var_out, int const *zw_idx, int const *zud_idx, int woho,
    int fo, int wihi, int fi, int ki, int nr, int n, int B, int pad_idx,
    float *delta_mu, float *delta_var);

__global__ void permmute_jacobian_cuda(float const *jcb_0, int wihi, int fi,
                                       int batch_size, float *jcb);

__global__ void conv2d_bwd_delta_w_cuda(float const *var_w, float const *mu_a,
                                        float const *delta_mu_out,
                                        float const *delta_var_out,
                                        int const *aidx, int B, int k, int woho,
                                        int wihi, int fi, int ki, int pad_idx,
                                        float *delta_mu_w, float *delta_var_w);

__global__ void conv2d_bwd_delta_b_cuda(float const *var_b,
                                        float const *delta_mu_out,
                                        const float *delta_var_out, int n,
                                        int k, float *delta_mu_b,
                                        float *delta_var_b);

__global__ void permute_delta_cuda(float const *delta_mu_0,
                                   float const *delta_var_0, int woho, int kp,
                                   int batch_size, float *delta_mu,
                                   float *delta_var);
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "data_struct.h"
#include "param_init.h"
#include "row_segments.cuh"

__global__ void linear_fwd_mean_var(float const *mu_w, float const *var_w,
                                    float const *mu_b, float const *var_b,
                                    const float *mu_a, const float *var_a,
                                    size_t input_size, size_t output_size,
                                    int batch_size, bool bias, float *mu_z,
                                    float *var_z);

__global__ void linear_fwd_full_cov(float const *mu_w, float const *var_a_f,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *var_z_fp);

__global__ void linear_fwd_full_var(float const *mu_w, float const *var_w,
                                    float const *var_b, float const *mu_a,
                                    float const *var_a, float const *var_z_fp,
                                    size_t input_size, size_t output_size,
                                    int batch_size, float *var_z,
                                    float *var_z_f);

__global__ void linear_fwd_delta_z(float const *mu_w, float const *jcb,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_in,
                                   float *delta_var_in);

__global__ void linear_bwd_delta_w(float const *var_w, float const *mu_a,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_w,
                                   float *delta_var_w);

__global__ void linear_bwd_delta_b(float const *var_b,
                                   float const *delta_mu_out,
                                   float const *delta_var_out,
                                   size_t input_size, size_t output_size,
                                   int batch_size, float *delta_mu_b,
                                   float *delta_var_b);

// cuBLAS path of one linear projection: its flags and the workspace of the
// variance GEMMs. d_w_sq holds mu_w^2, written by each forward and reused by
// the backward that follows it; d_a_sq holds a squared activation of the
// current batch. use_cublas off runs the tiled kernels instead.
struct LinearGemm {
    bool use_cublas = true;
    bool use_tf32 = false;
    float *d_w_sq = nullptr;
    float *d_a_sq = nullptr;

    LinearGemm() = default;
    ~LinearGemm();
    LinearGemm(const LinearGemm &) = delete;
    LinearGemm &operator=(const LinearGemm &) = delete;

    // Grows the buffers to the given sizes; keeps them otherwise.
    void reserve(size_t num_weights, size_t num_activations);

   private:
    size_t w_capacity = 0;
    size_t a_capacity = 0;
};

// Kernel launchers shared with layers that embed a linear projection
// (e.g. MultiheadAttentionCuda).
void linear_forward_cuda(HiddenStateCuda *&cu_input_states,
                         HiddenStateCuda *&cu_output_states,
                         const float *d_mu_w, const float *d_var_w,
                         const float *d_mu_b, const float *d_var_b,
                         size_t input_size, size_t output_size, int batch_size,
                         bool bias, LinearGemm &gemm);

void linear_state_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                DeltaStateCuda *&cu_output_delta_states,
                                BackwardStateCuda *&cu_next_bwd_states,
                                const float *d_mu_w, size_t input_size,
                                size_t output_size, int batch_size,
                                LinearGemm &gemm);

void linear_weight_backward_cuda(DeltaStateCuda *&cu_input_delta_states,
                                 DeltaStateCuda *&cu_output_delta_states,
                                 BackwardStateCuda *&cu_next_bwd_states,
                                 const float *d_var_w, size_t input_size,
                                 size_t output_size, int batch_size,
                                 float *d_delta_mu_w, float *d_delta_var_w,
                                 LinearGemm &gemm);

class LinearCuda : public BaseLayerCuda {
   public:
    float gain_w;
    float gain_b;
    std::string init_method;
    LinearGemm gemm;

    LinearCuda(size_t ip_size, size_t op_size, bool bias = true,
               float gain_weight = 1.0f, float gain_bias = 1.0f,
               std::string method = "He", int device_idx = 0);

    // Delete copy constructor and copy assignment
    LinearCuda(const LinearCuda &) = delete;
    LinearCuda &operator=(const LinearCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    LinearCuda(LinearCuda &&) = default;
    LinearCuda &operator=(LinearCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states, bool state_udapte) override;

    std::unique_ptr<BaseLayer> to_host() override;

    void set_output_indices(const std::vector<int> &unit_idx,
                            int n_obs) override;

    void set_tf32(bool on) override;

    void update_weights() override;

    void update_biases() override;

   protected:
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;

   private:
    // Sparse readout: 0-based path units [tokens, n_obs] for the next forward
    // live in segments.d_row; backward groups the (token, k) pairs by unit.
    RowSegments segments;
    int num_output_idx = 0;
    int n_obs = 0;
    bool output_idx_set = false;
    bool sparse_pass = false;

    void sparse_forward(HiddenStateCuda &input_states,
                        HiddenStateCuda &output_states, int num_tokens);
    void sparse_backward(DeltaStateCuda &input_delta_states,
                         DeltaStateCuda &output_delta_states,
                         BackwardStateCuda &bwd_states, int num_tokens,
                         bool state_udapte);
};

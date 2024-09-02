///////////////////////////////////////////////////////////////////////////////
// File:         activation_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 04, 2023
// Updated:      August 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#include "../include/activation.h"
#include "../include/activation_cuda.cuh"

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////
ReLUCuda::ReLUCuda() {}
ReLUCuda::~ReLUCuda() {}

std::string ReLUCuda::get_layer_info() const
/*
 */
{
    return "Relu()";
}

std::string ReLUCuda::get_layer_name() const
/*
 */
{
    return "ReLUCuda";
}

LayerType ReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void ReLUCuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    constexpr unsigned int THREADS = 256;
    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks = (num_states + THREADS - 1) / THREADS;

    relu_mean_var_cuda<<<blocks, THREADS>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }
}

std::unique_ptr<BaseLayer> ReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<ReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
SigmoidCuda::SigmoidCuda() {}
SigmoidCuda::~SigmoidCuda() {}

std::string SigmoidCuda::get_layer_info() const
/*
 */
{
    return "Sigmoid()";
}

std::string SigmoidCuda::get_layer_name() const
/*
 */
{
    return "SigmoidCuda";
}

LayerType SigmoidCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SigmoidCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    sigmoid_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SigmoidCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Sigmoid>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
TanhCuda::TanhCuda() {}
TanhCuda::~TanhCuda() {}

std::string TanhCuda::get_layer_info() const
/*
 */
{
    return "Tanh()";
}

std::string TanhCuda::get_layer_name() const
/*
 */
{
    return "TanhCuda";
}

LayerType TanhCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void TanhCuda::forward(BaseHiddenStates &input_states,
                       BaseHiddenStates &output_states,
                       BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    tanh_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> TanhCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Tanh>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Relu
////////////////////////////////////////////////////////////////////////////////
MixtureReLUCuda::MixtureReLUCuda() {}
MixtureReLUCuda ::~MixtureReLUCuda() {}

std::string MixtureReLUCuda::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureReLUCuda::get_layer_name() const
/*
 */
{
    return "MixtureReLUCuda";
}

LayerType MixtureReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureReLUCuda::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_relu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    // cu_output_states->to_device();

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
MixtureSigmoidCuda::MixtureSigmoidCuda() {}
MixtureSigmoidCuda ::~MixtureSigmoidCuda() {}

std::string MixtureSigmoidCuda::get_layer_info() const
/*
 */
{
    return "MixtureSigmoid()";
}

std::string MixtureSigmoidCuda::get_layer_name() const
/*
 */
{
    return "MixtureSigmoidCuda";
}

LayerType MixtureSigmoidCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureSigmoidCuda::forward(BaseHiddenStates &input_states,
                                 BaseHiddenStates &output_states,
                                 BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_sigmoid_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureSigmoidCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureSigmoid>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
MixtureTanhCuda::MixtureTanhCuda() {}
MixtureTanhCuda ::~MixtureTanhCuda() {}

std::string MixtureTanhCuda::get_layer_info() const
/*
 */
{
    return "MixtureTanh()";
}

std::string MixtureTanhCuda::get_layer_name() const
/*
 */
{
    return "MixtureTanhCuda";
}

LayerType MixtureTanhCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureTanhCuda::forward(BaseHiddenStates &input_states,
                              BaseHiddenStates &output_states,
                              BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    mixture_tanh_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> MixtureTanhCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureTanh>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
SoftplusCuda::SoftplusCuda() {}
SoftplusCuda::~SoftplusCuda() {}

std::string SoftplusCuda::get_layer_info() const
/*
 */
{
    return "Softplus()";
}

std::string SoftplusCuda::get_layer_name() const
/*
 */
{
    return "SoftplusCuda";
}

LayerType SoftplusCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SoftplusCuda::forward(BaseHiddenStates &input_states,
                           BaseHiddenStates &output_states,
                           BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    softplus_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, num_states,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SoftplusCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Softplus>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// LeakyRelu
////////////////////////////////////////////////////////////////////////////////
LeakyReLUCuda::LeakyReLUCuda() {}
LeakyReLUCuda::~LeakyReLUCuda() {}

std::string LeakyReLUCuda::get_layer_info() const
/*
 */
{
    return "leakyRelu()";
}

std::string LeakyReLUCuda::get_layer_name() const
/*
 */
{
    return "leakyReluCuda";
}

LayerType LeakyReLUCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyReLUCuda::forward(BaseHiddenStates &input_states,
                            BaseHiddenStates &output_states,
                            BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    leakyrelu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->alpha,
        num_states, cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> LeakyReLUCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<LeakyReLU>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// Softmax
////////////////////////////////////////////////////////////////////////////////
SoftmaxCuda::SoftmaxCuda() {}
SoftmaxCuda::~SoftmaxCuda() {}

std::string SoftmaxCuda::get_layer_info() const
/*
 */
{
    return "Softmax()";
}

std::string SoftmaxCuda::get_layer_name() const
/*
 */
{
    return "SoftmaxCuda";
}

LayerType SoftmaxCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void SoftmaxCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);
    // TempStateCuda *cu_temp_states = dynamic_cast<TempStateCuda
    // *>(&temp_states);

    unsigned int blocks =
        (input_states.block_size + this->num_cuda_threads - 1) /
        this->num_cuda_threads;

    softmax_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->actual_size, cu_input_states->block_size,
        cu_output_states->d_mu_a, cu_output_states->d_jcb,
        cu_output_states->d_var_a);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> SoftmaxCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<Softmax>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
/// EvenExp
////////////////////////////////////////////////////////////////////////////////
EvenExpCuda::EvenExpCuda() {}
EvenExpCuda::~EvenExpCuda() {}

std::string EvenExpCuda::get_layer_info() const
/*
 */
{
    return "EvenExp()";
}

std::string EvenExpCuda::get_layer_name() const
/*
 */
{
    return "EvenExpCuda";
}

LayerType EvenExpCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void EvenExpCuda::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/*
 */
{
    // New poitner will point to the same memory location when casting
    HiddenStateCuda *cu_input_states =
        dynamic_cast<HiddenStateCuda *>(&input_states);
    HiddenStateCuda *cu_output_states =
        dynamic_cast<HiddenStateCuda *>(&output_states);

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    even_exp_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
        cu_input_states->d_mu_a, cu_input_states->d_var_a,
        cu_input_states->d_jcb, num_states, cu_output_states->d_mu_a,
        cu_output_states->d_var_a, cu_output_states->d_jcb);

    if (this->input_size != input_states.actual_size) {
        this->input_size = input_states.actual_size;
        this->output_size = input_states.actual_size;
    }

    // Update number of actual states.
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;
}

std::unique_ptr<BaseLayer> EvenExpCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<EvenExp>();
    host_layer->input_size = this->input_size;
    host_layer->output_size = this->output_size;

    return host_layer;
}

////////////////////////////////////////////////////////////////////////////////
// CUDA kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        float tmp = fmaxf(mu_z[col], 0.0f);
        mu_a[col] = tmp;

        bool is_zero = (tmp == 0.0f);
        jcb[col] = is_zero ? 0.0f : 1.0f;
        var_a[col] = is_zero ? 0.0f : var_z[col];
    }
}

__global__ void relu_mean_var_cuda_vectorized(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_idx = idx * 4;

    if (vec_idx < num_states) {
        float4 mu_z_vec, var_z_vec, mu_a_vec, jcb_vec, var_a_vec;

        // Load 4 float values into float4 vectors
        mu_z_vec.x = mu_z[vec_idx];
        mu_z_vec.y = vec_idx + 1 < num_states ? mu_z[vec_idx + 1] : 0.0f;
        mu_z_vec.z = vec_idx + 2 < num_states ? mu_z[vec_idx + 2] : 0.0f;
        mu_z_vec.w = vec_idx + 3 < num_states ? mu_z[vec_idx + 3] : 0.0f;

        var_z_vec.x = var_z[vec_idx];
        var_z_vec.y = vec_idx + 1 < num_states ? var_z[vec_idx + 1] : 0.0f;
        var_z_vec.z = vec_idx + 2 < num_states ? var_z[vec_idx + 2] : 0.0f;
        var_z_vec.w = vec_idx + 3 < num_states ? var_z[vec_idx + 3] : 0.0f;

        // Process the data
        mu_a_vec.x = fmaxf(mu_z_vec.x, 0.0f);
        mu_a_vec.y = fmaxf(mu_z_vec.y, 0.0f);
        mu_a_vec.z = fmaxf(mu_z_vec.z, 0.0f);
        mu_a_vec.w = fmaxf(mu_z_vec.w, 0.0f);

        jcb_vec.x = (mu_a_vec.x == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.y = (mu_a_vec.y == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.z = (mu_a_vec.z == 0.0f) ? 0.0f : 1.0f;
        jcb_vec.w = (mu_a_vec.w == 0.0f) ? 0.0f : 1.0f;

        var_a_vec.x = (mu_a_vec.x == 0.0f) ? 0.0f : var_z_vec.x;
        var_a_vec.y = (mu_a_vec.y == 0.0f) ? 0.0f : var_z_vec.y;
        var_a_vec.z = (mu_a_vec.z == 0.0f) ? 0.0f : var_z_vec.z;
        var_a_vec.w = (mu_a_vec.w == 0.0f) ? 0.0f : var_z_vec.w;

        // Store the results back as individual floats
        mu_a[vec_idx] = mu_a_vec.x;
        jcb[vec_idx] = jcb_vec.x;
        var_a[vec_idx] = var_a_vec.x;

        if (vec_idx + 1 < num_states) {
            mu_a[vec_idx + 1] = mu_a_vec.y;
            jcb[vec_idx + 1] = jcb_vec.y;
            var_a[vec_idx + 1] = var_a_vec.y;
        }

        if (vec_idx + 2 < num_states) {
            mu_a[vec_idx + 2] = mu_a_vec.z;
            jcb[vec_idx + 2] = jcb_vec.z;
            var_a[vec_idx + 2] = var_a_vec.z;
        }

        if (vec_idx + 3 < num_states) {
            mu_a[vec_idx + 3] = mu_a_vec.w;
            jcb[vec_idx + 3] = jcb_vec.w;
            var_a[vec_idx + 3] = var_a_vec.w;
        }
    }
}

__global__ void sigmoid_mean_var_cuda(float const *mu_z, float const *var_z,
                                      int num_states, float *mu_a, float *jcb,
                                      float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;

    if (col < num_states) {
        tmp = 1.0f / (1.0f + expf(-mu_z[col]));
        mu_a[col] = tmp;
        jcb[col] = tmp * (1.0f - tmp);
        var_a[col] = tmp * (1.0f - tmp) * var_z[col] * tmp * (1.0f - tmp);
    }
}

__global__ void tanh_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = tanhf(mu_z[col]);
        float tmp_2 = tmp * tmp;
        mu_a[col] = tmp;
        jcb[col] = (1.0f - tmp_2);
        var_a[col] = (1.0f - tmp_2) * var_z[col] * (1.0f - tmp_2);
    }
}

__device__ float normcdf_cuda(float x)
/*
Normal cumulative distribution function
 */
{
    return 0.5f * erfcf(-x * 0.7071067811865475f);
}

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr float SQRT_2PI = 2.5066282746310002f;
    if (col < num_states) {
        // Reused components for moments calculations
        float tmp_mu_z = mu_z[col];
        float std_z = powf(var_z[col], 0.5);
        float alpha = tmp_mu_z / std_z;
        float pdf_alpha = (1.0f / SQRT_2PI) * expf(-0.5f * alpha * alpha);
        float cdf_alpha = normcdf_cuda(alpha);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = mu_z[col] * cdf_alpha + std_z * pdf_alpha;
        mu_a[col] = tmp_mu_a;
        var_a[col] = -tmp_mu_a * tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z -
                     tmp_mu_z * std_z * pdf_alpha +
                     (var_z[col] - tmp_mu_z * tmp_mu_z) * cdf_alpha;
        jcb[col] = cdf_alpha;
    }
}

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    constexpr float SQRT_2PI = 2.5066282746310002f;

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        std_z = powf(var_z[col], 0.5);
        alpha_l = (1.0f + mu_z[col]) / std_z;  // Lower truncation
        alpha_u = (1.0f - mu_z[col]) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_z = mu_z[col];
        float tmp_mu_z_2 = tmp_mu_z * tmp_mu_z;
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = tmp_mu_a;
        var_a[col] =
            max(0.000001f,
                (cdf_l * (var_z[col] - tmp_mu_z_2 - 2 * tmp_mu_z - 1) +
                 cdf_u * (var_z[col] - tmp_mu_z_2 + 2 * tmp_mu_z - 1) +
                 std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                 tmp_mu_a * tmp_mu_a + 2 * mu_a[col] * tmp_mu_z +
                 tmp_mu_z * tmp_mu_z - var_z[col] + 2) /
                    4.0f);
        mu_a[col] = tmp_mu_a / 2.0f + 0.5f;
        jcb[col] = (cdf_u + cdf_l - 1) / 2.0f;
    }
}

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float std_z, alpha_l, alpha_u, pdf_l, pdf_u, cdf_l, cdf_u;
    constexpr float SQRT_2PI = 2.5066282746310002f;

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        float tmp_mu_z = mu_z[col];
        std_z = powf(var_z[col], 0.5);
        alpha_l = (1.0f + tmp_mu_z) / std_z;  // Lower truncation
        alpha_u = (1.0f - tmp_mu_z) / std_z;  // Upper truncation
        cdf_l = normcdf_cuda(alpha_l);
        cdf_u = normcdf_cuda(alpha_u);
        pdf_l = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_l * alpha_l);
        pdf_u = (1.0f / SQRT_2PI) * expf(-0.5f * alpha_u * alpha_u);

        // Moments calculations (L. Alric, 2024)
        float tmp_mu_a = (tmp_mu_z + 1) * cdf_l + (tmp_mu_z - 1) * cdf_u +
                         std_z * (pdf_l - pdf_u) - tmp_mu_z;

        mu_a[col] = tmp_mu_a;
        var_a[col] = max(
            0.000001f,
            cdf_l * (var_z[col] - tmp_mu_z * tmp_mu_z - 2 * tmp_mu_z - 1) +
                cdf_u * (var_z[col] - tmp_mu_z * tmp_mu_z + 2 * tmp_mu_z - 1) +
                std_z * (pdf_u * (tmp_mu_z - 1) - pdf_l * (tmp_mu_z + 1)) -
                tmp_mu_a + 2 * tmp_mu_a * tmp_mu_z + tmp_mu_z - var_z[col] + 2);

        jcb[col] = cdf_u + cdf_l - 1;
    }
}

__global__ void softplus_mean_var_cuda(float const *mu_z, float const *var_z,
                                       int num_states, float *mu_a, float *jcb,
                                       float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float tmp = 0;
    if (col < num_states) {
        mu_a[col] = logf(1 + expf(mu_z[col]));
        tmp = 1 / (1 + expf(-mu_z[col]));
        jcb[col] = tmp;
        var_a[col] = tmp * var_z[col] * tmp;
    }
}

__global__ void leakyrelu_mean_var_cuda(float const *mu_z, float const *var_z,
                                        float alpha, int num_states,
                                        float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float zero_pad = 0.0f;
    float one_pad = 1.0f;
    float tmp = 0.0f;
    if (col < num_states) {
        tmp = max(mu_z[col], zero_pad);
        if (tmp == 0) {
            mu_a[col] = alpha * mu_z[col];
            jcb[col] = alpha;
            var_a[col] = alpha * var_z[col] * alpha;

        } else {
            mu_a[col] = tmp;
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
        }
    }
}

__global__ void softmax_mean_var_cuda(float const *mu_z, float *var_z,
                                      size_t output_size, int batch_size,
                                      float *mu_a, float *jcb, float *var_a)
/*
 */
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    float max_mu = mu_z[0];
    float max_var = var_z[0];

    for (int j = 1; j < output_size; j++) {
        if (mu_z[j + i * output_size] > max_mu) {
            max_mu = mu_z[j + i * output_size];
            max_var = var_z[j + i * output_size];
        }
    }

    float sum_mu = 0.0f;
    for (int j = 0; j < output_size; j++) {
        sum_mu += expf(mu_z[j + i * output_size] - max_mu);
    }

    float tmp_mu;
    for (int j = 0; j < output_size; j++) {
        tmp_mu = expf(mu_z[j + output_size * i] - max_mu) / sum_mu;

        mu_a[j + i * output_size] = tmp_mu;

        jcb[j + output_size * i] = tmp_mu * (1 - tmp_mu);

        var_a[j + output_size * i] = jcb[j + output_size * i] *
                                     (var_z[j + output_size * i] + max_var) *
                                     jcb[j + output_size * i];
    }
}

__global__ void even_exp_mean_var_cuda(float const *mu_z, float const *var_z,
                                       float const *jcb_z, int num_states,
                                       float *mu_a, float *var_a, float *jcb_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < num_states) {
        if (col % 2 == 0) {
            mu_a[col] = mu_z[col];
            var_a[col] = var_z[col];
            jcb_a[col] = jcb_z[col];
        } else {
            mu_a[col] = expf(mu_z[col] + 0.5f * var_z[col]);
            var_a[col] =
                expf(2.0f * mu_z[col] + var_z[col]) * (expf(var_z[col]) - 1.0f);
            jcb_a[col] = var_z[col] * mu_a[col];
        }
    }
}
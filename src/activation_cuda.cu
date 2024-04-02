///////////////////////////////////////////////////////////////////////////////
// File:         activation_cuda.cuh
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      December 04, 2023
// Updated:      April 02, 2024
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

    int num_states = input_states.actual_size * input_states.block_size;
    unsigned int blocks =
        (num_states + this->num_cuda_threads - 1) / this->num_cuda_threads;

    // Assign output dimensions
    cu_output_states->height = cu_input_states->height;
    cu_output_states->depth = cu_input_states->depth;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->block_size = cu_input_states->block_size;
    cu_output_states->actual_size = cu_input_states->actual_size;

    relu_mean_var_cuda<<<blocks, this->num_cuda_threads>>>(
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
MixtureReluCuda::MixtureReluCuda() {}
MixtureReluCuda ::~MixtureReluCuda() {}

std::string MixtureReluCuda::get_layer_info() const
/*
 */
{
    return "MixtureReLU()";
}

std::string MixtureReluCuda::get_layer_name() const
/*
 */
{
    return "MixtureReluCuda";
}

LayerType MixtureReluCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void MixtureReluCuda::forward(BaseHiddenStates &input_states,
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
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->omega_tol,
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

std::unique_ptr<BaseLayer> MixtureReluCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<MixtureRelu>();
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
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->omega_tol,
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
        cu_input_states->d_mu_a, cu_input_states->d_var_a, this->omega_tol,
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
LeakyReluCuda::LeakyReluCuda() {}
LeakyReluCuda::~LeakyReluCuda() {}

std::string LeakyReluCuda::get_layer_info() const
/*
 */
{
    return "leakyRelu()";
}

std::string LeakyReluCuda::get_layer_name() const
/*
 */
{
    return "leakyReluCuda";
}

LayerType LeakyReluCuda::get_layer_type() const
/*
 */
{
    return LayerType::Activation;
}

void LeakyReluCuda::forward(BaseHiddenStates &input_states,
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

    cu_input_states->to_device();

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

std::unique_ptr<BaseLayer> LeakyReluCuda::to_host()
/* Transfer to cpu version
 */
{
    std::unique_ptr<BaseLayer> host_layer = std::make_unique<LeakyRelu>();
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
// CUDA kernels
////////////////////////////////////////////////////////////////////////////////

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float one_pad = 1.0f;
    float zero_pad = 0;
    float tmp = 0;
    if (col < num_states) {
        tmp = max(mu_z[col], zero_pad);
        mu_a[col] = tmp;
        if (tmp == 0) {
            jcb[col] = zero_pad;
            var_a[col] = zero_pad;
        } else {
            jcb[col] = one_pad;
            var_a[col] = var_z[col];
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
        jcb[col] = tmp * (1 - tmp);
        var_a[col] = tmp * (1 - tmp) * var_z[col] * tmp * (1 - tmp);
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
        mu_a[col] = tmp;
        jcb[col] = (1 - powf(tmp, 2));
        var_a[col] = (1 - powf(tmp, 2)) * var_z[col] * (1 - powf(tmp, 2));
    }
}

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, float omega_tol,
                                           int num_states, float *mu_a,
                                           float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha, beta, omega, kappa, mu_z_til, var_z_til;
    float pi = 3.141592;  // pi number
    if (col < num_states) {
        // Hyper-parameters for Gaussian mixture
        alpha = -mu_z[col] / powf(var_z[col], 0.5);
        omega = max(1.0f - normcdff(alpha), omega_tol);
        beta = (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha, 2) / 2.0f) /
               omega;
        kappa = 1.0f + alpha * beta - powf(beta, 2);

        // Gaussian mixture's parameters
        mu_z_til = mu_z[col] + beta * powf(var_z[col], 0.5);
        var_z_til = kappa * var_z[col];

        // Activation distribution
        if (omega * mu_z_til > omega_tol) {
            mu_a[col] = omega * mu_z_til;
            var_a[col] =
                omega * var_z_til + omega * (1.0f - omega) * powf(mu_z_til, 2);
            // jcb[col] = powf(omega * kappa, 0.5); //Approx. formulation
            jcb[col] =  // Exact form. (Huber, 2020)
                ((powf(mu_z[col], 2) + var_z[col]) * normcdff(-alpha) +
                 mu_z[col] * powf(var_z[col], 0.5) *
                     (1.0f / powf(2.0f * pi, 0.5)) *
                     expf(-powf(-alpha, 2) / 2.0f) -
                 (mu_a[col] * mu_z[col])) /
                var_z[col];
        } else {
            mu_a[col] = omega_tol;
            var_a[col] =
                omega * var_z_til + omega * (1.0f - omega) * powf(omega_tol, 2);
            jcb[col] = 0.0f;  // TODO replace by 1.0f
        }
    }
}

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              float omega_tol, int num_states,
                                              float *mu_a, float *jcb,
                                              float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        alpha_upper = (1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                (pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[col] - beta * powf(var_z[col], 0.5);
        var_z_til = kappa * var_z[col];

        // Activation distribution
        mu_a[col] =
            (omega * mu_z_til - cdf_lower + (1 - cdf_upper)) / 2.0f + 0.5f;

        var_a[col] =
            (omega * var_z_til + omega * powf(mu_z_til - mu_a[col], 2) +
             cdf_lower * powf(1 + mu_a[col], 2) +
             (1 - cdf_upper) * powf(1 - mu_a[col], 2)) /
            4.0f;

        jcb[col] = omega * 0.5;
    }
}

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, float omega_tol,
                                           int num_states, float *mu_a,
                                           float *jcb, float *var_a)
/*
 */
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha_lower, alpha_upper, omega, beta, kappa, mu_z_til, var_z_til,
        cdf_lower, cdf_upper, pdf_lower, pdf_upper;
    float pi = 3.141592;  // pi number

    if (col < num_states) {
        // cdf and pdf for truncated normal distribution
        alpha_lower = (-1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        alpha_upper = (1.0f - mu_z[col]) / powf(var_z[col], 0.5);
        cdf_lower = normcdff(alpha_lower);
        cdf_upper = normcdff(alpha_upper);
        pdf_lower =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_lower, 2) / 2.0f);
        pdf_upper =
            (1.0f / powf(2.0f * pi, 0.5)) * expf(-powf(alpha_upper, 2) / 2.0f);

        // Truncated distribution's parameters
        omega = max(cdf_upper - cdf_lower, omega_tol);
        beta = (pdf_upper - pdf_lower) / omega;
        kappa = 1 -
                (pdf_upper * alpha_upper - pdf_lower * alpha_lower) / omega -
                powf(beta, 2);

        // Gaussian mixture's paramters
        mu_z_til = mu_z[col] - beta * powf(var_z[col], 0.5);
        var_z_til = kappa * var_z[col];

        // Activation distribution
        mu_a[col] = omega * mu_z_til - cdf_lower + (1 - cdf_upper);
        var_a[col] = omega * var_z_til + omega * powf(mu_z_til - mu_a[col], 2) +
                     cdf_lower * powf(1 + mu_a[col], 2) +
                     (1 - cdf_upper) * powf(1 - mu_a[col], 2);
        jcb[col] = omega;
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

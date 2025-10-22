#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "base_layer_cuda.cuh"

__global__ void relu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void relu_mean_var_cuda_vectorized(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a);

__global__ void sigmoid_mean_var_cuda(float const *mu_z, float const *var_z,
                                      int num_states, float *mu_a, float *jcb,
                                      float *var_a);

__global__ void tanh_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void mixture_relu_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a);

__global__ void mixture_sigmoid_mean_var_cuda(float const *mu_z,
                                              float const *var_z,
                                              int num_states, float *mu_a,
                                              float *jcb, float *var_a);

__global__ void mixture_tanh_mean_var_cuda(float const *mu_z,
                                           float const *var_z, int num_states,
                                           float *mu_a, float *jcb,
                                           float *var_a);

__global__ void celu_mean_var_cuda(float const *mu_z, float const *var_z,
                                   int num_states, float *mu_a, float *jcb,
                                   float *var_a);

__global__ void softplus_mean_var_cuda(float const *mu_z, float const *var_z,
                                       int num_states, float *mu_a, float *jcb,
                                       float *var_a);

__global__ void leakyrelu_mean_var_cuda(float const *mu_z, float const *var_z,
                                        float alpha, int num_states,
                                        float *mu_a, float *jcb, float *var_a);

__global__ void softmax_mean_var_cuda(float const *mu_z, float *var_z,
                                      size_t output_size, int batch_size,
                                      float *mu_a, float *jcb, float *var_a);

__global__ void exp_mean_var_cuda(float const *mu_z, float const *var_z,
                                  float const *jcb_z, int num_states,
                                  float *mu_a, float *var_a, float *jcb_a,
                                  float scale, float shift);

__global__ void split_stream_kernel(const float *d_in_mu, const float *d_in_var,
                                    const float *d_in_jcb, int half_size,
                                    float *d_even_mu, float *d_even_var,
                                    float *d_even_jcb, float *d_odd_mu,
                                    float *d_odd_var, float *d_odd_jcb);

__global__ void merge_stream_kernel(
    const float *d_even_mu, const float *d_even_var, const float *d_even_jcb,
    const float *d_odd_mu, const float *d_odd_var, const float *d_odd_jcb,
    int half_size, float *d_out_mu, float *d_out_var, float *d_out_jcb);

__global__ void agvi_extract_odd_stream_kernel(
    const float *d_input_mu, const float *d_input_var, const float *d_input_jcb,
    int half_size, float *d_odd_mu, float *d_odd_var, float *d_odd_jcb);

__global__ void agvi_forward_combine_kernel(
    const float *d_input_mu, const float *d_input_var, const float *d_input_jcb,
    const float *d_inner_output_mu, int half_size, float *d_output_mu,
    float *d_output_var, float *d_output_jcb, float *d_inner_output_jcb,
    bool agvi);

__global__ void agvi_backward_kernel(
    const float *d_incoming_delta_mu, const float *d_incoming_delta_var,
    const float *d_stored_output_mu_a, const float *d_stored_output_var_a,
    const float *d_stored_inner_mu_a, const float *d_stored_inner_var_a,
    const float *d_stored_inner_jcb, const float *d_stored_even_var_a,
    const float *d_stored_even_jcb, int half_size, float *d_output_delta_mu,
    float *d_output_delta_var, bool overfit_mu);

////////////////////////////////////////////////////////////////////////////////
/// Relu
////////////////////////////////////////////////////////////////////////////////
class ReLUCuda : public BaseLayerCuda {
   public:
    ReLUCuda();
    ~ReLUCuda();

    // Delete copy constructor and copy assignment
    ReLUCuda(const ReLUCuda &) = delete;
    ReLUCuda &operator=(const ReLUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    ReLUCuda(ReLUCuda &&) = default;
    ReLUCuda &operator=(ReLUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class SigmoidCuda : public BaseLayerCuda {
   public:
    SigmoidCuda();
    ~SigmoidCuda();

    // Delete copy constructor and copy assignment
    SigmoidCuda(const SigmoidCuda &) = delete;
    SigmoidCuda &operator=(const SigmoidCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    SigmoidCuda(SigmoidCuda &&) = default;
    SigmoidCuda &operator=(SigmoidCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class TanhCuda : public BaseLayerCuda {
   public:
    TanhCuda();

    ~TanhCuda();

    // Delete copy constructor and copy assignment
    TanhCuda(const TanhCuda &) = delete;
    TanhCuda &operator=(const TanhCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    TanhCuda(TanhCuda &&) = default;
    TanhCuda &operator=(TanhCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureReLUCuda : public BaseLayerCuda {
   public:
    MixtureReLUCuda();
    ~MixtureReLUCuda();

    // Delete copy constructor and copy assignment
    MixtureReLUCuda(const MixtureReLUCuda &) = delete;
    MixtureReLUCuda &operator=(const MixtureReLUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureReLUCuda(MixtureReLUCuda &&) = default;
    MixtureReLUCuda &operator=(MixtureReLUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureSigmoidCuda : public BaseLayerCuda {
   public:
    MixtureSigmoidCuda();
    ~MixtureSigmoidCuda();

    // Delete copy constructor and copy assignment
    MixtureSigmoidCuda(const MixtureSigmoidCuda &) = delete;
    MixtureSigmoidCuda &operator=(const MixtureSigmoidCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureSigmoidCuda(MixtureSigmoidCuda &&) = default;
    MixtureSigmoidCuda &operator=(MixtureSigmoidCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class MixtureTanhCuda : public BaseLayerCuda {
   public:
    MixtureTanhCuda();
    ~MixtureTanhCuda();

    // Delete copy constructor and copy assignment
    MixtureTanhCuda(const MixtureTanhCuda &) = delete;
    MixtureTanhCuda &operator=(const MixtureTanhCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureTanhCuda(MixtureTanhCuda &&) = default;
    MixtureTanhCuda &operator=(MixtureTanhCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class CELUCuda : public BaseLayerCuda {
   public:
    CELUCuda();
    ~CELUCuda();

    // Delete copy constructor and copy assignment
    CELUCuda(const CELUCuda &) = delete;
    CELUCuda &operator=(const CELUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    CELUCuda(CELUCuda &&) = default;
    CELUCuda &operator=(CELUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class SoftplusCuda : public BaseLayerCuda {
   public:
    SoftplusCuda();
    ~SoftplusCuda();

    // Delete copy constructor and copy assignment
    SoftplusCuda(const SoftplusCuda &) = delete;
    SoftplusCuda &operator=(const SoftplusCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    SoftplusCuda(SoftplusCuda &&) = default;
    SoftplusCuda &operator=(SoftplusCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class LeakyReLUCuda : public BaseLayerCuda {
   public:
    float alpha = 0.1f;
    LeakyReLUCuda();
    ~LeakyReLUCuda();

    // Delete copy constructor and copy assignment
    LeakyReLUCuda(const LeakyReLUCuda &) = delete;
    LeakyReLUCuda &operator=(const LeakyReLUCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    LeakyReLUCuda(LeakyReLUCuda &&) = default;
    LeakyReLUCuda &operator=(LeakyReLUCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

class SoftmaxCuda : public BaseLayerCuda {
   public:
    SoftmaxCuda();
    ~SoftmaxCuda();

    // Delete copy constructor and copy assignment
    SoftmaxCuda(const SoftmaxCuda &) = delete;
    SoftmaxCuda &operator=(const SoftmaxCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    SoftmaxCuda(SoftmaxCuda &&) = default;
    SoftmaxCuda &operator=(SoftmaxCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;
};

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
class RemaxCuda : public BaseLayerCuda {
   public:
    float alpha = 0.1f;
    std::vector<float> mu_m;
    std::vector<float> var_m;
    std::vector<float> jcb_m;
    std::vector<float> mu_log_m;
    std::vector<float> var_log_m;
    std::vector<float> mu_mt;
    std::vector<float> var_mt;
    std::vector<float> mu_log_mt;
    std::vector<float> var_log_mt;
    std::vector<float> cov_log_m_mt;
    float threshold = 1e-10;
    int batch_size_ = 0;
    float *d_mu_m = nullptr;
    float *d_var_m = nullptr;
    float *d_jcb_m = nullptr;
    float *d_mu_log_m = nullptr;
    float *d_var_log_m = nullptr;
    float *d_mu_mt = nullptr;
    float *d_var_mt = nullptr;
    float *d_mu_log_mt = nullptr;
    float *d_var_log_mt = nullptr;
    float *d_cov_log_m_mt = nullptr;

    RemaxCuda();
    ~RemaxCuda();

    // Delete copy constructor and copy assignment
    RemaxCuda(const RemaxCuda &) = delete;
    RemaxCuda &operator=(const RemaxCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    RemaxCuda(RemaxCuda &&) = default;
    RemaxCuda &operator=(RemaxCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;

    void data_to_host();
    void data_to_device();

   private:
    void allocate_memory(int hidden_size, int batch_size);
    void deallocate_memory();
};

////////////////////////////////////////////////////////////////////////////////
/// ClosedFormSoftmax
////////////////////////////////////////////////////////////////////////////////
class ClosedFormSoftmaxCuda : public BaseLayerCuda {
   public:
    std::vector<float> mu_e_sum;
    std::vector<float> var_e_sum;
    std::vector<float> cov_z_log_e_sum;
    std::vector<float> mu_log_e_sum;
    std::vector<float> var_log_e_sum;
    std::vector<float> cov_log_a_z;
    std::vector<float> mu_log_a;
    std::vector<float> var_log_a;

    int batch_size_ = 0;
    float *d_mu_e_sum = nullptr;
    float *d_var_e_sum = nullptr;
    float *d_cov_z_log_e_sum = nullptr;
    float *d_mu_log_e_sum = nullptr;
    float *d_var_log_e_sum = nullptr;
    float *d_cov_log_a_z = nullptr;
    float *d_mu_log_a = nullptr;
    float *d_var_log_a = nullptr;

    ClosedFormSoftmaxCuda();
    ~ClosedFormSoftmaxCuda();

    // Delete copy constructor and copy assignment
    ClosedFormSoftmaxCuda(const ClosedFormSoftmaxCuda &) = delete;
    ClosedFormSoftmaxCuda &operator=(const ClosedFormSoftmaxCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    ClosedFormSoftmaxCuda(ClosedFormSoftmaxCuda &&) = default;
    ClosedFormSoftmaxCuda &operator=(ClosedFormSoftmaxCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;

    void data_to_host();
    void data_to_device();

   private:
    void allocate_memory(int hidden_size, int batch_size);
    void deallocate_memory();
};

////////////////////////////////////////////////////////////////////////////////
/// SplitActivationCuda
////////////////////////////////////////////////////////////////////////////////
class SplitActivationCuda : public BaseLayerCuda {
   public:
    SplitActivationCuda(std::unique_ptr<BaseLayer> odd_layer,
                        std::unique_ptr<BaseLayer> even_layer = nullptr);
    ~SplitActivationCuda();

    // Delete copy constructor and copy assignment
    SplitActivationCuda(const SplitActivationCuda &) = delete;
    SplitActivationCuda &operator=(const SplitActivationCuda &) = delete;

    // Default move constructor and move assignment
    SplitActivationCuda(SplitActivationCuda &&) = default;
    SplitActivationCuda &operator=(SplitActivationCuda &&) = default;

    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    using BaseLayer::backward;

    std::unique_ptr<BaseLayer> to_host() override;

    // Methods that do nothing for this layer type
    void allocate_param_delta() override {};
    void update_weights() override {};
    void update_biases() override {};
    void save(std::ofstream &file) override {};
    void load(std::ifstream &file) override {};

   private:
    std::unique_ptr<BaseLayer> odd_layer;
    std::unique_ptr<BaseLayer> even_layer;
};

class ExpCuda : public BaseLayerCuda {
   public:
    ExpCuda(float scale = 1.0f, float shift = 0.0f);
    ~ExpCuda();

    unsigned int num_cuda_threads = 16;

    // Delete copy constructor and copy assignment
    ExpCuda(const ExpCuda &) = delete;
    ExpCuda &operator=(const ExpCuda &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    ExpCuda(ExpCuda &&) = default;
    ExpCuda &operator=(ExpCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

    std::unique_ptr<BaseLayer> to_host() override;

    float scale;
    float shift;
};

////////////////////////////////////////////////////////////////////////////////
/// AGVI Cuda
////////////////////////////////////////////////////////////////////////////////
class AGVICuda : public BaseLayerCuda {
   public:
    /**
     * @brief Construct a new AGVICuda object.
     *
     * @param odd_layer A unique_ptr to a CUDA-enabled activation layer for the
     * odd stream. The AGVICuda instance takes ownership.
     * @param even_layer An optional unique_ptr to a CUDA-enabled activation
     * layer for the even stream. Defaults to an Identity operation.
     * @param overfit_mu If true, uses a different Kalman gain for the mean
     * delta to encourage overfitting.
     * @param agvi If true, uses the AGVI learned noise model. Defaults to true.
     */
    explicit AGVICuda(std::unique_ptr<BaseLayer> odd_layer,
                      std::unique_ptr<BaseLayer> even_layer = nullptr,
                      bool overfit_mu = true, bool agvi = true);
    ~AGVICuda();

    // Standard move semantics are enabled.
    AGVICuda(AGVICuda &&) = default;
    AGVICuda &operator=(AGVICuda &&) = default;

    // Copy semantics are disabled to prevent duplicate ownership of resources.
    AGVICuda(const AGVICuda &) = delete;
    AGVICuda &operator=(const AGVICuda &) = delete;

    // Overridden BaseLayer methods
    std::string get_layer_info() const override;
    std::string get_layer_name() const override;
    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_update = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    // Methods that do nothing for this layer type as it has no trainable
    // parameters
    void allocate_param_delta() override {};
    void update_weights() override {};
    void update_biases() override {};
    void save(std::ofstream &file) override {};
    void load(std::ifstream &file) override {};

    // Runtime configuration setters and getters
    void set_overfit_mu(bool overfit_mu) { m_overfit_mu = overfit_mu; }
    bool get_overfit_mu() const { return m_overfit_mu; }
    void set_agvi(bool agvi) { m_agvi = agvi; }
    bool get_agvi() const { return m_agvi; }

   private:
    std::unique_ptr<BaseLayer> m_odd_layer;
    std::unique_ptr<BaseLayer> m_even_layer;
    bool m_overfit_mu;
    bool m_agvi;

    // Pointers to device memory from other states, stored for backward pass.
    // These do not own the memory.
    const float *d_stored_output_mu_a = nullptr;
    const float *d_stored_output_var_a = nullptr;

    // The full state objects for intermediate results are stored here to
    // correctly manage their device memory.
    HiddenStateCuda
        m_stored_inner_output_states;  // Output of the odd activation
    HiddenStateCuda m_stored_even_output_states;  // Output of the (optional)
                                                  // even activation
};

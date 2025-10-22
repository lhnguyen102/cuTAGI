#pragma once
#include <algorithm>
#include <memory>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "common.h"
#include "data_struct.h"

void relu_mean_var(std::vector<float> const &mu_z,
                   std::vector<float> const &var_z, int start_chunk,
                   int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a);

void relu_mean_var_mp(std::vector<float> const &mu_z,
                      std::vector<float> const &var_z, int n,
                      unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void sigmoid_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int start_chunk, int end_chunk, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void sigmoid_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                         int n, unsigned int num_threads,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a);

void tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                   int start_chunk, int end_chunk, std::vector<float> &mu_a,
                   std::vector<float> &jcb, std::vector<float> &var_a);

void tanh_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int n, unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void mixture_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a);

void mixture_relu_mean_var_mp(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int n,
                              unsigned int num_threads,
                              std::vector<float> &mu_a, std::vector<float> &jcb,
                              std::vector<float> &var_a);

void mixture_sigmoid_mean_var(std::vector<float> &mu_z,
                              std::vector<float> &var_z, int start_chunk,
                              int end_chunk, std::vector<float> &mu_a,
                              std::vector<float> &jcb,
                              std::vector<float> &var_a);

void mixture_sigmoid_mean_var_mp(std::vector<float> &mu_z,
                                 std::vector<float> &var_z, int n,
                                 unsigned int num_threads,
                                 std::vector<float> &mu_a,
                                 std::vector<float> &jcb,
                                 std::vector<float> &var_a);

void mixture_tanh_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                           int start_chunk, int end_chunk,
                           std::vector<float> &mu_a, std::vector<float> &jcb,
                           std::vector<float> &var_a);

void celu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int n, unsigned int num_threads, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void softplus_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                       int start_chunk, int end_chunk, std::vector<float> &mu_a,
                       std::vector<float> &jcb, std::vector<float> &var_a);

void softplus_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                          int n, unsigned int num_threads,
                          std::vector<float> &mu_a, std::vector<float> &jcb,
                          std::vector<float> &var_a);

void leaky_relu_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                         float alpha, int start_chunk, int end_chunk,
                         std::vector<float> &mu_a, std::vector<float> &jcb,
                         std::vector<float> &var_a);

void leaky_relu_mean_var_mp(std::vector<float> &mu_z, std::vector<float> &var_z,
                            float alpha, int n, unsigned int num_threads,
                            std::vector<float> &mu_a, std::vector<float> &jcb,
                            std::vector<float> &var_a);

void softmax_mean_var(std::vector<float> &mu_z, std::vector<float> &var_z,
                      int no, int batch_size, std::vector<float> &mu_a,
                      std::vector<float> &jcb, std::vector<float> &var_a);

void exp_mean_var(std::vector<float> const &mu_z,
                  std::vector<float> const &var_z, std::vector<float> &jcb_z,
                  int start_chunk, int end_chunk, std::vector<float> &mu_a,
                  std::vector<float> &var_a, std::vector<float> &jcb_a,
                  float scale, float shift);

void exp_mean_var_mp(std::vector<float> const &mu_z,
                     std::vector<float> const &var_z,
                     std::vector<float> const &jcb_z, int n,
                     unsigned int num_threads, std::vector<float> &mu_a,
                     std::vector<float> &var_a, std::vector<float> &jcb_a,
                     float scale, float shift);

void agvi_backward_chunk(int start_chunk, int end_chunk,
                         BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         const BaseHiddenStates &stored_output_states,
                         const BaseHiddenStates &stored_inner_output_states,
                         const BaseHiddenStates &stored_input_states,
                         const BaseHiddenStates &stored_even_output_states,
                         bool overfit_mu, bool has_even_layer);

void agvi_backward_mp(int n, unsigned int num_threads,
                      BaseDeltaStates &input_delta_states,
                      BaseDeltaStates &output_delta_states,
                      const BaseHiddenStates &stored_output_states,
                      const BaseHiddenStates &stored_inner_output_states,
                      const BaseHiddenStates &stored_input_states,
                      const BaseHiddenStates &stored_even_output_states,
                      bool overfit_mu, bool has_even_layer);

////////////////////////////////////////////////////////////////////////////////
/// ReLU
////////////////////////////////////////////////////////////////////////////////
class ReLU : public BaseLayer {
   public:
    ReLU();
    ~ReLU();

    // Delete copy constructor and copy assignment
    ReLU(const ReLU &) = delete;
    ReLU &operator=(const ReLU &) = delete;

    // Optionally implement move constructor and move assignment
    ReLU(ReLU &&) = default;
    ReLU &operator=(ReLU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Sigmoid
////////////////////////////////////////////////////////////////////////////////
class Sigmoid : public BaseLayer {
   public:
    Sigmoid();
    ~Sigmoid();

    // Delete copy constructor and copy assignment
    Sigmoid(const Sigmoid &) = delete;
    Sigmoid &operator=(const Sigmoid &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Sigmoid(Sigmoid &&) = default;
    Sigmoid &operator=(Sigmoid &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Tanh
////////////////////////////////////////////////////////////////////////////////
class Tanh : public BaseLayer {
   public:
    Tanh();
    ~Tanh();

    // Delete copy constructor and copy assignment
    Tanh(const Tanh &) = delete;
    Tanh &operator=(const Tanh &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Tanh(Tanh &&) = default;
    Tanh &operator=(Tanh &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void allocate_param_delta() override {};

    using BaseLayer::backward;

    void update_weights() override {};

    void update_biases() override {};

    void save(std::ofstream &file) override {};

    void load(std::ifstream &file) override {};

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture ReLU
////////////////////////////////////////////////////////////////////////////////
class MixtureReLU : public BaseLayer {
   public:
    MixtureReLU();
    ~MixtureReLU();

    // Delete copy constructor and copy assignment
    MixtureReLU(const MixtureReLU &) = delete;
    MixtureReLU &operator=(const MixtureReLU &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureReLU(MixtureReLU &&) = default;
    MixtureReLU &operator=(MixtureReLU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Sigmoid
////////////////////////////////////////////////////////////////////////////////
class MixtureSigmoid : public BaseLayer {
   public:
    MixtureSigmoid();
    ~MixtureSigmoid();

    // Delete copy constructor and copy assignment
    MixtureSigmoid(const MixtureSigmoid &) = delete;
    MixtureSigmoid &operator=(const MixtureSigmoid &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureSigmoid(MixtureSigmoid &&) = default;
    MixtureSigmoid &operator=(MixtureSigmoid &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Mixture Tanh
////////////////////////////////////////////////////////////////////////////////
class MixtureTanh : public BaseLayer {
   public:
    MixtureTanh();
    ~MixtureTanh();

    // Delete copy constructor and copy assignment
    MixtureTanh(const MixtureTanh &) = delete;
    MixtureTanh &operator=(const MixtureTanh &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    MixtureTanh(MixtureTanh &&) = default;
    MixtureTanh &operator=(MixtureTanh &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// CELU
////////////////////////////////////////////////////////////////////////////////
class CELU : public BaseLayer {
   public:
    CELU();
    ~CELU();

    // Delete copy constructor and copy assignment
    CELU(const CELU &) = delete;
    CELU &operator=(const CELU &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    CELU(CELU &&) = default;
    CELU &operator=(CELU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Softplus
////////////////////////////////////////////////////////////////////////////////
class Softplus : public BaseLayer {
   public:
    Softplus();
    ~Softplus();

    // Delete copy constructor and copy assignment
    Softplus(const Softplus &) = delete;
    Softplus &operator=(const Softplus &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Softplus(Softplus &&) = default;
    Softplus &operator=(Softplus &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Leaky ReLU
////////////////////////////////////////////////////////////////////////////////
class LeakyReLU : public BaseLayer {
   public:
    float alpha = 0.1f;
    LeakyReLU();
    ~LeakyReLU();

    // Delete copy constructor and copy assignment
    LeakyReLU(const LeakyReLU &) = delete;
    LeakyReLU &operator=(const LeakyReLU &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    LeakyReLU(LeakyReLU &&) = default;
    LeakyReLU &operator=(LeakyReLU &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Stable Softmax
////////////////////////////////////////////////////////////////////////////////
class Softmax : public BaseLayer {
   public:
    float alpha = 0.1f;
    Softmax();
    ~Softmax();

    // Delete copy constructor and copy assignment
    Softmax(const Softmax &) = delete;
    Softmax &operator=(const Softmax &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Softmax(Softmax &&) = default;
    Softmax &operator=(Softmax &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Remax
////////////////////////////////////////////////////////////////////////////////
class Remax : public BaseLayer {
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
    Remax();
    ~Remax();

    // Delete copy constructor and copy assignment
    Remax(const Remax &) = delete;
    Remax &operator=(const Remax &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    Remax(Remax &&) = default;
    Remax &operator=(Remax &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// ClosedFormSoftmax
////////////////////////////////////////////////////////////////////////////////
class ClosedFormSoftmax : public BaseLayer {
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

    ClosedFormSoftmax();
    ~ClosedFormSoftmax();

    // Delete copy constructor and copy assignment
    ClosedFormSoftmax(const ClosedFormSoftmax &) = delete;
    ClosedFormSoftmax &operator=(const ClosedFormSoftmax &) = delete;

    // Optionally implement move constructor and move assignment. This is
    // required for bwd_states
    ClosedFormSoftmax(ClosedFormSoftmax &&) = default;
    ClosedFormSoftmax &operator=(ClosedFormSoftmax &&) = default;

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

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// Exp
////////////////////////////////////////////////////////////////////////////////
class Exp : public BaseLayer {
   public:
    Exp(float scale = 1.0f, float shift = 0.0f);
    ~Exp();

    // Delete copy constructor and copy assignment
    Exp(const Exp &) = delete;
    Exp &operator=(const Exp &) = delete;

    // Optionally implement move constructor and move assignment
    Exp(Exp &&) = default;
    Exp &operator=(Exp &&) = default;

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

    float scale;
    float shift;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
};

////////////////////////////////////////////////////////////////////////////////
/// SplitActivation (formerly EvenExp)
////////////////////////////////////////////////////////////////////////////////
class SplitActivation : public BaseLayer {
   public:
    SplitActivation(std::shared_ptr<BaseLayer> odd_layer,
                    std::shared_ptr<BaseLayer> even_layer = nullptr);
    ~SplitActivation();

    // Delete copy constructor and copy assignment
    SplitActivation(const SplitActivation &) = delete;
    SplitActivation &operator=(const SplitActivation &) = delete;

    // Optionally implement move constructor and move assignment
    SplitActivation(SplitActivation &&) = default;
    SplitActivation &operator=(SplitActivation &&) = default;

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

    void save(std::ofstream &file) override;

    void load(std::ifstream &file) override;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif
   private:
    std::shared_ptr<BaseLayer> odd_layer;
    std::shared_ptr<BaseLayer> even_layer;
};

////////////////////////////////////////////////////////////////////////////////
/// AGVI (Approximate Gaussian Variance Inference)
////////////////////////////////////////////////////////////////////////////////
class AGVI : public BaseLayer {
   public:
    /**
     * @brief Construct a new AGVI object
     *
     * @param activation_layer The inner activation layer to be used.
     * @param overfit_mu If true, uses a different Jacobian for the mean delta
     * to encourage overfitting. Defaults to true.
     * @param agvi If true, uses the AGVI learned noise model. Defaults to true.
     */
    explicit AGVI(std::shared_ptr<BaseLayer> odd_layer,
                  std::shared_ptr<BaseLayer> even_layer = nullptr,
                  bool overfit_mu = true, bool agvi = true);

    ~AGVI();

    AGVI(const AGVI &) = delete;
    AGVI &operator=(const AGVI &) = delete;

    // Default move constructor and move assignment.
    AGVI(AGVI &&) = default;
    AGVI &operator=(AGVI &&) = default;

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

    void allocate_param_delta() override {};
    void update_weights() override {};
    void update_biases() override {};
    void save(std::ofstream &file) override {};
    void load(std::ifstream &file) override {};
#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif

    // Runtime configuration
    void set_overfit_mu(bool overfit_mu) { m_overfit_mu = overfit_mu; }
    bool get_overfit_mu() const { return m_overfit_mu; }
    void set_agvi(bool agvi) { m_agvi = agvi; }
    bool get_agvi() const { return m_agvi; }

   private:
    std::shared_ptr<BaseLayer> m_odd_layer;
    std::shared_ptr<BaseLayer> m_even_layer;
    bool m_overfit_mu;
    bool m_agvi;

    // Stored hidden states for backward pass usage
    BaseHiddenStates m_stored_inner_output_states;
    BaseHiddenStates m_stored_output_states;
    BaseHiddenStates m_stored_input_states;
    BaseHiddenStates m_stored_even_output_states;
};

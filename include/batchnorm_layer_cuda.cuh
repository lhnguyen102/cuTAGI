#pragma once

#include "base_layer_cuda.cuh"
#include "batchnorm_layer.h"
#include "config.h"
#include "cuda_error_checking.cuh"
#include "custom_logger.h"

class BatchNorm2dCuda : public BaseLayerCuda {
   public:
    int num_features;
    std::vector<float> mu_ra, var_ra, mu_norm_batch, var_norm_batch;
    float *d_mu_ra, *d_var_ra, *d_mu_norm_batch, *d_var_norm_batch;
    float epsilon;
    float momentum;
    float gain_w, gain_b;

    // momentum of running average of first batch is set to zero
    bool first_batch = true;

    BatchNorm2dCuda(int num_features, float eps = 1e-5, float mometum = 0.9,
                    bool bias = true, float gain_weight = 1.0,
                    float gain_bias = 1.0);
    ~BatchNorm2dCuda();

    // Delete copy constructor and copy assignment
    BatchNorm2dCuda(const BatchNorm2dCuda &) = delete;
    BatchNorm2dCuda &operator=(const BatchNorm2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    BatchNorm2dCuda(BatchNorm2dCuda &&) = default;
    BatchNorm2dCuda &operator=(BatchNorm2dCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias();

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    get_norm_mean_var() override;

   protected:
    void allocate_running_mean_var();
    void deallocate_running_mean_var();
    void running_mean_var_to_host();
    void running_mean_var_to_device();
    void lazy_init();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};
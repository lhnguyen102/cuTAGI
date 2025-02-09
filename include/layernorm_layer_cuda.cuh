#pragma once

#include "base_layer_cuda.cuh"
#include "config.h"
#include "cuda_error_checking.cuh"
#include "custom_logger.h"
#include "layernorm_layer.h"

class LayerNormCuda : public BaseLayerCuda {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> mu_ra, var_ra;
    float *d_mu_ra = nullptr, *d_var_ra = nullptr;

    float epsilon;
    int _batch_size = 0;

    LayerNormCuda(const std::vector<int> &normalized_shape, float eps = 1e-5,
                  bool bias = true);
    ~LayerNormCuda();

    // Delete copy constructor and copy assignment
    LayerNormCuda(const LayerNormCuda &) = delete;
    LayerNormCuda &operator=(const LayerNormCuda &) = delete;

    // Optionally implement move constructor and move assignment
    LayerNormCuda(LayerNormCuda &&) = default;
    LayerNormCuda &operator=(LayerNormCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void init_weight_bias() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    std::unique_ptr<BaseLayer> to_host() override;

    // DEBUG
    std::tuple<std::vector<float>, std::vector<float>> get_running_mean_var()
        override;

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    get_norm_mean_var() override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

   protected:
    void allocate_running_mean_var();
    void deallocate_running_mean_var();
    void running_mean_var_to_host();
    void running_mean_var_to_device();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};

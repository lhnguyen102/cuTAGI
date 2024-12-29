#pragma once
#include <iostream>
#include <memory>
#include <vector>

#include "base_layer.h"
#include "custom_logger.h"
#include "data_struct_cuda.cuh"

__global__ void fill_bwd_states_on_device(float const *mu_a_in,
                                          float const *jcb_in, int size,
                                          float *mu_a, float *jcb);

__global__ void fill_output_states_on_device(int size, float *jcb);

class BaseLayerCuda : public BaseLayer {
   public:
    float *d_mu_w = nullptr;
    float *d_var_w = nullptr;
    float *d_mu_b = nullptr;
    float *d_var_b = nullptr;
    float *d_delta_mu_w = nullptr;
    float *d_delta_var_w = nullptr;
    float *d_delta_mu_b = nullptr;
    float *d_delta_var_b = nullptr;
    int *d_neg_var_count = nullptr;
    unsigned int num_cuda_threads = 16;

    BaseLayerCuda();

    ~BaseLayerCuda();

    // Delete copy constructor and copy assignment
    BaseLayerCuda(const BaseLayerCuda &) = delete;
    BaseLayerCuda &operator=(const BaseLayerCuda &) = delete;

    // Optionally implement move constructor and move assignment
    BaseLayerCuda(BaseLayerCuda &&) = default;
    BaseLayerCuda &operator=(BaseLayerCuda &&) = default;

    using BaseLayer::forward;

    void allocate_param_delta() override;

    void update_weights() override;

    void update_biases() override;

    void raw_update_weights() override;

    void raw_update_biases() override;

    virtual void set_cuda_threads(int);

    virtual std::unique_ptr<BaseLayer> to_host();
    virtual void params_to_device();
    virtual void params_to_host();
    virtual void delta_params_to_host();

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

    // Get Parameters
    ParameterMap get_parameters_as_map(std::string suffix = "") override;
    void load_parameters_from_map(const ParameterMap &param_map,
                                  const std::string &suffix = "") override;
    std::vector<ParameterTuple> parameters() override;

    void copy_params_from(const BaseLayer &source);

   protected:
    virtual void allocate_param_memory();
    virtual void store_states_for_training_cuda(HiddenStateCuda &input_states,
                                                HiddenStateCuda &output_states);
};

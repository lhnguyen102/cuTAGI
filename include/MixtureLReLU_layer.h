#pragma once
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
//#include "param_init.h"

void MixtureLReLU_fwd_mean_var(float slope,
                         std::vector<float> &mu_z, std::vector<float> &var_z,
                         int start_chunk, int end_chunk, size_t input_size,
                         int batch_size,
                         std::vector<float> &mu_a, std::vector<float> &var_a);

void MixtureLReLU_fwd_mean_var_mp(float slope,
                            std::vector<float> &mu_z, std::vector<float> &var_z,
                            size_t input_size,
                            int batch_size, unsigned int num_threads,
                            std::vector<float> &mu_a,
                            std::vector<float> &var_a);

void MixtureLReLU_bwd_delta_z(float slope,
                           std::vector<float> &mu_a, std::vector<float> &var_a,
                           std::vector<float> &mu_z, std::vector<float> &var_z,
                           std::vector<float> &delta_mu,
                           std::vector<float> &delta_var, size_t input_size,
                           int B, int start_chunk,
                           int end_chunk, std::vector<float> &delta_mu_z,
                           std::vector<float> &delta_var_z);

void MixtureLReLU_bwd_delta_z_mp(float slope,
                            std::vector<float> &mu_a, std::vector<float> &var_a,
                            std::vector<float> &mu_z, std::vector<float> &var_z,
                            std::vector<float> &delta_mu,
                            std::vector<float> &delta_var, size_t input_size,
                            int batch_size,
                            unsigned int num_threads,
                            std::vector<float> &delta_mu_z,
                            std::vector<float> &delta_var_z);

class MixtureLReLU : public BaseLayer
{
public:
    float slope;

    MixtureLReLU(size_t ip_size, float slope = 0.1f);

    ~MixtureLReLU();

    // Delete copy constructor and copy assignment
    MixtureLReLU(const MixtureLReLU &) = delete;
    MixtureLReLU &operator=(const MixtureLReLU &) = delete;

    // Optionally implement move constructor and move assignment
    MixtureLReLU(MixtureLReLU &&) = default;
    MixtureLReLU &operator=(MixtureLReLU &&) = default;

    virtual std::string get_layer_info() const override;

    virtual std::string get_layer_name() const override;

    virtual LayerType get_layer_type() const override;

    //void init_weight_bias() override;

    virtual void forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states) override;

    virtual void backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states,
                          bool state_udapte) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif
};

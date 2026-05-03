#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "base_layer.h"
#include "base_layer_cuda.cuh"
#include "rmsnorm_layer.h"

class RMSNormCuda : public BaseLayerCuda {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> rms_ra;
    float *d_rms_ra = nullptr;

    float epsilon;
    float gain_w;
    int _batch_size = 0;
    bool debug = false;
    int debug_interval = 1;
    int _debug_step = 0;

    RMSNormCuda(const std::vector<int> &normalized_shape, float eps = 1e-5f,
                float gain_w = 1.0f, int device_idx = 0);
    ~RMSNormCuda();

    RMSNormCuda(const RMSNormCuda &) = delete;
    RMSNormCuda &operator=(const RMSNormCuda &) = delete;
    RMSNormCuda(RMSNormCuda &&) = default;
    RMSNormCuda &operator=(RMSNormCuda &&) = default;

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

    void update_weights() override;

    std::unique_ptr<BaseLayer> to_host() override;

   protected:
    void allocate_running_rms();
    void deallocate_running_rms();
    using BaseLayerCuda::allocate_param_memory;
    using BaseLayerCuda::params_to_device;
};

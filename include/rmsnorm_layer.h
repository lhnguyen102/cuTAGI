#pragma once

#include "base_layer.h"
#include "common.h"

std::tuple<int, int> get_number_params_rms_norm(
    const std::vector<int> &normalized_shape);

class RMSNorm : public BaseLayer {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> rms_ra;
    float epsilon;
    int _batch_size = 0;

    RMSNorm(const std::vector<int> &normalized_shape, float eps = 1e-6,
            int device_idx = 0);
    ~RMSNorm();

    RMSNorm(const RMSNorm &) = delete;
    RMSNorm &operator=(const RMSNorm &) = delete;

    RMSNorm(RMSNorm &&) = default;
    RMSNorm &operator=(RMSNorm &&) = default;

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

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda(int device_idx = 0) override;
#endif

   protected:
    void allocate_running_rms();
};

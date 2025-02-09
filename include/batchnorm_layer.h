#pragma once

#include <tuple>

#include "base_layer.h"
#include "common.h"

class BatchNorm2d : public BaseLayer {
   public:
    int num_features;
    std::vector<float> mu_ra, var_ra, mu_norm_batch, var_norm_batch;
    float epsilon;
    float momentum;
    float gain_w, gain_b;

    // momentum of running average of first batch is set to zero
    bool first_batch = true;

    BatchNorm2d(int num_features, float eps = 1e-5, float mometum = 0.9,
                bool bias = true, float gain_weight = 1.0f,
                float gain_bias = 1.0f);
    ~BatchNorm2d();

    // Delete copy constructor and copy assignment
    BatchNorm2d(const BatchNorm2d &) = delete;
    BatchNorm2d &operator=(const BatchNorm2d &) = delete;

    // Optionally implement move constructor and move assignment
    BatchNorm2d(BatchNorm2d &&) = default;
    BatchNorm2d &operator=(BatchNorm2d &&) = default;

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

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    get_norm_mean_var() override;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

   protected:
    void allocate_running_mean_var();
};
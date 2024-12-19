#pragma once

#include "base_layer.h"
#include "common.h"

std::tuple<int, int> get_number_params_layer_norm(
    const std::vector<int> &normalized_shape);

class LayerNorm : public BaseLayer {
   public:
    std::vector<int> normalized_shape;
    std::vector<float> mu_ra, var_ra;
    float epsilon;
    bool bias;
    int _batch_size = 0;

    LayerNorm(const std::vector<int> &normalized_shape, float eps = 1e-5,
              bool bias = true);
    ~LayerNorm();

    // Delete copy constructor and copy assignment
    LayerNorm(const LayerNorm &) = delete;
    LayerNorm &operator=(const LayerNorm &) = delete;

    // Optionally implement move constructor and move assignment
    LayerNorm(LayerNorm &&) = default;
    LayerNorm &operator=(LayerNorm &&) = default;

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

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    get_norm_mean_var() override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    // DEBUG
    std::tuple<std::vector<float>, std::vector<float>> get_running_mean_var()
        override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

   protected:
    void allocate_running_mean_var();
};
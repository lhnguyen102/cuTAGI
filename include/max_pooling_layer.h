#pragma once
#include "include/base_layer.h"

class MaxPool2d : public BaseLayer {
   public:
    MaxPool2d(size_t kernel_size, int stride = -1, int padding = 0,
              int padding_type = 0);
    size_t kernel_size = 0;
    int stride = 0;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> pool_idx, z_ud_idx;
    bool overlap = true;

    ~MaxPool2d();

    // Delete copy constructor and copy assignment
    MaxPool2d(const MaxPool2d &) = delete;
    MaxPool2d &operator=(const MaxPool2d &) = delete;

    // Optionally implement move constructor and move assignment
    MaxPool2d(MaxPool2d &&) = default;
    MaxPool2d &operator=(MaxPool2d &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    void compute_input_output_size(const InitArgs &args) override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_udapte = true) override;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    using BaseLayer::storing_states_for_training;
    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void preinit_layer() override;

   protected:
    void lazy_index_init();
};
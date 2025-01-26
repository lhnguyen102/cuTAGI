#pragma once
#include "base_layer_cuda.cuh"

class MaxPool2dCuda : public BaseLayerCuda {
   public:
    size_t kernel_size = 0;
    int stride = 0;
    int padding_type = 1;
    int padding = 0;
    std::vector<int> pool_idx, max_pool_idx;
    size_t row_zw = 0, col_z_ud = 0;
    bool overlap = true;
    int _batch_size = 0;

    int *d_pool_idx, *d_max_pool_idx;

    MaxPool2dCuda(size_t kernel_size, int stride = 1, int padding = 0,
                  int padding_type = 0);

    ~MaxPool2dCuda();

    // Delete copy constructor and copy assignment
    MaxPool2dCuda(const MaxPool2dCuda &) = delete;
    MaxPool2dCuda &operator=(const MaxPool2dCuda &) = delete;

    // Optionally implement move constructor and move assignment
    MaxPool2dCuda(MaxPool2dCuda &&) = default;
    MaxPool2dCuda &operator=(MaxPool2dCuda &&) = default;

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
                  bool state_update = true) override;

    void allocate_param_delta() override {};

    void update_weights() override {};

    void update_biases() override {};

    void preinit_layer() override;

   protected:
    void lazy_index_init();
    void allocate_maxpool2d_index();
    void maxpool2d_index_to_device();
    void allocate_max_val_index(int batch_size);
    void max_val_index_to_device();
    void max_val_index_to_host();
};
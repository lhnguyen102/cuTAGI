#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

class LayerBlock : public BaseLayer {
   public:
    std::vector<std::shared_ptr<BaseLayer>> layers;

    // Variadic template
    template <typename... Layers>
    LayerBlock(Layers &&...layers) {
        add_layers(std::forward<Layers>(layers)...);
    }
    // Recursive variadic template
    template <typename T, typename... Rest>
    void add_layers(T &&first, Rest &&...rest) {
        static_assert(
            std::is_base_of<BaseLayer, typename std::decay<T>::type>::value,
            "Type T must be derived from BaseLayer");

        add_layer(std::make_shared<typename std::remove_reference<T>::type>(
            std::forward<T>(first)));
        add_layers(std::forward<Rest>(rest)...);
    }
    // Base case for recursive variadic template. This function is called after
    // the last argument
    void add_layers();
    void add_layer(std::shared_ptr<BaseLayer> layer);

    LayerBlock();

    ~LayerBlock();

    // Delete copy constructor and copy assignment
    LayerBlock(const LayerBlock &) = delete;
    LayerBlock &operator=(const LayerBlock &) = delete;

    // Optionally implement move constructor and move assignment
    LayerBlock(LayerBlock &&) = default;
    LayerBlock &operator=(LayerBlock &&) = default;

    void switch_to_cuda();

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    int get_max_num_states() override;

    std::string get_device() override;

    void init_weight_bias() override;

    void set_threads(int num) override;

    void train() override;
    void eval() override;

#ifdef USE_CUDA
    void set_cuda_threads(int num);
#endif

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_update = true) override;

    void update_weights() override;
    void update_biases() override;

    void compute_input_output_size(const InitArgs &args) override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

    using BaseLayer::to_cuda;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void preinit_layer() override;
};
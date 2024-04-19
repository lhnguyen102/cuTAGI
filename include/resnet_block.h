#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"

class LayerBlock : public BaseLayer {
   public:
    std::vector<std::shared_ptr<BaseLayer>> layers;

    // Variadic template. Note that for the template function the definition of
    // template must be included in the herder
    template <typename... Layers>
    LayerBlock(Layers &&...layers) {
        add_layers(std::forward<Layers>(layers)...);
    }
    // Recursive variadic template
    template <typename T, typename... Rest>
    void add_layers(T &&first, Rest &&...rest) {
        // Runtime check to verify if T is derived from BaseLayer
        if (!std::is_base_of<BaseLayer,
                             typename std::remove_reference<T>::type>::value) {
            std::cerr << "Error in file: " << __FILE__
                      << " at line: " << __LINE__
                      << ". Reason: Type T must be derived from BaseLayer.\n";
            throw std::invalid_argument(
                "Error: Type T must be derived from BaseLayer");
        }

        // Add layer using shared_ptr
        add_layer(std::make_shared<T>(std::forward<T>(first)));

        // Recursively adding next layer
        add_layers(std::forward<Rest>(rest)...);
    }
    // Base case for recursive variadic template. This function is called after
    // the last argument
    void add_layers();

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

    void init_weight_bias() override;

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
};

class ResNetBlock : public BaseLayer {
   private:
    std::shared_ptr<LayerBlock> main_block;
    std::shared_ptr<BaseLayer> shortcut;
    int _batch_size = 0;

   public:
    std::shared_ptr<BaseHiddenStates> shortcut_output_z;
    std::shared_ptr<BaseDeltaStates> shortcut_output_delta_z;

    ResNetBlock(std::shared_ptr<BaseLayer> main_block_layer,
                std::shared_ptr<BaseLayer> shortcut_layer = nullptr);
    ~ResNetBlock();

    // Delete copy constructor and copy assignment
    ResNetBlock(const ResNetBlock &) = delete;
    ResNetBlock &operator=(const ResNetBlock &) = delete;

    // Optionally implement move constructor and move assignment
    ResNetBlock(ResNetBlock &&) = default;
    ResNetBlock &operator=(ResNetBlock &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    int get_max_num_states() override;

    void init_shortcut_state();

    void init_shortcut_delta_state();

    void init_weight_bias();

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
};
}
;
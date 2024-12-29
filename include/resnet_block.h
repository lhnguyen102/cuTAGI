#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "base_layer.h"
#include "data_struct.h"
#include "layer_block.h"

class ResNetBlock : public BaseLayer {
   private:
    int _batch_size = 0;

   public:
    std::shared_ptr<BaseLayer> main_block;
    std::shared_ptr<BaseLayer> shortcut;
    std::shared_ptr<BaseHiddenStates> input_z;
    std::shared_ptr<BaseDeltaStates> input_delta_z;
    std::shared_ptr<BaseHiddenStates> shortcut_output_z;
    std::shared_ptr<BaseDeltaStates> shortcut_output_delta_z;

    // Template to accept any type of LayerBlock and shortcut
    template <typename MainBlock, typename Shortcut = BaseLayer>
    ResNetBlock(MainBlock &&main, Shortcut &&shortcut_layer = Shortcut()) {
        static_assert(
            std::is_base_of<BaseLayer,
                            typename std::decay<MainBlock>::type>::value,
            "MainBlock must be derived from BaseLayer");
        static_assert(
            std::is_base_of<BaseLayer,
                            typename std::decay<Shortcut>::type>::value,
            "Shortcut must be derived from BaseLayer");

        // main_block = std::make_shared<LayerBlock>(std::move(main));
        // main_block = std::make_shared<BaseLayer>(std::move(main));
        main_block = std::make_shared<typename std::decay<MainBlock>::type>(
            std::move(main));

        bool is_shortcut_exist =
            !std::is_same<typename std::decay<Shortcut>::type,
                          BaseLayer>::value;
        if (is_shortcut_exist) {
            shortcut = std::make_shared<typename std::decay<Shortcut>::type>(
                std::forward<Shortcut>(shortcut_layer));
        } else {
            shortcut = nullptr;
        }

        // Set input & output sizes
        this->input_size = this->main_block->input_size;
        this->output_size = this->main_block->output_size;
    };

    template <typename MainBlock, typename Shortcut = BaseLayer>
    ResNetBlock(std::shared_ptr<MainBlock> main,
                std::shared_ptr<Shortcut> shortcut_layer = nullptr) {
        static_assert(std::is_base_of<BaseLayer, MainBlock>::value,
                      "MainBlock must be derived from BaseLayer");
        static_assert(std::is_base_of<BaseLayer, Shortcut>::value,
                      "Shortcut must be derived from BaseLayer");

        this->main_block = std::move(main);

        if (shortcut_layer) {
            this->shortcut = std::move(shortcut_layer);
        }

        this->input_size = main_block->input_size;
        this->output_size = main_block->output_size;
    }

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

    std::string get_device() override;

    void compute_input_output_size(const InitArgs &args) override;

    void init_shortcut_state();

    void init_shortcut_delta_state();

    void init_input_buffer();

    void init_weight_bias() override;

    void set_threads(int num) override;

    void train() override;

    void eval() override;

    void forward(BaseHiddenStates &input_states,
                 BaseHiddenStates &output_states,
                 BaseTempStates &temp_states) override;

    void backward(BaseDeltaStates &input_delta_states,
                  BaseDeltaStates &output_delta_states,
                  BaseTempStates &temp_states,
                  bool state_update = true) override;

    void update_weights() override;
    void update_biases() override;

    void save(std::ofstream &file) override;
    void load(std::ifstream &file) override;

    // Get/load parameters
    ParameterMap get_parameters_as_map(std::string suffix = "") override;
    void load_parameters_from_map(const ParameterMap &param_map,
                                  const std::string &suffix = "") override;
    std::vector<ParameterTuple> parameters() override;

    using BaseLayer::to_cuda;

    // DEBUG
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
               std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    get_norm_mean_var() override;

#ifdef USE_CUDA
    std::unique_ptr<BaseLayer> to_cuda() override;
#endif

    void preinit_layer() override;
};

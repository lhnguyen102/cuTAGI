#pragma once

#include "base_layer_cuda.cuh"
#include "layer_block.h"

class ResNetBlockCuda : public BaseLayerCuda {
   private:
    std::shared_ptr<BaseLayer> main_block;
    std::shared_ptr<BaseLayer> shortcut;
    int _batch_size = 0;

   public:
    std::shared_ptr<BaseHiddenStates> input_z;
    std::shared_ptr<BaseDeltaStates> input_delta_z;
    std::shared_ptr<BaseHiddenStates> shortcut_output_z;
    std::shared_ptr<BaseDeltaStates> shortcut_output_delta_z;

    // Template to accept any type of LayerBlock and shortcut
    template <typename MainBlock, typename Shortcut = BaseLayer>
    ResNetBlockCuda(MainBlock &&main, Shortcut &&shortcut_layer = Shortcut()) {
        static_assert(
            std::is_base_of<BaseLayer,
                            typename std::decay<MainBlock>::type>::value,
            "MainBlock must be derived from BaseLayer");
        static_assert(
            std::is_base_of<BaseLayer,
                            typename std::decay<Shortcut>::type>::value,
            "Shortcut must be derived from BaseLayer");

        auto cu_main = main->to_cuda();
        main_block = std::make_shared<typename std::decay<MainBlock>::type>(
            std::move(cu_main));

        bool is_shortcut_exist =
            !std::is_same<typename std::decay<Shortcut>::type,
                          BaseLayer>::value;
        if (is_shortcut_exist) {
            this->shortcut =
                std::make_shared<Shortcut>(std::move(shortcut_layer));
            this->shortcut->to_cuda();
        }

        // Set input & output sizes
        this->input_size = this->main_block->input_size;
        this->output_size = this->main_block->output_size;
    };

    template <typename MainBlock, typename Shortcut = BaseLayer>
    ResNetBlockCuda(std::shared_ptr<MainBlock> main,
                    std::shared_ptr<Shortcut> shortcut_layer = nullptr) {
        static_assert(std::is_base_of<BaseLayer, MainBlock>::value,
                      "MainBlock must be derived from BaseLayer");
        static_assert(std::is_base_of<BaseLayer, Shortcut>::value,
                      "Shortcut must be derived from BaseLayer");

        if (main->device != "cuda") {
            auto cu_main = main->to_cuda();
            this->main_block = std::move(cu_main);
        } else {
            this->main_block = std::move(main);
        }

        if (shortcut_layer) {
            auto cu_layer = shortcut_layer->to_cuda();
            this->shortcut = std::move(cu_layer);
        }

        this->input_size = main_block->input_size;
        this->output_size = main_block->output_size;
    }

    ~ResNetBlockCuda();

    // Delete copy constructor and copy assignment
    ResNetBlockCuda(const ResNetBlockCuda &) = delete;
    ResNetBlockCuda &operator=(const ResNetBlockCuda &) = delete;

    // Optionally implement move constructor and move assignment
    ResNetBlockCuda(ResNetBlockCuda &&) = default;
    ResNetBlockCuda &operator=(ResNetBlockCuda &&) = default;

    std::string get_layer_info() const override;

    std::string get_layer_name() const override;

    LayerType get_layer_type() const override;

    int get_max_num_states() override;

    std::string get_device() override;

    void compute_input_output_size(const InitArgs &args) override;

    void init_shortcut_state();

    void init_shortcut_delta_state();

    void init_input_buffer();

    void init_weight_bias();

    void set_threads(int num) override;

    void set_cuda_threads(int num) override;

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

    using BaseLayer::to_cuda;

    std::unique_ptr<BaseLayer> to_host() override;
    void preinit_layer() override;
};

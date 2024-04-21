#include "../include/layer_block.h"

LayerBlock::~LayerBlock() {}

LayerBlock::LayerBlock() {}

void LayerBlock::add_layers()
/*
 */
{}

void LayerBlock::add_layer(std::shared_ptr<BaseLayer> layer)
/*
NOTE: The output buffer size is determinated based on the output size for each
layer assuming that batch size = 1. If the batch size in the forward pass > 1,
it will be corrected at the first run in the forward pass.
 */
{
    // Stack layer
    if (this->device.compare("cpu") == 0) {
        this->layers.push_back(layer);
    } else if (this->device.compare("cuda") == 0) {
        this->layers.push_back(layer->to_cuda());
    } else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void LayerBlock::switch_to_cuda() {
    for (size_t i = 0; i < this->layers.size(); ++i) {
        auto cuda_layer = layers[i]->to_cuda();
        layers[i] = std::move(cuda_layer);
    }
}

std::string LayerBlock::get_layer_info() const
/*
 */
{
    return "LayerBlock(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string LayerBlock::get_layer_name() const
/*
 */
{
    return "LayerBlock";
}

LayerType LayerBlock::get_layer_type() const
/*
 */
{
    return LayerType::LayerBlock;
}

int LayerBlock::get_max_num_states()
/**/
{
    int max_size = 0;
    for (const auto &layer : this->layers) {
        int layer_max_size = layer->get_max_num_states();
        max_size = std::max(layer_max_size, max_size);
    }
    return max_size;
}

void LayerBlock::init_weight_bias()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->init_weight_bias();
    }
}

void LayerBlock::forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states)
/*
 */
{
    // Forward pass for all layers
    for (auto &layer : this->layers) {
        auto *current_layer = layer.get();

        current_layer->forward(input_states, output_states, temp_states);

        std::swap(input_states, output_states);
    }

    std::swap(output_states, input_states);
}

void LayerBlock::backward(BaseDeltaStates &input_delta_states,
                          BaseDeltaStates &output_delta_states,
                          BaseTempStates &temp_states, bool state_update)
/*
 */
{
    // Hidden layers
    for (auto layer = this->layers.rbegin(); layer != this->layers.rend() - 1;
         ++layer) {
        auto *current_layer = layer->get();

        // Backward pass for hidden states
        current_layer->backward(input_delta_states, output_delta_states,
                                temp_states);

        // Pass new input data for next iteration
        if (current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(input_delta_states, output_delta_states);
        }
    }

    // State update for input layer
    this->layers[0]->backward(input_delta_states, output_delta_states,
                              temp_states, state_update);

    if (this->layers[0]->get_layer_type() == LayerType::Activation ||
        !state_update) {
        std::swap(output_delta_states, input_delta_states);
    }
}

void LayerBlock::update_weights()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->update_weights();
    }
}

void LayerBlock::update_biases()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->update_biases();
    }
}

void LayerBlock::compute_input_output_size(const InitArgs &args)
/*
 */
{
    InitArgs tmp = InitArgs(args.width, args.height, args.depth);

    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i]->compute_input_output_size(tmp);

        tmp.width = this->layers[i]->out_width;
        tmp.height = this->layers[i]->out_height;
        tmp.depth = this->layers[i]->out_channels;
    }
}

void LayerBlock::save(std::ofstream &file)
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->save(file);
    }
}

void LayerBlock::load(std::ifstream &file)
/*
 */
{
    for (auto &layer : this->layers) {
        layer->load(file);
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LayerBlock::to_cuda() {
    this->device = "cuda";
    this->switch_to_cuda();
    return std::make_unique<LayerBlock>(std::move(*this));
}
#endif
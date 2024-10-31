#include "../include/layer_block.h"

#ifdef USE_CUDA
#include "../include/base_layer_cuda.cuh"
#endif

LayerBlock::~LayerBlock() {}

LayerBlock::LayerBlock() {}

void LayerBlock::add_layers()
/*
 */
{
    this->input_size = this->layers.front()->input_size;

    auto layer_type = this->layers.back()->get_layer_type();
    int num_layers = this->layers.size();
    int i = num_layers - 2;
    while (layer_type == LayerType::Activation && i >= 0) {
        this->output_size = this->layers[i]->output_size;
        layer_type = this->layers[i]->get_layer_type();
        i--;
    }
}

void LayerBlock::add_layer(std::shared_ptr<BaseLayer> layer)
/*
NOTE: The output buffer size is determinated based on the output size for each
layer assuming that batch size = 1. If the batch size in the forward pass > 1,
it will be corrected at the first run in the forward pass.
 */
{
    // Stack layer
    if (this->device.compare("cpu") == 0) {
        this->layers.push_back(std::move(layer));
    } else if (this->device.compare("cuda") == 0) {
        this->layers.push_back(std::move(layer->to_cuda()));
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

std::string LayerBlock::get_device()
/*
 */
{
    for (auto &layer : this->layers) {
        auto layer_device = layer->get_device();
        if (layer_device != this->device) {
            return layer_device;
        }
    }
    return this->device;
}

void LayerBlock::init_weight_bias()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->init_weight_bias();
    }
}

void LayerBlock::set_threads(int num)
/*
 */
{
    for (auto &layer : this->layers) {
        layer->set_threads(num);
    }
}

void LayerBlock::train()
/*
 */
{
    for (auto &layer : this->layers) {
        layer->train();
    }
}

void LayerBlock::eval()
/*
 */
{
    for (auto &layer : this->layers) {
        layer->eval();
    }
}

#ifdef USE_CUDA
void LayerBlock::set_cuda_threads(int num)
/*
 */
{
    for (auto &layer : this->layers) {
        BaseLayerCuda *cu_layer = dynamic_cast<BaseLayerCuda *>(layer.get());
        cu_layer->set_cuda_threads(num);
    }
}
#endif

void LayerBlock::forward(BaseHiddenStates &input_states,
                         BaseHiddenStates &output_states,
                         BaseTempStates &temp_states)
/*
 */
{
    // Cast input and outputs objects to pointers for  an efficient loop
    // swapping. It avoids swapping each members in their objects
    BaseHiddenStates *casted_input_states =
        dynamic_cast<BaseHiddenStates *>(&input_states);
    BaseHiddenStates *casted_output_states =
        dynamic_cast<BaseHiddenStates *>(&output_states);

    // Forward pass for all layers
    int batch_size = input_states.block_size;
    int num_layers = this->layers.size();

    for (int i = 0; i < num_layers; ++i) {
        auto *current_layer = this->layers[i].get();

        current_layer->forward(*casted_input_states, *casted_output_states,
                               temp_states);

        std::swap(casted_input_states, casted_output_states);
    }

    // Ensure output states contains the output of the last layer in the block
    if (num_layers % 2 == 0) {
        output_states.swap(input_states);
    }

    output_states.width = this->out_width;
    output_states.height = this->out_height;
    output_states.depth = this->out_channels;
    output_states.block_size = batch_size;
    output_states.actual_size = this->output_size;
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
            input_delta_states.swap(output_delta_states);
        }
    }

    // State update for input layer
    if (state_update && this->layers.size() > 1) {
        this->layers[0]->backward(input_delta_states, output_delta_states,
                                  temp_states, state_update);
    }

    if (this->layers[0]->get_layer_type() == LayerType::Activation ||
        !state_update || this->layers.size() == 1) {
        output_delta_states.swap(input_delta_states);
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
    this->in_channels = args.depth;
    this->in_height = args.height;
    this->in_width = args.width;

    InitArgs tmp = InitArgs(args.width, args.height, args.depth);

    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i]->compute_input_output_size(tmp);

        tmp.width = this->layers[i]->out_width;
        tmp.height = this->layers[i]->out_height;
        tmp.depth = this->layers[i]->out_channels;
    }

    this->out_channels = this->layers.back()->out_channels;
    this->out_height = this->layers.back()->out_height;
    this->out_width = this->layers.back()->out_width;

    this->input_size = this->in_width * this->in_width * this->in_channels;
    this->output_size = this->out_width * this->out_height * this->out_channels;
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

void LayerBlock::preinit_layer() {
    for (auto &layer : this->layers) {
        layer->preinit_layer();
    }
}
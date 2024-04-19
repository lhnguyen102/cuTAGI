#include "../include/resnet_block.h"

LayerBlock::~LayerBlock() {}

LayerBlock::LayerBlock() {}

void LayerBlock::add_layers()
/*
 */
{}

void LayerBlock::switch_to_cuda() {
    if (this->device == "cuda") {
        for (size_t i = 0; i < this->layers.size(); ++i) {
            layers[i] = layers[i]->to_cuda();
        }
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
        current_layer->backward(*this->input_delta_z_buffer,
                                *this->output_delta_z_buffer,
                                *this->temp_states);

        // Pass new input data for next iteration
        if (current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(this->input_delta_z_buffer, this->output_delta_z_buffer);
        }
    }

    // State update for input layer
    this->layers[0]->backward(*this->input_delta_z_buffer,
                              *this->output_delta_z_buffer, *this->temp_states,
                              this->input_state_update, state_update);

    if (this->layers[0]->get_layer_type() == LayerType::Activation ||
        !state_update) {
        std::swap(this->output_delta_z_buffer, this->input_delta_z_buffer);
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

void LayerBlock::update_weights()
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
    for (size_t i = 0; i < this->layers.size(); i++) {
        this->layers[i]->compute_input_output_size(args);

        args.width = this->layers[i]->out_width;
        args.height = this->layers[i]->out_height;
        args.depth = this->layers[i]->out_channels;
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

void LayerBlock::load(std::ofstream &file)
/*
 */
{
    for (auto &layer : this->layers) {
        layer->load(file);
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LayerBlock::to_cuda() {
    auto clone = std::make_unique<LayerBlock>(*this);
    clone->device = "cuda";
    clone->switch_to_cuda();
    return clone;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// Resnet Block
////////////////////////////////////////////////////////////////////////////////

ResNetBlock::ResNetBlock(std::shared_ptr<BaseLayer> main_block_layer,
                         std::shared_ptr<BaseLayer> shortcut_layer)
    : main_block(main_block_layer),
      shortcut(shortcut_layer)
/**/
{}
ResNetBlock::~ResNetBlock() {}

std::string ResNetBlock::get_layer_info() const
/*
 */
{
    return "ResNetBlock(" + std::to_string(this->input_size) + "," +
           std::to_string(this->output_size) + ")";
}

std::string ResNetBlock::get_layer_name() const
/*
 */
{
    return "ResNetBlock";
}

LayerType ResNetBlock::get_layer_type() const
/*
 */
{
    return LayerType::ResnetBlock;
}

int ResNetBlock::get_max_num_states()
/**/
{
    auto max_main_block = this->main_block->get_max_num_states();
    int max_shortcut = 0;
    if (this->shortcut != nullptr) {
        max_shortcut = this->shortcut->get_max_num_states();
    }

    return std::max(max_main_block, max_shortcut);
}

void ResNetBlock::init_shortcut_state()
/*
 */
{
    if (this->device.compare("cpu") == 0) {
        this->shortcut_output_z = std::make_shared<BaseHiddenStates>(
            this->shortcut->get_max_num_states(), this->_batch_size);
    }
#ifdef USE_CUDA
    else if (this->device.compare("cuda") == 0) {
        this->shortcut_output_z = std::make_shared<HiddenStateCuda>(
            this->shortcut->get_max_num_states(), batch_size);
    }
#endif
    else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void ResNetBlock::init_shortcut_delta_state()
/*
 */
{
    if (this->device.compare("cpu") == 0) {
        this->shortcut_output_delta_z = std::make_shared<BaseDeltaStates>(
            this->shortcut->get_max_num_states(), this->_batch_size);
    }
#ifdef USE_CUDA
    else if (this->device.compare("cuda") == 0) {
        this->shortcut_output_delta_z = std::make_shared<BaseDeltaStates>(
            this->shortcut->get_max_num_states(), batch_size);
    }
#endif
    else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void ResNetBlock::init_weight_bias()
/*
 */
{
    this->main_block->init_weight_bias();
    if (this->shortcut != nullptr) {
        this->shortcut->init_weight_bias();
    }
}

void ResNetBlock::forward(BaseHiddenStates &input_states,
                          BaseHiddenStates &output_states,
                          BaseTempStates &temp_states)
/**/

{
    int batch_size = input_states.block_size;

    // Main block
    if (batch_size > this->_batch_size && this->shortcut != nullptr) {
        this->_batch_size = batch_size;
        this->init_shortcut_state();
        if (this->training) {
            this->init_shortcut_delta_state();
        }
    }
    this->main_block->forward(input_states, output_states, temp_states);
    int num_states = output_states.block_size * output_states.actual_size;

    // Shortcut
    if (this->shortcut != nullptr) {
        this->shortcut->forward(input_states, shortcut_output_z, temp_states);
        for (int i = 0; i < num_states; i++) {
            output_states.mu_a[] += shortcut_output_z->mu_a[i];
            output_states.var_a[] += shortcut_output_z->var_a[i];
        }
    } else {
        for (int i = 0; i < num_states; i++) {
            output_states.mu_a[] += input_states.mu_a[i];
            output_states.var_a[] += input_states.var_a[i];
        }
    }
}

void ResNetBlock::backward(BaseDeltaStates &input_delta_states,
                           BaseDeltaStates &output_delta_states,
                           BaseTempStates &temp_states, bool state_update)
/**/
{
    this->main_block->backward(input_delta_states, output_delta_states,
                               temp_states, state_update);

    if (this->shortcut != nullptr) {
        this->shortcut->backward(input_delta_states,
                                 this->shortcut_output_delta_z, temp_states,
                                 state_update);
    }
}

void ResNetBlock::update_weights()
/*
 */
{
    this->main_block->update_weights();
    if (this->shortcut != nullptr) {
        this->shortcut->update_weights();
    }
}

void ResNetBlock::update_biases()
/*
 */
{
    this->main_block->update_biases();
    if (this->shortcut != nullptr) {
        this->shortcut->update_biases();
    }
}

void ResNetBlock::save(std::ofstream &file)
/*
 */
{
    this->main_block->save(file);
    if (this->shortcut != nullptr) {
        this->shortcut->save(file);
    }
}

void ResNetBlock::load(std::ofstream &file)
/*
 */
{
    this->main_block->load(file);
    if (this->shortcut != nullptr) {
        this->shortcut->load(file);
    }
}

#ifdef USE_CUDA
std::unique_ptr<BaseLayer> LayerBlock::to_cuda() {
    auto clone = std::make_unique<LayerBlock>(*this);
    clone->device = "cuda";
    clone->main_block = clone->main_block->to_cuda();
    if (clone->shortcut != nullptr) {
        clone->shortcut = clone->shortcut->to_cuda();
    }
    return clone;
}
#endif
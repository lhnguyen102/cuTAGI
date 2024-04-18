#include "../include/resnet_block.h"

LayerBlock::~LayerBlock() {}

void LayerBlock::add_layers()
/*
 */
{}

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

void LayerBlock::state_backward(BaseBackwardStates &next_bwd_states,
                                BaseDeltaStates &input_delta_states,
                                BaseDeltaStates &output_delta_states,
                                BaseTempStates &temp_states)
/*
 */
{
    // Hidden layers
    for (auto layer = this->layers.rbegin(); layer != this->layers.rend();
         ++layer) {
        auto *current_layer = layer->get();

        // Backward pass for hidden states
        current_layer->state_backward(*current_layer->bwd_states,
                                      input_delta_states, output_delta_states,
                                      temp_states);

        // Pass new input data for next iteration
        if (current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(input_delta_states, output_delta_states);
        }
    }
}

void LayerBlock::param_backward(BaseBackwardStates &next_bwd_states,
                                BaseDeltaStates &delta_states,
                                BaseTempStates &temp_states)
/**/
{
    // // Hidden layers
    // for (auto layer = this->layers.rbegin(); layer != this->layers.rend();
    //      ++layer) {
    //     auto *current_layer = layer->get();

    //     // Backward pass for parameters and hidden states

    //     current_layer->param_backward(*current_layer->bwd_states,
    //                                   *this->input_delta_z_buffer,
    //                                   *this->temp_states);

    //     // Backward pass for hidden states
    //     current_layer->state_backward(
    //         *current_layer->bwd_states, *this->input_delta_z_buffer,
    //         *this->output_delta_z_buffer, *this->temp_states);

    //     // Pass new input data for next iteration
    //     if (current_layer->get_layer_type() != LayerType::Activation) {
    //         std::swap(this->input_delta_z_buffer,
    //         this->output_delta_z_buffer);
    //     }
    // }

    // // Parameter update for input layer
    // if (this->param_update) {
    //     this->layers[0]->param_backward(*this->layers[0]->bwd_states,
    //                                     *this->input_delta_z_buffer,
    //                                     *this->temp_states);
    // }

    // // State update for input layer
    // if (this->input_state_update) {
    //     this->layers[0]->state_backward(
    //         *this->layers[0]->bwd_states, *this->input_delta_z_buffer,
    //         *this->output_delta_z_buffer, *this->temp_states);
    // }
}
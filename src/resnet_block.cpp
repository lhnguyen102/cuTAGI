#include "../include/resnet_block.h"

LayerBlock::~LayerBlock() {}

void LayerBlock::LayerBlock()
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

void LayerBlock::init_weight_bias()
/*
 */
{
    for (const auto& layer : this->layers) {
        layer->init_weight_bias();
    }
}
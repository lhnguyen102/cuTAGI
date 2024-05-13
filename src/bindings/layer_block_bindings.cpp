#include "../include/bindings/layer_block_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/base_layer.h"
#include "../include/layer_block.h"

void bind_layer_block(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<LayerBlock, std::shared_ptr<LayerBlock>, BaseLayer>(
        modo, "LayerBlock")
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](const std::vector<std::shared_ptr<BaseLayer>>& layers) {
                auto block = std::make_shared<LayerBlock>();
                for (const auto& layer : layers) {
                    block->add_layer(layer);
                }

                // Perform the pre-computation of the network's parameters
                block->add_layers();
                return block;
            }))
        .def("add_layer", &LayerBlock::add_layer)
        .def("switch_to_cuda", &LayerBlock::switch_to_cuda)
        .def_readwrite("layers", &LayerBlock::layers);
}
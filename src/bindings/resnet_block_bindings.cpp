#include "../include/bindings/resnet_block_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/base_layer.h"
#include "../include/layer_block.h"
#include "../include/resnet_block.h"

void bind_resnet_block(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<ResNetBlock, std::shared_ptr<ResNetBlock>, BaseLayer>(
        modo, "ResNetBlock")
        .def(pybind11::init(
            [](std::shared_ptr<LayerBlock> main,
               pybind11::object shortcut_layer = pybind11::none()) {
                if (!shortcut_layer.is_none()) {
                    auto shortcut_ptr =
                        shortcut_layer.cast<std::shared_ptr<BaseLayer>>();
                    return std::make_shared<ResNetBlock>(main, shortcut_ptr);
                } else {
                    return std::make_shared<ResNetBlock>(main);
                }
            }))
        .def_readwrite("main_block", &ResNetBlock::main_block)
        .def_readwrite("shortcut", &ResNetBlock::shortcut)
        .def("init_shortcut_state", &ResNetBlock::init_shortcut_state)
        .def("init_shortcut_delta_state",
             &ResNetBlock::init_shortcut_delta_state)
        .def("init_input_buffer", &ResNetBlock::init_input_buffer)
        .def_readwrite("input_z", &ResNetBlock::input_z)
        .def_readwrite("input_delta_z", &ResNetBlock::input_delta_z)
        .def_readwrite("shortcut_output_z", &ResNetBlock::shortcut_output_z)
        .def_readwrite("shortcut_output_delta_z",
                       &ResNetBlock::shortcut_output_delta_z);
}
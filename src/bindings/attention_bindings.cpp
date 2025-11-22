#include "../include/bindings/attention_bindings.h"

void bind_attention_layer(pybind11::module_& modo) {
    pybind11::class_<MultiheadAttention, std::shared_ptr<MultiheadAttention>,
                     BaseLayer>(modo, "MultiheadAttention")
        .def(pybind11::init<size_t, size_t, size_t, bool, float, float,
                            std::string, int>(),
             pybind11::arg("embed_dim"), pybind11::arg("num_heads"),
             pybind11::arg("num_kv_heads"), pybind11::arg("bias") = true,
             pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f,
             pybind11::arg("method") = "Xavier",
             pybind11::arg("device_idx") = 0)
        .def("get_layer_info", &MultiheadAttention::get_layer_info)
        .def("get_layer_name", &MultiheadAttention::get_layer_name)
        .def_readwrite("gain_w", &MultiheadAttention::gain_w)
        .def_readwrite("gain_b", &MultiheadAttention::gain_b)
        .def_readwrite("init_method", &MultiheadAttention::init_method)
        .def("init_weight_bias", &MultiheadAttention::init_weight_bias)
        .def("forward", &MultiheadAttention::forward)
        .def("state_backward", &MultiheadAttention::backward);
}

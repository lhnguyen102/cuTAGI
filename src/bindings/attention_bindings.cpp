#include "../include/bindings/attention_bindings.h"

void bind_attention_layer(pybind11::module_& modo) {
    pybind11::class_<MultiheadAttention, std::shared_ptr<MultiheadAttention>,
                     BaseLayer>(modo, "MultiheadAttention")
        .def(pybind11::init<size_t, size_t, size_t, size_t, bool, float, float,
                            std::string, std::string, float, size_t, bool,
                            int>(),
             pybind11::arg("embed_dim"), pybind11::arg("num_heads"),
             pybind11::arg("num_kv_heads"), pybind11::arg("seq_len") = 1,
             pybind11::arg("bias") = true, pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f,
             pybind11::arg("method") = "Xavier",
             pybind11::arg("pos_emb") = "rope",
             pybind11::arg("rope_theta") = 10000.0f,
             pybind11::arg("max_seq_len") = 2048,
             pybind11::arg("use_causal_mask") = true,
             pybind11::arg("device_idx") = 0)
        .def("get_layer_info", &MultiheadAttention::get_layer_info)
        .def("get_layer_name", &MultiheadAttention::get_layer_name)
        .def_readwrite("gain_w", &MultiheadAttention::gain_w)
        .def_readwrite("gain_b", &MultiheadAttention::gain_b)
        .def_readwrite("init_method", &MultiheadAttention::init_method)
        .def_readwrite("debug", &MultiheadAttention::debug)
        .def_readwrite("debug_interval", &MultiheadAttention::debug_interval)
        .def("init_weight_bias", &MultiheadAttention::init_weight_bias)
        .def("forward", &MultiheadAttention::forward)
        .def("state_backward", &MultiheadAttention::backward);

    pybind11::class_<MultiheadAttentionV2,
                     std::shared_ptr<MultiheadAttentionV2>, BaseLayer>(
        modo, "MultiheadAttentionV2")
        .def(pybind11::init<size_t, size_t, size_t, size_t, bool, float, float,
                            std::string, std::string, float, size_t, bool,
                            int>(),
             pybind11::arg("embed_dim"), pybind11::arg("num_heads"),
             pybind11::arg("num_kv_heads"), pybind11::arg("seq_len") = 1,
             pybind11::arg("bias") = true, pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f,
             pybind11::arg("method") = "Xavier",
             pybind11::arg("pos_emb") = "rope",
             pybind11::arg("rope_theta") = 10000.0f,
             pybind11::arg("max_seq_len") = 2048,
             pybind11::arg("use_causal_mask") = true,
             pybind11::arg("device_idx") = 0)
        .def("get_layer_info", &MultiheadAttentionV2::get_layer_info)
        .def("get_layer_name", &MultiheadAttentionV2::get_layer_name)
        .def_readwrite("gain_w", &MultiheadAttentionV2::gain_w)
        .def_readwrite("gain_b", &MultiheadAttentionV2::gain_b)
        .def_readwrite("init_method", &MultiheadAttentionV2::init_method)
        .def_readwrite("debug", &MultiheadAttentionV2::debug)
        .def_readwrite("debug_interval", &MultiheadAttentionV2::debug_interval)
        .def("init_weight_bias", &MultiheadAttentionV2::init_weight_bias)
        .def("forward", &MultiheadAttentionV2::forward)
        .def("state_backward", &MultiheadAttentionV2::backward);
}

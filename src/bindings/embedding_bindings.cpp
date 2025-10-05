#include "../include/bindings/embedding_bindings.h"

void bind_embedding(pybind11::module_& modo) {
    pybind11::class_<Embedding, std::shared_ptr<Embedding>, BaseLayer>(
        modo, "Embedding")
        .def(pybind11::init<int, int, int, float, int, int>(),
             pybind11::arg("num_embeddings"), pybind11::arg("embedding_dim"),
             pybind11::arg("input_size") = 0, pybind11::arg("scale") = 1.0f,
             pybind11::arg("padding_idx") = -1, pybind11::arg("device_idx") = 0)
        .def("get_layer_info", &Embedding::get_layer_info)
        .def("get_layer_name", &Embedding::get_layer_name)
        .def_readwrite("embedding_dim", &Embedding::embedding_dim)
        .def_readwrite("num_embeddings", &Embedding::num_embeddings)
        .def_readwrite("scale", &Embedding::scale)
        .def_readwrite("padding_idx", &Embedding::padding_idx)
        .def("init_weight_bias", &Embedding::init_weight_bias)
        .def("forward", &Embedding::forward)
        .def("state_backward", &Embedding::backward);
}

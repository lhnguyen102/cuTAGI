#include "../../include/bindings/positional_encoding_bindings.h"

#include "../../include/positional_encoding.h"

void bind_positional_encoding(pybind11::module_& modo) {
    pybind11::class_<PositionalEncoding, std::shared_ptr<PositionalEncoding>,
                     BaseLayer>(modo, "PositionalEncoding")
        .def(pybind11::init<int, size_t, int>(), pybind11::arg("embed_dim"),
             pybind11::arg("max_seq_len") = 2048,
             pybind11::arg("device_idx") = 0)
        .def("get_layer_info", &PositionalEncoding::get_layer_info)
        .def("get_layer_name", &PositionalEncoding::get_layer_name)
        .def("forward", &PositionalEncoding::forward);
}

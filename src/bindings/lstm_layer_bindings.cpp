#include "../include/bindings/lstm_layer_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/lstm_layer.h"

void bind_lstm_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<LSTM, std::shared_ptr<LSTM>, BaseLayer>(modo, "LSTM")
        .def(pybind11::init<size_t, size_t, int, bool, float, float,
                            std::string>(),
             pybind11::arg("input_size"), pybind11::arg("output_size"),
             pybind11::arg("seq_len"), pybind11::arg("bias"),
             pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f, pybind11::arg("method") = "He")
        .def("get_layer_info", &LSTM::get_layer_info)
        .def("get_layer_name", &LSTM::get_layer_name)
        .def_readwrite("gain_w", &LSTM::gain_w)
        .def_readwrite("gain_b", &LSTM::gain_b)
        .def_readwrite("init_method", &LSTM::init_method)
        .def("init_weight_bias", &LSTM::init_weight_bias)
        .def("forward", &LSTM::forward)
        .def("backward", &LSTM::backward);
}

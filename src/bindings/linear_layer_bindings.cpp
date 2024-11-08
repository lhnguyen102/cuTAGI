#include "../include/bindings/linear_layer_bindings.h"

void bind_linear_layer(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Linear, std::shared_ptr<Linear>, BaseLayer>(modo, "Linear")
        .def(pybind11::init<size_t, size_t, bool, float, float, std::string>(),
             pybind11::arg("ip_size"), pybind11::arg("op_size"),
             pybind11::arg("bias"), pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f, pybind11::arg("method") = "He")
        .def("get_layer_info", &Linear::get_layer_info)
        .def("get_layer_name", &Linear::get_layer_name)
        .def_readwrite("gain_w", &Linear::gain_w)
        .def_readwrite("gain_b", &Linear::gain_b)
        .def_readwrite("init_method", &Linear::init_method)
        .def("init_weight_bias", &Linear::init_weight_bias)
        .def("forward", &Linear::forward)
        .def("state_backward", &Linear::backward);
}

// PYBIND11_MODULE(neural_net_module, m) {
//     py::class_<Linear>(m, "Linear")
//         .def(py::init<int, int>())
//         // Binding the overloaded forward methods
//         .def("forward", (void (Linear::*)(BaseHiddenStates&,
//         BaseHiddenStates&, BaseTempStates&)) &Linear::forward)
//         .def("forward", (std::vector<float> (Linear::*)(const
//         std::vector<float>&)) &Linear::forward);
// }
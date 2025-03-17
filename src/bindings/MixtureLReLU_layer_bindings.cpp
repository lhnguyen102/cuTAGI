#include "../include/bindings/MixtureLReLU_layer_bindings.h"

void bind_MixtureLReLU_layer(pybind11::module_ &modo)
/*
 */
{
    pybind11::class_<MixtureLReLU, std::shared_ptr<MixtureLReLU>, BaseLayer>(modo, "MixtureLReLU")
        .def(pybind11::init<size_t, float>(),
             pybind11::arg("ip_size"), pybind11::arg("slope") = 1.0f)
        .def("get_layer_info", &MixtureLReLU::get_layer_info)
        .def("get_layer_name", &MixtureLReLU::get_layer_name)
        .def("forward", &MixtureLReLU::forward)
        .def("state_backward", &MixtureLReLU::backward);
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
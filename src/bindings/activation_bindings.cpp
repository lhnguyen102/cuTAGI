
#include "../include/bindings/activation_bindings.h"

void bind_relu(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<ReLU, std::shared_ptr<ReLU>, BaseLayer>(modo, "ReLU")
        .def(pybind11::init<>())
        .def("get_layer_info", &ReLU::get_layer_info)
        .def("get_layer_name", &ReLU::get_layer_name)
        .def("forward", &ReLU::forward)
        .def("update_weights", &ReLU::update_weights)
        .def("update_biases", &ReLU::update_biases)
        .def("load", &ReLU::load)
        .def("save", &ReLU::save)
        .def("to_cuda", &ReLU::to_cuda);
}

void bind_sigmoid(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Sigmoid, std::shared_ptr<Sigmoid>, BaseLayer>(modo,
                                                                   "Sigmoid")
        .def(pybind11::init<>())
        .def("get_layer_info", &Sigmoid::get_layer_info)
        .def("get_layer_name", &Sigmoid::get_layer_name)
        .def("forward", &Sigmoid::forward)
        .def("update_weights", &Sigmoid::update_weights)
        .def("update_biases", &Sigmoid::update_biases)
        .def("load", &Sigmoid::load)
        .def("save", &Sigmoid::save)
        .def("to_cuda", &Sigmoid::to_cuda);
}

void bind_tanh(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Tanh, std::shared_ptr<Tanh>, BaseLayer>(modo, "Tanh")
        .def(pybind11::init<>())
        .def("get_layer_info", &Tanh::get_layer_info)
        .def("get_layer_name", &Tanh::get_layer_name)
        .def("forward", &Tanh::forward)
        .def("update_weights", &Tanh::update_weights)
        .def("update_biases", &Tanh::update_biases)
        .def("load", &Tanh::load)
        .def("save", &Tanh::save)
        .def("to_cuda", &Tanh::to_cuda);
}

void bind_mixture_relu(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MixtureReLU, std::shared_ptr<MixtureReLU>, BaseLayer>(
        modo, "MixtureReLU")
        .def(pybind11::init<>())
        .def("get_layer_info", &MixtureReLU::get_layer_info)
        .def("get_layer_name", &MixtureReLU::get_layer_name)
        .def("forward", &MixtureReLU::forward)
        .def("update_weights", &MixtureReLU::update_weights)
        .def("update_biases", &MixtureReLU::update_biases)
        .def("load", &MixtureReLU::load)
        .def("save", &MixtureReLU::save)
        .def("to_cuda", &MixtureReLU::to_cuda);
}

void bind_mixture_sigmoid(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MixtureSigmoid, std::shared_ptr<MixtureSigmoid>,
                     BaseLayer>(modo, "MixtureSigmoid")
        .def(pybind11::init<>())
        .def("get_layer_info", &MixtureSigmoid::get_layer_info)
        .def("get_layer_name", &MixtureSigmoid::get_layer_name)
        .def("forward", &MixtureSigmoid::forward)
        .def("update_weights", &MixtureSigmoid::update_weights)
        .def("update_biases", &MixtureSigmoid::update_biases)
        .def("load", &MixtureSigmoid::load)
        .def("save", &MixtureSigmoid::save)
        .def("to_cuda", &MixtureSigmoid::to_cuda);
}

void bind_mixture_tanh(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<MixtureTanh, std::shared_ptr<MixtureTanh>, BaseLayer>(
        modo, "MixtureTanh")
        .def(pybind11::init<>())
        .def("get_layer_info", &MixtureTanh::get_layer_info)
        .def("get_layer_name", &MixtureTanh::get_layer_name)
        .def("forward", &MixtureTanh::forward)
        .def("update_weights", &MixtureTanh::update_weights)
        .def("update_biases", &MixtureTanh::update_biases)
        .def("load", &MixtureTanh::load)
        .def("save", &MixtureTanh::save)
        .def("to_cuda", &MixtureTanh::to_cuda);
}

void bind_softplus(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Softplus, std::shared_ptr<Softplus>, BaseLayer>(
        modo, "MixtureSoftplus")
        .def(pybind11::init<>())
        .def("get_layer_info", &Softplus::get_layer_info)
        .def("get_layer_name", &Softplus::get_layer_name)
        .def("forward", &Softplus::forward)
        .def("update_weights", &Softplus::update_weights)
        .def("update_biases", &Softplus::update_biases)
        .def("load", &Softplus::load)
        .def("save", &Softplus::save)
        .def("to_cuda", &Softplus::to_cuda);
}

void bind_leakyrelu(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<LeakyReLU, std::shared_ptr<LeakyReLU>, BaseLayer>(
        modo, "LeakyReLU")
        .def(pybind11::init<>())
        .def("get_layer_info", &LeakyReLU::get_layer_info)
        .def("get_layer_name", &LeakyReLU::get_layer_name)
        .def("forward", &LeakyReLU::forward)
        .def("update_weights", &LeakyReLU::update_weights)
        .def("update_biases", &LeakyReLU::update_biases)
        .def("load", &LeakyReLU::load)
        .def("save", &LeakyReLU::save)
        .def("to_cuda", &LeakyReLU::to_cuda);
}

void bind_softmax(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Softmax, std::shared_ptr<Softmax>, BaseLayer>(modo,
                                                                   "Softmax")
        .def(pybind11::init<>())
        .def("get_layer_info", &Softmax::get_layer_info)
        .def("get_layer_name", &Softmax::get_layer_name)
        .def("forward", &Softmax::forward)
        .def("update_weights", &Softmax::update_weights)
        .def("update_biases", &Softmax::update_biases)
        .def("load", &Softmax::load)
        .def("save", &Softmax::save)
        .def("to_cuda", &Softmax::to_cuda);
}

void bind_remax(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Remax, std::shared_ptr<Remax>, BaseLayer>(modo, "Remax")
        .def(pybind11::init<>())
        .def("get_layer_info", &Remax::get_layer_info)
        .def("get_layer_name", &Remax::get_layer_name)
        .def("forward", &Remax::forward)
        .def("update_weights", &Remax::update_weights)
        .def("update_biases", &Remax::update_biases)
        .def("load", &Remax::load)
        .def("save", &Remax::save)
        .def("to_cuda", &Remax::to_cuda);
}

void bind_closed_form_softmax(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<ClosedFormSoftmax, std::shared_ptr<ClosedFormSoftmax>,
                     BaseLayer>(modo, "ClosedFormSoftmax")
        .def(pybind11::init<>())
        .def("get_layer_info", &ClosedFormSoftmax::get_layer_info)
        .def("get_layer_name", &ClosedFormSoftmax::get_layer_name)
        .def("forward", &ClosedFormSoftmax::forward);
}

void bind_split_activation(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<SplitActivation, std::shared_ptr<SplitActivation>,
                     BaseLayer>(modo, "SplitActivation")
        // Bind the new constructor
        .def(pybind11::init<std::shared_ptr<BaseLayer>,
                            std::shared_ptr<BaseLayer>>(),
             "Initializes the SplitActivation layer.",
             pybind11::arg("odd_layer"),
             pybind11::arg("even_layer") =
                 nullptr)  // Define named, optional argument

        // The rest of the functions are bound as before
        .def("get_layer_info", &SplitActivation::get_layer_info)
        .def("get_layer_name", &SplitActivation::get_layer_name)
        .def("get_layer_type", &SplitActivation::get_layer_type)
        .def("forward", &SplitActivation::forward)
        .def("update_weights", &SplitActivation::update_weights)
        .def("update_biases", &SplitActivation::update_biases)
        .def("load", &SplitActivation::load)
        .def("save", &SplitActivation::save)
        .def("to_cuda", &SplitActivation::to_cuda);
}

void bind_exp(pybind11::module_& modo)
/*
 */
{
    pybind11::class_<Exp, std::shared_ptr<Exp>, BaseLayer>(modo, "Exp")
        .def(pybind11::init<float, float>(), pybind11::arg("scale") = 1.0f,
             pybind11::arg("shift") = 0.0f)
        .def("get_layer_info", &Exp::get_layer_info)
        .def("get_layer_name", &Exp::get_layer_name)
        .def("get_layer_type", &Exp::get_layer_type)
        .def("forward", &Exp::forward)
        .def("update_weights", &Exp::update_weights)
        .def("update_biases", &Exp::update_biases)
        .def("load", &Exp::load)
        .def("save", &Exp::save)
        .def("to_cuda", &Exp::to_cuda);
}

void bind_agvi(pybind11::module_& modo) {
    pybind11::class_<AGVI, std::shared_ptr<AGVI>, BaseLayer>(modo, "AGVI")
        .def(pybind11::init<std::shared_ptr<BaseLayer>, bool, bool>(),
             pybind11::arg("activation_layer"),
             pybind11::arg_v(
                 "overfit_mu", true,
                 "If true, use a different Jacobian for the mean delta"),
             pybind11::arg_v("agvi", true,
                             "If true, use the AGVI learned noise model"))

        .def("get_layer_info", &AGVI::get_layer_info)
        .def("get_layer_name", &AGVI::get_layer_name)
        .def("get_layer_type", &AGVI::get_layer_type)
        .def("forward", &AGVI::forward)
        .def("backward", &AGVI::backward)
        .def("update_weights", &AGVI::update_weights)
        .def("update_biases", &AGVI::update_biases)
        .def("load", &AGVI::load)
        .def("save", &AGVI::save)
        .def("to_cuda", &AGVI::to_cuda);
}

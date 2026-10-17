#include "../include/bindings/attention_bindings.h"

namespace {
pybind11::array_t<float> scores_to_numpy(const AttentionScores& scores,
                                         const std::vector<float>& values) {
    std::vector<pybind11::ssize_t> shape = {scores.batch_size, scores.num_heads,
                                            scores.timestep, scores.timestep};
    return pybind11::array_t<float>(shape, values.data());
}
}  // namespace

void bind_attention_layer(pybind11::module_& modo) {
    pybind11::class_<AttentionScores>(modo, "AttentionScores")
        .def_readonly("batch_size", &AttentionScores::batch_size)
        .def_readonly("num_heads", &AttentionScores::num_heads)
        .def_readonly("timestep", &AttentionScores::timestep)
        .def_property_readonly("mu_qk",
                               [](const AttentionScores& s) {
                                   return scores_to_numpy(s, s.mu_qk);
                               })
        .def_property_readonly("var_qk",
                               [](const AttentionScores& s) {
                                   return scores_to_numpy(s, s.var_qk);
                               })
        .def_property_readonly("mu_att",
                               [](const AttentionScores& s) {
                                   return scores_to_numpy(s, s.mu_att);
                               })
        .def_property_readonly("var_att", [](const AttentionScores& s) {
            return scores_to_numpy(s, s.var_att);
        });

    pybind11::class_<MultiheadAttention, std::shared_ptr<MultiheadAttention>,
                     BaseLayer>(modo, "MultiheadAttention")
        .def(
            pybind11::init<size_t, size_t, size_t, size_t, bool, float, float,
                           std::string, bool, float, size_t, bool, bool, int>(),
            pybind11::arg("embed_dim"), pybind11::arg("num_heads"),
            pybind11::arg("num_kv_heads"), pybind11::arg("seq_len") = 1,
            pybind11::arg("bias") = true, pybind11::arg("gain_weight") = 1.0f,
            pybind11::arg("gain_bias") = 1.0f,
            pybind11::arg("method") = "Xavier",
            pybind11::arg("use_rope") = true,
            pybind11::arg("rope_theta") = 10000.0f,
            pybind11::arg("max_seq_len") = 2048,
            pybind11::arg("use_causal_mask") = true,
            pybind11::arg("center_score_delta") = false,
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

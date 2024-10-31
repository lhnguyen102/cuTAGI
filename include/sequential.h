///////////////////////////////////////////////////////////////////////////////
// File:         sequential.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      March 18, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "base_layer.h"
#include "common.h"
#include "data_struct.h"
#include "linear_layer.h"
#include "lstm_layer.h"
#include "output_layer_update_cpu.h"
#include "slinear_layer.h"
#include "slstm_layer.h"
#ifdef USE_CUDA
#include "data_struct_cuda.cuh"
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>

class Sequential {
   public:
    bool valid_ = true;
    std::shared_ptr<BaseHiddenStates> output_z_buffer;
    std::shared_ptr<BaseHiddenStates> input_z_buffer;
    std::shared_ptr<BaseDeltaStates> output_delta_z_buffer;
    std::shared_ptr<BaseDeltaStates> input_delta_z_buffer;
    std::shared_ptr<BaseTempStates> temp_states;

    int z_buffer_size = 0;        // e.g., batch size x input size
    int z_buffer_block_size = 0;  // e.g., batch size
    int num_samples = 0;          // number of training samples

    int input_size = 0;
    bool training = true;
    bool param_update = true;
    bool input_state_update = false;
    unsigned num_threads = 1;
    std::string device = "cpu";
    std::vector<std::shared_ptr<BaseLayer>> layers;

    // Variadic template
    template <typename... Layers>
    Sequential(Layers&&... layers) {
        add_layers(std::forward<Layers>(layers)...);
    }
    // Recursive variadic template
    template <typename T, typename... Rest>
    void add_layers(T&& first, Rest&&... rest) {
        static_assert(
            std::is_base_of<BaseLayer, typename std::decay<T>::type>::value,
            "Type T must be derived from BaseLayer");

        add_layer(std::make_shared<typename std::remove_reference<T>::type>(
            std::move(first)));
        add_layers(std::forward<Rest>(rest)...);
    }

    // Base case for recursive variadic template. This function is called after
    // the last argument
    void add_layers();

    void add_layer(std::shared_ptr<BaseLayer> layer);

    Sequential() = default;

    ~Sequential();

    void switch_to_cuda();

    void to_device(const std::string& new_device);

    void params_to_host();

    void params_to_device();

    void set_buffer_size();

    void init_output_state_buffer();

    void init_delta_state_buffer();

    void set_threads(unsigned num_threads);

    void train();
    void eval();

    std::string get_device();

    void forward(const std::vector<float>& mu_a,
                 const std::vector<float>& var_a = std::vector<float>());

    void forward(BaseHiddenStates& input_states);

    void backward();

    std::tuple<std::vector<float>, std::vector<float>> smoother();

    void step();

    // DEBUG
    void output_to_host();
    void delta_z_to_host();
    void preinit_layer();

    // Utility function to get layer stack info
    std::string get_layer_stack_info() const;

    // Saving and loading
    void save(const std::string& filename);

    void load(const std::string& filename);

    void save_csv(const std::string& filename);

    void load_csv(const std::string& filename);

    std::vector<std::reference_wrapper<std::vector<float>>> parameters();

    std::map<std::string, std::tuple<std::vector<float>, std::vector<float>,
                                     std::vector<float>, std::vector<float>>>
    get_state_dict();

    void load_state_dict(
        const std::map<std::string,
                       std::tuple<std::vector<float>, std::vector<float>,
                                  std::vector<float>, std::vector<float>>>&
            state_dict);

    // Copy model params
    void params_from(const Sequential& ref_model);

    // Python Wrapper
    void forward_py(
        pybind11::array_t<float> mu_a_np,
        pybind11::array_t<float> var_a_np = pybind11::array_t<float>());

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_outputs();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_outputs_smoother();

    std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
    get_input_states();

   private:
    void compute_input_output_size();
};

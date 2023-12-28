///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      December 17, 2023
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
#include "output_layer_update_cpu.h"
#include "struct_var.h"
#ifdef USE_CUDA
#include "data_struct_cuda.cuh"
#endif

class Sequential {
   public:
    std::unique_ptr<BaseHiddenStates> output_z_buffer;
    std::unique_ptr<BaseHiddenStates> input_z_buffer;
    std::unique_ptr<BaseDeltaStates> output_delta_z_buffer;
    std::unique_ptr<BaseDeltaStates> input_delta_z_buffer;
    std::unique_ptr<BaseTempStates> temp_states;
    int z_buffer_size = 0;        // e.g., batch size x input size
    int z_buffer_block_size = 0;  // e.g., batch size
    int input_size = 0;
    bool training = true;
    bool param_update = true;
    bool input_hidden_state_update = false;
    unsigned num_threads = 1;
    std::string device = "cpu";

    // Variadic template. Note that for the template function the definition of
    // template must be included in the herder
    template <typename... Layers>
    Sequential(Layers&&... layers) {
        add_layers(std::forward<Layers>(layers)...);
    }

    Sequential();

    ~Sequential();

    void switch_to_cuda();

    void to_device(const std::string& new_device);

    void add_layer(std::unique_ptr<BaseLayer> layer);

    void init_output_state_buffer();

    void init_delta_state_buffer();

    void set_threads(unsigned num_threads);

    void forward(const std::vector<float>& mu_a,
                 const std::vector<float>& var_a = std::vector<float>());

    void to_z_buffer(const std::vector<float>& mu_x,
                     const std::vector<float>& var_x,
                     BaseHiddenStates& hidden_states);

    void backward();

    void step();

    // DEBUG
    void output_to_host();
    void delta_z_to_host();

    // Utility function to get layer stack info
    std::string get_layer_stack_info() const;

    // Saving and loading
    void save(const std::string& filename);

    void load(const std::string& filename);

    void save_csv(const std::string& filename);

    void load_csv(const std::string& filename);

    // Copy model params
    void params_from(const Sequential& ref_model);

   private:
    std::vector<std::unique_ptr<BaseLayer>> layers;

    // Recursive variadic template
    template <typename T, typename... Rest>
    void add_layers(T&& first, Rest&&... rest) {
        // Runtime check to verify if T is derived from BaseLayer
        if (!std::is_base_of<BaseLayer,
                             typename std::remove_reference<T>::type>::value) {
            std::cerr << "Error in file: " << __FILE__
                      << " at line: " << __LINE__
                      << ". Reason: Type T must be derived from BaseLayer.\n";
            throw std::invalid_argument(
                "Error: Type T must be derived from BaseLayer");
        }

        // Add layer
        add_layer(std::make_unique<T>(std::forward<T>(first)));

        // Recursively adding next layer
        add_layers(std::forward<Rest>(rest)...);
    }

    // Base case for recursive variadic template
    void add_layers(){};
};

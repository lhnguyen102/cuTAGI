///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack.h
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      November 25, 2023
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
#include "output_layer_update_cpu.h"
#include "struct_var.h"

class LayerStack {
   public:
    HiddenStates output_z_buffer;
    HiddenStates input_z_buffer;
    DeltaStates output_delta_z_buffer;
    DeltaStates input_delta_z_buffer;
    TempStates temp_states;
    int z_buffer_size = 0;        // e.g., batch size x input size
    int z_buffer_block_size = 1;  // e.g., batch size
    int input_size = 0;
    bool training = true;
    bool param_update = true;
    bool input_hidden_state_update = false;
    unsigned num_threads = 1;

    // Variadic template. Note that for the template function the definition of
    // template must be included in the herder
    template <typename... Layers>
    LayerStack(Layers&&... layers) {
        add_layers(std::forward<Layers>(layers)...);
    }

    LayerStack();

    ~LayerStack();

    void add_layer(std::unique_ptr<BaseLayer> layer);

    void init_output_state_buffer();

    void init_delta_state_buffer();

    void set_threads(unsigned num_threads);

    void forward(const std::vector<float>& mu_a,
                 const std::vector<float>& var_a = std::vector<float>());

    void to_z_buffer(const std::vector<float>& mu_x,
                     const std::vector<float>& var_x,
                     HiddenStates& hidden_states);

    void backward();

    void step();

    // Utility function to get layer stack info
    std::string get_layer_stack_info() const;

    // Saving and loading
    void save(const std::string& filename);

    void load(const std::string& filename);

    void save_csv(const std::string& filename);

    void load_csv(const std::string& filename);

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

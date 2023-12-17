///////////////////////////////////////////////////////////////////////////////
// File:         layer_stack_cpu.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      December 17, 2023
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/layer_stack_cpu.h"

LayerStack::LayerStack() {}
LayerStack::~LayerStack() {}

void LayerStack::add_layer(std::unique_ptr<BaseLayer> layer)
/*
NOTE: The output buffer size is determinated based on the output size for each
layer assuming that batch size = 1. If the batch size in the forward pass > 1,
it will be corrected at the first run in the forward pass.
 */
{
    // Get buffer size
    int output_size = layer->get_output_size();
    int input_size = layer->get_input_size();
    this->z_buffer_size = std::max(output_size, this->z_buffer_size);
    this->z_buffer_size = std::max(input_size, this->z_buffer_size);

    // Stack layer
    if (this->device.compare("cpu") == 0) {
        this->layers.push_back(std::move(layer));
    } else if (this->device.compare("cuda") == 0) {
        this->layers.push_back(std::move(layer->to_cuda()));
    } else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void LayerStack::init_output_state_buffer()
/*
 */
{
    if (this->device.compare("cpu") == 0) {
        this->output_z_buffer = std::make_unique<BaseHiddenStates>(
            this->z_buffer_size, this->z_buffer_block_size);
        this->input_z_buffer = std::make_unique<BaseHiddenStates>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#ifdef USE_CUDA
    else if (this->device.compare("cuda") == 0) {
        this->output_z_buffer = std::make_unique<HiddenStateCuda>(
            this->z_buffer_size, this->z_buffer_block_size);
        this->input_z_buffer = std::make_unique<HiddenStateCuda>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#endif
    else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void LayerStack::init_delta_state_buffer()
/*
 */
{
    if (this->device.compare("cpu") == 0) {
        this->output_delta_z_buffer = std::make_unique<BaseDeltaStates>(
            this->z_buffer_size, this->z_buffer_block_size);
        this->input_delta_z_buffer = std::make_unique<BaseDeltaStates>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#ifdef USE_CUDA
    else if (this->device.compare("cuda") == 0) {
        this->output_delta_z_buffer = std::make_unique<DeltaStateCuda>(
            this->z_buffer_size, this->z_buffer_block_size);
        this->input_delta_z_buffer = std::make_unique<DeltaStateCuda>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#endif
    else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void LayerStack::set_threads(unsigned int num_threads)
/*
 */
{
    this->num_threads = num_threads;
    for (auto &layer : this->layers) {
        layer->num_threads = num_threads;
    }
}

void LayerStack::to_z_buffer(const std::vector<float> &mu_x,
                             const std::vector<float> &var_x,
                             BaseHiddenStates &hidden_states)
/*
 */
{
    int data_size = mu_x.size();
    for (int i = 0; i < data_size; i++) {
        hidden_states.mu_z[i] = mu_x[i];
        hidden_states.mu_a[i] = mu_x[i];
    }
    if (var_x.size() == data_size) {
        for (int i = 0; i < data_size; i++) {
            hidden_states.var_z[i] = var_x[i];
            hidden_states.var_a[i] = var_x[i];
        }
    }
    hidden_states.size = data_size;
    hidden_states.block_size = data_size / this->layers.front()->input_size;
    hidden_states.actual_size = this->layers.front()->input_size;
}

void LayerStack::forward(const std::vector<float> &mu_x,
                         const std::vector<float> &var_x)
/*
 */
{
    // Batch size
    int batch_size = mu_x.size() / this->layers.front()->input_size;

    // Only initialize if batch size changes
    if (batch_size != this->z_buffer_block_size) {
        this->z_buffer_block_size = batch_size;
        this->z_buffer_size = batch_size * this->z_buffer_size;
        init_output_state_buffer();
        if (this->training) {
            init_delta_state_buffer();
        }
    }

    // Merge input data to the input buffer
    this->to_z_buffer(mu_x, var_x, *this->input_z_buffer);

    // Forward pass for all layers
    for (size_t i = 0; i < this->layers.size(); i++) {
        // Current layer
        BaseLayer *current_layer = this->layers[i].get();
        // if ((i + 1) % 2 != 0) {
        //     current_layer->forward(*this->input_z_buffer,
        //                            *this->output_z_buffer,
        //                            *this->temp_states);
        // } else {
        //     current_layer->forward(*this->output_z_buffer,
        //                            *this->input_z_buffer,
        //                            *this->temp_states);
        // }
        current_layer->forward(*this->input_z_buffer, *this->output_z_buffer,
                               *this->temp_states);
        *this->input_z_buffer = *this->output_z_buffer;
    }
    if (this->layers.size() % 2 == 0) {
        *this->output_z_buffer = *this->input_z_buffer;
    }
}

void LayerStack::backward()
/*
 */
{
    //  Output layer
    int last_layer_idx = this->layers.size() - 1;

    // Hidden layers
    for (int i = last_layer_idx, j = 0; i > 0; --i, ++j) {
        // Current layer
        BaseLayer *current_layer = this->layers[i].get();

        // // Backward pass for parameters and hidden states
        // if ((j + 1) % 2 != 0) {
        //     if (this->param_update) {
        //         current_layer->param_backward(current_layer->bwd_states,
        //                                       *this->input_delta_z_buffer,
        //                                       *this->temp_states);
        //     }

        //     // Backward pass for hidden states
        //     current_layer->state_backward(
        //         current_layer->bwd_states, *this->input_delta_z_buffer,
        //         *this->output_delta_z_buffer, *this->temp_states);

        // } else {
        //     if (this->param_update) {
        //         current_layer->param_backward(current_layer->bwd_states,
        //                                       *this->output_delta_z_buffer,
        //                                       *this->temp_states);
        //     }
        //     current_layer->state_backward(
        //         current_layer->bwd_states, *this->output_delta_z_buffer,
        //         *this->input_delta_z_buffer, *this->temp_states);
        // }

        if (this->param_update) {
            current_layer->param_backward(current_layer->bwd_states,
                                          *this->input_delta_z_buffer,
                                          *this->temp_states);
        }

        // Backward pass for hidden states
        current_layer->state_backward(
            current_layer->bwd_states, *this->input_delta_z_buffer,
            *this->output_delta_z_buffer, *this->temp_states);

        // Pass new input data for next iteration
        // std::swap(this->input_delta_z_buffer, this->output_delta_z_buffer);
        *this->input_delta_z_buffer = *this->output_delta_z_buffer;
    }

    // Parameter update for input layer
    if (this->param_update) {
        this->layers[0]->param_backward(this->layers[0]->bwd_states,
                                        *this->input_delta_z_buffer,
                                        *this->temp_states);
    }

    // // State update for input layer
    // if (this->input_hidden_state_update) {
    //     this->layers[0]->state_backward(
    //         this->layers[0]->bwd_states, *this->input_delta_z_buffer,
    //         *this->output_delta_z_buffer, *this->temp_states);
    // }
}

void LayerStack::step()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->update_weights();
        layer->update_biases();
    }
}

// Utility function to get layer stack info
std::string LayerStack::get_layer_stack_info() const {
    std::stringstream ss;
    for (const auto &layer : this->layers) {
        if (layer) {
            ss << layer->get_layer_info() << "\n";
        }
    }
    return ss.str();
}

void LayerStack::save(const std::string &filename)
/**/
{
    // Extract the directory path from the filename
    std::string directory = filename.substr(0, filename.find_last_of("\\/"));
    create_directory(directory);

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for saving");
    }

    for (const auto &layer : layers) {
        layer->save(file);
    }
    file.close();
}

void LayerStack::load(const std::string &filename)
/**/
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error in file: " + std::string(__FILE__) +
                                 " at line: " + std::to_string(__LINE__) +
                                 ". Failed to open file for loading");
    }

    for (auto &layer : layers) {
        layer->load(file);
    }
    file.close();
}

void LayerStack::save_csv(const std::string &filename)
/*
This allows saving network's parameters in csv so that
    (1) we can test on the previous version
    (2) we have a human-readable of weights and biases
*/
{
    // Extract the directory path from the filename
    std::string directory = filename.substr(0, filename.find_last_of("\\/"));
    create_directory(directory);

    // Initialize the size counters
    size_t total_mu_w_size = 0, total_var_w_size = 0, total_mu_b_size = 0,
           total_var_b_size = 0;

    // Calculate the total size needed for each vector
    for (const auto &layer : this->layers) {
        total_mu_w_size += layer->mu_w.size();
        total_var_w_size += layer->var_w.size();
        total_mu_b_size += layer->mu_b.size();
        total_var_b_size += layer->var_b.size();
    }

    // Allocate data vectors
    std::vector<float> mu_w, var_w, mu_b, var_b;
    mu_w.reserve(total_mu_w_size);
    var_w.reserve(total_var_w_size);
    mu_b.reserve(total_mu_b_size);
    var_b.reserve(total_var_b_size);

    // Concatenate layer parameters
    for (const auto &layer : this->layers) {
        mu_w.insert(mu_w.end(), layer->mu_w.begin(), layer->mu_w.end());
        var_w.insert(var_w.end(), layer->var_w.begin(), layer->var_w.end());
        mu_b.insert(mu_b.end(), layer->mu_b.begin(), layer->mu_b.end());
        var_b.insert(var_b.end(), layer->var_b.begin(), layer->var_b.end());
    }

    // Save parameters to csv
    std::string mu_w_path = filename + "_mu_w.csv";
    std::string var_w_path = filename + "_var_w.csv";
    std::string mu_b_path = filename + "_mu_b.csv";
    std::string var_b_path = filename + "_var_b.csv";

    write_csv(mu_w_path, mu_w);
    write_csv(var_w_path, var_w);
    write_csv(mu_b_path, mu_b);
    write_csv(var_b_path, var_b);
}

void LayerStack::load_csv(const std::string &filename)
/*
 */
{
    // Count number of weights & biases for the entire network
    int num_weights = 0, num_biases = 0;
    for (auto &layer : this->layers) {
        num_weights += layer->mu_w.size();
        num_biases += layer->mu_b.size();
    }

    // Define the global weight & bias vectors
    std::vector<float> mu_w(num_weights);
    std::vector<float> var_w(num_weights);
    std::vector<float> mu_b(num_biases);
    std::vector<float> var_b(num_biases);

    // Read data from csv
    std::string mu_w_path = filename + "_mu_w.csv";
    std::string var_w_path = filename + "_var_w.csv";
    std::string mu_b_path = filename + "_mu_b.csv";
    std::string var_b_path = filename + "_var_b.csv";

    read_csv(mu_w_path, mu_w, 1, false);
    read_csv(var_w_path, var_w, 1, false);
    read_csv(mu_b_path, mu_b, 1, false);
    read_csv(var_b_path, var_b, 1, false);

    // Distribute parameter for each layer
    int weight_start_idx = 0, bias_start_idx = 0;
    for (auto &layer : this->layers) {
        std::copy(mu_w.begin() + weight_start_idx,
                  mu_w.begin() + weight_start_idx + layer->mu_w.size(),
                  layer->mu_w.begin());
        std::copy(var_w.begin() + weight_start_idx,
                  var_w.begin() + weight_start_idx + layer->var_w.size(),
                  layer->var_w.begin());
        std::copy(mu_b.begin() + bias_start_idx,
                  mu_b.begin() + bias_start_idx + layer->mu_b.size(),
                  layer->mu_b.begin());
        std::copy(var_b.begin() + bias_start_idx,
                  var_b.begin() + bias_start_idx + layer->var_b.size(),
                  layer->var_b.begin());

        weight_start_idx += layer->mu_w.size();
        bias_start_idx += layer->mu_b.size();
    }
}
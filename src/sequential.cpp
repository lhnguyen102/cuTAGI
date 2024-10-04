///////////////////////////////////////////////////////////////////////////////
// File:         sequential.cpp
// Description:  ...
// Authors:      Luong-Ha Nguyen & James-A. Goulet
// Created:      October 09, 2023
// Updated:      July 19, 2024
// Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
// License:      This code is released under the MIT License.
////////////////////////////////////////////////////////////////////////////////

#include "../include/sequential.h"

#include "../include/config.h"
#include "../include/conv2d_layer.h"
#include "../include/pooling_layer.h"
// #include "slinear_layer.h"
#ifdef USE_CUDA
#include "../include/base_layer_cuda.cuh"
#endif
#include <memory>

// Sequential::Sequential() {}
Sequential::~Sequential() { this->valid_ = false; }

void Sequential::switch_to_cuda() {
    for (size_t i = 0; i < this->layers.size(); ++i) {
        auto cuda_layer = layers[i]->to_cuda();
        layers[i] = std::move(cuda_layer);
    }
}

void Sequential::to_device(const std::string &new_device) {
    this->device = new_device;
    if (this->device == "cuda") {
        switch_to_cuda();
    }

    // TODO: We should not run this again when switching device
    this->compute_input_output_size();
    this->set_buffer_size();
}
#ifdef USE_CUDA
void Sequential::params_to_host() {
    for (auto &layer : this->layers) {
        auto cuda_layer = dynamic_cast<BaseLayerCuda *>(layer.get());
        if (cuda_layer) {
            cuda_layer->params_to_host();
        }
    }
}

void Sequential::params_to_device() {
    for (auto &layer : this->layers) {
        auto cuda_layer = dynamic_cast<BaseLayerCuda *>(layer.get());
        if (cuda_layer) {
            cuda_layer->params_to_device();
        }
    }
}
#else
void Sequential::params_to_host() {
    // No CUDA support, do nothing
}

void Sequential::params_to_device() {
    // No CUDA support, do nothing
}
#endif

void Sequential::add_layers()
/*
 */
{
    // After variadic template, meanings vector of layers has formed
    if (this->device == "cpu") {
        this->compute_input_output_size();
        this->set_buffer_size();
    }
}

void Sequential::add_layer(std::shared_ptr<BaseLayer> layer)
/*
NOTE: The output buffer size is determinated based on the output size for each
layer assuming that batch size = 1. If the batch size in the forward pass > 1,
it will be corrected at the first run in the forward pass.
 */
{
    // Stack layer
    if (this->device.compare("cpu") == 0) {
        this->layers.push_back(layer);
    } else if (this->device.compare("cuda") == 0) {
        this->layers.push_back(layer->to_cuda());
    } else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void Sequential::set_buffer_size()
/*
 */
{
    for (auto &layer : this->layers) {
        int max_size = layer->get_max_num_states();
        this->z_buffer_size = std::max(max_size, this->z_buffer_size);
    }

    // Convert to the size that is multiple of PACK_SIZE
    if (this->z_buffer_size % PACK_SIZE != 0) {
        this->z_buffer_size =
            ((this->z_buffer_size + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE;
    }
}

void Sequential::compute_input_output_size()
/* TODO: this function is prone to error. Needs to come up with something more
 * robust.
 */
{
    int in_width = this->layers.front()->in_width;
    int in_height = this->layers.front()->in_height;
    int in_depth = this->layers.front()->in_channels;

    for (size_t i = 0; i < this->layers.size(); i++) {
        InitArgs args = InitArgs(in_width, in_height, in_depth);
        this->layers[i]->compute_input_output_size(args);

        // For next iteration
        in_width = this->layers[i]->out_width;
        in_height = this->layers[i]->out_height;
        in_depth = this->layers[i]->out_channels;
    }
}

void Sequential::init_output_state_buffer()
/*
 */
{
    if (this->device.compare("cpu") == 0) {
        if (this->layers[0]->get_layer_type() == LayerType::SLSTM) {
            this->output_z_buffer = std::make_shared<SmoothingHiddenStates>(
                this->z_buffer_size, this->z_buffer_block_size,
                this->num_samples);
            this->input_z_buffer = std::make_shared<SmoothingHiddenStates>(
                this->z_buffer_size, this->z_buffer_block_size,
                this->num_samples);
        } else {
            this->output_z_buffer = std::make_shared<BaseHiddenStates>(
                this->z_buffer_size, this->z_buffer_block_size);
            this->input_z_buffer = std::make_shared<BaseHiddenStates>(
                this->z_buffer_size, this->z_buffer_block_size);
        }
        this->temp_states = std::make_shared<BaseTempStates>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#ifdef USE_CUDA
    else if (this->device.compare("cuda") == 0) {
        if (this->layers[0]->get_layer_type() != LayerType::SLSTM) {
            this->output_z_buffer = std::make_shared<HiddenStateCuda>(
                this->z_buffer_size, this->z_buffer_block_size);
            this->input_z_buffer = std::make_shared<HiddenStateCuda>(
                this->z_buffer_size, this->z_buffer_block_size);
            this->temp_states = std::make_shared<TempStateCuda>(
                this->z_buffer_size, this->z_buffer_block_size);
        } else {
            throw std::invalid_argument(
                "Error in file: " + std::string(__FILE__) +
                " at line: " + std::to_string(__LINE__) +
                ". Smoothing feature does not support CUDA");
        }
    }
#endif
    else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void Sequential::init_delta_state_buffer()
/*
 */
{
    if (this->device.compare("cpu") == 0) {
        this->output_delta_z_buffer = std::make_shared<BaseDeltaStates>(
            this->z_buffer_size, this->z_buffer_block_size);
        this->input_delta_z_buffer = std::make_shared<BaseDeltaStates>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#ifdef USE_CUDA
    else if (this->device.compare("cuda") == 0) {
        this->output_delta_z_buffer = std::make_shared<DeltaStateCuda>(
            this->z_buffer_size, this->z_buffer_block_size);
        this->input_delta_z_buffer = std::make_shared<DeltaStateCuda>(
            this->z_buffer_size, this->z_buffer_block_size);
    }
#endif
    else {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Invalid device: [" + this->device + "]");
    }
}

void Sequential::set_threads(unsigned int num_threads)
/*
 */
{
    this->num_threads = num_threads;
    for (auto &layer : this->layers) {
        layer->set_threads(num_threads);
    }
}

void Sequential::train()
/*
 */
{
    for (auto &layer : this->layers) {
        layer->train();
    }
}

void Sequential::eval()
/*
 */
{
    for (auto &layer : this->layers) {
        layer->eval();
    }
}

std::string Sequential::get_device()
/*
 */
{
    for (auto &layer : this->layers) {
        auto layer_device = layer->get_device();
        if (layer_device != this->device) {
            return layer_device;
        }
    }
    return this->device;
}

void Sequential::forward(const std::vector<float> &mu_x,
                         const std::vector<float> &var_x)
/*
 */
{
    // Batch size
    int batch_size = mu_x.size() / this->layers.front()->get_input_size();

    // Lazy initialization
    if (this->z_buffer_block_size == 0) {
        this->z_buffer_block_size = batch_size;
        this->z_buffer_size = batch_size * this->z_buffer_size;

        init_output_state_buffer();
        if (this->training) {
            init_delta_state_buffer();
        }
    }

    // Reallocate the buffer if batch size changes
    if (batch_size != this->z_buffer_block_size) {
        this->z_buffer_size =
            batch_size * (this->z_buffer_size / this->z_buffer_block_size);
        this->z_buffer_block_size = batch_size;

        this->input_z_buffer->set_size(this->z_buffer_size, batch_size);
        if (this->training) {
            this->input_delta_z_buffer->set_size(this->z_buffer_size,
                                                 batch_size);
            this->output_delta_z_buffer->set_size(this->z_buffer_size,
                                                  batch_size);
        }
    }

    // Merge input data to the input buffer
    this->input_z_buffer->set_input_x(mu_x, var_x, batch_size);

    // Forward pass for all layers
    for (auto &layer : this->layers) {
        auto *current_layer = layer.get();

        current_layer->forward(*this->input_z_buffer, *this->output_z_buffer,
                               *this->temp_states);

        // Swap the pointer holding class
        std::swap(this->input_z_buffer, this->output_z_buffer);
    }

    // Output buffer is considered as the final output of network
    std::swap(this->output_z_buffer, this->input_z_buffer);
}

void Sequential::forward(BaseHiddenStates &input_states)
/*
 */
{
    // Batch size
    int batch_size = input_states.block_size;

    // Only initialize if batch size changes
    if (this->z_buffer_block_size == 0) {
        this->z_buffer_block_size = batch_size;
        this->z_buffer_size = batch_size * this->z_buffer_size;

        init_output_state_buffer();
        if (this->training) {
            init_delta_state_buffer();
        }
    }

    if (batch_size != this->z_buffer_block_size) {
        this->z_buffer_size =
            batch_size * (this->z_buffer_size / this->z_buffer_block_size);
        this->z_buffer_block_size = batch_size;

        this->input_z_buffer->set_size(this->z_buffer_size, batch_size);
        if (this->training) {
            this->input_delta_z_buffer->set_size(this->z_buffer_size,
                                                 batch_size);
            this->output_delta_z_buffer->set_size(this->z_buffer_size,
                                                  batch_size);
        }
    }

    auto *first_layer = this->layers[0].get();
    first_layer->forward(input_states, *this->input_z_buffer,
                         *this->temp_states);

    for (int i = 1; i < this->layers.size(); i++) {
        auto *current_layer = this->layers[i].get();

        current_layer->forward(*this->input_z_buffer, *this->output_z_buffer,
                               *this->temp_states);

        std::swap(this->input_z_buffer, this->output_z_buffer);
    }
    // Output buffer is considered as the final output of network
    std::swap(this->output_z_buffer, this->input_z_buffer);
}

void Sequential::backward()
/*
 */
{
    // Hidden layers
    for (auto layer = this->layers.rbegin(); layer != this->layers.rend() - 1;
         ++layer) {
        auto *current_layer = layer->get();

        // Backward pass for hidden states
        current_layer->backward(*this->input_delta_z_buffer,
                                *this->output_delta_z_buffer,
                                *this->temp_states);

        // Pass new input data for next iteration
        if (current_layer->get_layer_type() != LayerType::Activation) {
            std::swap(this->input_delta_z_buffer, this->output_delta_z_buffer);
        }
    }

    // State update for input layer
    this->layers[0]->backward(*this->input_delta_z_buffer,
                              *this->output_delta_z_buffer, *this->temp_states,
                              this->input_state_update);
}

std::tuple<std::vector<float>, std::vector<float>> Sequential::smoother()
/*
 */
{
    std::vector<float> mu_zo_smooths, var_zo_smooths;
    // Hidden layers
    for (auto layer = this->layers.begin(); layer != this->layers.end();
         layer++) {
        auto *current_layer = layer->get();
        if (current_layer->get_layer_type() == LayerType::SLSTM) {
            auto *slstm_layer = dynamic_cast<SLSTM *>(current_layer);
            slstm_layer->smoother();
        } else if (current_layer->get_layer_type() == LayerType::SLinear) {
            auto *slinear_layer = dynamic_cast<SLinear *>(current_layer);
            slinear_layer->smoother();
            mu_zo_smooths = slinear_layer->smooth_states.mu_zo_smooths;
            var_zo_smooths = slinear_layer->smooth_states.var_zo_smooths;
        }
    }
    return std::make_tuple(mu_zo_smooths, var_zo_smooths);
}

void Sequential::step()
/*
 */
{
    for (const auto &layer : this->layers) {
        layer->update_weights();
        layer->update_biases();
    }
}

void Sequential::output_to_host() {
#ifdef USE_CUDA
    if (this->device.compare("cuda") == 0) {
        HiddenStateCuda *cu_output_states =
            dynamic_cast<HiddenStateCuda *>(this->output_z_buffer.get());
        cu_output_states->to_host();
    }
#endif
}

void Sequential::delta_z_to_host() {
#ifdef USE_CUDA
    if (this->device.compare("cuda") == 0) {
        DeltaStateCuda *cu_input_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->input_delta_z_buffer.get());
        DeltaStateCuda *cu_output_delta_z =
            dynamic_cast<DeltaStateCuda *>(this->output_delta_z_buffer.get());

        cu_input_delta_z->to_host();
        cu_output_delta_z->to_host();
    }
#endif
}

// Utility function to get layer stack info
std::string Sequential::get_layer_stack_info() const {
    std::stringstream ss;
    for (const auto &layer : this->layers) {
        if (layer) {
            ss << layer->get_layer_info() << "\n";
        }
    }
    return ss.str();
}

void Sequential::preinit_layer() {
    for (const auto &layer : this->layers) {
        if (layer) {
            layer->preinit_layer();
        }
    }
}

void Sequential::save(const std::string &filename)
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

void Sequential::load(const std::string &filename)
/**/
{
    // Precalculate layer's properties e.g., number of parameres to load the
    // saved model
    this->preinit_layer();

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

void Sequential::save_csv(const std::string &filename)
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
    std::string mu_w_path = filename + "_1_mw.csv";
    std::string var_w_path = filename + "_2_Sw.csv";
    std::string mu_b_path = filename + "_3_mb.csv";
    std::string var_b_path = filename + "_4_Sb.csv";

    write_csv(mu_w_path, mu_w);
    write_csv(var_w_path, var_w);
    write_csv(mu_b_path, mu_b);
    write_csv(var_b_path, var_b);
}

void Sequential::load_csv(const std::string &filename)
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
    std::string mu_w_path = filename + "_1_mw.csv";
    std::string var_w_path = filename + "_2_Sw.csv";
    std::string mu_b_path = filename + "_3_mb.csv";
    std::string var_b_path = filename + "_4_Sb.csv";

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

std::vector<std::reference_wrapper<std::vector<float>>> Sequential::parameters()
/*Stored mu_w, var_w, mu_b, var_b in a vector of reference_wrapper
Example: A model of 5 layers leads to a params size of 5 * 4 = 20
 */
{
    if (!this->valid_) {
        throw std::runtime_error("Sequential object is no longer valid");
    }
    std::vector<std::reference_wrapper<std::vector<float>>> params;
    for (auto &layer : layers) {
        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            params.push_back(layer->mu_w);
            params.push_back(layer->var_w);
            params.push_back(layer->mu_b);
            params.push_back(layer->var_b);
        }
    }
    return params;
}

std::map<std::string, std::tuple<std::vector<float>, std::vector<float>,
                                 std::vector<float>, std::vector<float>>>
Sequential::get_state_dict()
/*
 */
{
    // Send the parameters to the host. TODO: for further speedup, we must to
    // avoid this step. One could use the copy data in gpu memory directly or
    // unified memory
    if (this->device.compare("cuda") == 0) {
        this->params_to_host();
    }

    std::map<std::string, std::tuple<std::vector<float>, std::vector<float>,
                                     std::vector<float>, std::vector<float>>>
        state_dict;

    for (size_t i = 0; i < this->layers.size(); ++i) {
        const auto &layer = this->layers[i];
        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            std::string layer_name =
                layer->get_layer_info() + "_" + std::to_string(i);
            state_dict[layer_name] = std::make_tuple(layer->mu_w, layer->var_w,
                                                     layer->mu_b, layer->var_b);
        }
    }
    return state_dict;
}

void Sequential::load_state_dict(
    const std::map<std::string,
                   std::tuple<std::vector<float>, std::vector<float>,
                              std::vector<float>, std::vector<float>>>
        &state_dict)
/*
 */
{
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto &layer = this->layers[i];

        if (layer->get_layer_type() != LayerType::Activation &&
            layer->get_layer_type() != LayerType::Pool2d) {
            std::string layer_name =
                layer->get_layer_info() + "_" + std::to_string(i);

            auto it = state_dict.find(layer_name);
            if (it == state_dict.end()) {
                throw std::runtime_error(
                    "Error in file: " + std::string(__FILE__) +
                    " at line: " + std::to_string(__LINE__) +
                    "Missing parameters for " + layer_name);
            }

            const auto &layer_params = it->second;

            // Check if the sizes match
            if (layer->mu_w.size() == std::get<0>(layer_params).size() &&
                layer->var_w.size() == std::get<1>(layer_params).size() &&
                layer->mu_b.size() == std::get<2>(layer_params).size() &&
                layer->var_b.size() == std::get<3>(layer_params).size()) {
                // Update layer parameters
                layer->mu_w = std::get<0>(layer_params);
                layer->var_w = std::get<1>(layer_params);
                layer->mu_b = std::get<2>(layer_params);
                layer->var_b = std::get<3>(layer_params);
            } else {
                throw std::runtime_error(
                    "Error in file: " + std::string(__FILE__) +
                    " at line: " + std::to_string(__LINE__) +
                    "Mismatch in layer sizes for " + layer_name);
            }
        }
    }

    // Send the parameters to the device. TODO: for further speedup, we must to
    // avoid this step. One could use the copy data in gpu memory directly or
    // unified memory
    if (this->device.compare("cuda") == 0) {
        this->params_to_device();
    }
}

void Sequential::params_from(const Sequential &model_ref) {
    if (this->layers.size() != model_ref.layers.size()) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". Model architecture is different");
    }

    // TODO: need to add more checks before copying
    for (int i = 0; i < this->layers.size(); i++) {
        if (this->layers[i]->mu_w.size() == 0) {
            this->layers[i]->mu_w.resize(model_ref.layers[i]->mu_w.size());
            this->layers[i]->var_w.resize(model_ref.layers[i]->var_w.size());
            this->layers[i]->mu_b.resize(model_ref.layers[i]->mu_b.size());
            this->layers[i]->var_b.resize(model_ref.layers[i]->var_b.size());
        }
        this->layers[i]->num_weights = model_ref.layers[i]->num_weights;
        this->layers[i]->num_biases = model_ref.layers[i]->num_biases;

        this->layers[i]->mu_w = model_ref.layers[i]->mu_w;
        this->layers[i]->var_w = model_ref.layers[i]->var_w;
        this->layers[i]->mu_b = model_ref.layers[i]->mu_b;
        this->layers[i]->var_b = model_ref.layers[i]->var_b;
    }
}

// Python Wrapper
void Sequential::forward_py(pybind11::array_t<float> mu_a_np,
                            pybind11::array_t<float> var_a_np)
/*
 */
{
    // Get pointers to the data in the arrays
    auto mu_a_buf = mu_a_np.request();
    float *mu_a_ptr = static_cast<float *>(mu_a_buf.ptr);
    std::vector<float> mu_a(mu_a_ptr, mu_a_ptr + mu_a_buf.size);

    if (!var_a_np.is_none()) {
        auto var_a_buf = var_a_np.request();
        float *var_a_ptr = static_cast<float *>(var_a_buf.ptr);
        std::vector<float> var_a(var_a_ptr, var_a_ptr + var_a_buf.size);
        this->forward(mu_a, var_a);
    } else {
        this->forward(mu_a);
    }
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
Sequential::get_outputs()
/*
 */
{
    if (this->device.compare("cuda") == 0) {
        this->output_to_host();
    }
    int batch_size = this->output_z_buffer->block_size;
    int num_outputs = this->layers.back()->output_size;
    std::vector<float> mu_a_output(batch_size * num_outputs);
    std::vector<float> var_a_output(batch_size * num_outputs);

    for (int j = 0; j < batch_size * num_outputs; j++) {
        mu_a_output[j] = this->output_z_buffer->mu_a[j];
        var_a_output[j] = this->output_z_buffer->var_a[j];
    }
    auto py_m_pred =
        pybind11::array_t<float>(mu_a_output.size(), mu_a_output.data());
    auto py_v_pred =
        pybind11::array_t<float>(var_a_output.size(), var_a_output.data());

    return {py_m_pred, py_v_pred};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
Sequential::get_outputs_smoother()
/*
 */
{
    auto last_layer = dynamic_cast<SLinear *>(this->layers.back().get());
    auto py_mu_zo_smooths = pybind11::array_t<float>(
        last_layer->smooth_states.mu_zo_smooths.size(),
        last_layer->smooth_states.mu_zo_smooths.data());

    auto py_var_zo_smooths = pybind11::array_t<float>(
        last_layer->smooth_states.var_zo_smooths.size(),
        last_layer->smooth_states.var_zo_smooths.data());

    return {py_mu_zo_smooths, py_var_zo_smooths};
}

std::tuple<pybind11::array_t<float>, pybind11::array_t<float>>
Sequential::get_input_states()
{
    // Check if input_state_update is enabled
    if (!this->input_state_update) {
        throw std::invalid_argument("Error in file: " + std::string(__FILE__) +
                                    " at line: " + std::to_string(__LINE__) +
                                    ". input_state_update is set to False");
    }

    // Define the slice input states size
    const size_t end_index = this->layers.front()->get_input_size() * this->input_z_buffer->block_size;

    // Slice delta_mu and delta_var
    std::vector<float> delta_mu_slice(
        this->output_delta_z_buffer->delta_mu.begin(),
        this->output_delta_z_buffer->delta_mu.begin() + end_index
    );

    std::vector<float> delta_var_slice(
        this->output_delta_z_buffer->delta_var.begin(),
        this->output_delta_z_buffer->delta_var.begin() + end_index
    );

    // Return the slices as pybind11::array_t
    auto py_delta_mu = pybind11::array_t<float>(
        delta_mu_slice.size(),
        delta_mu_slice.data());
    auto py_delta_var = pybind11::array_t<float>(
        delta_var_slice.size(),
        delta_var_slice.data());

    return {py_delta_mu, py_delta_var};
}


#include "../include/base_layer.h"

#include <cmath>

#include "../include/custom_logger.h"

InitArgs::InitArgs(size_t width, size_t height, size_t depth, int batch_size)
    : width(width), height(height), depth(depth), batch_size(batch_size) {}

BaseLayer::BaseLayer() {
    if (this->training) {
        this->bwd_states = std::make_unique<BaseBackwardStates>();
    }
}

const char *BaseLayer::get_layer_type_name() const {
    return typeid(*this).name();
}

std::string BaseLayer::get_layer_info() const { return "Base()"; }

std::string BaseLayer::get_layer_name() const { return "Base"; }

LayerType BaseLayer::get_layer_type() const { return LayerType::Base; };

int BaseLayer::get_input_size() { return static_cast<int>(this->input_size); }

int BaseLayer::get_output_size() { return static_cast<int>(this->output_size); }

int BaseLayer::get_max_num_states() {
    size_t max_size = std::max(this->output_size, this->input_size);
    return static_cast<int>(max_size);
}

std::string BaseLayer::get_device()
/*
 */
{
    return this->device + ":" + std::to_string(this->device_idx);
}

void BaseLayer::init_weight_bias() {}

void BaseLayer::forward(BaseHiddenStates &input_states,
                        BaseHiddenStates &output_states,
                        BaseTempStates &temp_states) {}

void BaseLayer::backward(BaseDeltaStates &input_delta_states,
                         BaseDeltaStates &output_delta_states,
                         BaseTempStates &temp_states, bool state_udapte) {}

// void BaseLayer::smoother(BaseTempStates &temp_states) {}

void BaseLayer::allocate_param_delta()
/*
 */
{
    this->delta_mu_w.resize(this->num_weights, 0.0f);
    this->delta_var_w.resize(this->num_weights, 0.0f);
    this->delta_mu_b.resize(this->num_biases, 0.0f);
    this->delta_var_b.resize(this->num_biases, 0.0f);
}

void BaseLayer::allocate_bwd_vector(int new_size)
/*
 */
{
    if (new_size <= 0) {
        LOG(LogLevel::ERROR, "Invalid size: " + std::to_string(new_size));
    }
    this->bwd_states->set_size(new_size);
}

void BaseLayer::fill_output_states(BaseHiddenStates &output_states)
/**/
{
    for (int j = 0; j < output_states.actual_size * output_states.block_size;
         j++) {
        output_states.jcb[j] = 1.0f;
    }
}

void BaseLayer::fill_bwd_vector(BaseHiddenStates &input_states)
/*
 */
{
    for (int i = 0; i < input_states.actual_size * input_states.block_size;
         i++) {
        this->bwd_states->mu_a[i] = input_states.mu_a[i];
        this->bwd_states->jcb[i] = input_states.jcb[i];
    }
}

void BaseLayer::raw_update_weights()
/*
 */
{
    for (int i = 0; i < this->mu_w.size(); i++) {
        this->mu_w[i] += this->delta_mu_w[i];
        this->var_w[i] += this->delta_var_w[i];
    }
}

void BaseLayer::raw_update_biases()
/*
 */
{
    if (this->bias) {
        for (int i = 0; i < this->mu_b.size(); i++) {
            this->mu_b[i] += this->delta_mu_b[i];
            this->var_b[i] += this->delta_var_b[i];
        }
    }
}

void BaseLayer::update_weights()
/*
 */
{
    this->neg_var_w_counter = 0;
    for (int i = 0; i < this->mu_w.size(); i++) {
        float delta_mu_sign =
            (this->delta_mu_w[i] > 0) - (this->delta_mu_w[i] < 0);
        float delta_var_sign =
            (this->delta_var_w[i] > 0) - (this->delta_var_w[i] < 0);
        float delta_bar = powf(this->var_w[i], 0.5) / this->cap_factor_update;

        this->mu_w[i] +=
            delta_mu_sign * std::min(std::abs(delta_mu_w[i]), delta_bar);
        this->var_w[i] +=
            delta_var_sign * std::min(std::abs(delta_var_w[i]), delta_bar);
        if (var_w[i] <= 0.0f) {
            var_w[i] = 1E-5f;  // TODO: replace by a parameter
            this->neg_var_w_counter++;
        }
    }
}

void BaseLayer::update_biases()
/*

 */
{
    if (this->bias) {
        for (int i = 0; i < this->mu_b.size(); i++) {
            float delta_mu_sign =
                (delta_mu_b[i] > 0) - (this->delta_mu_b[i] < 0);
            float delta_var_sign = (delta_var_b[i] > 0) - (delta_var_b[i] < 0);
            float delta_bar =
                powf(this->var_b[i], 0.5) / this->cap_factor_update;

            this->mu_b[i] += delta_mu_sign *
                             std::min(std::abs(this->delta_mu_b[i]), delta_bar);
            this->var_b[i] +=
                delta_var_sign *
                std::min(std::abs(this->delta_var_b[i]), delta_bar);
            if (var_b[i] <= 0.0f) {
                var_b[i] = 1E-5f;  // TODO: replace by a parameter
            }
        }
    }
}

void BaseLayer::set_cap_factor_udapte(int batch_size)
/*Get the cap factor w.r.t the bacth size.
We define a cap factor for regularizing the updating quantities of the
parameters when the batch size is large. NOTE: the current values of the cap
factor is based on what we have tested in practice.

Args:
    batch_size: Size of minibatches

Returns:
    cap_factor: Cap factor
*/
{
    // TODO: Heuristic values!!
    if (batch_size == 1) {
        this->cap_factor_update = 0.1f;
    }
    if (batch_size > 1 && batch_size < 256) {
        this->cap_factor_update = 2.0f;
    }
    if (batch_size >= 256) {
        this->cap_factor_update = 3.0f;
    }
}

void BaseLayer::set_threads(int num)
/**/
{
    this->num_threads = num;
}

void BaseLayer::compute_input_output_size(const InitArgs &args)
/*
 */
{
    // TODO: find a way to add specialized paramters for each layer
    this->in_width = args.width;
    this->in_height = args.height;
    this->in_channels = args.depth;

    this->out_width = args.width;
    this->out_height = args.height;
    this->out_channels = args.depth;
}

void BaseLayer::train() { this->training = true; }
void BaseLayer::eval() { this->training = false; }

void BaseLayer::storing_states_for_training(BaseHiddenStates &input_states,
                                            BaseHiddenStates &output_states)
/*
 */
{
    int act_size = input_states.actual_size * input_states.block_size;
    if (this->bwd_states->size != act_size) {
        this->allocate_bwd_vector(act_size);
    }

    // Activation's jacobian and mean from the previous layer
    this->fill_bwd_vector(input_states);

    // Send a copy of activation's mean and variance to the output buffer
    // for the current layer.
    this->fill_output_states(output_states);
}

void BaseLayer::save(std::ofstream &file)
/*
 */
{
    if (!file.is_open()) {
        LOG(LogLevel::ERROR, "Failed to open file for saving");
    }

    // Save the name length and name
    // TODO remove get_layer_name because cuda and cpu version
    auto layer_name = this->get_layer_info();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    for (const auto &m_w : this->mu_w) {
        file.write(reinterpret_cast<const char *>(&m_w), sizeof(m_w));
    }
    for (const auto &v_w : this->var_w) {
        file.write(reinterpret_cast<const char *>(&v_w), sizeof(v_w));
    }
    for (const auto &m_b : this->mu_b) {
        file.write(reinterpret_cast<const char *>(&m_b), sizeof(m_b));
    }
    for (const auto &v_b : this->var_b) {
        file.write(reinterpret_cast<const char *>(&v_b), sizeof(v_b));
    }
}

void BaseLayer::load(std::ifstream &file)
/*
 */
{
    if (!file.is_open()) {
        LOG(LogLevel::ERROR, "Failed to open file for loading");
    }
    // Load the name length and name
    auto layer_name = this->get_layer_info();
    std::string loaded_name;
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));
    loaded_name.resize(name_length);
    file.read(&loaded_name[0], name_length);

    // Check layer name
    if (layer_name != loaded_name) {
        LOG(LogLevel::ERROR, "Layer name are not match. Expected: " +
                                 layer_name + ", Found: " + loaded_name);
    }

    for (auto &m_w : this->mu_w) {
        file.read(reinterpret_cast<char *>(&m_w), sizeof(m_w));
    }
    for (auto &v_w : this->var_w) {
        file.read(reinterpret_cast<char *>(&v_w), sizeof(v_w));
    }
    for (auto &m_b : this->mu_b) {
        file.read(reinterpret_cast<char *>(&m_b), sizeof(m_b));
    }
    for (auto &v_b : this->var_b) {
        file.read(reinterpret_cast<char *>(&v_b), sizeof(v_b));
    }
    this->num_weights = this->mu_w.size();
    this->num_biases = this->mu_b.size();
    if (this->training) {
        this->allocate_param_delta();
    }
}

ParameterMap BaseLayer::get_parameters_as_map(std::string suffix) {
    std::string key = this->get_layer_name();
    if (!suffix.empty()) {
        key += "." + suffix;
    }

    ParameterTuple parameters =
        std::make_tuple(this->mu_w, this->var_w, this->mu_b, this->var_b);

    return {{key, parameters}};
}

void BaseLayer::load_parameters_from_map(const ParameterMap &param_map,
                                         const std::string &suffix) {
    // Generate the key for this layer
    std::string key = this->get_layer_name();
    if (!suffix.empty()) {
        key += "." + suffix;
    }

    // Find the key in the provided map
    auto it = param_map.find(key);
    if (it == param_map.end()) {
        LOG(LogLevel::ERROR, "Key " + key + " not found in parameter map.");
    }

    // Extract the parameters from the map
    // print key
    const auto &params = it->second;
    if (!std::is_same<std::decay_t<decltype(params)>, ParameterTuple>::value) {
        LOG(LogLevel::ERROR, "Parameter tuple for key " + key +
                                 " must contain exactly 4 vectors.");
    }

    this->mu_w = std::get<0>(params);
    this->var_w = std::get<1>(params);
    this->mu_b = std::get<2>(params);
    this->var_b = std::get<3>(params);
}

std::vector<ParameterTuple> BaseLayer::parameters() {
    return {{this->mu_w, this->var_w, this->mu_b, this->var_b}};
}

std::tuple<std::vector<float>, std::vector<float>>
BaseLayer::get_running_mean_var()
/*
 */
{
    return {std::vector<float>(), std::vector<float>()};
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>,
           std::vector<std::vector<float>>, std::vector<std::vector<float>>>
BaseLayer::get_norm_mean_var()
/*
 */
{
    return {
        std::vector<std::vector<float>>(), std::vector<std::vector<float>>(),
        std::vector<std::vector<float>>(), std::vector<std::vector<float>>()};
}

void BaseLayer::preinit_layer()
/* Pre-initialize the layer property e.g., number of weights & biases
 */
{
    // We do nothing by default
}

int BaseLayer::get_neg_var_w_counter() { return this->neg_var_w_counter; }
